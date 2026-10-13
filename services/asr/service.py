from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any
import uuid

from .config import (
    AsrSettings,
    _DEFAULT_INITIAL_PROMPT_EN,
    _DEFAULT_INITIAL_PROMPT_ZH,
    _DEFAULT_HOTWORDS_EN,
    _DEFAULT_HOTWORDS_ZH,
)


LOGGER = logging.getLogger("sandbox.asr")


class SpeechRecognizer:
    """启动时加载并预热 faster-whisper，限制并发 GPU 推理数。"""

    def __init__(
        self,
        settings: AsrSettings,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self._model_factory = model_factory
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._warmup_lock = threading.Lock()
        self._semaphore = asyncio.Semaphore(settings.concurrency)
        self.loaded = False
        self.warmed_up = False
        self.warmup_ms: int | None = None
        self.last_error: str | None = None
        self.latest_transcript: dict[str, Any] | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model
            factory = self._model_factory
            if factory is None:
                from faster_whisper import WhisperModel

                factory = WhisperModel
            try:
                self._model = factory(
                    self.settings.model,
                    device=self.settings.device,
                    compute_type=self.settings.compute_type,
                    download_root=self.settings.download_root,
                )
            except Exception as exc:
                self.last_error = str(exc)
                raise
            self.loaded = True
            self.last_error = None
            return self._model

    async def warmup(self) -> None:
        if not self.settings.enabled:
            return
        async with self._semaphore:
            if not self.warmed_up:
                await asyncio.to_thread(self._warmup_sync)

    def _warmup_sync(self) -> None:
        import numpy as np

        with self._warmup_lock:
            if self.warmed_up:
                return
            started = time.perf_counter()
            try:
                model = self._load_model()
                audio = np.zeros(16000, dtype=np.float32)
                # VAD would discard silence before Whisper sees it. Bypass it for
                # the GPU pass and consume the lazy iterator to run the decoder.
                segments, _ = model.transcribe(
                    audio,
                    language=self.settings.language,
                    task="transcribe",
                    beam_size=self.settings.beam_size,
                    vad_filter=False,
                    initial_prompt=None,
                    hotwords=None,
                    temperature=0.0,
                    max_new_tokens=16,
                )
                list(segments)
                # Also initialize Silero VAD, which real requests use. Its silent
                # output is discarded and never becomes a business transcript.
                segments, _ = model.transcribe(
                    audio, language=self.settings.language, task="transcribe",
                    beam_size=self.settings.beam_size, vad_filter=True,
                    initial_prompt=None, hotwords=None,
                    temperature=0.0, max_new_tokens=16,
                )
                list(segments)
            except Exception as exc:
                self.last_error = str(exc)
                raise
            self.warmup_ms = int((time.perf_counter() - started) * 1000)
            self.warmed_up = True
            self.last_error = None

    async def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None,
        session_id: str,
        task_id: str,
        source: str,
        stop_reason: str | None,
        original_filename: str,
    ) -> dict[str, Any]:
        if not self.settings.enabled:
            raise RuntimeError("ASR service is disabled")
        await self.warmup()
        async with self._semaphore:
            return await asyncio.to_thread(
                self._transcribe_sync,
                audio_path,
                language=language,
                session_id=session_id,
                task_id=task_id,
                source=source,
                stop_reason=stop_reason,
                original_filename=original_filename,
            )

    def _transcribe_sync(
        self,
        audio_path: Path,
        *,
        language: str | None,
        session_id: str,
        task_id: str,
        source: str,
        stop_reason: str | None,
        original_filename: str,
    ) -> dict[str, Any]:
        model = self._load_model()
        started = time.perf_counter()
        created_at_ms = int(time.time() * 1000)
        effective_language = (language or self.settings.language or "").strip().lower()
        initial_prompt, hotwords = _recognition_hints(
            effective_language,
            self.settings.initial_prompt,
            self.settings.hotwords,
        )
        try:
            segment_iter, info = model.transcribe(
                str(audio_path),
                beam_size=self.settings.beam_size,
                language=effective_language or None,
                task="transcribe",
                vad_filter=True,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
            )
            raw_segments = list(segment_iter)
        except Exception as exc:
            self.last_error = str(exc)
            raise

        segments = [
            {
                "startSec": float(segment.start),
                "endSec": float(segment.end),
                "text": str(segment.text).strip(),
            }
            for segment in raw_segments
            if str(segment.text).strip()
        ]
        text = _collapse_repeated_text(" ".join(item["text"] for item in segments).strip())
        duration_s = float(getattr(info, "duration", 0.0) or 0.0)
        if not duration_s and segments:
            duration_s = max(item["endSec"] for item in segments)
        self.last_error = None
        result = {
            "transcriptId": uuid.uuid4().hex,
            "sessionId": session_id,
            "taskId": task_id,
            "source": source,
            "text": text,
            "language": getattr(info, "language", language),
            "languageProbability": getattr(info, "language_probability", None),
            "durationMs": int(duration_s * 1000),
            "processingMs": int((time.perf_counter() - started) * 1000),
            "createdAtMs": created_at_ms,
            "stopReason": stop_reason,
            "segments": segments,
            "audioFilename": original_filename,
        }
        self.latest_transcript = result
        LOGGER.info(
            "transcription completed session_id=%s task_id=%s source=%s "
            "language=%s duration_ms=%s processing_ms=%s text=%s",
            session_id,
            task_id,
            source,
            result["language"],
            result["durationMs"],
            result["processingMs"],
            json.dumps(text, ensure_ascii=False),
        )
        return result

    def health(self) -> dict[str, Any]:
        ready = self.settings.enabled and self.warmed_up and self.last_error is None
        if not self.settings.enabled:
            status = "disabled"
        elif self.last_error is not None:
            status = "error"
        elif not self.warmed_up:
            status = "warming_up"
        else:
            status = "ready"
        return {
            "ok": ready,
            "ready": ready,
            "status": status,
            "service": "asr",
            "enabled": self.settings.enabled,
            "loaded": self.loaded,
            "warmedUp": self.warmed_up,
            "warmupMs": self.warmup_ms,
            "model": self.settings.model,
            "modelName": self.settings.model,
            "device": self.settings.device,
            "computeType": self.settings.compute_type,
            "lastError": self.last_error,
            "latestTranscript": self.latest_transcript,
        }


def _recognition_hints(
    language: str,
    initial_prompt: str | None,
    hotwords: str | None,
) -> tuple[str | None, str | None]:
    """Avoid feeding a bilingual prompt into a single-language decode."""
    if language.startswith("en"):
        prompt = _DEFAULT_INITIAL_PROMPT_EN if _contains_cjk(initial_prompt) else initial_prompt
        words = _DEFAULT_HOTWORDS_EN if _contains_cjk(hotwords) else hotwords
        return prompt, words
    if language.startswith("zh"):
        prompt = _DEFAULT_INITIAL_PROMPT_ZH if _contains_latin(initial_prompt) else initial_prompt
        words = _DEFAULT_HOTWORDS_ZH if _contains_latin(hotwords) else hotwords
        return prompt, words
    # In auto-detect mode hints can bias Whisper into emitting the prompt's
    # other language, so leave both optional hints out.
    return None, None


def _contains_cjk(value: str | None) -> bool:
    return bool(value) and any("\u3400" <= char <= "\u9fff" for char in value)


def _contains_latin(value: str | None) -> bool:
    return bool(value) and any(("a" <= char.lower() <= "z") for char in value)


def _collapse_repeated_text(text: str) -> str:
    """Collapse only an exact two-copy transcript; preserve real repetition."""
    if not text:
        return text
    compact = " ".join(text.split())
    if len(compact) % 2 == 0:
        midpoint = len(compact) // 2
        if compact[:midpoint] == compact[midpoint:]:
            return compact[:midpoint].strip()
    parts = compact.split(" ")
    if len(parts) % 2 == 0:
        midpoint = len(parts) // 2
        if parts[:midpoint] == parts[midpoint:]:
            return " ".join(parts[:midpoint]).strip()
    return compact
