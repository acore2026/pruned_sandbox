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

from .config import AsrSettings


LOGGER = logging.getLogger("sandbox.asr")


class SpeechRecognizer:
    """惰性加载 faster-whisper，并限制并发 GPU 推理数。"""

    def __init__(
        self,
        settings: AsrSettings,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self._model_factory = model_factory
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._semaphore = asyncio.Semaphore(settings.concurrency)
        self.loaded = False
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
        try:
            segment_iter, info = model.transcribe(
                str(audio_path),
                beam_size=self.settings.beam_size,
                language=language or self.settings.language,
                task="transcribe",
                vad_filter=True,
                initial_prompt=self.settings.initial_prompt,
                hotwords=self.settings.hotwords,
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
        text = " ".join(item["text"] for item in segments).strip()
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
        ready = self.settings.enabled and self.last_error is None
        if not self.settings.enabled:
            status = "disabled"
        elif self.last_error is not None:
            status = "error"
        else:
            status = "ready"
        return {
            "ok": ready,
            "ready": ready,
            "status": status,
            "service": "asr",
            "enabled": self.settings.enabled,
            "loaded": self.loaded,
            "model": self.settings.model,
            "modelName": self.settings.model,
            "device": self.settings.device,
            "computeType": self.settings.compute_type,
            "lastError": self.last_error,
            "latestTranscript": self.latest_transcript,
        }
