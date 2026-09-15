from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from services.asr.config import AsrSettings
from services.asr.main import _json_response, _public_intent
from services.asr.service import SpeechRecognizer


class FakeWhisperModel:
    def __init__(self, *_args, **kwargs) -> None:
        self.load_kwargs = kwargs
        self.transcribe_kwargs = None

    def transcribe(self, _path: str, **kwargs):
        self.transcribe_kwargs = kwargs
        return (
            iter(
                [
                    SimpleNamespace(start=0.0, end=0.8, text=" 帮我找"),
                    SimpleNamespace(start=0.8, end=1.5, text="黄色的狗 "),
                ]
            ),
            SimpleNamespace(language="zh", language_probability=0.99, duration=1.5),
        )


class SpeechRecognizerTest(IsolatedAsyncioTestCase):
    async def test_json_response_keeps_chinese_readable(self) -> None:
        response = _json_response({"text": "派机器狗巡逻园区内A区域。"})
        self.assertIn('"text": "派机器狗巡逻园区内A区域。"', response.text)
        self.assertNotIn("\\u", response.text)

    async def test_public_patrol_intent_uses_business_slots(self) -> None:
        self.assertEqual(
            {
                "executor": "robot dog",
                "intent": "security patrol",
                "area": "A",
                "matched": True,
                "backend": "rules",
            },
            _public_intent("patrol", "A区域", "rules"),
        )

    async def test_transcribe_returns_legacy_compatible_payload(self) -> None:
        created: list[FakeWhisperModel] = []

        def factory(*args, **kwargs):
            model = FakeWhisperModel(*args, **kwargs)
            created.append(model)
            return model

        with patch.dict(os.environ, {"ASR_MODEL": "tiny", "ASR_LANGUAGE": "zh"}, clear=True):
            settings = AsrSettings.from_env()
        recognizer = SpeechRecognizer(settings, model_factory=factory)
        with self.assertLogs("sandbox.asr", level="INFO") as captured:
            with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
                result = await recognizer.transcribe(
                    Path(audio.name),
                    language=None,
                    session_id="session-1",
                    task_id="task-1",
                    source="test",
                    stop_reason=None,
                    original_filename="speech.wav",
                )

        self.assertEqual("帮我找 黄色的狗", result["text"])
        self.assertEqual(1500, result["durationMs"])
        self.assertEqual("speech.wav", result["audioFilename"])
        self.assertEqual("zh", created[0].transcribe_kwargs["language"])
        self.assertIn("园区巡逻", created[0].transcribe_kwargs["initial_prompt"])
        self.assertIn("机器狗", created[0].transcribe_kwargs["hotwords"])
        self.assertIn("向左", created[0].transcribe_kwargs["hotwords"])
        self.assertTrue(recognizer.health()["loaded"])
        health = recognizer.health()
        self.assertTrue(health["ready"])
        self.assertEqual("ready", health["status"])
        self.assertEqual("tiny", health["modelName"])
        self.assertEqual(result, health["latestTranscript"])
        self.assertIn("session_id=session-1", captured.output[0])
        self.assertIn('text="帮我找 黄色的狗"', captured.output[0])
