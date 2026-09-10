from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from services.asr.config import AsrSettings
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
    async def test_transcribe_returns_legacy_compatible_payload(self) -> None:
        created: list[FakeWhisperModel] = []

        def factory(*args, **kwargs):
            model = FakeWhisperModel(*args, **kwargs)
            created.append(model)
            return model

        with patch.dict(os.environ, {"ASR_MODEL": "tiny", "ASR_LANGUAGE": "zh"}, clear=True):
            settings = AsrSettings.from_env()
        recognizer = SpeechRecognizer(settings, model_factory=factory)
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
        self.assertIn("智能眼镜语音助手", created[0].transcribe_kwargs["initial_prompt"])
        self.assertEqual("橙汁 葡萄汁 可乐 盲盒 拉布布", created[0].transcribe_kwargs["hotwords"])
        self.assertTrue(recognizer.health()["loaded"])
