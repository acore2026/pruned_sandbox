from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

from services.asr.config import AsrSettings
from services.asr.main import _discovery_skills, _json_response, _public_intent, create_app
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
                "type": "TASK",
                "parameters": {"area": "A"},
            },
            _public_intent("patrol", "A区域", "discovery"),
        )

    async def test_discovery_intent_uses_scene_document_mappings(self) -> None:
        self.assertEqual(
            {
                "type": "VIDEO_TASK",
                "parameters": {},
            },
            _public_intent("video_task", "", "discovery"),
        )
        self.assertEqual(
            {
                "type": "OBJECT_RECOGNITION",
                "parameters": {},
            },
            _public_intent("object_recognition", "", "discovery"),
        )
        self.assertEqual(
            ["patrol", "camera"],
            _discovery_skills({"type": "TASK"}),
        )
        self.assertEqual(
            ["patrol", "camera"],
            _discovery_skills({"type": "VIDEO_TASK"}),
        )
        self.assertEqual(
            ["camera"],
            _discovery_skills({"type": "OBJECT_RECOGNITION"}),
        )

    async def test_runtime_public_intent_unifies_defense_commands(self) -> None:
        self.assertEqual(
            {
                "executor": "robot dog",
                "intent": "movement",
                "direction": "forward",
                "matched": True,
                "backend": "rules",
            },
            _public_intent("defense", "suspect", "runtime"),
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


class AsrPublicApiTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ASR_ENABLED": "true",
                "ASR_MODEL": "mock-whisper",
                "ASR_INTENT_URL": "",
                "ASR_INTENT_PROFILE": "discovery",
            },
            clear=True,
        ):
            settings = AsrSettings.from_env()
        recognizer = SpeechRecognizer(
            settings, model_factory=lambda *_args, **_kwargs: FakeWhisperModel()
        )
        self.client = TestClient(TestServer(create_app(settings, recognizer)))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_discovery_response_uses_only_public_fields(self) -> None:
        upload = FormData()
        upload.add_field("request_id", "asr-001")
        upload.add_field("language", "zh")
        upload.add_field("file", b"mock-wave", filename="voice.wav")
        response = await self.client.post("/api/v1/transcribe", data=upload)

        self.assertEqual(200, response.status, await response.text())
        self.assertEqual(
            {
                "request_id": "asr-001",
                "text": "帮我找 黄色的狗",
                "intent": {
                    "type": "UNKNOWN",
                    "parameters": {},
                },
                "required_skills": [],
            },
            await response.json(),
        )

    async def test_request_id_is_required(self) -> None:
        upload = FormData()
        upload.add_field("file", b"mock-wave", filename="voice.wav")
        response = await self.client.post("/api/v1/transcribe", data=upload)
        self.assertEqual(400, response.status)
        self.assertEqual(
            "bad_request",
            (await response.json())["error"]["code"],
        )
