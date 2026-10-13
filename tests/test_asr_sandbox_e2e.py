from __future__ import annotations

import os
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

from services.asr.config import AsrSettings
from services.asr.main import create_app as create_asr_app
from services.asr.service import SpeechRecognizer
from services.sandbox.api import canonical_digest
from services.sandbox.main import create_app as create_sandbox_app
from services.video.config import VideoSettings


class FakeWhisperModel:
    transcript_text = "威吓歹徒"
    last_language = "unset"

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def transcribe(self, _path: str, **_kwargs):
        FakeWhisperModel.last_language = _kwargs.get("language")
        return (
            iter([SimpleNamespace(start=0.0, end=1.0, text=self.transcript_text)]),
            SimpleNamespace(language="zh", language_probability=0.99, duration=1.0),
        )


class AsrToSandboxE2ETest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        FakeWhisperModel.transcript_text = "威吓歹徒"
        with patch.dict(
            os.environ,
            {
                "ASR_ENABLED": "true",
                "ASR_MODEL": "mock-whisper",
                "ASR_LANGUAGE": "",
                "ASR_INTENT_URL": "",
                "YOLO_ENABLED": "false",
                "SANDBOX_INTENT_URL": "",
                "WEBRTC_ICE_SERVERS": "",
            },
            clear=True,
        ):
            asr_settings = AsrSettings.from_env()
            video_settings = VideoSettings.from_env()
        recognizer = SpeechRecognizer(
            asr_settings, model_factory=lambda *_args, **_kwargs: FakeWhisperModel()
        )
        self.asr = TestClient(TestServer(create_asr_app(asr_settings, recognizer)))
        await self.asr.start_server()
        video_settings = replace(
            video_settings,
            asr_url=str(self.asr.make_url("/api/v1/transcribe")),
        )
        self.sandbox = TestClient(TestServer(create_sandbox_app(video_settings)))
        await self.sandbox.start_server()

    async def asyncTearDown(self) -> None:
        await self.asr.close()
        await self.sandbox.close()

    async def test_sandbox_accepts_audio_without_creating_control_action(self) -> None:
        configuration = {
            "compute_service_session_id": "css-voice-1",
            "connection_parameters": {
                "transport": "WEBRTC",
                "media_connections_path": "/v1/media-connections",
                "recognition_target_path_template": "/v1/recognition-targets/{compute_service_session_id}",
            },
            "participants_network_facts": [
                {
                    "role": "consumer",
                    "agent_id": "glasses",
                    "up_session_id": "ups-glasses",
                    "pdu_session_id": 1,
                    "ue_ipv4_address": "10.60.0.11",
                    "dnn": "internet",
                    "snssai": "1-010203",
                    "generation": "1",
                },
                {
                    "role": "producer",
                    "agent_id": "dog",
                    "up_session_id": "ups-dog",
                    "pdu_session_id": 1,
                    "ue_ipv4_address": "10.60.0.12",
                    "dnn": "internet",
                    "snssai": "1-010203",
                    "generation": "1",
                },
            ],
        }
        response = await self.sandbox.post(
            "/management/v1/compute-session-bindings:bind",
            json={
                "activation_idempotency_key": "activate-voice-1",
                "owner_ref": "ca/css-voice-1",
                "binding_ref": "binding-voice-1",
                "compute_service_session_id": "css-voice-1",
                "compute_instance_id": "ci-voice-1",
                "configuration": configuration,
                "configuration_digest": canonical_digest(configuration),
                "expected_binding_absent": True,
            },
        )
        self.assertEqual(200, response.status, await response.text())

        context = {
            "compute_service_session_id": "css-voice-1",
            "compute_instance_id": "ci-voice-1",
            "binding_ref": "binding-voice-1",
            "role": "consumer",
            "agent_id": "glasses",
        }
        for text, direction in (("Threaten the suspect.", "deter"), ("Deter the suspect.", "deter"), ("威吓歹徒", "deter"), ("Expel the suspect.", "expel"), ("Move forward.", "forward"),
                                ("Move backward.", "backward"), ("Turn left.", "left"),
                                ("Turn right.", "right"), ("驱逐歹徒", "expel"),
                                ("detered", "deter"), ("Detered the suspect.", "deter"),
                                ("deterred the suspect.", "deter"),
                                ("the turds I suspect.", "deter"),
                                ("expelled the suspect.", "expel"),
                                ("Move forwards.", "forward"), ("Turn write.", "right"),
                                ("Go straight", "forward"), ("Get forward", "forward"),
                                ("deters a suspect.", "deter"), ("deters the suspect", "deter")):
            FakeWhisperModel.transcript_text = text
            upload = FormData()
            upload.add_field("request_id", "audio-action-1")
            upload.add_field("computing_context", json.dumps(context))
            upload.add_field("file", b"mock-wave-bytes", filename="voice.wav", content_type="audio/wav")
            with patch("services.sandbox.api.SandboxApi._create_control_action_with_classified",
                       side_effect=AssertionError("ASR must not execute control")):
                response = await self.sandbox.post("/v1/audio-control-actions", data=upload)
            body = await response.json()
            self.assertEqual(200, response.status, body)
            self.assertEqual(text, body["text"])
            self.assertEqual(direction, body["intent"]["direction"])
            self.assertTrue(body["intent"]["matched"])
            self.assertEqual("rules", body["intent"]["backend"])
            self.assertIsNone(FakeWhisperModel.last_language)
            self.assertNotIn("action_id", body)

    async def test_non_action_audio_still_returns_transcription(self) -> None:
        configuration = {
            "compute_service_session_id": "css-voice-2",
            "connection_parameters": {
                "transport": "WEBRTC",
                "media_connections_path": "/v1/media-connections",
                "recognition_target_path_template": "/v1/recognition-targets/{compute_service_session_id}",
            },
            "participants_network_facts": [
                {"role": "consumer", "agent_id": "glasses", "up_session_id": "ups-glasses", "pdu_session_id": 1, "ue_ipv4_address": "10.60.0.11", "dnn": "internet", "snssai": "1-010203", "generation": "1"},
                {"role": "producer", "agent_id": "dog", "up_session_id": "ups-dog", "pdu_session_id": 1, "ue_ipv4_address": "10.60.0.12", "dnn": "internet", "snssai": "1-010203", "generation": "1"},
            ],
        }
        bind = await self.sandbox.post(
            "/management/v1/compute-session-bindings:bind",
            json={
                "activation_idempotency_key": "activate-voice-2",
                "owner_ref": "ca/css-voice-2",
                "binding_ref": "binding-voice-2",
                "compute_service_session_id": "css-voice-2",
                "compute_instance_id": "ci-voice-2",
                "configuration": configuration,
                "configuration_digest": canonical_digest(configuration),
                "expected_binding_absent": True,
            },
        )
        self.assertEqual(200, bind.status, await bind.text())
        context = {
            "compute_service_session_id": "css-voice-2",
            "compute_instance_id": "ci-voice-2",
            "binding_ref": "binding-voice-2",
            "role": "consumer",
            "agent_id": "glasses",
        }
        for text in ("今天天气不错", "Don't detered the suspect.",
                     "detered or expel the suspect.", "Yesterday we deterred the suspect.",
                     "defer the suspect", "write", "wave", "Please WAVE!",
                     "wave hello", "say hello", "hello", "挥手", "打招呼", "你好"):
            with self.subTest(text=text):
                FakeWhisperModel.transcript_text = text
                upload = FormData()
                upload.add_field("request_id", "audio-non-action-1")
                upload.add_field("computing_context", json.dumps(context))
                upload.add_field("file", b"mock-wave-bytes", filename="voice.wav", content_type="audio/wav")
                with patch("services.sandbox.api.SandboxApi._create_control_action_with_classified",
                           side_effect=AssertionError("ASR must not execute control")):
                    response = await self.sandbox.post("/v1/audio-control-actions", data=upload)
                body = await response.json()
                self.assertEqual(200, response.status, body)
                self.assertEqual(text, body["text"])
                self.assertEqual(
                    {"executor": None, "type": "other", "matched": False, "backend": "rules"},
                    body["intent"],
                )
