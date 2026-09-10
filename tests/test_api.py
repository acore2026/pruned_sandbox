from __future__ import annotations

import os
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

from services.asr.config import AsrSettings
from services.asr.main import create_app as create_asr_app
from services.intent.config import IntentSettings
from services.intent.main import create_app as create_intent_app
from services.video.config import VideoSettings
from services.video.main import create_app as create_video_app


class SeparateServiceApiTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WEBRTC_ICE_SERVERS": "",
                "YOLO_ENABLED": "false",
                "ASR_ENABLED": "false",
                "INTENT_BACKEND": "rules",
            },
            clear=True,
        ):
            asr_settings = AsrSettings.from_env()
            intent_settings = IntentSettings.from_env()
            video_settings = VideoSettings.from_env()
        self.asr = TestClient(TestServer(create_asr_app(asr_settings)))
        self.intent = TestClient(TestServer(create_intent_app(intent_settings)))
        self.video = TestClient(TestServer(create_video_app(video_settings)))
        await self.asr.start_server()
        await self.intent.start_server()
        await self.video.start_server()

    async def asyncTearDown(self) -> None:
        await self.asr.close()
        await self.intent.close()
        await self.video.close()

    async def test_each_process_has_own_health_endpoint(self) -> None:
        asr_response = await self.asr.get("/health")
        intent_response = await self.intent.get("/health")
        video_response = await self.video.get("/health")

        self.assertEqual("asr", (await asr_response.json())["service"])
        self.assertEqual("intent", (await intent_response.json())["service"])
        self.assertEqual("video", (await video_response.json())["service"])

    async def test_intent_service_keeps_old_semantic_route_contract(self) -> None:
        response = await self.intent.post(
            "/api/v1/semantic/route",
            json={"intent_payload": "帮我找黄色的狗"},
        )
        payload = await response.json()

        self.assertEqual(200, response.status)
        self.assertEqual("find_object", payload["scene"])
        self.assertEqual("yellow dog", payload["normalized_argument"])

    async def test_intent_route_does_not_exist_on_other_services(self) -> None:
        asr_response = await self.asr.post("/api/v1/intent", json={"text": "向前走"})
        video_response = await self.video.post("/api/v1/intent", json={"text": "向前走"})

        self.assertEqual(404, asr_response.status)
        self.assertEqual(404, video_response.status)
