from __future__ import annotations

import asyncio
from fractions import Fraction
import os
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from aiohttp import ClientSession, web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import numpy as np

from services.video.config import VideoSettings
from services.video.main import VideoRuntime, create_app


class SyntheticVideoTrack(VideoStreamTrack):
    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        image[:, :] = (15, 30, 45)
        frame = VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base or Fraction(1, 90000)
        return frame


async def wait_ice(pc: RTCPeerConnection) -> None:
    if pc.iceGatheringState == "complete":
        return
    ready = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_state() -> None:
        if pc.iceGatheringState == "complete":
            ready.set()

    await asyncio.wait_for(ready.wait(), 8)


class OrangeVideoServerTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VIDEO_PUBLIC_IP": "127.0.0.1",
                "VIDEO_PORT": "28500",
                "VIDEO_SOURCE_WAIT_SECONDS": "8",
                "WEBRTC_ICE_SERVERS": "",
                "YOLO_ENABLED": "false",
                "VIDEO_WIDTH": "320",
                "VIDEO_HEIGHT": "180",
                "VIDEO_FPS": "10",
            },
            clear=True,
        ):
            self.settings = VideoSettings.from_env()
        self.runtime = VideoRuntime.build(self.settings)
        self.runner = web.AppRunner(create_app(self.settings, self.runtime))
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        socket = self.site._server.sockets[0]
        self.port = socket.getsockname()[1]
        object.__setattr__(self.settings, "port", self.port)
        self.base = f"http://127.0.0.1:{self.port}"
        self.http = ClientSession()
        self.peer_connections: list[RTCPeerConnection] = []

    async def asyncTearDown(self) -> None:
        await asyncio.gather(
            *(pc.close() for pc in self.peer_connections),
            return_exceptions=True,
        )
        await self.http.close()
        await self.runner.cleanup()

    async def allocate(self) -> dict:
        response = await self.http.post(
            f"{self.base}/compute/v1/offloading-sessions",
            json={
                "request_id": "create-1",
                "agent_id": "agent-b",
                "group_id": "group-ab",
                "workload_type": "video",
            },
        )
        self.assertEqual(201, response.status, await response.text())
        return await response.json()

    async def test_orange_health_and_auth_errors(self) -> None:
        response = await self.http.get(f"{self.base}/healthz")
        self.assertEqual(
            {
                "status": "ok",
                "service": "agent-sdk-mock-video-server",
                "video_server_ip": "127.0.0.1",
                "sessions": 0,
            },
            await response.json(),
        )

        allocated = await self.allocate()
        session_id = allocated["session_id"]
        response = await self.http.post(
            f"{self.base}/video/v1/sessions/{session_id}/source",
            json={"action": "create_offer"},
        )
        self.assertEqual(401, response.status)
        self.assertEqual("MISSING_BEARER", (await response.json())["error"])

        response = await self.http.post(
            f"{self.base}/compute/v1/offloading-sessions/{session_id}/consumers",
            json={
                "agent_id": "agent-b",
                "group_id": "group-ab",
                "target_agent_ids": ["agent-a"],
            },
        )
        self.assertEqual(409, response.status)
        self.assertEqual("SOURCE_NOT_CONNECTED", (await response.json())["error"])

    async def test_end_to_end_source_yolo_pipeline_and_consumer(self) -> None:
        def mark_as_processed(image: np.ndarray):
            annotated = image.copy()
            annotated[:16, :16] = (210, 30, 210)
            return annotated, []

        self.runtime.detector.process = mark_as_processed
        allocated = await self.allocate()
        session_id = allocated["session_id"]
        producer = allocated["producer"]
        producer_headers = {
            "Authorization": f"Bearer {producer['access_token']}"
        }
        self.assertEqual(
            f"{self.base}/video/v1/sessions/{session_id}/source",
            producer["source_start_url"],
        )

        response = await self.http.post(
            producer["source_start_url"],
            headers=producer_headers,
            json={"action": "create_offer"},
        )
        self.assertEqual(200, response.status, await response.text())
        source_offer = (await response.json())["sdp_offer"]
        self.assertEqual("offer", source_offer["type"])

        source_pc = RTCPeerConnection()
        self.peer_connections.append(source_pc)
        source_pc.addTrack(SyntheticVideoTrack())
        await source_pc.setRemoteDescription(
            RTCSessionDescription(**source_offer)
        )
        answer = await source_pc.createAnswer()
        await source_pc.setLocalDescription(answer)
        await wait_ice(source_pc)
        response = await self.http.post(
            producer["source_start_url"],
            headers=producer_headers,
            json={
                "sdp_answer": {
                    "type": source_pc.localDescription.type,
                    "sdp": source_pc.localDescription.sdp,
                }
            },
        )
        self.assertEqual(200, response.status, await response.text())
        connected = await response.json()
        self.assertEqual("SOURCE_CONNECTED", connected["state"])
        self.assertGreaterEqual(connected["frames_seen"], 1)
        await asyncio.wait_for(
            self.runtime.server.sessions[session_id].pipeline.processed_frame.wait(),
            8,
        )

        response = await self.http.post(
            (
                f"{self.base}/compute/v1/offloading-sessions/"
                f"{session_id}/consumers"
            ),
            json={
                "agent_id": "agent-b",
                "group_id": "group-ab",
                "target_agent_ids": ["agent-a", "agent-c"],
            },
        )
        self.assertEqual(200, response.status, await response.text())
        grants = (await response.json())["consumers"]
        self.assertNotEqual(
            grants["agent-a"]["access_ticket"],
            grants["agent-c"]["access_ticket"],
        )
        self.assertEqual("non-trickle", grants["agent-a"]["signaling"])

        consumer_pc = RTCPeerConnection()
        self.peer_connections.append(consumer_pc)
        received = asyncio.get_running_loop().create_future()

        @consumer_pc.on("track")
        def on_track(track) -> None:
            if not received.done():
                received.set_result(track)

        consumer_pc.addTransceiver("video", direction="recvonly")
        consumer_offer = await consumer_pc.createOffer()
        await consumer_pc.setLocalDescription(consumer_offer)
        await wait_ice(consumer_pc)
        response = await self.http.post(
            grants["agent-a"]["offer_url"],
            headers={
                "Authorization": (
                    f"Bearer {grants['agent-a']['access_ticket']}"
                )
            },
            json={
                "sdp_offer": {
                    "type": consumer_pc.localDescription.type,
                    "sdp": consumer_pc.localDescription.sdp,
                }
            },
        )
        self.assertEqual(200, response.status, await response.text())
        processed = await response.json()
        self.assertEqual("STREAMING", processed["state"])
        await consumer_pc.setRemoteDescription(
            RTCSessionDescription(**processed["sdp_answer"])
        )
        track = await asyncio.wait_for(received, 8)
        frame = await asyncio.wait_for(track.recv(), 8)
        image = frame.to_ndarray(format="bgr24")
        self.assertEqual((120, 160), (frame.height, frame.width))
        self.assertGreater(int(image[2, 2, 0]), 150)
        self.assertGreater(int(image[2, 2, 2]), 150)
        self.assertLess(abs(int(image[60, 80, 0]) - 15), 12)
        self.assertLess(abs(int(image[60, 80, 1]) - 30), 12)
        self.assertLess(abs(int(image[60, 80, 2]) - 45), 12)

        debug_response = await self.http.get(
            f"{self.base}/debug/v1/sessions"
        )
        debug = await debug_response.json()
        self.assertGreaterEqual(debug["sessions"][0]["frames_seen"], 1)
        self.assertEqual("SOURCE_CONNECTED", debug["sessions"][0]["state"])

        response = await self.http.post(
            producer["source_stop_url"],
            headers=producer_headers,
            json={},
        )
        self.assertEqual(200, response.status, await response.text())
        self.assertEqual("STOPPED", (await response.json())["state"])


if __name__ == "__main__":
    import unittest

    unittest.main()
