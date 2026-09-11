from __future__ import annotations

import asyncio
from fractions import Fraction
import os
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from aiohttp import ClientSession, web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.codecs import h264 as aiortc_h264
from aiortc.mediastreams import MediaStreamTrack
from av import CodecContext, VideoFrame
import numpy as np

from services.video.config import VideoSettings
from services.sandbox.main import VideoRuntime, create_app
from services.video.rtc_server import ApiError, require_h264_high_answer
from services.sandbox.api import canonical_digest


class HighProfilePacketTrack(MediaStreamTrack):
    """Send real Annex-B H.264 High packets without aiortc re-encoding."""

    kind = "video"

    def __init__(self) -> None:
        super().__init__()
        self._codec = CodecContext.create("libx264", "w")
        self._codec.width = 160
        self._codec.height = 120
        self._codec.bit_rate = 300_000
        self._codec.pix_fmt = "yuv420p"
        self._codec.framerate = Fraction(15, 1)
        self._codec.time_base = Fraction(1, 15)
        self._codec.options = {
            "level": "31",
            "profile": "high",
            "preset": "medium",
            "tune": "zerolatency",
        }
        self._frame_number = 0
        self.first_access_unit = b""

    async def recv(self):
        if self._frame_number:
            await asyncio.sleep(1 / 15)
        while True:
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            image[:, :] = (15, 30, 45)
            frame = VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = self._frame_number
            frame.time_base = Fraction(1, 15)
            self._frame_number += 1
            packets = self._codec.encode(frame)
            if packets:
                if not self.first_access_unit:
                    self.first_access_unit = bytes(packets[0])
                return packets[0]


class RawVideoTrack(MediaStreamTrack):
    """A terminal-like raw camera source for the standard Offer flow."""

    kind = "video"

    def __init__(self) -> None:
        super().__init__()
        self._frame_number = 0

    async def recv(self) -> VideoFrame:
        if self._frame_number:
            await asyncio.sleep(1 / 15)
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        image[:, :] = (15, 30, 45)
        frame = VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = self._frame_number
        frame.time_base = Fraction(1, 15)
        self._frame_number += 1
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
                "VIDEO_H264_RTP_PAYLOAD_BYTES": "1150",
                "WEBRTC_ICE_SERVERS": "",
                "YOLO_ENABLED": "false",
                "VIDEO_WIDTH": "160",
                "VIDEO_HEIGHT": "120",
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

    async def test_health_and_no_offloading_allocation_routes(self) -> None:
        response = await self.http.get(f"{self.base}/healthz")
        health = await response.json()
        self.assertEqual(200, response.status)
        self.assertEqual("orange-yolo-video-server", health["service"])
        self.assertEqual(10.0, health["output_fps"])
        self.assertEqual(1150, health["h264_rtp_payload_bytes"])
        self.assertEqual(1150, aiortc_h264.PACKET_MAX)

        response = await self.http.post(
            f"{self.base}/compute/v1/offloading-sessions",
            json={"workload_type": "video"},
        )
        self.assertEqual(404, response.status)

        response = await self.http.post(
            f"{self.base}/compute/v1/offloading-sessions/core-session/consumers",
            json={},
        )
        self.assertEqual(404, response.status)

        response = await self.http.delete(
            f"{self.base}/api/v1/webrtc/sessions/core-session"
        )
        self.assertEqual(404, response.status)

    def test_source_answer_rejects_h264_baseline_fallback(self) -> None:
        baseline_answer = "\r\n".join(
            [
                "v=0",
                "m=video 9 UDP/TLS/RTP/SAVPF 99 100",
                "a=rtpmap:99 H264/90000",
                "a=fmtp:99 level-asymmetry-allowed=1;"
                "packetization-mode=1;profile-level-id=42e01f",
                "a=rtpmap:100 rtx/90000",
                "a=fmtp:100 apt=99",
                "",
            ]
        )
        with self.assertRaises(ApiError) as raised:
            require_h264_high_answer(baseline_answer)
        self.assertEqual("SOURCE_CODEC_MISMATCH", raised.exception.code)

    async def test_standard_api_accepts_terminal_offers_for_both_roles(self) -> None:
        configuration = {
            "compute_service_session_id": "css-standard",
            "connection_parameters": {
                "transport": "WEBRTC",
                "media_connections_path": "/v1/media-connections",
                "recognition_target_path_template": "/v1/recognition-targets/{compute_service_session_id}",
            },
            "participants_network_facts": [
                {
                    "role": "consumer", "agent_id": "glasses",
                    "up_session_id": "ups-glasses", "pdu_session_id": 1,
                    "ue_ipv4_address": "10.60.0.11", "dnn": "internet",
                    "snssai": "1-010203", "generation": "1",
                },
                {
                    "role": "producer", "agent_id": "dog",
                    "up_session_id": "ups-dog", "pdu_session_id": 1,
                    "ue_ipv4_address": "10.60.0.12", "dnn": "internet",
                    "snssai": "1-010203", "generation": "1",
                },
            ],
        }
        response = await self.http.post(
            f"{self.base}/management/v1/compute-session-bindings:bind",
            json={
                "activation_idempotency_key": "activate-standard",
                "owner_ref": "ca/css-standard",
                "binding_ref": "binding-standard",
                "compute_service_session_id": "css-standard",
                "compute_instance_id": "ci-standard",
                "configuration": configuration,
                "configuration_digest": canonical_digest(configuration),
                "expected_binding_absent": True,
            },
        )
        self.assertEqual(200, response.status, await response.text())

        consumer_pc = RTCPeerConnection()
        self.peer_connections.append(consumer_pc)
        consumer_pc.addTransceiver("video", direction="recvonly")
        offer = await consumer_pc.createOffer()
        await consumer_pc.setLocalDescription(offer)
        await wait_ice(consumer_pc)
        consumer_context = {
            "compute_service_session_id": "css-standard",
            "compute_instance_id": "ci-standard",
            "binding_ref": "binding-standard",
            "role": "consumer",
            "agent_id": "glasses",
        }
        response = await self.http.post(
            f"{self.base}/v1/media-connections",
            json={
                "request_id": "media-consumer-standard",
                "computing_context": consumer_context,
                "offer": {
                    "type": "offer",
                    "sdp": consumer_pc.localDescription.sdp,
                },
            },
        )
        self.assertEqual(201, response.status, await response.text())
        consumer_answer = (await response.json())["answer"]
        await consumer_pc.setRemoteDescription(RTCSessionDescription(**consumer_answer))

        producer_pc = RTCPeerConnection()
        self.peer_connections.append(producer_pc)
        producer_pc.addTrack(RawVideoTrack())
        offer = await producer_pc.createOffer()
        await producer_pc.setLocalDescription(offer)
        await wait_ice(producer_pc)
        producer_context = dict(consumer_context, role="producer", agent_id="dog")
        response = await self.http.post(
            f"{self.base}/v1/media-connections",
            json={
                "request_id": "media-producer-standard",
                "computing_context": producer_context,
                "offer": {
                    "type": "offer",
                    "sdp": producer_pc.localDescription.sdp,
                },
            },
        )
        self.assertEqual(201, response.status, await response.text())
        producer_answer = (await response.json())["answer"]
        await producer_pc.setRemoteDescription(RTCSessionDescription(**producer_answer))

        media_state = {}
        for _ in range(80):
            response = await self.http.get(
                f"{self.base}/management/v1/compute-session-bindings/binding-standard/media-state"
            )
            media_state = await response.json()
            if media_state.get("producer_connected") and media_state.get("consumer_connected"):
                break
            await asyncio.sleep(0.1)
        self.assertTrue(
            media_state["producer_connected"],
            self.runtime.server.sessions["binding-standard"].pipeline.health(),
        )
        self.assertTrue(media_state["consumer_connected"])

    async def test_core_session_media_flow_and_yolo_switch(self) -> None:
        def mark_as_processed(
            image: np.ndarray, classes: tuple[str, ...] | None = None
        ):
            annotated = image.copy()
            annotated[:16, :16] = (210, 30, 210)
            return annotated, []

        self.runtime.detector.process = mark_as_processed
        session_id = "core-session-1"

        consumer_pc = RTCPeerConnection()
        self.peer_connections.append(consumer_pc)
        track_ready = asyncio.get_running_loop().create_future()

        @consumer_pc.on("track")
        def on_track(track) -> None:
            if not track_ready.done():
                track_ready.set_result(track)

        consumer_pc.addTransceiver("video", direction="recvonly")
        consumer_offer = await consumer_pc.createOffer()
        await consumer_pc.setLocalDescription(consumer_offer)
        await wait_ice(consumer_pc)
        response = await self.http.post(
            f"{self.base}/video/v1/sessions/{session_id}/processed",
            json={
                "sdp_offer": {
                    "type": consumer_pc.localDescription.type,
                    "sdp": consumer_pc.localDescription.sdp,
                }
            },
        )
        self.assertEqual(200, response.status, await response.text())
        consumer_answer = await response.json()
        self.assertEqual("SOURCE_PENDING", consumer_answer["state"])
        await consumer_pc.setRemoteDescription(
            RTCSessionDescription(**consumer_answer["sdp_answer"])
        )
        remote_track = await asyncio.wait_for(track_ready, 8)
        original_track_id = remote_track.id
        placeholder = await asyncio.wait_for(remote_track.recv(), 8)
        placeholder_image = placeholder.to_ndarray(format="bgr24")
        self.assertLess(int(placeholder_image[2, 2, 0]), 100)

        async def wait_for_processed_frame() -> tuple[VideoFrame, list[int]]:
            while True:
                candidate = await remote_track.recv()
                marker = candidate.to_ndarray(format="bgr24")[2, 2].tolist()
                if marker[0] >= 150 and marker[2] >= 150:
                    return candidate, marker

        processed_frame_task = asyncio.create_task(wait_for_processed_frame())

        response = await self.http.post(
            f"{self.base}/video/v1/sessions/{session_id}/source",
            json={"action": "create_offer"},
        )
        self.assertEqual(200, response.status, await response.text())
        source_offer = (await response.json())["sdp_offer"]
        self.assertIn("profile-level-id=64001f", source_offer["sdp"])
        self.assertIn("profile-level-id=640c1f", source_offer["sdp"])
        self.assertNotIn("profile-level-id=42e01f", source_offer["sdp"])

        source_pc = RTCPeerConnection()
        self.peer_connections.append(source_pc)
        high_profile_track = HighProfilePacketTrack()
        source_pc.addTrack(high_profile_track)
        await source_pc.setRemoteDescription(RTCSessionDescription(**source_offer))
        source_answer = await source_pc.createAnswer()
        await source_pc.setLocalDescription(source_answer)
        self.assertIn(
            "profile-level-id=64001f",
            require_h264_high_answer(source_pc.localDescription.sdp),
        )
        await wait_ice(source_pc)
        response = await self.http.post(
            f"{self.base}/video/v1/sessions/{session_id}/source",
            json={
                "sdp_answer": {
                    "type": source_pc.localDescription.type,
                    "sdp": source_pc.localDescription.sdp,
                }
            },
        )
        self.assertEqual(200, response.status, await response.text())
        self.assertEqual("SOURCE_CONNECTED", (await response.json())["state"])
        self.assertIn(b"\x00\x00\x00\x01\x67\x64", high_profile_track.first_access_unit)

        processed_frame, marker = await asyncio.wait_for(processed_frame_task, 8)
        self.assertEqual(original_track_id, remote_track.id)
        self.assertEqual((120, 160), (processed_frame.height, processed_frame.width))
        self.assertGreaterEqual(marker[0], 150)
        self.assertGreaterEqual(marker[2], 150)

        debug = await (
            await self.http.get(f"{self.base}/debug/v1/sessions")
        ).json()
        session_debug = debug["sessions"][0]
        self.assertEqual("SOURCE_CONNECTED", session_debug["state"])
        self.assertGreaterEqual(session_debug["frames_seen"], 1)
        self.assertGreaterEqual(session_debug["frames_processed"], 1)
        self.assertIn("profile-level-id=64001f", session_debug["source_codec"])
        consumer_debug = session_debug["consumers"]["consumer-1"]
        self.assertGreater(consumer_debug["placeholder_frames_sent"], 0)
        self.assertGreater(consumer_debug["frames_processed"], 0)
        self.assertTrue(consumer_debug["first_source_frame_sent"])

        response = await self.http.post(
            f"{self.base}/video/v1/sessions/{session_id}/source/stop",
            json={},
        )
        self.assertEqual(200, response.status, await response.text())
        self.assertEqual("STOPPED", (await response.json())["state"])


if __name__ == "__main__":
    import unittest

    unittest.main()
