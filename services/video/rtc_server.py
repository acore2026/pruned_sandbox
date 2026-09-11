from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiohttp import web
from aiortc import (
    RTCPeerConnection,
    RTCRtpReceiver,
    RTCRtpSender,
    RTCSessionDescription,
)
from aiortc.codecs import CODECS
from aiortc.codecs import h264 as aiortc_h264
from aiortc.mediastreams import MediaStreamTrack
from aiortc.rtcrtpparameters import RTCRtcpFeedback, RTCRtpCodecParameters
from av import VideoFrame

from .config import VideoSettings
from .detector import YoloDetector
from .ice import build_rtc_configuration, wait_for_ice_gathering
from .pipeline import FramePipeline


LOGGER = logging.getLogger("sandbox.video.rtc_server")

H264_HIGH_PROFILE_LEVEL_IDS = ("64001f", "640c1f")
H264_BASELINE_PROFILE_LEVEL_IDS = ("42e01f", "42001f")
H264_PACKETIZATION_MODE = "1"


def register_h264_high_profiles() -> None:
    """Add High and Constrained High to aiortc's process-wide capabilities."""
    existing = {
        str(codec.parameters.get("profile-level-id", "")).lower()
        for codec in CODECS["video"]
        if codec.mimeType.lower() == "video/h264"
        and str(codec.parameters.get("packetization-mode", "0"))
        == H264_PACKETIZATION_MODE
    }
    used_payload_types = {codec.payloadType for codec in CODECS["video"]}
    next_payload_type = next(
        payload_type
        for payload_type in range(96, 127, 2)
        if payload_type not in used_payload_types
        and payload_type + 1 not in used_payload_types
    )
    for profile_level_id in H264_HIGH_PROFILE_LEVEL_IDS:
        if profile_level_id in existing:
            continue
        feedback = [
            RTCRtcpFeedback(type="nack"),
            RTCRtcpFeedback(type="nack", parameter="pli"),
            RTCRtcpFeedback(type="goog-remb"),
        ]
        CODECS["video"].extend(
            [
                RTCRtpCodecParameters(
                    mimeType="video/H264",
                    clockRate=90000,
                    payloadType=next_payload_type,
                    rtcpFeedback=feedback,
                    parameters={
                        "level-asymmetry-allowed": "1",
                        "packetization-mode": H264_PACKETIZATION_MODE,
                        "profile-level-id": profile_level_id,
                    },
                ),
                RTCRtpCodecParameters(
                    mimeType="video/rtx",
                    clockRate=90000,
                    payloadType=next_payload_type + 1,
                    parameters={"apt": next_payload_type},
                ),
            ]
        )
        next_payload_type += 2


register_h264_high_profiles()


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def selected_video_format(sdp: str) -> tuple[str, dict[str, str]]:
    video_payloads: list[str] = []
    codec_by_payload: dict[str, str] = {}
    fmtp_by_payload: dict[str, dict[str, str]] = {}
    in_video_section = False
    for raw_line in sdp.splitlines():
        line = raw_line.strip()
        if line.startswith("m="):
            parts = line.split()
            in_video_section = line.startswith("m=video ")
            if in_video_section and len(parts) > 3:
                video_payloads = parts[3:]
            continue
        if not in_video_section or not line.startswith("a=rtpmap:"):
            if in_video_section and line.startswith("a=fmtp:"):
                payload_and_fmtp = line[len("a=fmtp:") :].split(None, 1)
                if len(payload_and_fmtp) == 2:
                    fmtp_by_payload[payload_and_fmtp[0]] = {
                        key.strip().lower(): value.strip().lower()
                        for item in payload_and_fmtp[1].split(";")
                        if "=" in item
                        for key, value in [item.split("=", 1)]
                    }
            continue
        payload_and_codec = line[len("a=rtpmap:") :].split(None, 1)
        if len(payload_and_codec) == 2:
            codec_by_payload[payload_and_codec[0]] = payload_and_codec[1]
    for payload in video_payloads:
        codec = codec_by_payload.get(payload)
        if codec and not codec.lower().startswith("rtx/"):
            return codec, fmtp_by_payload.get(payload, {})
    return "<unknown>", {}


def selected_video_codec(sdp: str) -> str:
    return selected_video_format(sdp)[0]


def require_h264_high_answer(sdp: str) -> str:
    """Reject a source answer which silently falls back from H.264 High."""
    codec, parameters = selected_video_format(sdp)
    profile_level_id = parameters.get("profile-level-id", "<missing>")
    packetization_mode = parameters.get("packetization-mode", "0")
    if (
        codec.lower() != "h264/90000"
        or profile_level_id not in H264_HIGH_PROFILE_LEVEL_IDS
        or packetization_mode != H264_PACKETIZATION_MODE
    ):
        raise ApiError(
            409,
            "SOURCE_CODEC_MISMATCH",
            "source must negotiate H264 High Profile with packetization-mode=1; "
            f"got codec={codec} profile-level-id={profile_level_id} "
            f"packetization-mode={packetization_mode}",
        )
    return (
        f"{codec};profile-level-id={profile_level_id};"
        f"packetization-mode={packetization_mode}"
    )


def prefer_h264_high(transceiver: Any) -> None:
    """Require H.264 High on the Orange SDK source leg."""
    codecs = RTCRtpSender.getCapabilities("video").codecs
    h264 = [
        codec
        for codec in codecs
        if codec.mimeType.lower() == "video/h264"
        and str(codec.parameters.get("profile-level-id", "")).lower()
        in H264_HIGH_PROFILE_LEVEL_IDS
        and str(codec.parameters.get("packetization-mode", "0"))
        == H264_PACKETIZATION_MODE
    ]
    if len(h264) != len(H264_HIGH_PROFILE_LEVEL_IDS):
        raise RuntimeError("H264 High Profile capabilities were not registered")
    retransmission = [
        codec for codec in codecs if codec.mimeType.lower() == "video/rtx"
    ]
    transceiver.setCodecPreferences(h264 + retransmission)


def prefer_h264_baseline(transceiver: Any) -> None:
    """Keep server-encoded processed video on aiortc's real Baseline profile."""
    codecs = RTCRtpSender.getCapabilities("video").codecs
    h264 = [
        codec
        for codec in codecs
        if codec.mimeType.lower() == "video/h264"
        and str(codec.parameters.get("profile-level-id", "")).lower()
        in H264_BASELINE_PROFILE_LEVEL_IDS
    ]
    if not h264:
        return
    retransmission = [
        codec for codec in codecs if codec.mimeType.lower() == "video/rtx"
    ]
    fallback = [
        codec
        for codec in codecs
        if codec.mimeType.lower() not in {"video/h264", "video/rtx"}
    ]
    transceiver.setCodecPreferences(h264 + retransmission + fallback)


@dataclass(slots=True)
class ConsumerConnection:
    pc: RTCPeerConnection
    state: str = "new"
    codec: str = "<pending>"
    frames_processed: int = 0
    packets_sent: int = 0
    bytes_sent: int = 0
    placeholder_frames_sent: int = 0
    first_frame_logged: bool = False
    first_source_frame_logged: bool = False
    sender: RTCRtpSender | None = None
    keyframes_requested: int = 0
    stats_task: asyncio.Task[None] | None = None


MediaStateCallback = Callable[[str, bool, str], Awaitable[None]]


@dataclass(slots=True)
class StandardMediaConnection:
    """A real PeerConnection owned by one standard API media resource."""

    role: str
    pc: RTCPeerConnection
    consumer: ConsumerConnection | None = None


async def refresh_consumer_stats(connection: ConsumerConnection) -> None:
    try:
        report = await connection.pc.getStats()
    except Exception as error:
        LOGGER.debug("consumer RTP stats unavailable: %s", error)
        return
    outbound = [
        stat
        for stat in report.values()
        if getattr(stat, "type", "") == "outbound-rtp"
        and getattr(stat, "kind", "") == "video"
    ]
    connection.packets_sent = sum(
        int(getattr(stat, "packetsSent", 0)) for stat in outbound
    )
    connection.bytes_sent = sum(
        int(getattr(stat, "bytesSent", 0)) for stat in outbound
    )


class ConsumerVideoTrack(MediaStreamTrack):
    """Count placeholder and YOLO frames without changing the remote track."""

    kind = "video"

    def __init__(
        self,
        source: MediaStreamTrack,
        pipeline: FramePipeline,
        session_id: str,
        consumer_id: str,
        connection: ConsumerConnection,
    ) -> None:
        super().__init__()
        self._source = source
        self._pipeline = pipeline
        self._session_id = session_id
        self._consumer_id = consumer_id
        self._connection = connection

    async def recv(self) -> VideoFrame:
        frame = await self._source.recv()
        is_source = self._pipeline.source_frame_available
        if is_source:
            self._connection.frames_processed += 1
        else:
            self._connection.placeholder_frames_sent += 1
        if not self._connection.first_frame_logged:
            self._connection.first_frame_logged = True
            LOGGER.info(
                "first consumer frame id=%s connection=%s kind=%s codec=%s",
                self._session_id,
                self._consumer_id,
                "source" if is_source else "placeholder",
                self._connection.codec,
            )
        if is_source and not self._connection.first_source_frame_logged:
            self._connection.first_source_frame_logged = True
            LOGGER.info(
                "first YOLO frame sent id=%s connection=%s codec=%s",
                self._session_id,
                self._consumer_id,
                self._connection.codec,
            )
        return frame


@dataclass(slots=True)
class VideoSession:
    """Process-local media state for a session ID allocated by the core network."""

    session_id: str
    pipeline: FramePipeline
    state: str = "ALLOCATED"
    producer_pc: RTCPeerConnection | None = None
    source_track: MediaStreamTrack | None = None
    source_keyframe_task: asyncio.Task[None] | None = None
    processed_keyframe_task: asyncio.Task[None] | None = None
    source_codec: str = "<pending>"
    source_keyframes_requested: int = 0
    source_ready_task: asyncio.Task[None] | None = None
    consumer_connections: list[ConsumerConnection] = field(default_factory=list)

    @property
    def frames_seen(self) -> int:
        return self.pipeline.received_frames

    async def close(self) -> None:
        self.state = "STOPPED"
        if self.source_keyframe_task is not None:
            self.source_keyframe_task.cancel()
            await asyncio.gather(self.source_keyframe_task, return_exceptions=True)
            self.source_keyframe_task = None
        if self.processed_keyframe_task is not None:
            self.processed_keyframe_task.cancel()
            await asyncio.gather(
                self.processed_keyframe_task,
                return_exceptions=True,
            )
            self.processed_keyframe_task = None
        if self.source_ready_task is not None:
            self.source_ready_task.cancel()
            await asyncio.gather(self.source_ready_task, return_exceptions=True)
            self.source_ready_task = None
        connections = list(self.consumer_connections)
        await asyncio.gather(
            *(refresh_consumer_stats(connection) for connection in connections),
            return_exceptions=True,
        )
        for connection in connections:
            if connection.stats_task is not None:
                connection.stats_task.cancel()
        await asyncio.gather(
            *(
                connection.stats_task
                for connection in connections
                if connection.stats_task is not None
            ),
            return_exceptions=True,
        )
        await self.pipeline.close()
        peers = {connection.pc for connection in connections}
        if self.producer_pc is not None:
            peers.add(self.producer_pc)
        await asyncio.gather(*(pc.close() for pc in peers), return_exceptions=True)
        self.producer_pc = None
        self.source_track = None


class OrangeVideoServer:
    """Orange SDK media signaling with per-session YOLO processing.

    The core network allocates and manages offloading sessions. This service only
    creates process-local media state when one of those session IDs reaches a
    source or processed-video endpoint.
    """

    def __init__(self, settings: VideoSettings, detector: YoloDetector) -> None:
        self.settings = settings
        self.detector = detector
        self.sessions: dict[str, VideoSession] = {}
        # Bound H.264 FU-A RTP payloads so the encrypted UDP/IP packet remains
        # below the 1280-byte Android TUN MTU, matching mock-video-server.
        aiortc_h264.PACKET_MAX = settings.h264_rtp_payload_bytes

    async def health(self, _: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "service": "orange-yolo-video-server",
                "video_server_ip": self.settings.public_ip,
                "output_fps": self.settings.video_fps,
                "h264_rtp_payload_bytes": self.settings.h264_rtp_payload_bytes,
                "sessions": len(self.sessions),
            }
        )

    async def list_sessions(self, _: web.Request) -> web.Response:
        connections = [
            connection
            for session in self.sessions.values()
            for connection in session.consumer_connections
        ]
        await asyncio.gather(
            *(refresh_consumer_stats(connection) for connection in connections),
            return_exceptions=True,
        )
        return web.json_response(
            {
                "sessions": [
                    {
                        "session_id": session.session_id,
                        "state": session.state,
                        "frames_seen": session.frames_seen,
                        "source_codec": session.source_codec,
                        "source_keyframes_requested": session.source_keyframes_requested,
                        "frames_processed": session.pipeline.processed_frames,
                        "frames_dropped": session.pipeline.dropped_frames,
                        "output_fps": self.settings.video_fps,
                        "source_frame_available": session.pipeline.source_frame_available,
                        "consumer_connections": sum(
                            connection.state not in {"failed", "closed"}
                            for connection in session.consumer_connections
                        ),
                        "consumers": {
                            f"consumer-{index}": {
                                "frames_processed": connection.frames_processed,
                                "placeholder_frames_sent": connection.placeholder_frames_sent,
                                "packets_sent": connection.packets_sent,
                                "bytes_sent": connection.bytes_sent,
                                "codec": connection.codec,
                                "first_frame_sent": connection.first_frame_logged,
                                "first_source_frame_sent": connection.first_source_frame_logged,
                                "keyframes_requested": connection.keyframes_requested,
                            }
                            for index, connection in enumerate(
                                session.consumer_connections,
                                start=1,
                            )
                        },
                    }
                    for session in self.sessions.values()
                ]
            }
        )

    def ensure_binding(self, binding_ref: str) -> VideoSession:
        """Create the media pipeline allocated by an accepted M-BIND."""
        return self._get_session(binding_ref, create=True)

    def set_recognition_target(self, binding_ref: str, prompt: str | None) -> None:
        self._get_session(binding_ref).pipeline.set_recognition_target(prompt)

    async def search_object(
        self, binding_ref: str, prompt: str, timeout_ms: int
    ) -> list[dict[str, Any]]:
        return await self._get_session(binding_ref).pipeline.search_object(
            prompt, timeout_ms
        )

    async def create_standard_media_connection(
        self,
        binding_ref: str,
        role: str,
        offer_sdp: str,
        state_callback: MediaStateCallback,
    ) -> tuple[StandardMediaConnection, dict[str, str]]:
        """Apply a terminal Offer and return the real non-trickle Answer."""
        session = self._get_session(binding_ref)
        if session.state in {"STOPPED", "FAILED"}:
            raise ApiError(409, "binding-not-streamable", "binding is not streamable")
        if role == "producer":
            return await self._create_standard_producer(
                session, offer_sdp, state_callback
            )
        if role == "consumer":
            return await self._create_standard_consumer(
                session, offer_sdp, state_callback
            )
        raise ApiError(400, "invalid-role", "role must be producer or consumer")

    async def close_standard_media_connection(
        self,
        binding_ref: str,
        connection: StandardMediaConnection,
    ) -> None:
        session = self.sessions.get(binding_ref)
        if connection.role == "producer":
            if session is not None and session.producer_pc is connection.pc:
                await self._reset_source(session)
            elif connection.pc.connectionState != "closed":
                await connection.pc.close()
            return
        consumer = connection.consumer
        if consumer is not None:
            if consumer.stats_task is not None:
                consumer.stats_task.cancel()
                await asyncio.gather(consumer.stats_task, return_exceptions=True)
            if session is not None and consumer in session.consumer_connections:
                session.consumer_connections.remove(consumer)
        if connection.pc.connectionState != "closed":
            await connection.pc.close()

    async def close_binding(self, binding_ref: str) -> None:
        session = self.sessions.pop(binding_ref, None)
        if session is not None:
            await session.close()

    async def _create_standard_producer(
        self,
        session: VideoSession,
        offer_sdp: str,
        state_callback: MediaStateCallback,
    ) -> tuple[StandardMediaConnection, dict[str, str]]:
        await self._reset_source(session)
        session.state = "WAITING_FOR_SOURCE"
        pc = RTCPeerConnection(configuration=build_rtc_configuration(self.settings))
        session.producer_pc = pc

        @pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if track.kind != "video":
                return
            session.source_track = track
            asyncio.create_task(
                session.pipeline.attach_source(track, f"source:{session.session_id}")
            )

            @track.on("ended")
            async def on_track_ended() -> None:
                await state_callback("producer", False, "media-ended")

        @pc.on("connectionstatechange")
        async def on_connection_state_change() -> None:
            if pc.connectionState in {"failed", "closed"}:
                await state_callback(
                    "producer", False, f"webrtc-{pc.connectionState}"
                )
            elif pc.connectionState == "connected":
                receiver = next(
                    (
                        transceiver.receiver
                        for transceiver in pc.getTransceivers()
                        if transceiver.kind == "video"
                    ),
                    None,
                )
                if receiver is not None and session.source_keyframe_task is None:
                    session.source_keyframe_task = asyncio.create_task(
                        self._request_source_keyframes(session, receiver)
                    )

        async def wait_for_first_frame() -> None:
            try:
                await session.pipeline.first_frame.wait()
                if session.producer_pc is pc:
                    session.state = "SOURCE_CONNECTED"
                    await state_callback("producer", True, "")
                    if session.processed_keyframe_task is None:
                        session.processed_keyframe_task = asyncio.create_task(
                            self._request_consumer_keyframes_when_processed(session)
                        )
            except asyncio.CancelledError:
                raise

        try:
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=offer_sdp, type="offer")
            )
            video_transceiver = next(
                (
                    transceiver
                    for transceiver in pc.getTransceivers()
                    if transceiver.kind == "video"
                ),
                None,
            )
            if video_transceiver is None:
                raise ApiError(400, "invalid-sdp", "producer Offer has no video")
            prefer_h264_high(video_transceiver)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await self._wait_ice(pc)
            local = pc.localDescription
            session.source_codec = selected_video_codec(local.sdp)
            session.source_ready_task = asyncio.create_task(wait_for_first_frame())
            return StandardMediaConnection("producer", pc), {
                "type": local.type,
                "sdp": local.sdp,
            }
        except Exception:
            if session.producer_pc is pc:
                session.producer_pc = None
            await pc.close()
            raise

    async def _create_standard_consumer(
        self,
        session: VideoSession,
        offer_sdp: str,
        state_callback: MediaStateCallback,
    ) -> tuple[StandardMediaConnection, dict[str, str]]:
        pc = RTCPeerConnection(configuration=build_rtc_configuration(self.settings))
        consumer = ConsumerConnection(pc=pc)
        session.consumer_connections.append(consumer)

        @pc.on("connectionstatechange")
        async def on_connection_state_change() -> None:
            consumer.state = pc.connectionState
            if pc.connectionState == "connected":
                await state_callback("consumer", True, "")
            elif pc.connectionState in {"failed", "closed"}:
                await state_callback(
                    "consumer", False, f"webrtc-{pc.connectionState}"
                )

        try:
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=offer_sdp, type="offer")
            )
            output = session.pipeline.new_output_track()
            consumer.sender = pc.addTrack(
                ConsumerVideoTrack(
                    output,
                    session.pipeline,
                    session.session_id,
                    "consumer-standard",
                    consumer,
                )
            )
            transceiver = next(
                item
                for item in pc.getTransceivers()
                if item.sender is consumer.sender
            )
            prefer_h264_baseline(transceiver)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await self._wait_ice(pc)
            local = pc.localDescription
            consumer.codec = selected_video_codec(local.sdp)
            consumer.stats_task = asyncio.create_task(
                self._poll_consumer_stats(consumer)
            )
            return StandardMediaConnection("consumer", pc, consumer), {
                "type": local.type,
                "sdp": local.sdp,
            }
        except Exception:
            if consumer in session.consumer_connections:
                session.consumer_connections.remove(consumer)
            await pc.close()
            raise

    async def source(self, request: web.Request) -> web.Response:
        session = self._session(request, create=True)
        body = await self._json(request)
        sdp_type, sdp = self._sdp(body, "sdp_answer")

        if not sdp:
            await self._reset_source(session)
            session.state = "WAITING_FOR_SOURCE"
            pc = RTCPeerConnection(
                configuration=build_rtc_configuration(self.settings)
            )
            session.producer_pc = pc
            source_transceiver = pc.addTransceiver("video", direction="recvonly")
            prefer_h264_high(source_transceiver)

            @pc.on("track")
            def on_track(track: MediaStreamTrack) -> None:
                if track.kind != "video":
                    return
                session.source_track = track
                asyncio.create_task(
                    session.pipeline.attach_source(
                        track,
                        f"source:{session.session_id}",
                    )
                )
                LOGGER.info(
                    "source track negotiated id=%s track=%s",
                    session.session_id,
                    track.id,
                )

                @track.on("ended")
                async def on_track_ended() -> None:
                    if session.state not in {"STOPPED", "FAILED"}:
                        session.state = "SOURCE_ENDED"
                    if session.processed_keyframe_task is not None:
                        session.processed_keyframe_task.cancel()

            @pc.on("connectionstatechange")
            async def on_source_state_change() -> None:
                LOGGER.info(
                    "source pc id=%s state=%s",
                    session.session_id,
                    pc.connectionState,
                )
                if (
                    pc.connectionState in {"failed", "closed"}
                    and session.producer_pc is pc
                    and session.state != "STOPPED"
                ):
                    session.state = "FAILED"
                    if session.source_keyframe_task is not None:
                        session.source_keyframe_task.cancel()
                    if session.processed_keyframe_task is not None:
                        session.processed_keyframe_task.cancel()
                elif (
                    pc.connectionState == "connected"
                    and session.source_keyframe_task is None
                ):
                    session.source_keyframe_task = asyncio.create_task(
                        self._request_source_keyframes(
                            session,
                            source_transceiver.receiver,
                        )
                    )

            try:
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)
                await self._wait_ice(pc)
                local = pc.localDescription
                return web.json_response(
                    {
                        "session_id": session.session_id,
                        "state": session.state,
                        "sdp_offer": {"type": local.type, "sdp": local.sdp},
                    }
                )
            except Exception:
                if session.producer_pc is pc:
                    session.producer_pc = None
                await pc.close()
                raise

        if sdp_type != "answer" or session.producer_pc is None:
            raise ApiError(
                409,
                "SOURCE_SIGNALING_ORDER",
                "request a server offer before sending an answer",
            )
        session.source_codec = require_h264_high_answer(sdp)
        LOGGER.info(
            "source answer selected id=%s codec=%s",
            session.session_id,
            session.source_codec,
        )
        await session.producer_pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type=sdp_type)
        )
        try:
            await asyncio.wait_for(
                session.pipeline.first_frame.wait(),
                timeout=self.settings.source_wait_seconds,
            )
        except TimeoutError as error:
            session.state = "FAILED"
            raise ApiError(
                504,
                "SOURCE_FRAME_TIMEOUT",
                "no video frame arrived from the source",
            ) from error
        session.state = "SOURCE_CONNECTED"
        session.processed_keyframe_task = asyncio.create_task(
            self._request_consumer_keyframes_when_processed(session)
        )
        return web.json_response(
            {
                "session_id": session.session_id,
                "state": session.state,
                "track_id": session.source_track.id if session.source_track else "",
                "frames_seen": session.frames_seen,
            }
        )

    async def stop_source(self, request: web.Request) -> web.Response:
        session = self._session(request)
        await session.close()
        return web.json_response(
            {"session_id": session.session_id, "state": session.state}
        )

    async def processed(self, request: web.Request) -> web.Response:
        session = self._session(request, create=True)
        if session.state in {"STOPPED", "FAILED"}:
            raise ApiError(
                409,
                "SESSION_NOT_STREAMABLE",
                "processed stream is not available",
            )
        body = await self._json(request)
        sdp_type, sdp = self._sdp(body, "sdp_offer")
        if sdp_type != "offer" or not sdp:
            raise ApiError(
                400,
                "INVALID_SDP",
                "consumer request must contain an SDP offer",
            )

        pc = RTCPeerConnection(
            configuration=build_rtc_configuration(self.settings)
        )
        connection = ConsumerConnection(pc=pc)
        session.consumer_connections.append(connection)
        consumer_id = f"consumer-{len(session.consumer_connections)}"

        @pc.on("connectionstatechange")
        async def on_consumer_state_change() -> None:
            connection.state = pc.connectionState
            LOGGER.info(
                "consumer pc id=%s connection=%s state=%s",
                session.session_id,
                consumer_id,
                pc.connectionState,
            )
            if pc.connectionState in {"failed", "closed"}:
                await refresh_consumer_stats(connection)
            elif pc.connectionState == "connected":
                self._request_consumer_keyframe_if_ready(
                    session,
                    connection,
                    consumer_id,
                    reason="connected-and-source-ready",
                )

        try:
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=sdp, type=sdp_type)
            )
            output = session.pipeline.new_output_track()
            connection.sender = pc.addTrack(
                ConsumerVideoTrack(
                    output,
                    session.pipeline,
                    session.session_id,
                    consumer_id,
                    connection,
                )
            )
            consumer_transceiver = next(
                transceiver
                for transceiver in pc.getTransceivers()
                if transceiver.sender is connection.sender
            )
            prefer_h264_baseline(consumer_transceiver)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await self._wait_ice(pc)
            local = pc.localDescription
        except Exception:
            await pc.close()
            raise

        connection.codec = selected_video_codec(local.sdp)
        connection.stats_task = asyncio.create_task(
            self._poll_consumer_stats(connection)
        )
        LOGGER.info(
            "consumer answer ready id=%s connection=%s codec=%s",
            session.session_id,
            consumer_id,
            connection.codec,
        )
        return web.json_response(
            {
                "session_id": session.session_id,
                "consumer_id": consumer_id,
                "state": (
                    "SOURCE_CONNECTED"
                    if session.pipeline.first_frame.is_set()
                    else "SOURCE_PENDING"
                ),
                "sdp_answer": {"type": local.type, "sdp": local.sdp},
            }
        )

    async def close(self) -> None:
        await asyncio.gather(
            *(session.close() for session in self.sessions.values()),
            return_exceptions=True,
        )

    async def _reset_source(self, session: VideoSession) -> None:
        if session.source_ready_task is not None:
            session.source_ready_task.cancel()
            await asyncio.gather(session.source_ready_task, return_exceptions=True)
            session.source_ready_task = None
        if session.source_keyframe_task is not None:
            session.source_keyframe_task.cancel()
            await asyncio.gather(
                session.source_keyframe_task,
                return_exceptions=True,
            )
            session.source_keyframe_task = None
        if session.processed_keyframe_task is not None:
            session.processed_keyframe_task.cancel()
            await asyncio.gather(
                session.processed_keyframe_task,
                return_exceptions=True,
            )
            session.processed_keyframe_task = None
        old_pc = session.producer_pc
        session.producer_pc = None
        session.source_track = None
        session.source_codec = "<pending>"
        session.source_keyframes_requested = 0
        await session.pipeline.clear_source()
        if old_pc is not None:
            await old_pc.close()

    async def _request_source_keyframes(
        self,
        session: VideoSession,
        receiver: RTCRtpReceiver,
    ) -> None:
        send_pli = getattr(receiver, "_send_rtcp_pli", None)
        if not callable(send_pli):
            LOGGER.warning("source keyframe request unsupported id=%s", session.session_id)
            return
        try:
            for index, delay_seconds in enumerate(
                (0.0, 0.2, 0.5, 1.0, 2.0, 3.0),
                start=1,
            ):
                if delay_seconds:
                    await asyncio.sleep(delay_seconds)
                pc = session.producer_pc
                if (
                    pc is None
                    or pc.connectionState != "connected"
                    or session.pipeline.first_frame.is_set()
                ):
                    return
                sources = receiver.getSynchronizationSources()
                if not sources:
                    continue
                for source in sources:
                    await send_pli(source.source)
                    session.source_keyframes_requested += 1
                    LOGGER.info(
                        "source keyframe requested id=%s ssrc=%s "
                        "reason=decode-startup-%s count=%s",
                        session.session_id,
                        source.source,
                        index,
                        session.source_keyframes_requested,
                    )
        except asyncio.CancelledError:
            raise

    def _request_consumer_keyframe(
        self,
        connection: ConsumerConnection,
        session_id: str,
        consumer_id: str,
        reason: str,
    ) -> None:
        sender = connection.sender
        if sender is None:
            return
        request_keyframe = getattr(sender, "_send_keyframe", None)
        if not callable(request_keyframe):
            LOGGER.warning(
                "consumer keyframe request unsupported id=%s connection=%s reason=%s",
                session_id,
                consumer_id,
                reason,
            )
            return
        request_keyframe()
        connection.keyframes_requested += 1
        LOGGER.info(
            "consumer keyframe requested id=%s connection=%s reason=%s count=%s",
            session_id,
            consumer_id,
            reason,
            connection.keyframes_requested,
        )

    def _request_consumer_keyframe_if_ready(
        self,
        session: VideoSession,
        connection: ConsumerConnection,
        consumer_id: str,
        reason: str,
    ) -> None:
        if (
            connection.state != "connected"
            or not session.pipeline.source_frame_available
            or connection.keyframes_requested != 0
        ):
            return
        self._request_consumer_keyframe(
            connection,
            session.session_id,
            consumer_id,
            reason,
        )

    async def _request_consumer_keyframes_when_processed(
        self,
        session: VideoSession,
    ) -> None:
        try:
            await session.pipeline.processed_frame.wait()
            if session.state != "SOURCE_CONNECTED":
                return
            for index, connection in enumerate(
                session.consumer_connections,
                start=1,
            ):
                self._request_consumer_keyframe_if_ready(
                    session,
                    connection,
                    f"consumer-{index}",
                    reason="first-yolo-frame-ready",
                )
        except asyncio.CancelledError:
            raise

    async def _poll_consumer_stats(self, connection: ConsumerConnection) -> None:
        try:
            while connection.state not in {"failed", "closed"}:
                await asyncio.sleep(1.0)
                await refresh_consumer_stats(connection)
        except asyncio.CancelledError:
            await refresh_consumer_stats(connection)
            raise

    async def _wait_ice(self, pc: RTCPeerConnection) -> None:
        result = await wait_for_ice_gathering(
            pc,
            self.settings.rtc_ice_gather_timeout_s,
        )
        if result != "complete":
            raise ApiError(
                504,
                "ICE_GATHERING_TIMEOUT",
                "server ICE gathering did not complete",
            )

    def _session(
        self,
        request: web.Request,
        *,
        create: bool = False,
    ) -> VideoSession:
        return self._get_session(request.match_info["session_id"], create=create)

    def _get_session(
        self,
        session_id: str,
        *,
        create: bool = False,
    ) -> VideoSession:
        session = self.sessions.get(session_id)
        if session is None and create:
            session = VideoSession(
                session_id=session_id,
                pipeline=FramePipeline(self.settings, self.detector),
            )
            self.sessions[session_id] = session
            LOGGER.info("media state created for core session id=%s", session_id)
        if session is None:
            raise ApiError(
                404,
                "SESSION_NOT_FOUND",
                "media state for the offloading session was not found",
            )
        return session

    @staticmethod
    async def _json(request: web.Request) -> dict[str, Any]:
        try:
            value = await request.json()
        except Exception as error:
            raise ApiError(
                400,
                "INVALID_JSON",
                "request body must be a JSON object",
            ) from error
        if not isinstance(value, dict):
            raise ApiError(
                400,
                "INVALID_JSON",
                "request body must be a JSON object",
            )
        return value

    @staticmethod
    def _sdp(body: dict[str, Any], nested_field: str) -> tuple[str, str]:
        nested = body.get(nested_field)
        candidate = nested if isinstance(nested, dict) else body
        sdp_type = candidate.get("type")
        sdp = candidate.get("sdp")
        return (
            sdp_type if isinstance(sdp_type, str) else "",
            sdp if isinstance(sdp, str) else "",
        )

    def status(self) -> dict[str, Any]:
        return {
            "activeSessions": sum(
                session.state != "STOPPED" for session in self.sessions.values()
            ),
            "sessions": len(self.sessions),
            "states": {
                state: sum(
                    session.state == state for session in self.sessions.values()
                )
                for state in (
                    "ALLOCATED",
                    "WAITING_FOR_SOURCE",
                    "SOURCE_CONNECTED",
                    "SOURCE_ENDED",
                    "FAILED",
                    "STOPPED",
                )
            },
        }
