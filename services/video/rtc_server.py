from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
import secrets
from typing import Any
import uuid

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack

from .config import VideoSettings
from .detector import YoloDetector
from .ice import build_rtc_configuration, wait_for_ice_gathering
from .pipeline import FramePipeline


LOGGER = logging.getLogger("sandbox.video.rtc_server")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def rfc3339(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(slots=True)
class ConsumerGrant:
    agent_id: str
    ticket: str
    peer_connections: set[RTCPeerConnection] = field(default_factory=set)


@dataclass(slots=True)
class VideoSession:
    session_id: str
    sandbox_id: str
    group_id: str
    source_agent_id: str
    workload_type: str
    producer_token: str
    expires_at: datetime
    pipeline: FramePipeline
    state: str = "ALLOCATED"
    producer_pc: RTCPeerConnection | None = None
    source_track: MediaStreamTrack | None = None
    consumers: dict[str, ConsumerGrant] = field(default_factory=dict)

    @property
    def frames_seen(self) -> int:
        return self.pipeline.received_frames

    async def close(self) -> None:
        self.state = "STOPPED"
        await self.pipeline.close()
        peers: set[RTCPeerConnection] = set()
        if self.producer_pc is not None:
            peers.add(self.producer_pc)
        for grant in self.consumers.values():
            peers.update(grant.peer_connections)
        await asyncio.gather(*(pc.close() for pc in peers), return_exceptions=True)
        self.producer_pc = None
        self.source_track = None


class OrangeVideoServer:
    """Orange Agent SDK-compatible signaling with per-session YOLO processing."""

    def __init__(self, settings: VideoSettings, detector: YoloDetector) -> None:
        self.settings = settings
        self.detector = detector
        self.sessions: dict[str, VideoSession] = {}

    @property
    def base_url(self) -> str:
        return f"http://{self.settings.public_ip}:{self.settings.port}"

    async def health(self, _: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "service": "agent-sdk-mock-video-server",
                "video_server_ip": self.settings.public_ip,
                "sessions": len(self.sessions),
            }
        )

    async def list_sessions(self, _: web.Request) -> web.Response:
        return web.json_response(
            {
                "sessions": [
                    {
                        "session_id": session.session_id,
                        "group_id": session.group_id,
                        "source_agent_id": session.source_agent_id,
                        "state": session.state,
                        "frames_seen": session.frames_seen,
                        "consumer_agents": sorted(session.consumers),
                        "consumer_connections": sum(
                            len(grant.peer_connections)
                            for grant in session.consumers.values()
                        ),
                    }
                    for session in self.sessions.values()
                ]
            }
        )

    async def create_session(self, request: web.Request) -> web.Response:
        body = await self._json(request)
        agent_id = self._required_string(body, "agent_id")
        group_id = self._required_string(body, "group_id")
        workload_type = self._required_string(body, "workload_type")
        session_id = f"mock-{uuid.uuid4()}"
        sandbox_id = str(body.get("preferred_sandbox_id") or "mock-video-sandbox")
        session = VideoSession(
            session_id=session_id,
            sandbox_id=sandbox_id,
            group_id=group_id,
            source_agent_id=agent_id,
            workload_type=workload_type,
            producer_token=secrets.token_urlsafe(32),
            expires_at=utc_now() + timedelta(seconds=self.settings.session_ttl_seconds),
            pipeline=FramePipeline(self.settings, self.detector),
        )
        self.sessions[session_id] = session
        LOGGER.info(
            "session allocated id=%s source=%s group=%s",
            session_id,
            agent_id,
            group_id,
        )
        return web.json_response(
            {
                "session_id": session_id,
                "sandbox_id": sandbox_id,
                "group_id": group_id,
                "source_agent_id": agent_id,
                "workload_type": workload_type,
                "state": session.state,
                "expires_at": rfc3339(session.expires_at),
                "producer": {
                    "video_server_ip": self.settings.public_ip,
                    "source_start_url": (
                        f"{self.base_url}/video/v1/sessions/{session_id}/source"
                    ),
                    "source_stop_url": (
                        f"{self.base_url}/video/v1/sessions/{session_id}/source/stop"
                    ),
                    "access_token": session.producer_token,
                },
            },
            status=201,
        )

    async def create_consumers(self, request: web.Request) -> web.Response:
        session = self._session(request)
        body = await self._json(request)
        if body.get("agent_id") != session.source_agent_id:
            raise ApiError(
                403,
                "SOURCE_AGENT_MISMATCH",
                "agent_id is not the session source",
            )
        if body.get("group_id") != session.group_id:
            raise ApiError(
                409,
                "GROUP_MISMATCH",
                "group_id does not match the session",
            )
        targets = body.get("target_agent_ids")
        if (
            not isinstance(targets, list)
            or not targets
            or any(
                not isinstance(value, str) or not value.strip()
                for value in targets
            )
        ):
            raise ApiError(
                400,
                "INVALID_TARGETS",
                "target_agent_ids must be a non-empty string list",
            )
        if len(set(targets)) != len(targets):
            raise ApiError(
                400,
                "INVALID_TARGETS",
                "target_agent_ids contains duplicates",
            )
        if session.state != "SOURCE_CONNECTED":
            raise ApiError(
                409,
                "SOURCE_NOT_CONNECTED",
                "Video Server has not received a source frame",
            )

        response: dict[str, Any] = {}
        for target in targets:
            grant = ConsumerGrant(agent_id=target, ticket=secrets.token_urlsafe(32))
            session.consumers[target] = grant
            response[target] = {
                "video_server_ip": self.settings.public_ip,
                "offer_url": (
                    f"{self.base_url}/video/v1/sessions/{session.session_id}/processed"
                ),
                "access_ticket": grant.ticket,
                "protocol": "webrtc",
                "signaling": "non-trickle",
            }
        LOGGER.info("consumer grants id=%s targets=%s", session.session_id, targets)
        return web.json_response(
            {"session_id": session.session_id, "consumers": response}
        )

    async def source(self, request: web.Request) -> web.Response:
        session = self._session(request)
        self._require_bearer(request, session.producer_token)
        body = await self._json(request)
        sdp_type, sdp = self._sdp(body, "sdp_answer")

        if not sdp:
            await self._reset_source(session)
            session.state = "WAITING_FOR_SOURCE"
            pc = RTCPeerConnection(
                configuration=build_rtc_configuration(self.settings)
            )
            session.producer_pc = pc
            pc.addTransceiver("video", direction="recvonly")

            @pc.on("track")
            def on_track(track: MediaStreamTrack) -> None:
                if track.kind != "video":
                    return
                session.source_track = track
                asyncio.create_task(
                    session.pipeline.attach_source(track, f"source:{session.session_id}")
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

            try:
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)
                ice_result = await wait_for_ice_gathering(
                    pc,
                    self.settings.rtc_ice_gather_timeout_s,
                )
                if ice_result != "complete":
                    raise ApiError(
                        504,
                        "ICE_GATHERING_TIMEOUT",
                        "server ICE gathering did not complete",
                    )
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
        self._require_bearer(request, session.producer_token)
        await session.close()
        return web.json_response(
            {"session_id": session.session_id, "state": session.state}
        )

    async def processed(self, request: web.Request) -> web.Response:
        session = self._session(request)
        if session.state != "SOURCE_CONNECTED" or session.source_track is None:
            raise ApiError(
                409,
                "SOURCE_NOT_CONNECTED",
                "processed stream is not ready",
            )
        ticket = self._bearer(request)
        grant = next(
            (
                value
                for value in session.consumers.values()
                if value.ticket == ticket
            ),
            None,
        )
        if grant is None:
            raise ApiError(
                403,
                "INVALID_CONSUMER_TICKET",
                "consumer ticket is invalid",
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
        grant.peer_connections.add(pc)

        @pc.on("connectionstatechange")
        async def on_consumer_state_change() -> None:
            LOGGER.info(
                "consumer pc id=%s agent=%s state=%s",
                session.session_id,
                grant.agent_id,
                pc.connectionState,
            )
            if pc.connectionState in {"failed", "closed"}:
                grant.peer_connections.discard(pc)

        try:
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=sdp, type=sdp_type)
            )
            pc.addTrack(session.pipeline.new_output_track())
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            ice_result = await wait_for_ice_gathering(
                pc,
                self.settings.rtc_ice_gather_timeout_s,
            )
            if ice_result != "complete":
                raise ApiError(
                    504,
                    "ICE_GATHERING_TIMEOUT",
                    "server ICE gathering did not complete",
                )
            local = pc.localDescription
        except Exception:
            grant.peer_connections.discard(pc)
            await pc.close()
            raise

        LOGGER.info(
            "consumer connected id=%s agent=%s",
            session.session_id,
            grant.agent_id,
        )
        return web.json_response(
            {
                "session_id": session.session_id,
                "consumer_agent_id": grant.agent_id,
                "state": "STREAMING",
                "sdp_answer": {"type": local.type, "sdp": local.sdp},
            }
        )

    async def delete_session(self, request: web.Request) -> web.Response:
        session_id = request.match_info["session_id"]
        session = self.sessions.pop(session_id, None)
        if session is None:
            return web.json_response({"sessionId": session_id, "closed": False})
        await session.close()
        return web.json_response({"sessionId": session_id, "closed": True})

    async def close(self) -> None:
        await asyncio.gather(
            *(session.close() for session in self.sessions.values()),
            return_exceptions=True,
        )

    async def _reset_source(self, session: VideoSession) -> None:
        old_pc = session.producer_pc
        session.producer_pc = None
        session.source_track = None
        await session.pipeline.clear_source()
        if old_pc is not None:
            await old_pc.close()

    def _session(self, request: web.Request) -> VideoSession:
        session = self.sessions.get(request.match_info["session_id"])
        if session is None:
            raise ApiError(
                404,
                "SESSION_NOT_FOUND",
                "offloading session was not found",
            )
        if session.expires_at <= utc_now():
            raise ApiError(
                410,
                "SESSION_EXPIRED",
                "offloading session has expired",
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
    def _required_string(body: dict[str, Any], field_name: str) -> str:
        value = body.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ApiError(
                400,
                "INVALID_ARGUMENT",
                f"{field_name} must be a non-empty string",
            )
        return value

    @staticmethod
    def _bearer(request: web.Request) -> str:
        authorization = request.headers.get("Authorization", "")
        if not authorization.startswith("Bearer ") or not authorization[7:].strip():
            raise ApiError(
                401,
                "MISSING_BEARER",
                "Authorization Bearer credential is required",
            )
        return authorization[7:].strip()

    def _require_bearer(self, request: web.Request, expected: str) -> None:
        if not secrets.compare_digest(self._bearer(request), expected):
            raise ApiError(
                403,
                "INVALID_PRODUCER_TOKEN",
                "producer token is invalid",
            )

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
                state: sum(session.state == state for session in self.sessions.values())
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
