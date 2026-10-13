"""Per-peer aiortc policy for the high latency CONNECT-IP video path.

Only consent transactions get the longer response window. Initial ICE checks
keep aioice's normal retransmission policy. The probe clock is independent of
responses, and a missing response still expires consent after 30 seconds.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from types import MethodType
from typing import Any

from aioice import stun
from aiortc import RTCPeerConnection

LOGGER = logging.getLogger(__name__)
CONSENT_RESPONSE_TIMEOUT = 15.0
CONSENT_INTERVAL = 5.0
CONSENT_EXPIRY = 30.0
VIDEO_BITRATE_BPS = 250_000
# Packet pacing is separate from the encoded average bitrate. A higher pacing
# rate completes each frame sooner without increasing the encoder's target.
# Keep an environment override for deployments with constrained tunnel queues.
VIDEO_WIRE_BITRATE_BPS = int(os.getenv("VIDEO_WIRE_BITRATE_BPS", "1000000"))
if VIDEO_WIRE_BITRATE_BPS <= 0:
    raise ValueError("VIDEO_WIRE_BITRATE_BPS must be positive")


class _ConsentTransaction(stun.Transaction):
    def __init__(self, request: Any, addr: Any, protocol: Any, key: bytes,
                 timeout: float) -> None:
        super().__init__(request, addr, protocol, retransmissions=0)
        # aioice 0.10 exposes no per-transaction RTO argument. Keep this
        # compatibility access isolated, rather than changing RETRY_RTO.
        if not hasattr(self, "_Transaction__timeout_delay"):
            raise RuntimeError("unsupported aioice consent transaction API")
        self._Transaction__timeout_delay = timeout
        self._consent_addr = addr
        self._consent_key = key

    def raw_response_received(self, data: bytes, addr: Any) -> None:
        # Verify the original bytes. libwebrtc may include extension attributes
        # or nonzero padding that aioice's decoded Message cannot reproduce.
        if addr != self._consent_addr:
            return
        try:
            message = stun.parse_message(data, integrity_key=self._consent_key)
        except ValueError:
            return
        if (message.message_method == stun.Method.BINDING
                and message.message_class in {stun.Class.RESPONSE, stun.Class.ERROR}
                and "MESSAGE-INTEGRITY" in message.attributes):
            super().response_received(message, addr)

    def response_received(self, message: Any, addr: Any) -> None:
        if addr != self._consent_addr or message.message_method != stun.Method.BINDING:
            return
        if "MESSAGE-INTEGRITY" not in message.attributes:
            return
        try:
            stun.parse_message(bytes(message), integrity_key=self._consent_key)
        except ValueError:
            return
        super().response_received(message, addr)


async def _consent_request(connection: Any, pair: Any, timeout: float) -> None:
    request = connection.build_request(pair, nominate=False)
    key = connection.remote_password.encode("utf8")
    request.add_message_integrity(key)
    transaction = _ConsentTransaction(request, pair.remote_addr, pair.protocol, key, timeout)
    protocol = pair.protocol
    if not getattr(protocol, "_consent_raw_receiver", False):
        original = protocol.datagram_received

        def receive(self: Any, data: bytes, addr: Any) -> None:
            tx = self.transactions.get(data[8:20]) if len(data) >= 20 and data[4:8] == b"\x21\x12\xa4\x42" else None
            if isinstance(tx, _ConsentTransaction):
                tx.raw_response_received(data, (addr[0], addr[1]))
            else:
                original(data, addr)

        protocol.datagram_received = MethodType(receive, protocol)
        protocol._consent_raw_receiver = True
    pair.protocol.transactions[request.transaction_id] = transaction
    try:
        await transaction.run()
    finally:
        pair.protocol.transactions.pop(request.transaction_id, None)


async def query_consent(connection: Any, *, response_timeout: float = CONSENT_RESPONSE_TIMEOUT,
                        interval: float = CONSENT_INTERVAL,
                        expiry: float = CONSENT_EXPIRY) -> None:
    loop = asyncio.get_running_loop()
    last_response = {component: loop.time() for component in connection._nominated}
    pending: set[asyncio.Task[Any]] = set()
    wake = asyncio.Event()
    revoked = False

    async def probe(component: int, pair: Any) -> None:
        nonlocal revoked
        sent_at = loop.time()
        try:
            await _consent_request(connection, pair, response_timeout)
        except stun.TransactionFailed as exc:
            if exc.response.attributes.get("ERROR-CODE", (0,))[0] == 403:
                revoked = True
                wake.set()
        except stun.TransactionTimeout:
            LOGGER.warning("rtc_consent_probe_timeout connection=%s component=%s wait_seconds=%s", connection, component, response_timeout)
        else:
            now = loop.time()
            if connection._nominated.get(component) is pair and now - last_response[component] < expiry:
                last_response[component] = now
                LOGGER.info("rtc_consent_response connection=%s component=%s rtt_ms=%.1f", connection, component, 1000 * (now - sent_at))
                wake.set()

    next_probe = loop.time() + interval * random.uniform(0.8, 1.2)
    try:
        while connection._nominated:
            now = loop.time()
            if revoked or any(now - value >= expiry for value in last_response.values()):
                LOGGER.warning("rtc_consent_expired connection=%s reason=%s expiry_seconds=%s", connection, "revoked" if revoked else "no-valid-response", expiry)
                connection._query_consent_task = None
                await connection.close()
                return
            if now >= next_probe:
                for component, pair in list(connection._nominated.items()):
                    task = asyncio.create_task(probe(component, pair))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                next_probe = now + interval * random.uniform(0.8, 1.2)
            deadline = min(next_probe, min(last_response.values()) + expiry)
            wake.clear()
            try:
                await asyncio.wait_for(wake.wait(), max(0, deadline - loop.time()))
            except asyncio.TimeoutError:
                pass
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


def configure_consent(connection: Any) -> None:
    if getattr(connection, "_high_latency_consent", False):
        return
    if connection._query_consent_task is not None:
        raise RuntimeError("consent policy must be installed before ICE connects")
    connection.query_consent = MethodType(query_consent, connection)
    connection._high_latency_consent = True
    LOGGER.info("rtc_consent_policy response_timeout_seconds=%s interval_seconds=%s expiry_seconds=%s",
                CONSENT_RESPONSE_TIMEOUT, CONSENT_INTERVAL, CONSENT_EXPIRY)


class HighLatencyPeerConnection(RTCPeerConnection):
    def _configure_consent(self) -> None:
        transports = {item.sender.transport.transport for item in self.getTransceivers()}
        if self.sctp is not None:
            transports.add(self.sctp.transport.transport)
        for transport in transports:
            configure_consent(transport._connection)

    async def setLocalDescription(self, description: Any = None) -> None:
        self._configure_consent()
        await super().setLocalDescription(description)
        self._configure_consent()

    async def setRemoteDescription(self, description: Any) -> None:
        self._configure_consent()
        await super().setRemoteDescription(description)
        self._configure_consent()


class _BoundedEncoder:
    def __init__(self, encoder: Any, maximum: int) -> None:
        self._encoder = encoder
        self._maximum = maximum
        self._capture_timestamp: int | None = None
        self._encode_timestamp = 0
        self._last_encode_report = 0.0
        self.target_bitrate = maximum
        # H264's encoder floor is 500 kbps, VP8's is 250 kbps. Keep
        # receiver feedback bounded at the effective codec-specific floor.
        self._maximum = max(maximum, self.target_bitrate)

    @property
    def target_bitrate(self) -> int:
        return self._encoder.target_bitrate

    @target_bitrate.setter
    def target_bitrate(self, value: int) -> None:
        self._encoder.target_bitrate = min(value, self._maximum)

    def encode(self, frame: Any, force_keyframe: bool = False) -> Any:
        from av import VideoFrame
        from aiortc.codecs.vpx import Vp8Encoder
        from aiortc.mediastreams import VIDEO_TIME_BASE, convert_timebase

        if not isinstance(self._encoder, Vp8Encoder):
            return self._encoder.encode(frame, force_keyframe)
        started = time.monotonic()
        timestamp = convert_timebase(frame.pts, frame.time_base, VIDEO_TIME_BASE)
        if self._capture_timestamp is not None:
            # libvpx CBR uses PTS gaps to budget the next frame. After a
            # network stall this can produce an oversized frame and prolong
            # the stall. Cap that budget at 100 ms while retaining the real
            # capture clock on RTP. Faster sources keep their own cadence.
            self._encode_timestamp += max(1, min(9000, timestamp - self._capture_timestamp))
        self._capture_timestamp = timestamp
        # The source frame may also be used by another consumer. Never
        # rewrite its timestamps from the encoder's worker thread.
        yuv = frame.reformat(format="yuv420p")
        encoded_frame = VideoFrame(yuv.width, yuv.height, "yuv420p")
        for source, destination in zip(yuv.planes, encoded_frame.planes):
            if source.line_size == destination.line_size:
                destination.update(source)
            else:
                data = bytearray(destination.buffer_size)
                view = memoryview(source)
                for row in range(destination.height):
                    start = row * destination.line_size
                    offset = row * source.line_size
                    data[start:start + destination.width] = view[offset:offset + destination.width]
                destination.update(data)
        encoded_frame.pts = self._encode_timestamp
        encoded_frame.time_base = VIDEO_TIME_BASE
        payloads, _ = self._encoder.encode(encoded_frame, force_keyframe)
        if started - self._last_encode_report >= 5:
            self._last_encode_report = started
            LOGGER.info("rtc_video_encoded payload_bytes=%s packets=%s encode_ms=%.1f keyframe_requested=%s",
                        sum(map(len, payloads)), len(payloads),
                        (time.monotonic() - started) * 1000, force_keyframe)
        return payloads, timestamp

    def pack(self, *args: Any, **kwargs: Any) -> Any:
        return self._encoder.pack(*args, **kwargs)


def limit_sender_bitrate(sender: Any, maximum: int = VIDEO_BITRATE_BPS,
                         *, wire_bitrate: int = VIDEO_WIRE_BITRATE_BPS) -> None:
    """Enforce the encoder cap even when receiver REMB asks for more bandwidth."""
    if maximum < 250_000:
        raise ValueError("aiortc video requires at least 250000 bits/s")
    if getattr(sender, "_video_bitrate_limit", None) == maximum:
        return
    if not hasattr(sender, "_RTCRtpSender__encoder"):
        raise RuntimeError("unsupported aiortc encoder API")
    original = sender._next_encoded_frame

    async def next_frame(self: Any, codec: Any) -> Any:
        from aiortc.codecs import get_encoder
        encoder = self._RTCRtpSender__encoder
        if encoder is None:
            encoder = get_encoder(codec)
        if not isinstance(encoder, _BoundedEncoder):
            self._RTCRtpSender__encoder = _BoundedEncoder(encoder, maximum)
        return await original(codec)

    sender._next_encoded_frame = MethodType(next_frame, sender)
    sender._video_bitrate_limit = maximum
    pace_video_transport(sender.transport, wire_bitrate)
    LOGGER.info("rtc_encoder_bitrate_limit bits_per_second=%s", maximum)


class _PacketPacer:
    def __init__(self, bits_per_second: int) -> None:
        self._rate = bits_per_second
        self._next_send = 0.0
        self._lock = asyncio.Lock()

    async def wait(self, packet_bytes: int) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if self._next_send > now:
                await asyncio.sleep(self._next_send - now)
            # No accumulated credit: a delayed encoder/keyframe must not
            # burst into the core's 64-message mailbox after a pause.
            self._next_send = loop.time() + (packet_bytes + 64) * 8 / self._rate


def pace_video_transport(transport: Any, rate: int = VIDEO_WIRE_BITRATE_BPS) -> None:
    if getattr(transport, "_video_packet_pacer", None) is not None:
        return
    original = transport._send_rtp
    pacer = _PacketPacer(rate)

    async def send(self: Any, packet: bytes) -> None:
        if len(packet) >= 12 and packet[0] >> 6 == 2 and not 192 <= packet[1] <= 223:
            await pacer.wait(len(packet))
        await original(packet)

    transport._send_rtp = MethodType(send, transport)
    transport._video_packet_pacer = pacer
    LOGGER.info("rtc_video_packet_pacing wire_bits_per_second=%s", rate)
