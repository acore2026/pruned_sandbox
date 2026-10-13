from __future__ import annotations

import asyncio
from typing import Any

from aiortc import RTCConfiguration, RTCIceServer

from .config import VideoSettings


def public_host_candidates(sdp: str, public_ip: str) -> str:
    """Advertise the configured reachable interface, without VPN route churn.

    Keep discovery behavior when the configured address isn't a local ICE
    candidate (for example a NAT deployment using server-reflexive candidates).
    """
    lines = sdp.splitlines()
    public = []
    for line in lines:
        if line.startswith("a=candidate:"):
            parts = line.split()
            if len(parts) >= 8 and parts[4] == public_ip and parts[7] == "host":
                public.append(parts)
    if not public:
        return sdp
    result = []
    for line in lines:
        if line.startswith("a=candidate:"):
            parts = line.split()
            if len(parts) >= 8 and parts[7] == "host" and parts[4] != public_ip:
                continue
        result.append(line)
    return "\r\n".join(result) + "\r\n"


def build_rtc_configuration(settings: VideoSettings) -> RTCConfiguration:
    servers: list[RTCIceServer] = []
    for url in settings.rtc_ice_servers:
        kwargs: dict[str, Any] = {"urls": url}
        if settings.rtc_ice_username:
            kwargs["username"] = settings.rtc_ice_username
        if settings.rtc_ice_credential:
            kwargs["credential"] = settings.rtc_ice_credential
        servers.append(RTCIceServer(**kwargs))
    return RTCConfiguration(iceServers=servers)


async def wait_for_ice_gathering(pc: Any, timeout_s: float) -> str:
    if pc.iceGatheringState == "complete":
        return "complete"
    complete = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_ice_gathering_state_change() -> None:
        if pc.iceGatheringState == "complete":
            complete.set()

    try:
        await asyncio.wait_for(complete.wait(), timeout=timeout_s)
        return "complete"
    except asyncio.TimeoutError:
        return "timeout"
