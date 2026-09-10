from __future__ import annotations

import asyncio
from typing import Any

from aiortc import RTCConfiguration, RTCIceServer

from .config import VideoSettings


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
