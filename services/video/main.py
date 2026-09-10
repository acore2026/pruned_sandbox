from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import logging
import os
from typing import Any

from aiohttp import web

from .config import VideoSettings
from .detector import YoloDetector
from .rtc_server import ApiError, OrangeVideoServer


LOGGER = logging.getLogger("sandbox.video")


@dataclass(slots=True)
class VideoRuntime:
    settings: VideoSettings
    detector: YoloDetector
    server: OrangeVideoServer

    @classmethod
    def build(cls, settings: VideoSettings) -> "VideoRuntime":
        detector = YoloDetector(settings)
        return cls(
            settings=settings,
            detector=detector,
            server=OrangeVideoServer(settings, detector),
        )

    async def close(self) -> None:
        await self.server.close()


RUNTIME_KEY = web.AppKey("video_runtime", VideoRuntime)


def create_app(
    settings: VideoSettings | None = None,
    runtime: VideoRuntime | None = None,
) -> web.Application:
    active_settings = settings or VideoSettings.from_env()
    active_runtime = runtime or VideoRuntime.build(active_settings)
    app = web.Application(
        middlewares=[cors_middleware, error_middleware],
        client_max_size=2 * 1024 * 1024,
    )
    app[RUNTIME_KEY] = active_runtime
    app.add_routes(
        [
            # Orange Agent SDK-compatible contract.
            web.get("/healthz", active_runtime.server.health),
            web.get("/debug/v1/sessions", active_runtime.server.list_sessions),
            web.post(
                "/compute/v1/offloading-sessions",
                active_runtime.server.create_session,
            ),
            web.post(
                "/compute/v1/offloading-sessions/{session_id}/consumers",
                active_runtime.server.create_consumers,
            ),
            web.post(
                "/video/v1/sessions/{session_id}/source",
                active_runtime.server.source,
            ),
            web.post(
                "/video/v1/sessions/{session_id}/source/stop",
                active_runtime.server.stop_source,
            ),
            web.post(
                "/video/v1/sessions/{session_id}/processed",
                active_runtime.server.processed,
            ),
            # Sandbox diagnostics and YOLO target configuration extensions.
            web.get("/health", health),
            web.get("/api/health", health),
            web.get("/api/v1/detection/classes", get_detection_classes),
            web.post("/api/v1/detection/classes", set_detection_classes),
            web.delete(
                "/api/v1/webrtc/sessions/{session_id}",
                active_runtime.server.delete_session,
            ),
        ]
    )

    async def shutdown(_: web.Application) -> None:
        await active_runtime.close()

    app.on_shutdown.append(shutdown)
    return app


async def health(request: web.Request) -> web.Response:
    runtime = request.app[RUNTIME_KEY]
    pipelines = [session.pipeline for session in runtime.server.sessions.values()]
    return web.json_response(
        {
            "ok": True,
            "service": "video",
            "videoServerIp": runtime.settings.public_ip,
            "yolo": runtime.detector.health(),
            "webrtc": runtime.server.status(),
            "frames": {
                "received": sum(item.received_frames for item in pipelines),
                "processed": sum(item.processed_frames for item in pipelines),
                "dropped": sum(item.dropped_frames for item in pipelines),
            },
        }
    )


async def get_detection_classes(request: web.Request) -> web.Response:
    classes = request.app[RUNTIME_KEY].detector.health()["classes"]
    return web.json_response({"classes": classes})


async def set_detection_classes(request: web.Request) -> web.Response:
    payload = await request.app[RUNTIME_KEY].server._json(request)
    classes = payload.get("classes")
    if not isinstance(classes, list) or not all(
        isinstance(item, str) for item in classes
    ):
        raise ApiError(400, "INVALID_CLASSES", "classes must be a string array")
    normalized = request.app[RUNTIME_KEY].detector.set_classes(classes)
    return web.json_response({"classes": normalized})


@web.middleware
async def error_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    try:
        return await handler(request)
    except ApiError as exc:
        return web.json_response(
            {"error": exc.code, "message": exc.message},
            status=exc.status,
        )
    except web.HTTPException as exc:
        return web.json_response(
            {
                "error": exc.reason.replace(" ", "_").upper(),
                "message": exc.text,
            },
            status=exc.status,
        )
    except Exception as exc:
        LOGGER.exception("视频服务请求失败 path=%s", request.path)
        return web.json_response(
            {"error": "INTERNAL_ERROR", "message": str(exc)},
            status=500,
        )


@web.middleware
async def cors_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    response = web.Response(status=204) if request.method == "OPTIONS" else await handler(request)
    origin = request.app[RUNTIME_KEY].settings.cors_origin
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    return response


def parse_args(defaults: VideoSettings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orange Agent SDK-compatible WebRTC + YOLO video server"
    )
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--public-ip", default=defaults.public_ip)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    defaults = VideoSettings.from_env()
    args = parse_args(defaults)
    settings = replace(
        defaults,
        host=args.host,
        port=args.port,
        public_ip=args.public_ip,
    )
    web.run_app(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        print=None,
    )


if __name__ == "__main__":
    main()
