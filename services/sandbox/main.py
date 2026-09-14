from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
import hashlib
import json
import logging
import os
import signal
import time
from typing import Any

from aiohttp import web

from services.logging_config import configure_logging

from services.video.config import VideoSettings
from services.video.detector import YoloDetector
from services.video.rtc_server import ApiError, OrangeVideoServer

from .api import SandboxApi
from .control import HttpProducerControlAdapter


LOGGER = logging.getLogger("sandbox.service")


@dataclass(slots=True)
class VideoRuntime:
    settings: VideoSettings
    detector: YoloDetector
    server: OrangeVideoServer
    api: SandboxApi

    @classmethod
    def build(cls, settings: VideoSettings) -> "VideoRuntime":
        detector = YoloDetector(settings)
        server = OrangeVideoServer(settings, detector)
        return cls(
            settings=settings,
            detector=detector,
            server=server,
            api=SandboxApi(
                server,
                settings.management_token,
                settings.intent_url,
                HttpProducerControlAdapter(
                    settings.producer_control_url,
                    settings.producer_control_timeout_s,
                ),
            ),
        )

    async def close(self) -> None:
        await self.api.close()
        await self.server.close()


RUNTIME_KEY = web.AppKey("video_runtime", VideoRuntime)


def create_app(
    settings: VideoSettings | None = None,
    runtime: VideoRuntime | None = None,
) -> web.Application:
    active_settings = settings or VideoSettings.from_env()
    active_runtime = runtime or VideoRuntime.build(active_settings)
    app = web.Application(
        middlewares=[traffic_log_middleware, cors_middleware, error_middleware],
        client_max_size=max(2 * 1024 * 1024, active_settings.audio_max_upload_bytes + 1024 * 1024),
    )
    app[RUNTIME_KEY] = active_runtime
    app.add_routes(
        [
            # Orange Agent SDK media-plane contract. The core network owns
            # offloading-session allocation and lifecycle management.
            web.get("/healthz", active_runtime.server.health),
            web.get("/debug/v1/sessions", active_runtime.server.list_sessions),
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
        ]
    )
    active_runtime.api.register_routes(app)

    async def shutdown(_: web.Application) -> None:
        await active_runtime.close()

    app.on_shutdown.append(shutdown)
    return app


def create_management_app(runtime: VideoRuntime) -> web.Application:
    app = _base_app(runtime)
    app.add_routes([web.get("/healthz", management_health)])
    runtime.api.register_management_routes(app)
    return app


def create_user_app(runtime: VideoRuntime) -> web.Application:
    app = _base_app(runtime)
    app.add_routes(
        [
            web.get("/healthz", runtime.server.health),
            web.get("/debug/v1/sessions", runtime.server.list_sessions),
            web.post("/video/v1/sessions/{session_id}/source", runtime.server.source),
            web.post(
                "/video/v1/sessions/{session_id}/source/stop",
                runtime.server.stop_source,
            ),
            web.post(
                "/video/v1/sessions/{session_id}/processed",
                runtime.server.processed,
            ),
            web.get("/health", health),
            web.get("/api/health", health),
            web.get("/api/v1/detection/classes", get_detection_classes),
            web.post("/api/v1/detection/classes", set_detection_classes),
        ]
    )
    runtime.api.register_user_routes(app)
    return app


def _base_app(runtime: VideoRuntime) -> web.Application:
    app = web.Application(
        middlewares=[traffic_log_middleware, cors_middleware, error_middleware],
        client_max_size=max(
            2 * 1024 * 1024,
            runtime.settings.audio_max_upload_bytes + 1024 * 1024,
        ),
    )
    app[RUNTIME_KEY] = runtime
    return app


async def management_health(request: web.Request) -> web.Response:
    runtime = request.app[RUNTIME_KEY]
    binding_records = len(runtime.api.bindings)
    active_bindings = sum(
        record.state != "UNBOUND" for record in runtime.api.bindings.values()
    )
    return web.json_response(
        {
            "ok": True,
            "service": "sandbox-management",
            "bindings": active_bindings,
            "binding_records": binding_records,
        }
    )


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
async def traffic_log_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    """Log API payloads without persisting credentials or large media bodies."""
    if request.path in {"/healthz", "/health", "/api/health"}:
        return await handler(request)

    started = time.monotonic()
    request_body = await _logged_request_body(request)
    response = await handler(request)
    elapsed_ms = round((time.monotonic() - started) * 1000, 2)
    LOGGER.info(
        "api_exchange remote=%s method=%s path=%s status=%s duration_ms=%s "
        "request_id=%s request=%s response=%s",
        request.remote or "-",
        request.method,
        request.path_qs,
        response.status,
        elapsed_ms,
        request.headers.get("X-Request-ID", "-"),
        _compact_json(request_body),
        _compact_json(_logged_response_body(response)),
    )
    return response


async def _logged_request_body(request: web.Request) -> Any:
    content_type = request.content_type.lower()
    if content_type.startswith("multipart/") or content_type.startswith("audio/"):
        return {"body": "<binary omitted>", "content_length": request.content_length}
    if content_type != "application/json" and not content_type.endswith("+json"):
        return {"body": "<not JSON>", "content_length": request.content_length}
    raw = await request.read()
    if not raw:
        return None
    try:
        return _sanitize_log_value(json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"body": "<invalid JSON>", "bytes": len(raw)}


def _logged_response_body(response: web.StreamResponse) -> Any:
    if not isinstance(response, web.Response) or response.body is None:
        return {"body": "<streamed or empty>"}
    if response.content_type != "application/json":
        return {"body": "<non-JSON>", "bytes": len(response.body)}
    try:
        return _sanitize_log_value(json.loads(response.body))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"body": "<invalid JSON response>", "bytes": len(response.body)}


def _sanitize_log_value(value: Any, key: str = "") -> Any:
    lowered = key.lower()
    if any(marker in lowered for marker in ("authorization", "token", "password", "secret")):
        return "<redacted>"
    if lowered == "sdp" and isinstance(value, str):
        return {
            "omitted": "sdp",
            "bytes": len(value.encode("utf-8")),
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(item_key): _sanitize_log_value(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_log_value(item, key) for item in value]
    if isinstance(value, str) and len(value) > 1000:
        return {"omitted": "long string", "characters": len(value)}
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


@web.middleware
async def error_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    try:
        return await handler(request)
    except ApiError as exc:
        if request.path.startswith(("/v1/", "/management/v1/")):
            return web.json_response(
                {"error": {"code": exc.code, "message": exc.message}},
                status=exc.status,
            )
        return web.json_response(
            {"error": exc.code, "message": exc.message},
            status=exc.status,
        )
    except web.HTTPException as exc:
        code = exc.reason.replace(" ", "-").lower()
        if request.path.startswith(("/v1/", "/management/v1/")):
            return web.json_response(
                {"error": {"code": code, "message": exc.text}},
                status=exc.status,
            )
        return web.json_response(
            {
                "error": exc.reason.replace(" ", "_").upper(),
                "message": exc.text,
            },
            status=exc.status,
        )
    except Exception as exc:
        LOGGER.exception("Sandbox服务请求失败 path=%s", request.path)
        if request.path.startswith(("/v1/", "/management/v1/")):
            return web.json_response(
                {
                    "error": {
                        "code": "internal-error",
                        "message": "internal Sandbox error",
                    }
                },
                status=500,
            )
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
    response.headers["Access-Control-Allow-Methods"] = (
        "GET, POST, PUT, DELETE, OPTIONS"
    )
    return response


def parse_args(defaults: VideoSettings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Free6GC computing Sandbox service"
    )
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument(
        "--user-port",
        "--port",
        dest="user_port",
        type=int,
        default=defaults.port,
        help="N6 user-plane port (--port is retained as a legacy alias)",
    )
    parser.add_argument("--management-host", default=defaults.management_host)
    parser.add_argument("--management-port", type=int, default=defaults.management_port)
    parser.add_argument("--public-ip", default=defaults.public_ip)
    return parser.parse_args()


async def serve(settings: VideoSettings) -> None:
    runtime = VideoRuntime.build(settings)
    management_runner = web.AppRunner(create_management_app(runtime))
    user_runner = web.AppRunner(create_user_app(runtime))
    await management_runner.setup()
    await user_runner.setup()
    await web.TCPSite(
        management_runner, settings.management_host, settings.management_port
    ).start()
    await web.TCPSite(user_runner, settings.host, settings.port).start()
    LOGGER.info(
        "Sandbox management=%s:%d user=%s:%d",
        settings.management_host,
        settings.management_port,
        settings.host,
        settings.port,
    )
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stopped.set)
        except NotImplementedError:
            pass
    try:
        await stopped.wait()
    finally:
        await management_runner.cleanup()
        await user_runner.cleanup()
        await runtime.close()


def main() -> None:
    configure_logging("sandbox")
    defaults = VideoSettings.from_env()
    args = parse_args(defaults)
    settings = replace(
        defaults,
        host=args.host,
        port=args.user_port,
        management_host=args.management_host,
        management_port=args.management_port,
        public_ip=args.public_ip,
    )
    asyncio.run(serve(settings))


if __name__ == "__main__":
    main()
