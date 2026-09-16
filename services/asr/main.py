from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
import signal
import tempfile
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web

from services.logging_config import configure_logging
from services.intent.classifier import RuleIntentClassifier

from .config import AsrSettings
from .service import SpeechRecognizer


LOGGER = logging.getLogger("sandbox.asr")
ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
RECOGNIZER_KEY = web.AppKey("recognizer", SpeechRecognizer)
SETTINGS_KEY = web.AppKey("settings", AsrSettings)


def create_app(
    settings: AsrSettings | None = None,
    recognizer: SpeechRecognizer | None = None,
) -> web.Application:
    active_settings = settings or AsrSettings.from_env()
    active_recognizer = recognizer or SpeechRecognizer(active_settings)
    app = web.Application(
        middlewares=[cors_middleware, error_middleware],
        client_max_size=active_settings.max_upload_bytes + 1024 * 1024,
    )
    app[SETTINGS_KEY] = active_settings
    app[RECOGNIZER_KEY] = active_recognizer
    app.add_routes(
        [
            web.get("/health", health),
            web.get("/api/health", health),
            web.post("/api/v1/transcribe", transcribe),
        ]
    )
    return app


async def health(request: web.Request) -> web.Response:
    return _json_response(request.app[RECOGNIZER_KEY].health())


async def transcribe(request: web.Request) -> web.Response:
    temp_path, filename, fields = await _read_audio_upload(request)
    try:
        request_id = _required_request_id(fields)
        result = await request.app[RECOGNIZER_KEY].transcribe(
            temp_path,
            language=_optional(fields.get("language")),
            session_id=request_id,
            task_id=request_id,
            source="glasses",
            stop_reason=None,
            original_filename=filename,
        )
        intent = await _resolve_intent(
            result["text"], request.app[SETTINGS_KEY]
        )
        response: dict[str, Any] = {
            "request_id": request_id,
            "text": result["text"],
            "intent": intent,
        }
        settings = request.app[SETTINGS_KEY]
        if settings.intent_profile == "discovery":
            response["required_skills"] = _discovery_skills(intent)
        return _json_response(response)
    finally:
        temp_path.unlink(missing_ok=True)


async def _read_audio_upload(request: web.Request) -> tuple[Path, str, dict[str, str]]:
    if not request.content_type.startswith("multipart/"):
        raise web.HTTPUnsupportedMediaType(text="multipart/form-data is required")
    reader = await request.multipart()
    settings = request.app[SETTINGS_KEY]
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    fields: dict[str, str] = {}
    temp_path: Path | None = None
    filename = ""
    size = 0
    try:
        async for field in reader:
            if field.name != "file" or not field.filename:
                fields[str(field.name)] = await field.text()
                continue
            if temp_path is not None:
                raise web.HTTPBadRequest(text="only one audio file is allowed")
            filename = Path(field.filename).name
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_AUDIO_SUFFIXES:
                raise web.HTTPBadRequest(text=f"unsupported audio type: {suffix or '<none>'}")
            descriptor, raw_path = tempfile.mkstemp(prefix="upload-", suffix=suffix, dir=settings.temp_dir)
            temp_path = Path(raw_path)
            with os.fdopen(descriptor, "wb") as output:
                while True:
                    chunk = await field.read_chunk(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise web.HTTPRequestEntityTooLarge(
                            max_size=settings.max_upload_bytes,
                            actual_size=size,
                        )
                    output.write(chunk)
        if temp_path is None:
            raise web.HTTPBadRequest(text="file field is required")
        if size == 0:
            raise web.HTTPBadRequest(text="audio file is empty")
        return temp_path, filename, fields
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


@web.middleware
async def error_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    try:
        return await handler(request)
    except web.HTTPException as exc:
        return _json_response(
            {
                "error": {
                    "code": exc.reason.replace(" ", "_").lower(),
                    "message": exc.text,
                }
            },
            status=exc.status,
        )
    except RuntimeError as exc:
        return _json_response(
            {"error": {"code": "service_unavailable", "message": str(exc)}},
            status=503,
        )
    except Exception as exc:
        LOGGER.exception("ASR 请求失败 path=%s", request.path)
        return _json_response(
            {"error": {"code": "internal_error", "message": str(exc)}}, status=500
        )


@web.middleware
async def cors_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    response = web.Response(status=204) if request.method == "OPTIONS" else await handler(request)
    response.headers["Access-Control-Allow-Origin"] = request.app[SETTINGS_KEY].cors_origin
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def _optional(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _required_request_id(fields: dict[str, str]) -> str:
    request_id = str(fields.get("request_id") or "").strip()
    if not request_id:
        raise web.HTTPBadRequest(text="request_id field is required")
    return request_id


def _json_response(payload: Any, *, status: int = 200) -> web.Response:
    return web.json_response(
        payload,
        status=status,
        dumps=lambda value: json.dumps(value, ensure_ascii=False),
    )


async def _resolve_intent(text: str, settings: AsrSettings) -> dict[str, Any]:
    payload: dict[str, Any] | None = None
    if settings.intent_url:
        try:
            async with ClientSession(timeout=ClientTimeout(total=15)) as client:
                async with client.post(settings.intent_url, json={"text": text}) as response:
                    if response.status == 200:
                        value = await response.json(content_type=None)
                        if isinstance(value, dict):
                            payload = value
        except Exception as exc:
            LOGGER.warning("intent service unavailable; using rules error=%s", exc)
    if payload is None:
        payload = RuleIntentClassifier().classify(text).to_dict()
    name = str(payload.get("intent") or payload.get("scene") or "other")
    argument = str(
        payload.get("normalized_argument") or payload.get("argument") or ""
    )
    backend = str(payload.get("backend") or "rules")
    return _public_intent(name, argument, settings.intent_profile, backend)


def _public_intent(
    scene: str, argument: str, profile: str, backend: str = "rules"
) -> dict[str, Any]:
    normalized_scene = str(scene or "other").strip().lower()
    if profile == "discovery" and normalized_scene == "patrol":
        parameters: dict[str, str] = {}
        area = _normalize_area(argument)
        if area:
            parameters["area"] = area
        return {"type": "TASK", "parameters": parameters}
    if profile == "discovery" and normalized_scene == "video_task":
        return {"type": "VIDEO_TASK", "parameters": {}}
    if profile == "discovery" and normalized_scene == "object_recognition":
        return {"type": "OBJECT_RECOGNITION", "parameters": {}}
    if profile == "runtime" and normalized_scene in {"defense", "movement"}:
        direction = "forward" if normalized_scene == "defense" else str(argument).strip()
        if direction not in {"forward", "backward", "left", "right", "wave"}:
            direction = "forward"
        return {
            "executor": "robot dog",
            "intent": "movement",
            "direction": direction,
            "matched": True,
            "backend": backend,
        }
    if profile == "discovery":
        return {"type": "UNKNOWN", "parameters": {}}
    return {
        "executor": None,
        "intent": "other",
        "matched": False,
        "backend": backend,
    }


def _discovery_skills(intent: dict[str, Any]) -> list[str]:
    intent_type = str(intent.get("type") or "").upper()
    if intent_type in {"TASK", "VIDEO_TASK"}:
        return ["patrol", "camera"]
    if intent_type == "OBJECT_RECOGNITION":
        return ["camera"]
    return []


def _normalize_area(argument: str) -> str:
    normalized = str(argument or "").strip()
    if normalized.endswith("区域"):
        normalized = normalized[:-2].strip()
    return normalized


async def _serve_dual_listeners(settings: AsrSettings) -> None:
    """Serve discovery and runtime APIs with one shared Whisper model."""
    discovery_settings = replace(
        settings,
        host=os.getenv("ASR_DISCOVERY_HOST", settings.host).strip() or settings.host,
        port=_listener_port("ASR_DISCOVERY_PORT", settings.port),
        intent_profile="discovery",
    )
    runtime_settings = replace(
        settings,
        host=os.getenv("ASR_RUNTIME_HOST", "127.0.0.1").strip() or "127.0.0.1",
        port=_listener_port("ASR_RUNTIME_PORT", 9005),
        intent_profile="runtime",
    )
    recognizer = SpeechRecognizer(runtime_settings)
    runners = [
        web.AppRunner(create_app(discovery_settings, recognizer)),
        web.AppRunner(create_app(runtime_settings, recognizer)),
    ]
    try:
        for runner in runners:
            await runner.setup()
        await asyncio.gather(
            web.TCPSite(
                runners[0], discovery_settings.host, discovery_settings.port
            ).start(),
            web.TCPSite(
                runners[1], runtime_settings.host, runtime_settings.port
            ).start(),
        )
        LOGGER.info(
            "ASR dual listeners ready discovery=http://%s:%s runtime=http://%s:%s",
            discovery_settings.host,
            discovery_settings.port,
            runtime_settings.host,
            runtime_settings.port,
        )
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signal_name in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signal_name, stopped.set)
            except (NotImplementedError, RuntimeError):
                pass
        await stopped.wait()
    finally:
        await asyncio.gather(*(runner.cleanup() for runner in runners))


def _listener_port(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def main() -> None:
    configure_logging("asr")
    defaults = AsrSettings.from_env()
    parser = argparse.ArgumentParser(description="Sandbox dual-listener ASR service")
    parser.parse_args()
    asyncio.run(_serve_dual_listeners(defaults))


if __name__ == "__main__":
    main()
