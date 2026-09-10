from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

from aiohttp import web

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
    return web.json_response(request.app[RECOGNIZER_KEY].health())


async def transcribe(request: web.Request) -> web.Response:
    temp_path, filename, fields = await _read_audio_upload(request)
    try:
        result = await request.app[RECOGNIZER_KEY].transcribe(
            temp_path,
            language=_optional(fields.get("language")),
            session_id=fields.get("session_id", "demo-room"),
            task_id=fields.get("task_id", "demo-room"),
            source=fields.get("source", "client"),
            stop_reason=_optional(fields.get("stop_reason")),
            original_filename=filename,
        )
        return web.json_response(result)
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
        return web.json_response(
            {"error": exc.reason.replace(" ", "_").lower(), "message": exc.text},
            status=exc.status,
        )
    except RuntimeError as exc:
        return web.json_response({"error": "service_unavailable", "message": str(exc)}, status=503)
    except Exception as exc:
        LOGGER.exception("ASR 请求失败 path=%s", request.path)
        return web.json_response({"error": "internal_error", "message": str(exc)}, status=500)


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


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    defaults = AsrSettings.from_env()
    parser = argparse.ArgumentParser(description="Sandbox ASR service")
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    args = parser.parse_args()
    web.run_app(create_app(defaults), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
