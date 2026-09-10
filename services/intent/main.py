from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from aiohttp import web

from .classifier import IntentService
from .config import IntentSettings


LOGGER = logging.getLogger("sandbox.intent")
SERVICE_KEY = web.AppKey("intent_service", IntentService)
SETTINGS_KEY = web.AppKey("intent_settings", IntentSettings)


def create_app(
    settings: IntentSettings | None = None,
    service: IntentService | None = None,
) -> web.Application:
    active_settings = settings or IntentSettings.from_env()
    app = web.Application(middlewares=[cors_middleware, error_middleware])
    app[SETTINGS_KEY] = active_settings
    app[SERVICE_KEY] = service or IntentService(active_settings)
    app.add_routes(
        [
            web.get("/health", health),
            web.get("/api/health", health),
            web.post("/api/v1/intent", classify),
            web.post("/api/v1/semantic/route", classify),
        ]
    )
    return app


async def health(request: web.Request) -> web.Response:
    return web.json_response(request.app[SERVICE_KEY].health())


async def classify(request: web.Request) -> web.Response:
    payload = await _json_body(request)
    text = str(payload.get("text") or payload.get("intent_payload") or payload.get("intent") or "").strip()
    if not text:
        raise web.HTTPBadRequest(text="text or intent_payload is required")
    return web.json_response(await request.app[SERVICE_KEY].classify(text))


async def _json_body(request: web.Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise web.HTTPBadRequest(text="valid JSON body is required") from exc
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="JSON body must be an object")
    return payload


@web.middleware
async def error_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    try:
        return await handler(request)
    except web.HTTPException as exc:
        return web.json_response(
            {"error": exc.reason.replace(" ", "_").lower(), "message": exc.text},
            status=exc.status,
        )
    except ValueError as exc:
        return web.json_response({"error": "bad_request", "message": str(exc)}, status=400)
    except Exception as exc:
        LOGGER.exception("意图识别请求失败 path=%s", request.path)
        return web.json_response({"error": "internal_error", "message": str(exc)}, status=500)


@web.middleware
async def cors_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    response = web.Response(status=204) if request.method == "OPTIONS" else await handler(request)
    response.headers["Access-Control-Allow-Origin"] = request.app[SETTINGS_KEY].cors_origin
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    defaults = IntentSettings.from_env()
    parser = argparse.ArgumentParser(description="Sandbox intent service")
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    args = parser.parse_args()
    web.run_app(create_app(defaults), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
