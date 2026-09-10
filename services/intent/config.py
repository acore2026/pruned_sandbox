from __future__ import annotations

from dataclasses import dataclass
import os


def _int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class IntentSettings:
    host: str
    port: int
    cors_origin: str
    backend: str
    model: str
    device: str
    max_new_tokens: int

    @classmethod
    def from_env(cls) -> "IntentSettings":
        return cls(
            host=os.getenv("INTENT_HOST", os.getenv("SEMANTIC_ROUTE_HOST", "0.0.0.0")),
            port=_int("INTENT_PORT", _legacy_port()),
            cors_origin=os.getenv("CORS_ORIGIN", "*"),
            backend=os.getenv("INTENT_BACKEND", "hybrid").strip().lower(),
            model=os.getenv("INTENT_MODEL", "/models/intent/Qwen2.5-0.5B-Instruct").strip(),
            device=os.getenv("INTENT_DEVICE", "auto").strip(),
            max_new_tokens=_int("INTENT_MAX_NEW_TOKENS", 128),
        )


def _legacy_port() -> int:
    try:
        return int(os.getenv("SEMANTIC_ROUTE_PORT", "8011"))
    except ValueError:
        return 8011
