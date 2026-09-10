from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class AsrSettings:
    host: str
    port: int
    cors_origin: str
    temp_dir: Path
    enabled: bool
    model: str
    download_root: str | None
    device: str
    compute_type: str
    language: str | None
    beam_size: int
    initial_prompt: str | None
    hotwords: str | None
    max_upload_bytes: int
    concurrency: int

    @classmethod
    def from_env(cls) -> "AsrSettings":
        language = os.getenv("ASR_LANGUAGE", "zh").strip()
        download_root = os.getenv("ASR_DOWNLOAD_ROOT", "/models/asr").strip()
        return cls(
            host=os.getenv("ASR_HOST", os.getenv("WHISPER_HOST", "0.0.0.0")),
            port=_int("ASR_PORT", _legacy_port()),
            cors_origin=os.getenv("CORS_ORIGIN", "*"),
            temp_dir=Path(os.getenv("ASR_TEMP_DIR", "/tmp/sandbox-asr")),
            enabled=_bool("ASR_ENABLED", True),
            model=os.getenv("ASR_MODEL", "/models/asr/whisper-large-v3").strip(),
            download_root=download_root or None,
            device=os.getenv("ASR_DEVICE", "cuda").strip(),
            compute_type=os.getenv("ASR_COMPUTE_TYPE", "float16").strip(),
            language=language or None,
            beam_size=_int("ASR_BEAM_SIZE", 5),
            initial_prompt=_optional("ASR_INITIAL_PROMPT", _DEFAULT_INITIAL_PROMPT),
            hotwords=_optional("ASR_HOTWORDS", _DEFAULT_HOTWORDS),
            max_upload_bytes=_int("ASR_MAX_UPLOAD_MB", 50) * 1024 * 1024,
            concurrency=_int("ASR_CONCURRENCY", 1),
        )


def _legacy_port() -> int:
    try:
        return int(os.getenv("WHISPER_PORT", "9004"))
    except ValueError:
        return 9004


def _optional(name: str, default: str) -> str | None:
    value = os.getenv(name, default).strip()
    return value or None


_DEFAULT_INITIAL_PROMPT = (
    "这是智能眼镜语音助手。它能让机器狗去超市，执行找、抓、夹、走等操作，"
    "超市里常见商品包括：橙汁、葡萄汁、可乐、盲盒。"
)
_DEFAULT_HOTWORDS = "橙汁 葡萄汁 可乐 盲盒 拉布布"
