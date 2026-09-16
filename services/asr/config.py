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
    intent_url: str
    intent_profile: str

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
            intent_url=os.getenv(
                "ASR_INTENT_URL",
                "http://127.0.0.1:8011/api/v1/intent",
            ).strip(),
            intent_profile=_intent_profile(),
        )


def _legacy_port() -> int:
    try:
        return int(os.getenv("WHISPER_PORT", "9004"))
    except ValueError:
        return 9004


def _optional(name: str, default: str) -> str | None:
    value = os.getenv(name, default).strip()
    return value or None


def _intent_profile() -> str:
    profile = os.getenv("ASR_INTENT_PROFILE", "runtime").strip().lower()
    if profile not in {"discovery", "runtime"}:
        raise ValueError("ASR_INTENT_PROFILE must be discovery or runtime")
    return profile


_DEFAULT_INITIAL_PROMPT = (
    "这是智能眼镜与机器狗协同执行园区巡逻的语音助手。首句任务可能是巡逻巡检、"
    "查看实时画面或可疑物识别；运行期可识别威吓歹徒、驱逐歹徒和机器狗移动方向。"
)
_DEFAULT_HOTWORDS = (
    "机器狗 巡逻 巡检 园区 区域 A区域 B区域 实时画面 查看现场 可疑物识别 "
    "威吓 驱逐 歹徒 向前 退后 向左 向右 左转 右转"
)
