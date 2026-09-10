from __future__ import annotations

from dataclasses import dataclass
import os


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


def _float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def _csv(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _model_path() -> str:
    value = os.getenv("YOLO_MODEL", "/models/yolo/yolov8s-worldv2.pt").strip()
    candidate = value or "/models/yolo/yolov8s-worldv2.pt"
    if "/" in candidate:
        return candidate
    if not candidate.endswith(".pt"):
        candidate = f"{candidate}.pt"
    return f"/models/yolo/{candidate}"


@dataclass(frozen=True, slots=True)
class VideoSettings:
    host: str
    port: int
    public_ip: str
    cors_origin: str
    yolo_enabled: bool
    yolo_model: str
    yolo_device: str
    yolo_confidence: float
    yolo_iou: float
    yolo_image_size: int
    yolo_classes: tuple[str, ...]
    video_width: int
    video_height: int
    video_fps: float
    rtc_ice_servers: tuple[str, ...]
    rtc_ice_username: str
    rtc_ice_credential: str
    rtc_ice_gather_timeout_s: float
    source_wait_seconds: float
    session_ttl_seconds: float

    @classmethod
    def from_env(cls) -> "VideoSettings":
        return cls(
            host=os.getenv(
                "VIDEO_HOST",
                os.getenv("MOCK_LISTEN_HOST", os.getenv("SANDBOX_HOST", "0.0.0.0")),
            ),
            port=_int("VIDEO_PORT", _legacy_port()),
            public_ip=os.getenv(
                "VIDEO_PUBLIC_IP",
                os.getenv("MOCK_VIDEO_SERVER_IP", "172.30.0.10"),
            ).strip(),
            cors_origin=os.getenv("CORS_ORIGIN", "*"),
            yolo_enabled=_bool("YOLO_ENABLED", True),
            yolo_model=_model_path(),
            yolo_device=os.getenv("YOLO_DEVICE", "0").strip(),
            yolo_confidence=_float("YOLO_CONFIDENCE", 0.3),
            yolo_iou=_float("YOLO_IOU", 0.4),
            yolo_image_size=_int("YOLO_IMAGE_SIZE", 640),
            yolo_classes=_csv("YOLO_CLASSES"),
            video_width=_int("VIDEO_WIDTH", 1280),
            video_height=_int("VIDEO_HEIGHT", 720),
            video_fps=_float("VIDEO_FPS", 15.0, minimum=1.0),
            rtc_ice_servers=_csv("WEBRTC_ICE_SERVERS"),
            rtc_ice_username=os.getenv("WEBRTC_ICE_USERNAME", "").strip(),
            rtc_ice_credential=os.getenv("WEBRTC_ICE_CREDENTIAL", "").strip(),
            rtc_ice_gather_timeout_s=_float("WEBRTC_ICE_GATHER_TIMEOUT_S", 8.0, minimum=0.1),
            source_wait_seconds=_float("VIDEO_SOURCE_WAIT_SECONDS", 12.0, minimum=0.1),
            session_ttl_seconds=_float("VIDEO_SESSION_TTL_SECONDS", 7200.0, minimum=1.0),
        )


def _legacy_port() -> int:
    for name in ("MOCK_VIDEO_PORT", "SANDBOX_PORT"):
        raw = os.getenv(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return 28500
