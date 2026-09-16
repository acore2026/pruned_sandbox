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
    management_host: str
    management_port: int
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
    h264_rtp_payload_bytes: int
    rtc_ice_servers: tuple[str, ...]
    rtc_ice_username: str
    rtc_ice_credential: str
    rtc_ice_gather_timeout_s: float
    source_wait_seconds: float
    management_token: str
    intent_url: str
    asr_url: str
    audio_max_upload_bytes: int
    producer_control_url: str
    producer_control_timeout_s: float

    @classmethod
    def from_env(cls) -> "VideoSettings":
        return cls(
            host=os.getenv(
                "VIDEO_HOST",
                os.getenv("MOCK_LISTEN_HOST", os.getenv("SANDBOX_HOST", "0.0.0.0")),
            ),
            port=_int("SANDBOX_USER_PORT", _legacy_port()),
            management_host=os.getenv("SANDBOX_MANAGEMENT_HOST", "0.0.0.0"),
            management_port=_int("SANDBOX_MANAGEMENT_PORT", 28501),
            public_ip=os.getenv(
                "VIDEO_PUBLIC_IP",
                os.getenv("MOCK_VIDEO_SERVER_IP", "127.0.0.1"),
            ).strip(),
            cors_origin=os.getenv("CORS_ORIGIN", "*"),
            yolo_enabled=_bool("YOLO_ENABLED", True),
            yolo_model=_model_path(),
            yolo_device=os.getenv("YOLO_DEVICE", "0").strip(),
            yolo_confidence=_float("YOLO_CONFIDENCE", 0.3),
            yolo_iou=_float("YOLO_IOU", 0.4),
            yolo_image_size=_int("YOLO_IMAGE_SIZE", 640),
            yolo_classes=_csv("YOLO_CLASSES"),
            video_width=_int("VIDEO_WIDTH", 640, minimum=2),
            video_height=_int("VIDEO_HEIGHT", 480, minimum=2),
            video_fps=_float("VIDEO_FPS", 30.0, minimum=1.0),
            h264_rtp_payload_bytes=_h264_rtp_payload_bytes(),
            rtc_ice_servers=_csv("WEBRTC_ICE_SERVERS"),
            rtc_ice_username=os.getenv("WEBRTC_ICE_USERNAME", "").strip(),
            rtc_ice_credential=os.getenv("WEBRTC_ICE_CREDENTIAL", "").strip(),
            rtc_ice_gather_timeout_s=_float("WEBRTC_ICE_GATHER_TIMEOUT_S", 8.0, minimum=0.1),
            source_wait_seconds=_float("VIDEO_SOURCE_WAIT_SECONDS", 12.0, minimum=0.1),
            management_token=os.getenv(
                "FREE6GC_COMPUTING_SANDBOX_MANAGEMENT_TOKEN", ""
            ).strip(),
            intent_url=os.getenv(
                "SANDBOX_INTENT_URL",
                "http://127.0.0.1:8011/api/v1/intent",
            ).strip(),
            asr_url=os.getenv(
                "SANDBOX_ASR_URL",
                "http://127.0.0.1:9005/api/v1/transcribe",
            ).strip(),
            audio_max_upload_bytes=_int("ASR_MAX_UPLOAD_MB", 50) * 1024 * 1024,
            producer_control_url=os.getenv(
                "SANDBOX_PRODUCER_CONTROL_URL", ""
            ).strip(),
            producer_control_timeout_s=_float(
                "SANDBOX_PRODUCER_CONTROL_TIMEOUT_S", 15.0, minimum=0.1
            ),
        )


def _legacy_port() -> int:
    for name in ("VIDEO_PORT", "MOCK_VIDEO_PORT", "SANDBOX_PORT"):
        raw = os.getenv(name)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return 28502


def _h264_rtp_payload_bytes() -> int:
    raw = os.getenv(
        "VIDEO_H264_RTP_PAYLOAD_BYTES",
        os.getenv("MOCK_VIDEO_H264_RTP_PAYLOAD_BYTES", "1150"),
    )
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("VIDEO_H264_RTP_PAYLOAD_BYTES must be an integer") from error
    if not 1100 <= value <= 1150:
        raise ValueError(
            "VIDEO_H264_RTP_PAYLOAD_BYTES must be between 1100 and 1150"
        )
    return value
