from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from .config import VideoSettings


class YoloDetector:
    """Ultralytics YOLO 适配器，模型在第一帧时加载。"""

    def __init__(self, settings: VideoSettings, model_factory: Any | None = None) -> None:
        self.settings = settings
        self._model_factory = model_factory
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._requested_classes = list(settings.yolo_classes)
        self._dynamic_classes = False
        self.loaded = False
        self.last_error: str | None = None
        self.last_inference_ms = 0
        self.last_detections: list[dict[str, Any]] = []

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            factory = self._model_factory
            if factory is None:
                from ultralytics import YOLO

                factory = YOLO
            try:
                self._model = factory(self.settings.yolo_model)
                self._try_set_dynamic_classes(self._requested_classes)
            except Exception as exc:
                self.last_error = str(exc)
                raise
            self.loaded = True
            self.last_error = None
            return self._model

    def _try_set_dynamic_classes(self, classes: list[str]) -> None:
        setter = getattr(self._model, "set_classes", None)
        if not classes or not callable(setter):
            self._dynamic_classes = False
            return
        try:
            setter(classes)
            self._dynamic_classes = True
        except (AttributeError, NotImplementedError):
            self._dynamic_classes = False

    def set_classes(self, classes: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(item.strip() for item in classes if item.strip()))
        self._requested_classes = normalized
        if self._model is not None:
            with self._inference_lock:
                self._try_set_dynamic_classes(normalized)
        return list(self._requested_classes)

    def process(self, image: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if not self.settings.yolo_enabled:
            return image, []
        model = self._load_model()
        started = time.perf_counter()
        try:
            with self._inference_lock:
                kwargs: dict[str, Any] = {
                    "source": image,
                    "conf": self.settings.yolo_confidence,
                    "iou": self.settings.yolo_iou,
                    "imgsz": self.settings.yolo_image_size,
                    "device": self.settings.yolo_device,
                    "verbose": False,
                }
                class_ids = self._resolve_class_ids(model)
                if class_ids is not None:
                    kwargs["classes"] = class_ids
                result = model.predict(**kwargs)[0]
            annotated = result.plot()
            detections = self._serialize_detections(result)
            self.last_inference_ms = int((time.perf_counter() - started) * 1000)
            self.last_detections = detections
            self.last_error = None
            return annotated, detections
        except Exception as exc:
            self.last_error = str(exc)
            raise

    def _resolve_class_ids(self, model: Any) -> list[int] | None:
        if not self._requested_classes or self._dynamic_classes:
            return None
        names = getattr(model, "names", {})
        if isinstance(names, list):
            names = dict(enumerate(names))
        wanted = {item.lower() for item in self._requested_classes}
        return [int(index) for index, name in names.items() if str(name).lower() in wanted]

    @staticmethod
    def _serialize_detections(result: Any) -> list[dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = getattr(result, "names", {})
        detections: list[dict[str, Any]] = []
        for box in boxes:
            class_id = int(box.cls[0].item())
            detections.append(
                {
                    "classId": class_id,
                    "label": str(names.get(class_id, class_id)),
                    "confidence": round(float(box.conf[0].item()), 4),
                    "box": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                }
            )
        return detections

    def health(self) -> dict[str, Any]:
        return {
            "enabled": self.settings.yolo_enabled,
            "loaded": self.loaded,
            "model": self.settings.yolo_model,
            "device": self.settings.yolo_device,
            "classes": list(self._requested_classes),
            "lastInferenceMs": self.last_inference_ms,
            "lastDetectionCount": len(self.last_detections),
            "lastError": self.last_error,
        }
