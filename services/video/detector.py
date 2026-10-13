from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from .classes import normalize_classes
from .config import VideoSettings


class YoloDetector:
    """Ultralytics YOLO 适配器，启动时通过一次推理触发内置预热。"""

    def __init__(self, settings: VideoSettings, model_factory: Any | None = None) -> None:
        self.settings = settings
        self._model_factory = model_factory
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._warmup_lock = threading.Lock()
        self._requested_classes = normalize_classes(settings.yolo_classes)
        self._dynamic_classes = False
        self.loaded = False
        self.warmed_up = False
        self.warmup_ms: int | None = None
        self.last_error: str | None = None
        self.last_inference_ms = 0
        self.last_detections: list[dict[str, Any]] = []

    def warmup(self) -> None:
        if not self.settings.yolo_enabled:
            return
        with self._warmup_lock:
            if self.warmed_up:
                return
            started = time.perf_counter()
            image = np.zeros(
                (self.settings.video_height, self.settings.video_width, 3),
                dtype=np.uint8,
            )
            # predict initializes the Ultralytics predictor and runs its
            # existing model.warmup. Exercise postprocessing as well.
            self.process(image)
            self.last_detections = []
            self.last_inference_ms = 0
            self.warmup_ms = int((time.perf_counter() - started) * 1000)
            self.warmed_up = True

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
                self._move_model_to_inference_device()
                self._try_set_dynamic_classes(self._requested_classes)
            except Exception as exc:
                self.last_error = str(exc)
                raise
            self.loaded = True
            self.last_error = None
            return self._model

    def _move_model_to_inference_device(self) -> None:
        """Move YOLO-World before encoding dynamic CLIP classes.

        ``set_classes`` creates and stores text embeddings.  If it runs while
        the wrapper is still on CPU and the first ``predict`` later moves the
        detector to CUDA, the cached text embeddings stay on CPU and inference
        fails with a mixed-device tensor error.
        """
        device = str(self.settings.yolo_device or "").strip().lower()
        if not device or device == "cpu":
            return
        mover = getattr(self._model, "to", None)
        if not callable(mover):
            return
        if device.isdigit():
            device = f"cuda:{device}"
        mover(device)

    def _try_set_dynamic_classes(self, classes: list[str]) -> None:
        # A plain YOLO wrapper can forward set_classes to a trained WorldModel.
        # Keep that checkpoint's cached text features and fixed class IDs intact;
        # only explicit open-vocabulary wrappers should replace their vocabulary.
        if not callable(getattr(type(self._model), "set_classes", None)):
            self._dynamic_classes = False
            return
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
        normalized = normalize_classes(classes)
        self._requested_classes = normalized
        if self._model is not None:
            with self._inference_lock:
                self._try_set_dynamic_classes(normalized)
        return list(self._requested_classes)

    def process(
        self,
        image: np.ndarray,
        classes: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if not self.settings.yolo_enabled:
            return image, []
        model = self._load_model()
        started = time.perf_counter()
        try:
            with self._inference_lock:
                requested_classes = (
                    normalize_classes(classes)
                    if classes is not None
                    else list(self._requested_classes)
                )
                # YOLO-World stores its vocabulary on the model. Set it while
                # holding the inference lock so concurrent bindings cannot see
                # one another's recognition target.
                self._try_set_dynamic_classes(requested_classes)
                kwargs: dict[str, Any] = {
                    "source": image,
                    "conf": self.settings.yolo_confidence,
                    "iou": self.settings.yolo_iou,
                    # Ultralytics expects (height, width); the service media
                    # contract is fixed at 640x480 (width x height).
                    "imgsz": list(self.settings.yolo_image_size),
                    "device": self.settings.yolo_device,
                    "verbose": False,
                }
                class_ids = self._resolve_class_ids(model, requested_classes)
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

    def _resolve_class_ids(
        self,
        model: Any,
        requested_classes: list[str],
    ) -> list[int] | None:
        if not requested_classes or self._dynamic_classes:
            return None
        names = getattr(model, "names", {})
        if isinstance(names, list):
            names = dict(enumerate(names))
        wanted = {item.lower() for item in requested_classes}
        return [int(index) for index, name in names.items() if str(name).lower() in wanted]

    @staticmethod
    def _serialize_detections(result: Any) -> list[dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = getattr(result, "names", {})
        if isinstance(names, list):
            names = dict(enumerate(names))
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
            "warmedUp": self.warmed_up,
            "warmupMs": self.warmup_ms,
            "model": self.settings.yolo_model,
            "device": self.settings.yolo_device,
            "classes": list(self._requested_classes),
            "lastInferenceMs": self.last_inference_ms,
            "lastDetectionCount": len(self.last_detections),
            "lastError": self.last_error,
        }
