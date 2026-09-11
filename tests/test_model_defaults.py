from __future__ import annotations

import os
from unittest import TestCase
from unittest.mock import patch

from services.asr.config import AsrSettings
from services.intent.config import IntentSettings
from services.video.config import VideoSettings


class OriginalModelDefaultsTest(TestCase):
    def test_original_models_are_the_runtime_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            asr = AsrSettings.from_env()
            intent = IntentSettings.from_env()
            video = VideoSettings.from_env()

        self.assertEqual("/models/asr/whisper-large-v3", asr.model)
        self.assertEqual("cuda", asr.device)
        self.assertEqual("float16", asr.compute_type)
        self.assertEqual("hybrid", intent.backend)
        self.assertEqual("/models/intent/Qwen2.5-0.5B-Instruct", intent.model)
        self.assertEqual("/models/yolo/yolov8s-worldv2.pt", video.yolo_model)
        self.assertEqual(0.3, video.yolo_confidence)
        self.assertEqual(0.4, video.yolo_iou)

    def test_custom_yolo_model_name_resolves_inside_image(self) -> None:
        with patch.dict(os.environ, {"YOLO_MODEL": "custom-patrol"}, clear=True):
            settings = VideoSettings.from_env()

        self.assertEqual("/models/yolo/custom-patrol.pt", settings.yolo_model)
