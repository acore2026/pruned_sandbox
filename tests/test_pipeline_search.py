from __future__ import annotations

import asyncio
import os
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from services.video.config import VideoSettings
from services.video.detector import YoloDetector
from services.video.pipeline import FramePipeline


class PipelineSearchTest(IsolatedAsyncioTestCase):
    async def test_one_shot_search_does_not_replace_persistent_target(self) -> None:
        with patch.dict(os.environ, {"YOLO_ENABLED": "false"}, clear=True):
            settings = VideoSettings.from_env()
        pipeline = FramePipeline(settings, YoloDetector(settings))
        pipeline.set_recognition_target("dog")

        search = asyncio.create_task(pipeline.search_object("cup", 1000))
        await asyncio.sleep(0)
        self.assertEqual(("dog", "cup"), pipeline._active_classes())

        async with pipeline._detections_changed:
            pipeline._last_detections = [
                {"label": "cup", "confidence": 0.94},
                {"label": "person", "confidence": 0.99},
            ]
            pipeline._detection_revision += 1
            pipeline._detections_changed.notify_all()

        self.assertEqual(
            [{"label": "cup", "confidence": 0.94}], await search
        )
        self.assertEqual(("dog",), pipeline._active_classes())
