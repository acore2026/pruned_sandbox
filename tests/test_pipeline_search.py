from __future__ import annotations

import asyncio
import os
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from services.video.config import VideoSettings
from services.video.detector import YoloDetector
from services.video.pipeline import FramePipeline


class PipelineSearchTest(IsolatedAsyncioTestCase):
    async def test_startup_animation_changes_and_yields_to_real_frame(self):
        import numpy as np
        with patch.dict(os.environ, {"YOLO_ENABLED": "false"}, clear=True):
            settings = VideoSettings.from_env()
        pipeline = FramePipeline(settings, YoloDetector(settings))
        pipeline._startup_at = 0.0
        with patch("services.video.pipeline.time.monotonic", return_value=0.0):
            first = pipeline.latest_image()
        with patch("services.video.pipeline.time.monotonic", return_value=0.5):
            second = pipeline.latest_image()
            self.assertEqual(first.shape, second.shape)
            self.assertFalse(np.array_equal(first, second))
            self.assertFalse(pipeline.source_frame_available)
            pipeline.processed_frame.set()
            pipeline._last_processed_at = 0.5
            scene = np.full_like(first, 123)
            pipeline._latest_image = scene
            self.assertIs(pipeline.latest_image(), scene)

    async def test_stalled_source_replaces_old_scene_and_recovers_on_new_frame(self):
        with patch.dict(os.environ, {"YOLO_ENABLED": "false"}, clear=True):
            settings = VideoSettings.from_env()
        pipeline = FramePipeline(settings, YoloDetector(settings))
        scene = pipeline._latest_image
        pipeline.processed_frame.set()
        pipeline._last_processed_at = 100.0
        with patch("services.video.pipeline.time.monotonic", return_value=102.0):
            self.assertTrue(pipeline.source_frame_available)
            self.assertIs(pipeline.latest_image(), scene)
        with patch("services.video.pipeline.time.monotonic", return_value=104.0):
            self.assertFalse(pipeline.source_frame_available)
            self.assertIs(pipeline.latest_image(), pipeline._stalled_image)
            pipeline._last_processed_at = 104.0
            self.assertTrue(pipeline.source_frame_available)
            self.assertIs(pipeline.latest_image(), scene)

    async def test_sword_alias_search_matches_the_trained_label(self) -> None:
        with patch.dict(os.environ, {"YOLO_ENABLED": "false"}, clear=True):
            settings = VideoSettings.from_env()
        pipeline = FramePipeline(settings, YoloDetector(settings))
        pipeline.set_recognition_target("红色光剑")
        search = asyncio.create_task(pipeline.search_object("light saber", 1000))
        await asyncio.sleep(0)
        self.assertEqual(("sword",), pipeline._active_classes())
        async with pipeline._detections_changed:
            pipeline._last_detections = [{"label": "sword", "confidence": 0.94}]
            pipeline._detection_revision += 1
            pipeline._detections_changed.notify_all()
        self.assertEqual([{"label": "sword", "confidence": 0.94}], await search)
        self.assertEqual(("sword",), pipeline._active_classes())

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


class OutputTimestampTest(IsolatedAsyncioTestCase):
    async def test_output_timestamp_tracks_wall_clock_after_sender_stall(self):
        import numpy as np
        from services.video.pipeline import OutputVideoTrack
        with patch.dict(os.environ, {"YOLO_ENABLED": "false"}, clear=True):
            settings = VideoSettings.from_env()
        track = OutputVideoTrack(FramePipeline(settings, YoloDetector(settings)), 10)
        with patch("services.video.pipeline.time.perf_counter", return_value=100.0):
            first = await track.recv()
        with patch("services.video.pipeline.time.perf_counter", return_value=103.0):
            latest = await track.recv()
        self.assertAlmostEqual(float(latest.time - first.time), 3.0, places=3)
        track.stop()


class FreshOutputTest(IsolatedAsyncioTestCase):
    def make_pipeline(self):
        with patch.dict(os.environ, {"YOLO_ENABLED": "false"}, clear=True):
            settings = VideoSettings.from_env()
        return FramePipeline(settings, YoloDetector(settings))

    async def publish(self, pipeline, value):
        import numpy as np
        import time
        pipeline._latest_image = np.full((480, 640, 3), value, dtype=np.uint8)
        pipeline.processed_frames += 1
        pipeline._last_processed_at = time.monotonic()
        pipeline.processed_frame.set()
        async with pipeline._detections_changed:
            pipeline._detections_changed.notify_all()

    async def test_no_duplicate_and_multiple_consumers_receive_newest(self):
        pipeline = self.make_pipeline()
        a, b = pipeline.new_output_track(), pipeline.new_output_track()
        await self.publish(pipeline, 10)
        await a.recv()
        await b.recv()
        pending = asyncio.create_task(a.recv())
        await asyncio.sleep(0.12)
        self.assertFalse(pending.done(), "Repeated previous camera frame")
        await self.publish(pipeline, 20)
        await self.publish(pipeline, 30)
        frame = await asyncio.wait_for(pending, 1)
        self.assertEqual(30, int(frame.to_ndarray(format="bgr24")[0,0,0]))
        frame_b = await b.recv()
        self.assertEqual(30, int(frame_b.to_ndarray(format="bgr24")[0,0,0]))
        a.stop()
        b.stop()

    async def test_rate_cap_and_stop_while_waiting(self):
        from aiortc.mediastreams import MediaStreamError
        import time
        from services.video.pipeline import OutputVideoTrack
        pipeline = self.make_pipeline()
        track = OutputVideoTrack(pipeline, 60)
        await self.publish(pipeline, 10)
        first = await track.recv()
        started = time.perf_counter()
        await self.publish(pipeline, 20)
        second = await track.recv()
        self.assertGreaterEqual(time.perf_counter() - started, 0.04)
        self.assertGreater(second.pts, first.pts)
        pending = asyncio.create_task(track.recv())
        await asyncio.sleep(0.07)
        track.stop()
        with self.assertRaises(MediaStreamError):
            await asyncio.wait_for(pending, 1)
