from __future__ import annotations

import asyncio
from dataclasses import replace
import os
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock, patch

from aiohttp.test_utils import TestClient, TestServer
import numpy as np

from services.asr.config import AsrSettings
from services.asr.main import create_app as create_asr_app
from services.asr.service import SpeechRecognizer
from services.sandbox.main import VideoRuntime, create_user_app
from services.video.config import VideoSettings
from services.video.detector import YoloDetector
from services.intent.classifier import IntentResult, IntentService
from services.intent.config import IntentSettings
from services.intent.main import create_app as create_intent_app


class WarmupWhisper:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.decoded: list[bool] = []
        self.fail = False

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        if not isinstance(audio, np.ndarray) or audio.dtype != np.float32:
            raise AssertionError("Warmup must use an in-memory waveform")

        def segments():
            if self.fail:
                raise RuntimeError("GPU decode failed")
            self.decoded.append(kwargs["vad_filter"])
            yield SimpleNamespace(start=0.0, end=1.0, text="synthetic warmup result")

        return segments(), SimpleNamespace(language="en", duration=1.0)


class AsrWarmupTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.settings = replace(AsrSettings.from_env(), concurrency=2)

    async def test_shared_listeners_wait_for_one_complete_warmup(self) -> None:
        model = WarmupWhisper()
        factory = Mock(return_value=model)
        recognizer = SpeechRecognizer(self.settings, factory)
        self.assertFalse(recognizer.health()["ready"])
        self.assertEqual("warming_up", recognizer.health()["status"])
        clients = [TestClient(TestServer(create_asr_app(
            replace(self.settings, intent_profile=profile), recognizer
        ))) for profile in ("discovery", "runtime")]
        try:
            await asyncio.gather(*(client.start_server() for client in clients))
            await recognizer.warmup()
            self.assertEqual(1, factory.call_count)
            self.assertEqual([False, True], model.decoded)
            self.assertTrue(all(call["beam_size"] == self.settings.beam_size for call in model.calls))
            self.assertTrue(all(call["initial_prompt"] is None for call in model.calls))
            for client in clients:
                response = await client.get("/health")
                health = await response.json()
                self.assertTrue(health["ready"])
                self.assertTrue(health["loaded"])
                self.assertTrue(health["warmedUp"])
                self.assertIsNotNone(health["warmupMs"])
                self.assertIsNone(health["latestTranscript"])
        finally:
            await asyncio.gather(*(client.close() for client in clients))

    async def test_lazy_decode_failure_is_not_reported_ready_and_can_retry(self) -> None:
        model = WarmupWhisper()
        model.fail = True
        recognizer = SpeechRecognizer(self.settings, Mock(return_value=model))
        with self.assertRaisesRegex(RuntimeError, "GPU decode failed"):
            await recognizer.warmup()
        self.assertFalse(recognizer.health()["ready"])
        self.assertFalse(recognizer.health()["warmedUp"])
        self.assertEqual("error", recognizer.health()["status"])
        model.fail = False
        await recognizer.warmup()
        self.assertTrue(recognizer.health()["ready"])
        self.assertIsNone(recognizer.health()["latestTranscript"])

    async def test_disabled_asr_does_not_load_model(self) -> None:
        factory = Mock(side_effect=AssertionError("Disabled ASR must not load"))
        recognizer = SpeechRecognizer(replace(self.settings, enabled=False), factory)
        await recognizer.warmup()
        self.assertEqual("disabled", recognizer.health()["status"])
        self.assertFalse(recognizer.health()["loaded"])
        factory.assert_not_called()


class YoloWarmupTest(IsolatedAsyncioTestCase):
    async def test_user_service_startup_runs_predict_and_discards_synthetic_detections(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = replace(VideoSettings.from_env(), yolo_device="cpu")
        model = Mock()
        model.names = {0: "sword"}
        image = np.zeros((settings.video_height, settings.video_width, 3), dtype=np.uint8)
        model.predict.return_value = [SimpleNamespace(
            names=model.names,
            boxes=[SimpleNamespace(cls=np.array([0]), conf=np.array([0.9]),
                                   xyxy=np.array([[1.0, 2.0, 3.0, 4.0]]))],
            plot=lambda: image,
        )]
        detector = YoloDetector(settings, model_factory=Mock(return_value=model))
        runtime = VideoRuntime.build(settings)
        runtime.detector = detector
        runtime.server.detector = detector
        client = TestClient(TestServer(create_user_app(runtime)))
        try:
            await client.start_server()
            await asyncio.to_thread(detector.warmup)
            self.assertEqual(1, model.predict.call_count)
            source = model.predict.call_args.kwargs["source"]
            self.assertEqual(image.shape, source.shape)
            self.assertEqual(list(settings.yolo_image_size), model.predict.call_args.kwargs["imgsz"])
            response = await client.get("/health")
            health = (await response.json())["yolo"]
            self.assertTrue(health["warmedUp"])
            self.assertTrue(health["loaded"])
            self.assertEqual(0, health["lastDetectionCount"])
            self.assertEqual(0, health["lastInferenceMs"])
            self.assertIsNotNone(health["warmupMs"])
        finally:
            await client.close()
            await runtime.close()

    async def test_disabled_yolo_skips_warmup(self) -> None:
        settings = replace(VideoSettings.from_env(), yolo_enabled=False)
        factory = Mock(side_effect=AssertionError("Disabled YOLO must not load"))
        detector = YoloDetector(settings, model_factory=factory)
        await asyncio.to_thread(detector.warmup)
        factory.assert_not_called()
        self.assertFalse(detector.health()["warmedUp"])

    async def test_predict_failure_does_not_mark_yolo_warmed_up(self) -> None:
        settings = replace(VideoSettings.from_env(), yolo_device="cpu")
        model = Mock()
        model.names = {0: "sword"}
        model.predict.side_effect = RuntimeError("YOLO GPU failed")
        detector = YoloDetector(settings, model_factory=lambda _: model)
        with self.assertRaisesRegex(RuntimeError, "YOLO GPU failed"):
            await asyncio.to_thread(detector.warmup)
        self.assertFalse(detector.health()["warmedUp"])
        self.assertEqual("YOLO GPU failed", detector.health()["lastError"])


class IntentWarmupTest(IsolatedAsyncioTestCase):
    async def test_hybrid_startup_generates_once_and_preserves_rules_priority(self) -> None:
        settings = replace(IntentSettings.from_env(), backend="hybrid")
        service = IntentService(settings)
        client = TestClient(TestServer(create_intent_app(settings, service)))
        try:
            with patch.object(service.qwen, "classify", return_value=IntentResult("other", "", 0.9, "qwen")) as classify:
                await client.start_server()
                await service.warmup()
                self.assertEqual(1, classify.call_count)
                self.assertTrue(service.health()["warmedUp"])
                result = await service.classify("detered")
                self.assertEqual("deter", result["argument"])
                self.assertEqual("rules", result["backend"])
                self.assertEqual(1, classify.call_count)
        finally:
            await client.close()

    async def test_rules_backend_does_not_load_qwen(self) -> None:
        service = IntentService(replace(IntentSettings.from_env(), backend="rules"))
        with patch.object(service.qwen, "classify", side_effect=AssertionError("Must not load")):
            await service.warmup()

    async def test_hybrid_warmup_failure_retains_rules_fallback(self) -> None:
        service = IntentService(replace(IntentSettings.from_env(), backend="hybrid"))
        with patch.object(service.qwen, "classify", side_effect=RuntimeError("Missing model")):
            await service.warmup()
        self.assertFalse(service.health()["warmedUp"])
        self.assertEqual("Missing model", service.health()["lastError"])
        self.assertEqual("deter", (await service.classify("detered"))["argument"])
