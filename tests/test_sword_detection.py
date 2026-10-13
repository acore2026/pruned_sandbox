from __future__ import annotations

from dataclasses import replace
import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from services.video.classes import normalize_class, normalize_classes
from services.video.config import VideoSettings
from services.video.detector import YoloDetector


class SwordClassTest(TestCase):
    def test_sword_names_and_descriptions(self) -> None:
        for value in (
            "光剑", "剑", "宝剑", "长剑", "玩具剑", "激光剑", "红色光剑", "蓝色的玩具剑",
            "sword", "swords", "SWORD", "lightsaber", "light saber", "LIGHT-SABER",
            "light   saber", "laser sword", "toy sword", "red lightsaber", "sabre",
        ):
            with self.subTest(value=value):
                self.assertEqual("sword", normalize_class(value))

    def test_unrelated_targets_are_preserved(self) -> None:
        for value in ("swordfish", "password", "剑桥", "剑鞘", "red doll", "dog"):
            with self.subTest(value=value):
                self.assertEqual(value, normalize_class(value))
        self.assertEqual(["sword", "dog"], normalize_classes(["光剑", "SWORD", "", " dog "]))

    def test_environment_classes_are_normalized(self) -> None:
        with patch.dict(os.environ, {"YOLO_CLASSES": "光剑,light saber,sword"}, clear=True):
            self.assertEqual(("sword",), VideoSettings.from_env().yolo_classes)


class TrainedSwordDetectorTest(TestCase):
    def test_trained_world_checkpoint_keeps_its_vocabulary_and_features(self) -> None:
        class TrainedModel:
            names = {0: "sword"}

            def __init__(self) -> None:
                self.requests = []
                self.text_features = object()

            def __getattr__(self, name):
                # Ultralytics YOLO forwards this method to its WorldModel.
                if name == "set_classes":
                    def reset_vocabulary(_classes):
                        raise AssertionError("trained vocabulary must not be replaced")
                    return reset_vocabulary
                raise AttributeError(name)

            def predict(self, **kwargs):
                self.requests.append(kwargs)
                boxes = [] if kwargs.get("classes") == [] else [SimpleNamespace(
                    cls=np.array([0]), conf=np.array([0.95]),
                    xyxy=np.array([[10.0, 20.0, 110.0, 220.0]]),
                )]
                return [SimpleNamespace(names=self.names, boxes=boxes, plot=lambda: kwargs["source"])]

        with patch.dict(os.environ, {}, clear=True):
            settings = replace(VideoSettings.from_env(), yolo_device="cpu")
        model = TrainedModel()
        original_features = model.text_features
        detector = YoloDetector(settings, model_factory=lambda path: model)
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        for classes in (None, ["光剑"], ["red light saber"], ["sword", "剑"]):
            with self.subTest(classes=classes):
                _, detections = detector.process(image, classes)
                self.assertEqual([0], model.requests[-1]["classes"])
                self.assertEqual("sword", detections[0]["label"])
                self.assertEqual(0, detections[0]["classId"])
        self.assertEqual(["sword"], detector.set_classes(["光剑", "sword"]))
        _, detections = detector.process(image, ["dog"])
        self.assertEqual([], model.requests[-1]["classes"])
        self.assertEqual([], detections)
        self.assertIs(original_features, model.text_features)
        self.assertEqual({0: "sword"}, model.names)
        self.assertFalse(detector._dynamic_classes)
