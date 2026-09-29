from __future__ import annotations

import os
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from services.intent.classifier import IntentService, RuleIntentClassifier
from services.intent.config import IntentSettings


class RuleIntentClassifierTest(TestCase):
    def setUp(self) -> None:
        self.classifier = RuleIntentClassifier()

    def test_classifies_english_find_object_and_normalizes_target(self) -> None:
        result = self.classifier.classify("Please find the yellow dog").to_dict()

        self.assertEqual("find_object", result["intent"])
        self.assertEqual("yellow dog", result["normalized_argument"])
        self.assertEqual({"zh": "", "en": "yellow dog"}, result["normalized_argument_i18n"])

    def test_classifies_movement(self) -> None:
        result = self.classifier.classify("please turn left")

        self.assertEqual("movement", result.intent)
        self.assertEqual("left", result.argument)

    def test_classifies_campus_patrol_direction_variants(self) -> None:
        expected = {
            "move forward": "forward",
            "move backward": "backward",
            "turn left": "left",
            "turn right": "right",
        }
        for command, direction in expected.items():
            with self.subTest(command=command):
                result = self.classifier.classify(command)
                self.assertEqual("movement", result.intent)
                self.assertEqual(direction, result.argument)

    def test_classifies_patrol_and_extracts_area(self) -> None:
        for text in (
            "Send the robot dog to patrol area A on campus.",
            "Please inspect zone A.",
            "Please patrol area B and locate the red doll.",
        ):
            with self.subTest(text=text):
                result = self.classifier.classify(text).to_dict()
                self.assertEqual("patrol", result["intent"])
                self.assertEqual("B" if "area B" in text else "A", result["argument"])
                self.assertEqual("robot dog", result["executor"])

    def test_other_has_no_executor_skill(self) -> None:
        result = self.classifier.classify("What is the weather today?").to_dict()

        self.assertIsNone(result["executor"])

    def test_classifies_threaten_and_expel_as_defense(self) -> None:
        for command in ("Threaten the suspect", "Expel the suspect"):
            with self.subTest(command=command):
                result = self.classifier.classify(command)
                self.assertEqual("defense", result.intent)
                self.assertEqual("suspect", result.argument)

    def test_classifies_scene_document_discovery_intents(self) -> None:
        classifier = RuleIntentClassifier()
        self.assertEqual("video_task", classifier.classify("Show the robot dog's live video").intent)
        self.assertEqual(
            "object_recognition",
            classifier.classify("Identify suspicious objects in the area").intent,
        )

    def test_classifies_grab_without_target(self) -> None:
        result = self.classifier.classify("Please grab")

        self.assertEqual("grab", result.intent)
        self.assertEqual("", result.argument)

    def test_unrelated_text_is_other(self) -> None:
        result = self.classifier.classify("What is the weather today?")

        self.assertEqual("other", result.intent)
        self.assertEqual("", result.argument)


class HybridIntentClassifierTest(IsolatedAsyncioTestCase):
    async def test_explicit_patrol_uses_rules_before_qwen(self) -> None:
        with patch.dict(os.environ, {"INTENT_BACKEND": "hybrid"}, clear=True):
            service = IntentService(IntentSettings.from_env())
        with patch.object(
            service.qwen,
            "classify",
            side_effect=AssertionError("Qwen must not override explicit patrol"),
        ):
            result = await service.classify("Send the robot dog to patrol area A on campus.")

        self.assertEqual("patrol", result["scene"])
        self.assertEqual("A", result["normalized_argument"])
        self.assertEqual("rules", result["backend"])
