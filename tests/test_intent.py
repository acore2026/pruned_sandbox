from __future__ import annotations

import os
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from services.intent.classifier import IntentService, RuleIntentClassifier
from services.intent.config import IntentSettings


class RuleIntentClassifierTest(TestCase):
    def setUp(self) -> None:
        self.classifier = RuleIntentClassifier()

    def test_classifies_chinese_find_object_and_normalizes_target(self) -> None:
        result = self.classifier.classify("帮我找黄色的狗").to_dict()

        self.assertEqual("find_object", result["intent"])
        self.assertEqual("yellow dog", result["normalized_argument"])
        self.assertEqual({"zh": "黄色的狗", "en": "yellow dog"}, result["normalized_argument_i18n"])

    def test_classifies_movement(self) -> None:
        result = self.classifier.classify("please turn left")

        self.assertEqual("movement", result.intent)
        self.assertEqual("left", result.argument)

    def test_classifies_campus_patrol_direction_variants(self) -> None:
        expected = {
            "向前": "forward",
            "退后": "backward",
            "向左": "left",
            "向右": "right",
        }
        for command, direction in expected.items():
            with self.subTest(command=command):
                result = self.classifier.classify(command)
                self.assertEqual("movement", result.intent)
                self.assertEqual(direction, result.argument)

    def test_classifies_patrol_and_extracts_area(self) -> None:
        for text in ("派机器狗巡逻园区内A区域", "请机器狗巡检园区内A区域"):
            with self.subTest(text=text):
                result = self.classifier.classify(text).to_dict()
                self.assertEqual("patrol", result["intent"])
                self.assertEqual("A区域", result["argument"])
                self.assertEqual("robot dog", result["executor"])

    def test_other_has_no_executor_skill(self) -> None:
        result = self.classifier.classify("今天天气怎么样").to_dict()

        self.assertIsNone(result["executor"])

    def test_classifies_threaten_and_expel_as_defense(self) -> None:
        for command in ("威吓歹徒", "驱逐歹徒"):
            with self.subTest(command=command):
                result = self.classifier.classify(command)
                self.assertEqual("defense", result.intent)
                self.assertEqual("suspect", result.argument)

    def test_classifies_scene_document_discovery_intents(self) -> None:
        classifier = RuleIntentClassifier()
        self.assertEqual("video_task", classifier.classify("查看机器狗实时画面").intent)
        self.assertEqual(
            "object_recognition", classifier.classify("识别园区内可疑物").intent
        )

    def test_classifies_grab_without_target(self) -> None:
        result = self.classifier.classify("请抓取")

        self.assertEqual("grab", result.intent)
        self.assertEqual("", result.argument)

    def test_unrelated_text_is_other(self) -> None:
        result = self.classifier.classify("今天天气怎么样")

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
            result = await service.classify("派机器狗巡逻园区内A区域")

        self.assertEqual("patrol", result["scene"])
        self.assertEqual("A区域", result["normalized_argument"])
        self.assertEqual("rules", result["backend"])
