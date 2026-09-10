from __future__ import annotations

from unittest import TestCase

from services.intent.classifier import RuleIntentClassifier


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
        self.assertEqual("turn_left", result.argument)

    def test_classifies_grab_without_target(self) -> None:
        result = self.classifier.classify("请抓取")

        self.assertEqual("grab", result.intent)
        self.assertEqual("", result.argument)

    def test_unrelated_text_is_other(self) -> None:
        result = self.classifier.classify("今天天气怎么样")

        self.assertEqual("other", result.intent)
        self.assertEqual("", result.argument)
