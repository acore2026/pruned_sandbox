from __future__ import annotations

import os
from dataclasses import replace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from services.asr.config import AsrSettings
from services.asr.main import _resolve_intent
from services.intent.classifier import IntentResult, IntentService, RuleIntentClassifier
from services.intent.config import IntentSettings
from services.intent.main import create_app


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

    def test_existing_chinese_patrol_requests_keep_area(self) -> None:
        for command in ("派机器狗巡逻园区内A区域。", "请派机器狗到A区域巡逻",
                        "让机器狗去A区域巡检", "请帮我巡逻A区域"):
            with self.subTest(command=command):
                result = self.classifier.classify(command)
                self.assertEqual(("patrol", "A区域"), (result.intent, result.argument))

    def test_classifies_threaten_and_expel_as_defense(self) -> None:
        for command, direction in (("Expel the suspect", "expel"), ("驱逐歹徒", "expel"),
                                   ("威吓", "deter"), ("请威吓歹徒", "deter"),
                                   ("Threaten the suspect", "deter"),
                                   ("Deter the suspect", "deter"),
                                   ("Please intimidate the suspect", "deter")):
            with self.subTest(command=command):
                result = self.classifier.classify(command)
                self.assertEqual("defense", result.intent)
                self.assertEqual(direction, result.argument)

    def test_classifies_scene_document_discovery_intents(self) -> None:
        classifier = RuleIntentClassifier()
        self.assertEqual("video_task", classifier.classify("Show the robot dog's live video").intent)
        self.assertEqual(
            "object_recognition",
            classifier.classify("Identify suspicious objects in the area").intent,
        )

    def test_asr_variants_match_the_correct_command(self) -> None:
        commands = {
            "Go straight.": ("movement", "forward"),
            "Please get forward!": ("movement", "forward"),
            "deters the suspect": ("defense", "deter"),
            "deters a suspect.": ("defense", "deter"),
            "deters": ("defense", "deter"),
            "detar a suspect": ("defense", "deter"),
            "detered": ("defense", "deter"),
            "Detered the suspect.": ("defense", "deter"),
            "Please deterred the suspect!": ("defense", "deter"),
            "Could you detering the suspect?": ("defense", "deter"),
            "Tell the robot dog to detered the suspect": ("defense", "deter"),
            "the turds I suspect.": ("defense", "deter"),
            "detour the suspect": ("defense", "deter"),
            "detar the suspect": ("defense", "deter"),
            "expelled the suspect": ("defense", "expel"),
            "ex pel the suspect": ("defense", "expel"),
            "ex-pel the suspect": ("defense", "expel"),
            "exple the suspect": ("defense", "expel"),
            "move forwards": ("movement", "forward"),
            "backwards": ("movement", "backward"),
            "go backwards": ("movement", "backward"),
            "turn write": ("movement", "right"),
            "turn lift": ("movement", "left"),
            "move forword": ("movement", "forward"),
            "patrole area B": ("patrol", "B"),
            "patrl zone A": ("patrol", "A"),
        }
        for command, expected in commands.items():
            with self.subTest(command=command):
                result = self.classifier.classify(command)
                self.assertEqual(expected, (result.intent, result.argument))

    def test_removed_greeting_commands_are_not_actions(self) -> None:
        for command in ("wave", "wave hello", "say hello", "hello", "挥手", "打招呼", "你好",
                        "Please WAVE!", "PLEASE wave hello.", "Please say hello?", "Hello!", "挥手。"):
            with self.subTest(command=command):
                result = self.classifier.classify(command)
                self.assertEqual(("other", ""), (result.intent, result.argument))

    def test_negated_and_conflicting_commands_are_not_actions(self) -> None:
        for command in (
            "Do not deter the suspect", "Don't detered the suspect",
            "Don’t deterred the suspect", "Never expel the suspect",
            "不要威吓歹徒", "别驱逐歹徒", "取消巡逻A区域",
            "deter or expel the suspect", "detered and expel the suspect",
            "deters a suspect and expel the suspect",
            "Don't go straight", "Don't get forward", "Don't deters a suspect",
            "Don't wave", "不要挥手", "Yesterday we said hello", "The wave is high",
            "turn left or right", "forward backward", "left右转",
            "forwards backwards", "detered expelled", "happy forward",
            "威吓歹徒然后驱逐歹徒",
            "write", "lift", "detour", "defer the suspect",
            "detar", "detrxxx the suspect",
        ):
            with self.subTest(command=command):
                self.assertEqual("other", self.classifier.classify(command).intent)

    def test_fuzzy_corrections_do_not_modify_object_slots(self) -> None:
        for command, intent, argument in (
            ("find the stop sign", "find_object", "stop sign"),
            ("find the no entry sign", "find_object", "no entry sign"),
            ("grab the forword cup", "grab", "forword cup"),
            ("patrl area B7", "patrol", "B7"),
            ("识别剑", "find_object", "剑"),
            ("寻找别针", "find_object", "别针"),
        ):
            with self.subTest(command=command):
                result = self.classifier.classify(command)
                self.assertEqual((intent, argument), (result.intent, result.argument))

    def test_keywords_match_anywhere_without_complete_commands(self) -> None:
        for text, intent, argument in (
            ("We patrol area A", "patrol", "A"),
            ("The next task is PATROL!", "patrol", ""),
            ("We need inspection of zone B7 today", "patrol", "B7"),
            ("今天去A区域进行巡检吧", "patrol", "A区域"),
            ("先去这里，之后巡逻", "patrol", ""),
            ("Could you move forward a little now", "movement", "forward"),
            ("We should head straight through here", "movement", "forward"),
            ("A little back please", "movement", "backward"),
            ("这里需要向左走一点", "movement", "left"),
            ("The word is HAPPY", "movement", "happy"),
            ("Please help me detered that suspect now", "defense", "deter"),
            ("这里需要detered嫌疑人", "defense", "deter"),
            ("Yesterday we deterred the suspect", "defense", "deter"),
            ("What does detered mean?", "defense", "deter"),
            ("The dog was expelled", "defense", "expel"),
            ("在这里威慑一下嫌疑人", "defense", "deter"),
            ("I said drive-away the suspect now", "defense", "expel"),
            ("This video is live", "video_task", ""),
            ("Please enable CAMERA now", "video_task", ""),
            ("帮我看看这里的实时画面吧", "video_task", ""),
            ("We need to recognize the threat", "object_recognition", ""),
            ("recognition", "object_recognition", ""),
            ("detect", "object_recognition", ""),
            ("这里有可疑物", "object_recognition", ""),
            ("Now locate the yellow dog", "find_object", "yellow dog"),
            ("find", "find_object", ""),
            ("identify", "find_object", ""),
            ("请现在寻找红色玩偶", "find_object", "red doll"),
            ("Please now grab the yellow dog", "grab", "yellow dog"),
            ("I want to pick-up a red cup", "grab", "red cup"),
            ("请现在抓取红色杯子", "grab", "red cup"),
            ("比如 patrol", "patrol", ""),
        ):
            with self.subTest(text=text):
                result = self.classifier.classify(text)
                self.assertEqual((intent, argument), (result.intent, result.argument))

    def test_keyword_boundaries_do_not_match_parts_of_english_words(self) -> None:
        for text in ("leftover", "bright", "background", "straightforward", "findings",
                     "grabber", "patroller", "expeller", "deterrent", "cameraman",
                     "videos", "recognizable", "deteredness", "patrol_area"):
            with self.subTest(text=text):
                self.assertEqual("other", self.classifier.classify(text).intent)

    def test_repeated_synonyms_are_one_action_and_intents_use_priority(self) -> None:
        for text, intent, argument in (
            ("go straight or get forward", "movement", "forward"),
            ("forward forwards 前进", "movement", "forward"),
            ("deter detered threaten 威慑", "defense", "deter"),
            ("patrol area A and deter the suspect", "patrol", "A"),
            ("patrol and move forward", "movement", "forward"),
            ("patrol and video", "patrol", ""),
            ("video recognize deter grab find", "video_task", ""),
            ("recognize deter grab find", "object_recognition", ""),
            ("find the detered doll", "defense", "deter"),
            ("grab and find a red cup", "grab", "and find a red cup"),
            ("巡逻A区域然后威慑嫌疑人", "patrol", "A区域"),
        ):
            with self.subTest(text=text):
                result = self.classifier.classify(text)
                self.assertEqual((intent, argument), (result.intent, result.argument))

    def test_classifies_grab_without_target(self) -> None:
        result = self.classifier.classify("Please grab")

        self.assertEqual("grab", result.intent)
        self.assertEqual("", result.argument)

    def test_unrelated_text_is_other(self) -> None:
        result = self.classifier.classify("What is the weather today?")

        self.assertEqual("other", result.intent)
        self.assertEqual("", result.argument)


class HybridIntentClassifierTest(IsolatedAsyncioTestCase):
    async def test_asr_corrections_and_negations_do_not_call_qwen(self) -> None:
        with patch.dict(os.environ, {"INTENT_BACKEND": "hybrid"}, clear=True):
            service = IntentService(IntentSettings.from_env())
        with patch.object(service.qwen, "classify", side_effect=AssertionError("Unexpected Qwen call")):
            for command, intent, argument in (
                ("detered", "defense", "deter"),
                ("deterred the suspect.", "defense", "deter"),
                ("turn write", "movement", "right"),
                ("Don't detered the suspect", "other", ""),
                ("deter or expel the suspect", "other", ""),
                ("Please help me detered that suspect now", "defense", "deter"),
                ("We need patrol area B now", "patrol", "B"),
                ("Please enable the camera now", "video_task", ""),
                ("left and right", "other", ""),
            ):
                with self.subTest(command=command):
                    result = await service.classify(command)
                    self.assertEqual((intent, argument, "rules"),
                                     (result["intent"], result["argument"], result["backend"]))

    async def test_qwen_cannot_guess_actions_from_unmatched_text(self) -> None:
        with patch.dict(os.environ, {"INTENT_BACKEND": "hybrid"}, clear=True):
            service = IntentService(IntentSettings.from_env())
        for intent, argument in (("movement", "wave"), ("defense", "deter"),
                                 ("defense", "expel")):
            with self.subTest(intent=intent, argument=argument):
                with patch.object(service.qwen, "classify",
                                  return_value=IntentResult(intent, argument, 0.9, "qwen")):
                    result = await service.classify("We are discussing what happened in the park")
                self.assertEqual("other", result["intent"])
                self.assertIsNone(result["executor"])

    async def test_qwen_non_action_semantics_remain_available(self) -> None:
        with patch.dict(os.environ, {"INTENT_BACKEND": "hybrid"}, clear=True):
            service = IntentService(IntentSettings.from_env())
        for text, intent, argument in (
            ("Show me what the lens sees", "video_task", ""),
            ("Have the dog check zone A", "patrol", "A"),
        ):
            with self.subTest(text=text):
                with patch.object(service.qwen, "classify",
                                  return_value=IntentResult(intent, argument, 0.9, "qwen")):
                    result = await service.classify(text)
                self.assertEqual((intent, argument, "qwen"),
                                 (result["intent"], result["argument"], result["backend"]))

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


class AsrIntentIntegrationTest(IsolatedAsyncioTestCase):
    async def test_remote_intent_response_and_local_fallback_agree(self) -> None:
        with patch.dict(os.environ, {"INTENT_BACKEND": "hybrid"}, clear=True):
            intent_settings = IntentSettings.from_env()
            asr_settings = replace(AsrSettings.from_env(), intent_profile="runtime")
        service = IntentService(intent_settings)
        client = TestClient(TestServer(create_app(intent_settings, service)))
        with patch.object(service.qwen, "classify", return_value=IntentResult("other", "", 1.0, "qwen")):
            await client.start_server()
        try:
            with patch.object(service.qwen, "classify",
                              side_effect=AssertionError("Known/blocked commands must use rules")):
                for command, direction in (
                    ("Go straight.", "forward"), ("Get forward", "forward"),
                    ("deters a suspect.", "deter"),
                    ("detered", "deter"), ("Please deterred the suspect.", "deter"),
                    ("Expelled the suspect.", "expel"), ("turn write", "right"),
                    ("Don't detered the suspect", None),
                    ("deter or expel the suspect", None),
                    ("Please help me detered that suspect now", "deter"),
                    ("Please move forward just a little", "forward"),
                    ("left and right", None),
                    ("We need patrol area A now", None),
                ):
                    with self.subTest(command=command):
                        remote = await _resolve_intent(command, replace(
                            asr_settings, intent_url=str(client.make_url("/api/v1/intent"))))
                        local = await _resolve_intent(command, replace(asr_settings, intent_url=""))
                        self.assertEqual(local, remote)
                        self.assertEqual(direction is not None, remote["matched"])
                        self.assertEqual(direction, remote.get("direction"))
        finally:
            await client.close()

    async def test_discovery_keywords_keep_existing_task_mapping(self) -> None:
        settings = replace(AsrSettings.from_env(), intent_profile="discovery", intent_url="")
        for text, expected in (
            ("We need patrol area B7 now", {"type": "TASK", "parameters": {"area": "B7"}}),
            ("The next task is patrol", {"type": "TASK", "parameters": {}}),
            ("Please open the camera", {"type": "VIDEO_TASK", "parameters": {}}),
            ("We should recognize this", {"type": "OBJECT_RECOGNITION", "parameters": {}}),
            ("Don't patrol", {"type": "UNKNOWN", "parameters": {}}),
            ("Move forward just a little", {"type": "UNKNOWN", "parameters": {}}),
        ):
            with self.subTest(text=text):
                self.assertEqual(expected, await _resolve_intent(text, settings))

    async def test_remote_failure_still_corrects_detered_locally(self) -> None:
        async def unavailable(_request):
            return web.Response(status=503)

        app = web.Application()
        app.router.add_post("/api/v1/intent", unavailable)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            settings = replace(AsrSettings.from_env(), intent_profile="runtime",
                               intent_url=str(client.make_url("/api/v1/intent")))
            result = await _resolve_intent("Detered the suspect.", settings)
            self.assertEqual((True, "deter", "rules"),
                             (result["matched"], result["direction"], result["backend"]))
        finally:
            await client.close()
