from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import re
import threading
from typing import Any

from .config import IntentSettings
from .command_normalization import command_is_blocked, normalize_command_text
from .command_normalization import (
    DEFENSE_KEYWORDS,
    MOVEMENT_KEYWORDS,
    keyword_arguments,
    keyword_pattern,
)


DEFAULT_INTENTS = {
    "patrol",
    "video_task",
    "object_recognition",
    "defense",
    "movement",
    "other",
}
VIDEO_TASK_KEYWORDS = (
    "video", "camera", "view scene", "view the scene", "view site", "view the site",
    "view live", "view the live", "实时画面", "查看现场",
)
OBJECT_RECOGNITION_KEYWORDS = (
    "recognize", "recognition", "detect", "suspicious object", "suspicious objects", "可疑物",
)
FIND_KEYWORDS = ("look for", "identify", "locate", "search for", "find", "寻找", "识别", "找")
GRAB_KEYWORDS = ("pick up", "grasp", "grab", "抓取", "夹取", "拿起", "拿", "抓")
PATROL_KEYWORDS = ("patrol", "inspect", "inspection", "巡逻", "巡检")
PATROL_AREA_PATTERN = re.compile(r"\b(?:area|zone)\s+([a-z0-9]+)\b", re.IGNORECASE)
LEGACY_PATROL_AREA_PATTERN = re.compile(r"([A-Za-z0-9一二三四五六七八九十]+区域)")
ZH_TO_EN = {
    "黄色": "yellow",
    "黄": "yellow",
    "红色": "red",
    "红": "red",
    "蓝色": "blue",
    "蓝": "blue",
    "绿色": "green",
    "绿": "green",
    "黑色": "black",
    "黑": "black",
    "白色": "white",
    "白": "white",
    "玩具熊": "toy bear",
    "小熊": "bear",
    "熊": "bear",
    "狗": "dog",
    "猫": "cat",
    "杯子": "cup",
    "瓶子": "bottle",
    "可乐": "cola",
    "娃娃": "doll",
    "玩偶": "doll",
    "盲盒": "mystery box",
    "拉布布": "labubu",
}


@dataclass(frozen=True, slots=True)
class IntentResult:
    intent: str
    argument: str
    confidence: float
    backend: str
    argument_i18n: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "success",
            "executor": "robot dog" if self.intent != "other" else None,
            "intent": self.intent,
            "scene": self.intent,
            "argument": self.argument,
            "normalized_argument": self.argument,
            "confidence": round(self.confidence, 4),
            "backend": self.backend,
        }
        if self.argument_i18n is not None:
            payload["normalized_argument_i18n"] = self.argument_i18n
        return payload


class RuleIntentClassifier:
    name = "rules"

    def classify(self, text: str) -> IntentResult:
        normalized = " ".join(str(text or "").strip().split())
        if not normalized or command_is_blocked(normalized):
            return IntentResult("other", "", 1.0, self.name)
        normalized = normalize_command_text(normalized)
        movements = keyword_arguments(normalized, MOVEMENT_KEYWORDS)
        if len(movements) == 1:
            return IntentResult("movement", movements.pop(), 1.0, self.name)
        patrol = self._patrol(normalized)
        if patrol is not None:
            return IntentResult("patrol", patrol, 1.0, self.name)
        if self._video_task(normalized):
            return IntentResult("video_task", "", 1.0, self.name)
        if self._object_recognition(normalized):
            return IntentResult("object_recognition", "", 1.0, self.name)
        defenses = keyword_arguments(normalized, DEFENSE_KEYWORDS)
        if len(defenses) == 1:
            return IntentResult("defense", defenses.pop(), 1.0, self.name)
        grab = self._argument_after_keyword(normalized, GRAB_KEYWORDS)
        if grab is not None:
            argument, i18n = self._normalize_object(grab)
            return IntentResult("grab", argument, 0.98, self.name, i18n)
        found = self._argument_after_keyword(normalized, FIND_KEYWORDS)
        if found is not None:
            argument, i18n = self._normalize_object(found)
            return IntentResult("find_object", argument, 0.98, self.name, i18n)
        return IntentResult("other", "", 0.8, self.name)

    @staticmethod
    def _patrol(text: str) -> str | None:
        lowered = text.lower()
        patrol_match = keyword_pattern(PATROL_KEYWORDS).search(text)
        if patrol_match:
            match = LEGACY_PATROL_AREA_PATTERN.search(text)
            if match:
                return match.group(1)
            match = PATROL_AREA_PATTERN.search(lowered)
            return match.group(1).upper() if match else ""
        return None

    @staticmethod
    def _video_task(text: str) -> bool:
        return keyword_pattern(VIDEO_TASK_KEYWORDS).search(text) is not None

    @staticmethod
    def _object_recognition(text: str) -> bool:
        return keyword_pattern(OBJECT_RECOGNITION_KEYWORDS).search(text) is not None

    @staticmethod
    def _argument_after_keyword(text: str, keywords: tuple[str, ...]) -> str | None:
        match = keyword_pattern(keywords).search(text)
        if match is None:
            return None
        return text[match.end():].strip(" ，。！？,.!?;：:")

    @staticmethod
    def _normalize_object(text: str) -> tuple[str, dict[str, str] | None]:
        original = text.strip(" ，。！？,.!?")
        if not original:
            return "", {"zh": "", "en": ""}
        if not any("\u4e00" <= char <= "\u9fff" for char in original):
            english = re.sub(r"^(the|a|an)\s+", "", " ".join(original.lower().split()))
            return english, {"zh": "", "en": english}
        compact = original.replace("一个", "").replace("一只", "").replace("的", "").replace(" ", "")
        translated = compact
        for source, target in sorted(ZH_TO_EN.items(), key=lambda item: len(item[0]), reverse=True):
            translated = translated.replace(source, f" {target} ")
        english = " ".join(translated.split()).strip().lower()
        if any("\u4e00" <= char <= "\u9fff" for char in english):
            english = original
        return english, {"zh": original, "en": english}


class QwenIntentClassifier:
    name = "qwen"
    _json_pattern = re.compile(r"\{.*?\}", re.DOTALL)

    def __init__(self, settings: IntentSettings) -> None:
        self.settings = settings
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return self._tokenizer, self._model
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = Path(self.settings.model)
            if not model_path.is_dir():
                raise FileNotFoundError(f"本地 Qwen 模型目录不存在: {model_path}")
            source = str(model_path)
            tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
            kwargs: dict[str, Any] = {"torch_dtype": "auto", "local_files_only": True}
            if self.settings.device == "auto":
                kwargs["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
            if self.settings.device != "auto":
                model = model.to(self.settings.device)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            return tokenizer, model

    def classify(self, text: str) -> IntentResult:
        tokenizer, model = self._load()
        prompt = (
            "This is a campus patrol scenario using smart glasses and a robot dog. Classify only "
            f"the request into one of {list(self.settings.candidates)}. Return JSON only "
            "with keys intent and argument. Movement argument must be forward, backward, left, "
            "right, or happy. Examples: move forward=forward, move backward=backward, "
            "turn left=left, turn right=right. "
            "A request to start a campus patrol or inspection must be patrol; its argument is "
            "the requested area. Requests for a live view or to view the site must be video_task. "
            "Requests for suspicious-object recognition must be object_recognition. Commands to "
            "threaten, deter or intimidate a suspect must be defense with argument deter. "
            "Expelling or driving away a suspect must be defense with argument expel. "
            "Never use happy or suspect as the argument for these defense requests. "
            f"Command: {json.dumps(text, ensure_ascii=False)}"
        )
        messages = [
            {"role": "system", "content": "You are a strict campus patrol robot intent classifier."},
            {"role": "user", "content": prompt},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
        with self._inference_lock:
            generated = model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
            )
        answer = tokenizer.decode(generated[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        match = self._json_pattern.search(answer)
        if match is None:
            raise ValueError("Qwen did not return JSON")
        payload = json.loads(match.group(0))
        intent = str(payload.get("intent") or payload.get("scene") or "other").strip().lower()
        argument = " ".join(str(payload.get("argument") or payload.get("normalized_argument") or "").split())
        if intent not in self.settings.candidates:
            raise ValueError(f"Unsupported intent from Qwen: {intent}")
        return IntentResult(intent, "" if intent == "other" else argument, 0.9, self.name)


class IntentService:
    def __init__(self, settings: IntentSettings) -> None:
        self.settings = settings
        self.rules = RuleIntentClassifier()
        self.qwen = QwenIntentClassifier(settings)
        self.last_backend = "rules"
        self.last_error: str | None = None
        self.warmed_up = False
        self._warmup_lock = asyncio.Lock()

    async def warmup(self) -> None:
        if self.settings.backend not in {"qwen", "hybrid"}:
            return
        async with self._warmup_lock:
            if self.warmed_up:
                return
            try:
                # Load weights and execute generation; discard the synthetic
                # classification instead of exposing it as a user request.
                await asyncio.to_thread(self.qwen.classify, "What is the weather today?")
            except Exception as exc:
                self.last_error = str(exc)
                if self.settings.backend == "qwen":
                    raise
                return  # Hybrid deployments retain their rules fallback.
            self.warmed_up = True
            self.last_error = None

    async def classify(self, text: str) -> dict[str, Any]:
        normalized = " ".join(str(text or "").strip().split())
        rule_result = self.rules.classify(normalized)
        if command_is_blocked(normalized) or self.settings.backend == "rules" or (
            rule_result.intent in self.settings.candidates and rule_result.intent != "other"
        ):
            if rule_result.intent not in self.settings.candidates:
                rule_result = IntentResult("other", "", 1.0, self.rules.name)
            self.last_backend = rule_result.backend
            self.last_error = None
            return rule_result.to_dict()
        if self.settings.backend in {"qwen", "hybrid"} and normalized:
            try:
                result = await asyncio.to_thread(self.qwen.classify, normalized)
                # Runtime actions are a bounded command vocabulary. A model
                # guess without a matching command must not turn chatter or
                # an ambiguous ASR fragment into a runtime action.
                if result.intent in {"defense", "movement"}:
                    result = IntentResult("other", "", 1.0, self.rules.name)
                self.last_backend = result.backend
                self.last_error = None
                return result.to_dict()
            except Exception as exc:
                self.last_error = str(exc)
                if self.settings.backend == "qwen":
                    raise
        result = rule_result
        if result.intent not in self.settings.candidates:
            result = IntentResult("other", "", 1.0, self.rules.name)
        self.last_backend = result.backend
        return result.to_dict()

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "intent",
            "backend": self.settings.backend,
            "lastBackend": self.last_backend,
            "warmedUp": self.warmed_up,
            "model": self.settings.model if self.settings.backend != "rules" else None,
            "lastError": self.last_error,
            "intents": list(self.settings.candidates),
        }
