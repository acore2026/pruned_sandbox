from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import re
import threading
from typing import Any

from .config import IntentSettings


DEFAULT_INTENTS = {
    "patrol",
    "video_task",
    "object_recognition",
    "defense",
    "movement",
    "other",
}
DEFENSE_KEYWORDS = ("威吓", "驱逐")
VIDEO_TASK_KEYWORDS = ("实时画面", "查看现场")
OBJECT_RECOGNITION_KEYWORDS = ("可疑物识别", "识别可疑物")
MOVEMENT_COMMANDS = {
    "向前": "forward",
    "向前走": "forward",
    "请向前走": "forward",
    "往前走": "forward",
    "前进": "forward",
    "move forward": "forward",
    "forward": "forward",
    "退后": "backward",
    "向后": "backward",
    "向后走": "backward",
    "请向后走": "backward",
    "往后退": "backward",
    "后退": "backward",
    "move back": "backward",
    "back": "backward",
    "backward": "backward",
    "向左": "left",
    "左转": "left",
    "向左转": "left",
    "turn left": "left",
    "left": "left",
    "向右": "right",
    "右转": "right",
    "向右转": "right",
    "turn right": "right",
    "right": "right",
    "挥手": "wave",
    "打招呼": "wave",
    "你好": "wave",
    "hello": "wave",
    "wave": "wave",
}
FIND_PREFIXES = (
    "请帮我寻找",
    "请帮我找",
    "帮我寻找",
    "帮我找",
    "帮我识别",
    "请识别",
    "寻找",
    "识别",
    "找",
    "look for",
    "identify",
    "locate",
    "find the",
    "find",
)
GRAB_PREFIXES = (
    "请帮我抓取",
    "帮我抓取",
    "请抓取",
    "抓取",
    "请帮我夹取",
    "帮我夹取",
    "夹取",
    "请帮我拿起",
    "帮我拿起",
    "请帮我拿",
    "帮我拿",
    "拿起",
    "请抓",
    "帮我抓",
    "抓",
    "please pick up",
    "please grasp",
    "please grab",
    "pick up",
    "grasp",
    "grab",
)
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
        if not normalized:
            return IntentResult("other", "", 1.0, self.name)
        movement = self._movement(normalized)
        if movement:
            return IntentResult("movement", movement, 1.0, self.name)
        patrol = self._patrol(normalized)
        if patrol is not None:
            return IntentResult("patrol", patrol, 1.0, self.name)
        if self._video_task(normalized):
            return IntentResult("video_task", "", 1.0, self.name)
        if self._object_recognition(normalized):
            return IntentResult("object_recognition", "", 1.0, self.name)
        if self._defense(normalized):
            return IntentResult("defense", "suspect", 1.0, self.name)
        grab = self._argument_after_prefix(normalized, GRAB_PREFIXES)
        if grab is not None:
            argument, i18n = self._normalize_object(grab)
            return IntentResult("grab", argument, 0.98, self.name, i18n)
        found = self._argument_after_prefix(normalized, FIND_PREFIXES)
        if found is not None and found:
            argument, i18n = self._normalize_object(found)
            return IntentResult("find_object", argument, 0.98, self.name, i18n)
        return IntentResult("other", "", 0.8, self.name)

    @staticmethod
    def _movement(text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", text.lower())
        cleaned = " ".join(cleaned.split())
        if cleaned.startswith("please "):
            cleaned = cleaned[7:]
        return MOVEMENT_COMMANDS.get(cleaned, "")

    @staticmethod
    def _patrol(text: str) -> str | None:
        if "巡逻" not in text and "巡检" not in text:
            return None
        match = re.search(r"([A-Za-z0-9一二三四五六七八九十]+区域)", text)
        return match.group(1) if match else ""

    @staticmethod
    def _defense(text: str) -> bool:
        return any(keyword in text for keyword in DEFENSE_KEYWORDS)

    @staticmethod
    def _video_task(text: str) -> bool:
        return any(keyword in text for keyword in VIDEO_TASK_KEYWORDS)

    @staticmethod
    def _object_recognition(text: str) -> bool:
        return any(keyword in text for keyword in OBJECT_RECOGNITION_KEYWORDS) or (
            "可疑物" in text and "识别" in text
        )

    @staticmethod
    def _argument_after_prefix(text: str, prefixes: tuple[str, ...]) -> str | None:
        lowered = text.lower()
        for prefix in prefixes:
            lowered_prefix = prefix.lower()
            if lowered == lowered_prefix:
                return ""
            if not lowered.startswith(lowered_prefix):
                continue
            remainder = text[len(prefix) :]
            if any("\u4e00" <= char <= "\u9fff" for char in prefix) or remainder.startswith(" "):
                return remainder.strip(" ，。！？,.!?")
        return None

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
            "right, or wave. Examples: 向前=forward, 退后=backward, 向左=left, 向右=right. "
            "A request to start a campus patrol or inspection must be patrol; its argument is "
            "the requested area. Requests for a live view or to view the site must be video_task. "
            "Requests for suspicious-object recognition must be object_recognition. Commands to "
            "threaten or expel a suspect must be defense; its argument is suspect. "
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

    async def classify(self, text: str) -> dict[str, Any]:
        normalized = " ".join(str(text or "").strip().split())
        rule_result = self.rules.classify(normalized)
        if (
            self.settings.backend == "rules"
            or rule_result.intent
            in {"patrol", "video_task", "object_recognition", "defense", "movement"}
        ):
            if rule_result.intent not in self.settings.candidates:
                rule_result = IntentResult("other", "", 1.0, self.rules.name)
            self.last_backend = rule_result.backend
            self.last_error = None
            return rule_result.to_dict()
        if self.settings.backend in {"qwen", "hybrid"} and normalized:
            try:
                result = await asyncio.to_thread(self.qwen.classify, normalized)
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
            "model": self.settings.model if self.settings.backend != "rules" else None,
            "lastError": self.last_error,
            "intents": list(self.settings.candidates),
        }
