"""Bounded corrections for ASR command text, never for the displayed transcript."""
from __future__ import annotations

import re
from functools import lru_cache


MOVEMENT_KEYWORDS = {
    "forward": ("forward", "straight", "向前", "前进"),
    "backward": ("backward", "back", "向后", "后退", "退后"),
    "left": ("left", "向左", "左转"),
    "right": ("right", "向右", "右转"),
    "happy": ("happy",),
}
DEFENSE_KEYWORDS = {
    "deter": ("deter", "threaten", "intimidate", "威吓", "威慑"),
    "expel": ("expel", "drive away", "驱逐"),
}


@lru_cache(maxsize=64)
def keyword_pattern(keywords: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = []
    for keyword in sorted(keywords, key=len, reverse=True):
        escaped = re.escape(keyword).replace(r"\ ", r"[^\w]+")
        if not re.search(r"[\u4e00-\u9fff]", keyword):
            escaped = r"(?<![a-z0-9_])" + escaped + r"(?![a-z0-9_])"
        alternatives.append(escaped)
    return re.compile("|".join(alternatives), re.IGNORECASE)


def keyword_arguments(text: str, keywords: dict[str, tuple[str, ...]]) -> set[str]:
    return {argument for argument, words in keywords.items()
            if keyword_pattern(words).search(text)}


WORD_ALIASES = {
    "deters": "deter",
    "detered": "deter", "deterred": "deter",
    "deterring": "deter", "detering": "deter",
    "expelled": "expel", "expelling": "expel",
    "threatened": "threaten", "intimidated": "intimidate",
    "forwards": "forward", "backwards": "backward",
    "patrole": "patrol", "patrolled": "patrol",
}
NEGATION = re.compile(
    r"\b(?:not|never|don't|dont|cannot|can't|cant|won't|wont)\b"
    r"|^(?:please\s+)?(?:no|cancel|stop)\b"
    r"|不要|(?:^|[，。！？\s]|请|机器狗)别|禁止|不许|不用|不能|取消|停止",
    re.IGNORECASE,
)
REQUEST_PREFIX = re.compile(
    r"^(?:(?:please|can you|could you|would you)\s+)*"
    r"(?:(?:ask|tell|send)\s+(?:the\s+)?(?:robot\s+)?dog\s+to\s+)?",
    re.IGNORECASE,
)
COMMAND_WORDS = ("deter", "expel", "threaten", "intimidate", "patrol", "inspect")
# These real words must not be changed just because they are one edit away.
FUZZY_EXCLUSIONS = {"defer"}


def command_is_blocked(text: str) -> bool:
    """Keep negation and conflicting directions out of keyword commands."""
    text = text.replace("’", "'")
    text = normalize_word_aliases(text)
    return bool(NEGATION.search(text)
                or len(keyword_arguments(text, MOVEMENT_KEYWORDS)) > 1
                or len(keyword_arguments(text, DEFENSE_KEYWORDS)) > 1)


def normalize_word_aliases(text: str) -> str:
    return keyword_pattern(tuple(WORD_ALIASES)).sub(
        lambda match: WORD_ALIASES[match[0].lower()], text)


def _one_edit_apart(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        differences = [index for index, (a, b) in enumerate(zip(left, right)) if a != b]
        return len(differences) == 1 or (
            len(differences) == 2 and differences[1] == differences[0] + 1
            and left[differences[0]] == right[differences[1]]
            and left[differences[1]] == right[differences[0]]
        )
    shorter, longer = sorted((left, right), key=len)
    return any(longer[:index] + longer[index + 1:] == shorter
               for index in range(len(longer)))


def normalize_command_text(text: str) -> str:
    """Correct known variants and a unique one-edit command verb in templates.

    Explicit aliases apply anywhere; fuzzy corrections remain restricted to
    the existing templates. Displayed transcripts are never changed.
    """
    if command_is_blocked(text):
        return text
    cleaned = normalize_word_aliases(" ".join(text.strip().split()).strip(" .!?，。！？"))
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        return cleaned
    cleaned = cleaned.lower()
    command = REQUEST_PREFIX.sub("", cleaned, count=1)
    # This full phrase occurred in glasses ASR logs. It is intentionally not
    # a global mapping of 'turds' or 'suspect' to a defense action.
    if command == "the turds i suspect":
        return "deter the suspect"
    if re.fullmatch(r"detour\s+(?:the\s+)?suspect", command):
        return "deter the suspect"
    command = re.sub(r"^ex[- ]pel(?=\s|$)", "expel", command)
    command = re.sub(
        r"^(move|go|turn)\s+(write|lift)$",
        lambda match: match[1] + (" right" if match[2] == "write" else " left"),
        command,
    )
    words = command.split()
    if not words:
        return text
    # Fuzzy corrections are limited to short command templates.
    if len(words) == 2 and words[0] in {"move", "go", "step", "turn"}:
        directions = ("forward", "backward", "left", "right")
        if words[1] not in directions and len(words[1]) >= 4:
            matches = [word for word in directions if _one_edit_apart(words[1], word)]
            if len(matches) == 1:
                words[1] = matches[0]
    # Fuzzy defense verbs require the suspect slot; fuzzy patrol verbs require
    # a named area/zone. This prevents correcting arbitrary English words.
    tail = " ".join(words[1:])
    candidates = ()
    if re.fullmatch(r"(?:(?:the|a) )?suspects?", tail):
        candidates = COMMAND_WORDS[:4]
    elif re.fullmatch(r"(?:area|zone) [a-z0-9]+", tail):
        candidates = COMMAND_WORDS[4:]
    if (words[0] not in COMMAND_WORDS and words[0] not in FUZZY_EXCLUSIONS
            and len(words[0]) >= 4):
        matches = [word for word in candidates if _one_edit_apart(words[0], word)]
        if len(matches) == 1:
            words[0] = matches[0]
    return " ".join(words)
