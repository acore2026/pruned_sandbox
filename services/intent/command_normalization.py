"""Bounded corrections for ASR command text, never for the displayed transcript."""
from __future__ import annotations

import re


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
NARRATIVE = re.compile(
    r"\b(?:yesterday|was|were|said|says|meaning|means|meant|example)\b"
    r"|昨天|说过|说了|是什么意思|比如|例如",
    re.IGNORECASE,
)
REQUEST_PREFIX = re.compile(
    r"^(?:(?:please|can you|could you|would you)\s+)*"
    r"(?:(?:ask|tell|send)\s+(?:the\s+)?(?:robot\s+)?dog\s+to\s+)?",
    re.IGNORECASE,
)
COMMAND_WORDS = ("deter", "expel", "threaten", "intimidate", "patrol", "inspect")
CONFLICTING_ACTION = re.compile(
    r"\b(?:and|or|then)\s+(?:(?:please|also)\s+)?"
    r"(?:deter(?:s|ed|red|ring|ing)?|expel(?:led|ling)?|threaten(?:ed)?|"
    r"intimidat(?:e|ed)|drive away|move|go|get forward|turn|forward|backward|left|right|"
    r"happy|patrol|inspect)\b"
    r"|(?:还是|或者|然后|并且|同时)(?:威吓|威慑|驱逐|前进|后退|左转|右转|巡逻|巡检)",
    re.IGNORECASE,
)
# These real words must not be changed just because they are one edit away.
FUZZY_EXCLUSIONS = {"defer"}


def command_is_blocked(text: str) -> bool:
    """Negated, reported or conflicting actions cannot become one action."""
    text = text.replace("’", "'")
    return bool(NEGATION.search(text) or NARRATIVE.search(text)
                or CONFLICTING_ACTION.search(text))


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

    Object names and area slots are never spell-corrected. Homophones such as
    write/lift are accepted only after a movement verb, never as bare words.
    """
    if command_is_blocked(text):
        return text
    cleaned = " ".join(text.strip().split()).strip(" .!?，。！？")
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
    # Only the leading action verb or a direction in a short movement
    # template may be corrected. Leave the rest of the sentence intact.
    words[0] = WORD_ALIASES.get(words[0], words[0])
    if len(words) == 2 and words[0] in {"move", "go", "step", "turn"}:
        words[1] = WORD_ALIASES.get(words[1], words[1])
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
