from __future__ import annotations

from collections.abc import Iterable
import re


# Match whole English words so "swordfish" and "password" stay unchanged.
_EN_SWORD = re.compile(
    r"\b(?:swords?|light[\s-]*sab(?:er|re)s?|laser[\s-]+swords?|sab(?:er|re)s?)\b",
    re.IGNORECASE,
)
_ZH_SWORD = re.compile(
    r"(?:(?:红色?|蓝色?|绿色?|黄色?|紫色?|白色?|黑色?|橙色?|粉色?|银色?|金色?|"
    r"玩具|塑料|金属|儿童|发光|激光|星球大战|一个|一把|那把|这把|的))*"
    r"(?:光剑|激光剑|宝剑|长剑|短剑|武士剑|双剑|剑)"
)


def normalize_class(value: str) -> str:
    """Map sword descriptions to the trained class, preserving other targets."""
    normalized = value.strip()
    compact = re.sub(r"\s+", "", normalized)
    if _EN_SWORD.search(normalized) or _ZH_SWORD.fullmatch(compact):
        return "sword"
    return normalized


def normalize_classes(values: Iterable[str]) -> list[str]:
    normalized = (normalize_class(value) for value in values)
    return list(dict.fromkeys(target for target in normalized if target))
