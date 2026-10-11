"""
Decide whether a turn's pipeline language is English.

The curated check-in pool and keyword tables are English-only, so on a non-English pipeline the
agent degrades to the plain streamed answer rather than speaking broken or English check-ins.
This module holds the one small predicate that decides it; the caller guards every check-in
injection point on the result.
"""

from __future__ import annotations


def is_english(language: str | None) -> bool:
    """
    Return whether ``language`` names English as its primary subtag.

    ``None`` (the AI-task path, which carries no pipeline language) counts as English so that
    check-ins are not suppressed where there is no language to disagree with. A BCP-47 tag is
    compared on its primary subtag only, case-insensitively: ``en``, ``en-US`` and ``en_GB`` are
    English; ``fr``, ``de`` and ``zh-Hans-CN`` are not.
    """
    if language is None:
        return True
    primary = language.strip().split("-", 1)[0].split("_", 1)[0]
    return primary.lower() == "en"
