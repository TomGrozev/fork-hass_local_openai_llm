"""
Estimate how long spoken text takes to say.

Check-in timing is measured against the estimated *end* of the audio already handed to TTS, not
against the moment a check-in was emitted: a short opener can still be speaking seconds later,
and a check-in dropped in then only delays the real answer. The estimate is deliberately crude -
the TTS engine's rate and start latency are unobservable from the entity - and lives in one pure
function so the debounce has a single number to compare against.
"""

from __future__ import annotations

from custom_components.local_openai.checkin.pool import strip_mood_tags


def estimate_speech_s(text: str, *, words_per_minute: float) -> float:
    """
    Estimate how many seconds ``text`` takes to speak at ``words_per_minute``.

    ``[mood]`` tags are stripped first (they are not spoken), then the remaining words are
    counted and divided by the rate. Empty or tag-only text estimates to 0.

    ponytail: ``words_per_minute`` is a user-set average with no TTS start latency and no pause
    tags counted. The upgrade path, if check-in timing ever feels off against a specific engine,
    is a measured or per-voice rate.
    """
    return len(strip_mood_tags(text).split()) / (words_per_minute / 60)
