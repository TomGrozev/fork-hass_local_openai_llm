"""
The single dead-air gate shared by every non-opener check-in.

One turn's check-ins all share one speech horizon (see ``delivery``). The opener is exempt: it
covers the model's first token and always fires. Every later check-in passes through
:class:`DeadAirDebounce` first, and only when the *estimated* silence after the horizon — the
end of the audio already handed to TTS — is at least the gap. When it is not, the check-in is
skipped outright, never queued: a late reassurance is worse than none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custom_components.local_openai.checkin.delivery import CheckinDelivery


def should_emit(*, horizon: float | None, now: float, gap_s: float) -> bool:
    """
    Return whether ``gap_s`` of estimated dead air has passed since the speech horizon.

    ``horizon`` is the estimated end of all audio handed to TTS so far; ``None`` means nothing
    has spoken yet, so silence is unbounded and the check-in is due.
    """
    if horizon is None:
        return True
    return now - horizon >= gap_s


class DeadAirDebounce:
    """Emit the non-opener check-ins, gating each against the delivery's speech horizon."""

    def __init__(self, delivery: CheckinDelivery, *, gap_s: float) -> None:
        """Hold ``delivery`` and the minimum estimated dead air between its check-ins."""
        self._delivery = delivery
        self._gap_s = gap_s

    async def emit_if_due(self, text: str) -> bool:
        """Emit ``text`` through the delivery when due; return whether it was emitted."""
        if not should_emit(
            horizon=self._delivery.horizon,
            now=self._delivery.now,
            gap_s=self._gap_s,
        ):
            return False
        await self._delivery.emit(text)
        return True
