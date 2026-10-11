"""
The single dead-air gate shared by every non-opener check-in.

One turn's check-ins all share one emission clock (see ``delivery``). The opener is exempt:
it covers the model's first token and always fires. Every later check-in passes through
:class:`DeadAirDebounce` first, so two never land back to back. When the gap has not elapsed
the check-in is skipped outright, never queued: a late reassurance is worse than none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custom_components.local_openai.checkin.delivery import CheckinDelivery


def should_emit(*, last_emission: float | None, now: float, gap_s: float) -> bool:
    """Return whether at least ``gap_s`` has elapsed since the last check-in's emission."""
    if last_emission is None:
        return True
    return now - last_emission >= gap_s


class DeadAirDebounce:
    """Emit the non-opener check-ins, gating each against the shared clock."""

    def __init__(self, delivery: CheckinDelivery, *, gap_s: float) -> None:
        """Hold ``delivery`` and the minimum gap between its check-ins."""
        self._delivery = delivery
        self._gap_s = gap_s

    async def emit_if_due(self, text: str) -> bool:
        """Emit ``text`` through the delivery when due; return whether it was emitted."""
        if not should_emit(
            last_emission=self._delivery.last_emission,
            now=self._delivery.now,
            gap_s=self._gap_s,
        ):
            return False
        await self._delivery.emit(text)
        return True
