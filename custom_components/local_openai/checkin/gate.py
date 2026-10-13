"""
The check-before-speak gate for the opener.

An opener is only worth speaking if the LLM is actually slow, so it is gated: ``answer_started``
is set by the stream transform on the first non-empty content delta, and ``run`` waits out the
grace window before deciding. If the answer started first the opener is suppressed silently —
nothing is spoken and the answer's delta stream is left untouched.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custom_components.local_openai.checkin.delivery import CheckinDelivery
    from custom_components.local_openai.checkin.pool import LinePool


class OpenerGate:
    """
    Speak a rendered opener unless the real answer beats the grace window.

    Constructed at chat-log-handler entry, so the grace window starts before context injection
    and the first completion. ``answer_started`` is handed to the stream transform; ``run`` is
    the opener task body.
    """

    def __init__(  # noqa: PLR0913 - one field per piece of per-turn context
        self,
        *,
        grace_s: float,
        pool: LinePool,
        delivery: CheckinDelivery,
        bucket: str,
        intent: str | None,
        expression_tags: bool,
    ) -> None:
        """Configure the gate for one turn's opener."""
        self._grace_s = grace_s
        self._pool = pool
        self._delivery = delivery
        self._bucket = bucket
        self._intent = intent
        self._expression_tags = expression_tags
        self._answer_started = asyncio.Event()

    @property
    def answer_started(self) -> asyncio.Event:
        """The event the stream transform sets on the first non-empty content delta."""
        return self._answer_started

    async def run(self) -> bool:
        """
        Wait the grace window, then emit the opener unless the answer started.

        Returns whether the opener was emitted (``False`` = silently suppressed).
        """
        try:
            await asyncio.wait_for(self._answer_started.wait(), timeout=self._grace_s)
        except TimeoutError:
            await self._delivery.emit(
                self._pool.opener(
                    self._bucket, self._intent, expression_tags=self._expression_tags
                )
            )
            return True
        return False
