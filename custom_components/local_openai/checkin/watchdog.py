"""
Capped reassurance during a long wait.

The opener covers the first token and the mid-chain marks a tool step; the watchdog covers the
silence in between, when a tool (or the model) is taking a long time. It shares the one
dead-air debounce with the mid-chain, so a reassurance can never land on top of the last
check-in, and it is capped: after ``max_extras`` reassurances it goes quiet and lets the real
tokens land whenever they arrive - no abort, no integration-level deadline. Once the real
answer has started it returns immediately, so a stall mid-answer can never splice a
reassurance into the spoken answer.

It is driven by the *loop* clock (one ``gap_s`` sleep per attempt) while the debounce reads the
*emission* clock; the two are advanced independently in tests.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custom_components.local_openai.checkin.debounce import DeadAirDebounce
    from custom_components.local_openai.checkin.pool import LinePool


class Watchdog:
    """Speak capped reassurances while a turn sits in dead air."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        gap_s: float,
        max_extras: int,
        pool: LinePool,
        debounce: DeadAirDebounce,
        expression_tags: bool,
        opener_task: asyncio.Task[bool] | None,
        answer_started: asyncio.Event,
    ) -> None:
        """Configure one turn's watchdog: one reassurance per ``gap_s`` of silence, capped."""
        self._gap_s = gap_s
        self._max_extras = max_extras
        self._pool = pool
        self._debounce = debounce
        self._expression_tags = expression_tags
        self._opener_task = opener_task
        self._answer_started = answer_started
        self._extras = 0
        self._progressed = False

    def note_progress(self) -> None:
        """Record that the model spoke (a chunk carrying choices) during the interval."""
        self._progressed = True

    async def run(self) -> None:
        """Emit a reassurance per silent gap until the cap, then return."""
        while self._extras < self._max_extras:
            self._progressed = False
            await asyncio.sleep(self._gap_s)
            # The real answer has begun: the watchdog's job is over. Its reassurance would be
            # spliced into the already-streaming answer, so it must stop even if the answer
            # then stalls for a whole gap.
            if self._answer_started.is_set():
                return
            # The opener is always spoken; an extra check-in must never pre-empt the opener.
            if self._opener_task is not None and not self._opener_task.done():
                continue
            # The model spoke during the interval, so this is not dead air.
            if self._progressed:
                continue
            if await self._debounce.emit_if_due(
                self._pool.watchdog(expression_tags=self._expression_tags)
            ):
                self._extras += 1
