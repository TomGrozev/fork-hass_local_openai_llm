"""
Emit a check-in delta into the chat log and record when it was emitted.

Every kind of check-in delivers the same way — a short assistant content delta on the chat
log's streaming path: the opener that covers the model's first token, the mid-chain fragment
before a tool step, the watchdog reassurance during a long wait. They also share one emission
clock, so a later check-in can be measured against the most recent one and skipped rather than
stacked on top of it.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from homeassistant.components import conversation


class CheckinDelivery:
    """
    Emit check-in deltas into one turn's chat log, stamping each emission on one clock.

    One instance per turn, wrapping that turn's ``chat_log``. ``emit`` writes ``text`` as an
    assistant content delta through HA's streaming path — the only supported channel from a
    conversation entity — and records the time, so a later check-in can be skipped rather than
    stacked on top of it.
    """

    def __init__(
        self,
        chat_log: conversation.ChatLog,
        entity_id: str,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wrap ``chat_log`` for one turn; emission times read from ``clock``."""
        self._chat_log = chat_log
        self._entity_id = entity_id
        self._clock = clock
        self._last_emission: float | None = None

    @property
    def last_emission(self) -> float | None:
        """Monotonic time of the most recent emitted check-in, or ``None`` if none yet."""
        return self._last_emission

    @property
    def now(self) -> float:
        """The current reading of the emission clock (the same clock ``emit`` stamps with)."""
        return self._clock()

    async def emit(self, text: str) -> None:
        """Append ``text`` to the chat log as an assistant delta, then stamp the emission time."""

        async def _deltas() -> AsyncIterator[conversation.AssistantContentDeltaDict]:
            # The ``role`` key is load-bearing: HA's assist pipeline remembers the latest delta
            # role and hands content to TTS only while it is ``assistant``. A role-less delta at
            # the start of a turn (no role seen yet) or after a tool result (role
            # ``tool_result``) is shown in the debug log but never spoken. Each call is its own
            # stream, so the check-in commits as its own ``AssistantContent`` ahead of the
            # answer's.
            yield {"role": "assistant", "content": text}

        stream = self._chat_log.async_add_delta_content_stream(
            self._entity_id, _deltas()
        )
        # The stream generator is lazy: it writes nothing unless it is consumed.
        async for _ in stream:
            pass
        self._last_emission = self._clock()
