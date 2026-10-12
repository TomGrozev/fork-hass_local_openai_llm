"""
Emit a check-in delta into the chat log and track the turn's speech horizon.

Every kind of check-in delivers the same way — a short assistant content delta on the chat
log's streaming path: the opener that covers the model's first token, the mid-chain fragment
before a tool step, the watchdog reassurance during a long wait. They also share one emission
clock, so a later check-in can be measured against the audio still playing and skipped rather
than stacked on top of it.

The clock alone is not enough to know whether it is dead air: a check-in (or the model's own
visible preamble) can still be speaking for seconds after it was handed to TTS. So the delivery
also keeps a *speech horizon* — an estimated time at which everything spoken so far finishes —
and every check-in, opener included, plus every visible chunk the model streams, pushes it out
by :func:`estimate_speech_s`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from custom_components.local_openai.checkin.speech import estimate_speech_s

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from homeassistant.components import conversation


class CheckinDelivery:
    """
    Emit check-in deltas into one turn's chat log and track when its audio will finish.

    One instance per turn, wrapping that turn's ``chat_log``. ``emit`` writes ``text`` as an
    assistant content delta through HA's streaming path — the only supported channel from a
    conversation entity — and pushes the speech horizon out, so a later check-in can be skipped
    rather than stacked on top of it. ``note_spoken_text`` does the same for the model's own
    visible chunks without counting as a check-in.
    """

    def __init__(
        self,
        chat_log: conversation.ChatLog,
        entity_id: str,
        *,
        words_per_minute: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wrap ``chat_log`` for one turn; speech estimated at ``words_per_minute``."""
        self._chat_log = chat_log
        self._entity_id = entity_id
        self._words_per_minute = words_per_minute
        self._clock = clock
        self._horizon: float | None = None

    @property
    def horizon(self) -> float | None:
        """
        Estimated monotonic time all spoken audio finishes, or ``None`` if nothing has spoken.

        ``None`` means no audio is queued this turn, so there is nothing to wait out.
        """
        return self._horizon

    @property
    def now(self) -> float:
        """The current reading of the emission clock (the same clock ``emit`` stamps with)."""
        return self._clock()

    def note_spoken_text(self, text: str) -> None:
        """
        Push the speech horizon out by the model's own spoken ``text``.

        Not a check-in: it records that the model is speaking, so the debounce measures dead air
        after the model's audio rather than after the last check-in.
        """
        self._extend_horizon(text)

    def _extend_horizon(self, text: str) -> None:
        """Push the horizon out by ``text``'s estimated speech, floored at the current time."""
        now = self._clock()
        base = now if self._horizon is None else max(self._horizon, now)
        self._horizon = base + estimate_speech_s(
            text, words_per_minute=self._words_per_minute
        )

    async def emit(self, text: str) -> None:
        """Append ``text`` to the chat log as an assistant delta, then push the horizon out."""

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
        self._extend_horizon(text)
