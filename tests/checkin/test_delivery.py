"""Unit tests for CheckinDelivery's chat-log emission and clock stamping."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from custom_components.local_openai.checkin.delivery import CheckinDelivery

ENTITY_ID = "conversation.local_openai"


class _MutableClock:
    """A settable emission clock the test advances directly."""

    def __init__(self, now: float) -> None:
        """Start the clock at ``now``."""
        self.now = now


class _FakeChatLog:
    """A minimal stand-in for ``conversation.ChatLog``'s streaming path.

    Records every delta handed to ``async_add_delta_content_stream`` so a test can see exactly
    what was written, mirroring the real method's lazy async-generator contract.
    """

    def __init__(self) -> None:
        """Start with nothing received."""
        self.agent_ids: list[str] = []
        self.received: list[dict[str, Any]] = []

    async def async_add_delta_content_stream(
        self, agent_id: str, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[None]:
        """Consume ``stream``, recording each delta, then yield a dummy completion."""
        self.agent_ids.append(agent_id)
        async for delta in stream:
            self.received.append(delta)
            yield None


class TestEmit:
    """emit writes one assistant-attributed content delta through the streaming path."""

    async def test_delta_is_attributed_to_the_assistant(self) -> None:
        """The delta carries ``role: assistant``; HA's pipeline speaks only assistant content."""
        chat_log = _FakeChatLog()
        delivery = CheckinDelivery(chat_log, ENTITY_ID, words_per_minute=150)

        await delivery.emit("hi")

        assert chat_log.received == [{"role": "assistant", "content": "hi"}]

    async def test_streams_under_the_turn_entity_id(self) -> None:
        """The delta stream is opened for the entity the delivery wraps."""
        chat_log = _FakeChatLog()
        delivery = CheckinDelivery(chat_log, ENTITY_ID, words_per_minute=150)

        await delivery.emit("hi")

        assert chat_log.agent_ids == [ENTITY_ID]


class TestNow:
    """now reads the injected emission clock emit stamps with."""

    def test_returns_the_injected_clock_reading(self) -> None:
        """A fixed clock reads back exactly what it was injected with."""
        delivery = CheckinDelivery(
            _FakeChatLog(), ENTITY_ID, words_per_minute=150, clock=lambda: 7.5
        )

        assert delivery.now == 7.5

    def test_follows_a_mutable_clock(self) -> None:
        """A moving clock is re-read on every access, not cached."""
        readings = iter([1.0, 2.0])
        delivery = CheckinDelivery(
            _FakeChatLog(),
            ENTITY_ID,
            words_per_minute=150,
            clock=lambda: next(readings),
        )

        assert delivery.now == 1.0
        assert delivery.now == 2.0


class TestHorizon:
    """horizon estimates when all audio handed to TTS so far finishes speaking."""

    def test_is_none_before_any_speech(self) -> None:
        """A fresh delivery has queued no audio, so there is nothing to wait out."""
        delivery = CheckinDelivery(
            _FakeChatLog(), ENTITY_ID, words_per_minute=150, clock=lambda: 42.0
        )

        assert delivery.horizon is None

    async def test_emit_extends_horizon_by_the_estimated_speech(self) -> None:
        """An emitted check-in pushes the horizon out by its own spoken duration."""
        delivery = CheckinDelivery(
            _FakeChatLog(), ENTITY_ID, words_per_minute=150, clock=lambda: 100.0
        )

        await delivery.emit("one two three four five")  # 5 words at 2.5 words/s -> 2 s

        assert delivery.horizon == pytest.approx(102.0)

    async def test_uses_the_injected_speaking_rate(self) -> None:
        """A faster rate estimates less speech time for the same words."""
        delivery = CheckinDelivery(
            _FakeChatLog(),
            ENTITY_ID,
            words_per_minute=300,
            clock=lambda: 100.0,
        )

        await delivery.emit("one two three four five")  # 5 words at 300 wpm -> 1 s

        assert delivery.horizon == pytest.approx(101.0)

    async def test_successive_emits_stack_onto_the_horizon(self) -> None:
        """Each spoken text extends the horizon from wherever the last one left it."""
        clock = _MutableClock(100.0)
        delivery = CheckinDelivery(
            _FakeChatLog(), ENTITY_ID, words_per_minute=150, clock=lambda: clock.now
        )

        await delivery.emit("one two")  # 2 words -> 0.8 s
        await delivery.emit("one two")  # a further 0.8 s, clock unchanged

        assert delivery.horizon == pytest.approx(101.6)

    def test_note_spoken_text_extends_the_horizon(self) -> None:
        """The model's own visible text is spoken, so it pushes the horizon out too."""
        delivery = CheckinDelivery(
            _FakeChatLog(), ENTITY_ID, words_per_minute=150, clock=lambda: 100.0
        )

        delivery.note_spoken_text("one two three four")  # 4 words -> 1.6 s

        assert delivery.horizon == pytest.approx(101.6)

    def test_note_spoken_text_is_not_a_checkin(self) -> None:
        """Model text never writes to the chat log; only ``emit`` speaks a check-in."""
        chat_log = _FakeChatLog()
        delivery = CheckinDelivery(
            chat_log, ENTITY_ID, words_per_minute=150, clock=lambda: 100.0
        )

        delivery.note_spoken_text("hello there")

        assert chat_log.received == []

    def test_a_stale_horizon_is_floored_at_the_current_time(self) -> None:
        """Speech that has already finished by ``now`` restarts from ``now``, not the past."""
        clock = _MutableClock(100.0)
        delivery = CheckinDelivery(
            _FakeChatLog(), ENTITY_ID, words_per_minute=150, clock=lambda: clock.now
        )

        delivery.note_spoken_text("one two")  # horizon -> 100.8
        clock.now = 200.0
        delivery.note_spoken_text("one two")  # max(100.8, 200.0) + 0.8

        assert delivery.horizon == pytest.approx(200.8)
