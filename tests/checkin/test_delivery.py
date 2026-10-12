"""Unit tests for CheckinDelivery's chat-log emission and clock stamping."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from custom_components.local_openai.checkin.delivery import CheckinDelivery

ENTITY_ID = "conversation.local_openai"


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
        delivery = CheckinDelivery(chat_log, ENTITY_ID)

        await delivery.emit("hi")

        assert chat_log.received == [{"role": "assistant", "content": "hi"}]

    async def test_streams_under_the_turn_entity_id(self) -> None:
        """The delta stream is opened for the entity the delivery wraps."""
        chat_log = _FakeChatLog()
        delivery = CheckinDelivery(chat_log, ENTITY_ID)

        await delivery.emit("hi")

        assert chat_log.agent_ids == [ENTITY_ID]


class TestNow:
    """now reads the injected emission clock emit stamps with."""

    def test_returns_the_injected_clock_reading(self) -> None:
        """A fixed clock reads back exactly what it was injected with."""
        delivery = CheckinDelivery(_FakeChatLog(), ENTITY_ID, clock=lambda: 7.5)

        assert delivery.now == 7.5

    def test_follows_a_mutable_clock(self) -> None:
        """A moving clock is re-read on every access, not cached."""
        readings = iter([1.0, 2.0])
        delivery = CheckinDelivery(
            _FakeChatLog(), ENTITY_ID, clock=lambda: next(readings)
        )

        assert delivery.now == 1.0
        assert delivery.now == 2.0


class TestLastEmission:
    """last_emission reports the injected clock's reading for the most recent emit."""

    def test_is_none_before_any_emit(self) -> None:
        """A fresh delivery has not stamped an emission yet."""
        delivery = CheckinDelivery(_FakeChatLog(), ENTITY_ID, clock=lambda: 42.0)

        assert delivery.last_emission is None

    async def test_stamps_the_clock_after_emit(self) -> None:
        """A successful emit records the clock's value."""
        delivery = CheckinDelivery(_FakeChatLog(), ENTITY_ID, clock=lambda: 42.0)

        await delivery.emit("hi")

        assert delivery.last_emission == 42.0

    async def test_stamps_the_clock_again_on_each_emit(self) -> None:
        """A later emit overwrites the previous stamp."""
        readings = iter([1.0, 2.0])
        delivery = CheckinDelivery(
            _FakeChatLog(), ENTITY_ID, clock=lambda: next(readings)
        )

        await delivery.emit("one")
        await delivery.emit("two")

        assert delivery.last_emission == 2.0
