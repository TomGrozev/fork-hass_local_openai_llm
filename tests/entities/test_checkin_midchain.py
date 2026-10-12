"""End-to-end tests for mid-chain check-ins on the conversation entity.

Drives the real ``_async_handle_chat_log`` with a real ``conversation.ChatLog`` and a scripted
LLM stream that yields one tool-call turn per tool round-trip and an answer turn last. The
check-in emission clock is injected, so the dead-air debounce is decided by the test's own
clock reading rather than by wall-clock time; the opener's grace deadline is advanced on the HA
clock with ``async_fire_time_changed`` exactly as the opener tests do.
"""

from __future__ import annotations

import asyncio
import functools
import random
from collections.abc import AsyncIterator, Callable
from datetime import timedelta
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from freezegun import freeze_time

from tests.conftest import spoken_texts

from custom_components.local_openai import entity as entity_module
from custom_components.local_openai.checkin.delivery import CheckinDelivery
from custom_components.local_openai.checkin.pool import (
    DEFAULT_MOOD,
    MIDCHAIN_LINES,
    OPENER_LINES,
    LinePool,
    render,
)
from custom_components.local_openai.checkin.selector import (
    bucket_for_tool,
    select_opener_bucket,
)
from custom_components.local_openai.const import (
    CONF_CHECK_INS_ENABLED,
    CONF_CHECKIN_GAP_MS,
    CONF_EXPRESSION_TAGS,
    CONF_GRACE_MS,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_MODEL
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm
from homeassistant.util import dt as dt_util
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from pytest_homeassistant_custom_component.common import async_fire_time_changed

USER_TEXT = "turn on the kitchen light"
ANSWER_TEXT = "Done."
TOOL_NAME = "HassTurnOn"
TOOL_BUCKET = bucket_for_tool(TOOL_NAME)
LIST_TOOL_NAME = "HassListAddItem"
LIST_TOOL_BUCKET = bucket_for_tool(LIST_TOOL_NAME)
GAP_MS = 1500
GRACE_MS = 500
#: The injected emission clock's starting reading; the debounce only compares deltas.
CLOCK_START = 100.0


class _EmissionClock:
    """A settable stand-in for the check-in emission clock, advanced by the test."""

    def __init__(self, now: float = CLOCK_START) -> None:
        """Start the clock at ``now``."""
        self.now = now


class _StubTool(llm.Tool):
    """The HA tool the scripted stream pretends to call."""

    def __init__(self, name: str) -> None:
        """Name the stub so the entity can format its schema."""
        self.name = name
        self.description = "Turn a device on."

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> llm.ToolResult:
        """Never invoked: the entity's stubbed LLM API supplies the tool result."""
        return llm.ToolResult(data={})


def _llm_api(tool_name: str = TOOL_NAME) -> Any:
    """A stubbed LLM API exposing the tool the scripted stream calls."""
    api = MagicMock()
    api.tools = [_StubTool(tool_name)]
    api.custom_serializer = None
    api.async_call_tool = AsyncMock(return_value=llm.ToolResult(data={}))
    return api


def _make_entity(
    hass: HomeAssistant,
    entry: ConfigEntry,
    *,
    enabled: bool = True,
    expression_tags: bool = False,
    gap_ms: int = GAP_MS,
    grace_ms: int = GRACE_MS,
) -> LocalAiConversationEntity:
    """Build a conversation entity whose subentry carries the given check-in options."""
    subentry = ConfigSubentry(
        subentry_id="test_conversation_subentry_id",
        subentry_type="conversation",
        title="Conversation Agent",
        data=MappingProxyType(
            {
                CONF_MODEL: "test-model",
                CONF_CHECK_INS_ENABLED: enabled,
                CONF_GRACE_MS: grace_ms,
                CONF_CHECKIN_GAP_MS: gap_ms,
                CONF_EXPRESSION_TAGS: expression_tags,
            }
        ),
        unique_id=None,
    )
    entity = LocalAiConversationEntity(entry, subentry)
    entity.hass = hass
    entity.entity_id = "conversation.test"
    return entity


def _inject_clock(monkeypatch: pytest.MonkeyPatch, clock: _EmissionClock) -> None:
    """Bind the entity's per-turn delivery to the test's settable emission clock."""
    monkeypatch.setattr(
        entity_module,
        "CheckinDelivery",
        functools.partial(CheckinDelivery, clock=lambda: clock.now),
    )


def _chunk(delta: ChoiceDelta, finish_reason: str | None = None) -> ChatCompletionChunk:
    """One streamed chunk carrying ``delta`` and an optional finish reason."""
    return ChatCompletionChunk(
        id="test",
        created=0,
        model="test",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=delta, finish_reason=finish_reason)],
    )


def _tool_turn(
    *,
    name: str = TOOL_NAME,
    arguments: str = '{"name": "kitchen"}',
    gate: asyncio.Event | None = None,
) -> Callable[[], AsyncIterator[ChatCompletionChunk]]:
    """A turn that calls ``name``; ``gate`` holds it open until the test releases it."""

    async def scripted() -> AsyncIterator[ChatCompletionChunk]:
        if gate is not None:
            await gate.wait()
        yield _chunk(
            ChoiceDelta(
                role="assistant",
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="call_1",
                        type="function",
                        function=ChoiceDeltaToolCallFunction(
                            name=name, arguments=arguments
                        ),
                    )
                ],
            )
        )
        yield _chunk(ChoiceDelta(), finish_reason="tool_calls")

    return scripted


def _answer_turn() -> Callable[[], AsyncIterator[ChatCompletionChunk]]:
    """A turn that answers directly, closing the loop."""

    async def scripted() -> AsyncIterator[ChatCompletionChunk]:
        yield _chunk(ChoiceDelta(role="assistant", content=ANSWER_TEXT))
        yield _chunk(ChoiceDelta(), finish_reason="stop")

    return scripted


def _script_stream(
    entity: LocalAiConversationEntity,
    turns: list[Callable[[], AsyncIterator[ChatCompletionChunk]]],
) -> None:
    """Yield one scripted turn per completion call, in order."""
    entity.entry.runtime_data.chat.completions.create = AsyncMock(
        side_effect=[turn() for turn in turns]
    )


def _chat_log(
    hass: HomeAssistant, *, tool_name: str = TOOL_NAME
) -> conversation.ChatLog:
    """Build a chat log already holding the user's utterance, as the handler expects."""
    log = conversation.ChatLog(
        hass, "test-conversation-id", llm_api=_llm_api(tool_name)
    )
    log.async_add_user_content(conversation.UserContent(content=USER_TEXT))
    return log


def _start_turn(
    entity: LocalAiConversationEntity,
    chat_log: conversation.ChatLog,
) -> asyncio.Task[None]:
    """Start ``_async_handle_chat_log`` as a task and return it."""
    user_input = conversation.ConversationInput(
        text=USER_TEXT,
        context=Context(),
        conversation_id=chat_log.conversation_id,
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id=entity.entity_id,
    )
    return entity.hass.async_create_task(
        entity._async_handle_chat_log(chat_log, user_input=user_input)
    )


async def _drain(iterations: int = 25) -> None:
    """Yield to the loop repeatedly so every waiting task gets to run."""
    for _ in range(iterations):
        await asyncio.sleep(0)


def _bucket_midchains(
    bucket: str,
    *,
    slots: dict[str, str] | None = None,
    expression_tags: bool = False,
) -> set[str]:
    """Every mid-chain text the pool can produce for one tool bucket."""
    return {
        render(
            line,
            slots=slots,
            default_mood=DEFAULT_MOOD[f"midchain:{bucket}"],
            expression_tags=expression_tags,
        )
        for line in MIDCHAIN_LINES[bucket]
    }


def _bucket_openers(bucket: str, *, expression_tags: bool = False) -> set[str]:
    """Every opener text the pool can produce for one bucket."""
    return {
        render(
            line,
            default_mood=DEFAULT_MOOD[f"opener:{bucket}"],
            expression_tags=expression_tags,
        )
        for line in OPENER_LINES[bucket]
    }


class TestMidchainBeforeToolStep:
    """A tool round-trip lets a mid-chain fragment out ahead of the next completion."""

    async def test_opener_then_midchain_then_answer(
        self,
        hass: HomeAssistant,
        mock_config_entry: ConfigEntry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A gated tool turn lets the opener speak, then the mid-chain, then the answer."""
        with freeze_time("2026-10-10 12:00:00"):
            clock = _EmissionClock()
            _inject_clock(monkeypatch, clock)
            entity = _make_entity(hass, mock_config_entry)
            gate = asyncio.Event()
            _script_stream(entity, [_tool_turn(gate=gate), _answer_turn()])
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log)
            await _drain()
            # The tool turn is still in flight, so the grace window closes and the opener speaks.
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GRACE_MS + 1)
            )
            await _drain()
            # The mid-chain may only speak once the gap since the opener has elapsed.
            clock.now = CLOCK_START + GAP_MS / 1000
            gate.set()
            await _drain()
            await task

            texts = spoken_texts(recorded)
            assert len(texts) == 3
            assert texts[0] in _bucket_openers(select_opener_bucket(USER_TEXT, ()))
            midchain_text = texts[1]
            assert midchain_text in _bucket_midchains(TOOL_BUCKET)
            assert texts[2] == ANSWER_TEXT
            assert any(
                isinstance(content, conversation.AssistantContent)
                and content.content == midchain_text
                for content in chat_log.content
            )


class TestMidchainYieldsToTheOpener:
    """The opener is always spoken; the mid-chain only a maybe, so it never jumps the queue."""

    async def test_midchain_is_skipped_while_the_opener_is_pending(
        self,
        hass: HomeAssistant,
        mock_config_entry: ConfigEntry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A fast tool-only turn skips its mid-chain: the opener never resolved first."""
        with freeze_time("2026-10-10 12:00:00"):
            clock = _EmissionClock()
            _inject_clock(monkeypatch, clock)
            entity = _make_entity(hass, mock_config_entry)
            _script_stream(entity, [_tool_turn(), _answer_turn()])
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log)
            await _drain()
            # No clock advance: the opener is still inside its grace window at the only
            # mid-chain decision, so the mid-chain is skipped and the opener is dropped at
            # turn end -- neither is ever written.
            await task

            assert spoken_texts(recorded) == [ANSWER_TEXT]


class TestMidchainDebounced:
    """A mid-chain landing inside the dead-air gap is skipped outright, never queued."""

    async def test_midchain_within_gap_of_the_opener_is_suppressed(
        self,
        hass: HomeAssistant,
        mock_config_entry: ConfigEntry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The opener's emission shares the clock, so a step inside the gap stays silent."""
        with freeze_time("2026-10-10 12:00:00"):
            clock = _EmissionClock()
            _inject_clock(monkeypatch, clock)
            entity = _make_entity(hass, mock_config_entry)
            gate = asyncio.Event()
            _script_stream(entity, [_tool_turn(gate=gate), _answer_turn()])
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log)
            await _drain()
            # The tool turn is still in flight, so the grace window closes and the opener
            # speaks through the shared delivery, stamping the emission clock at CLOCK_START.
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GRACE_MS + 1)
            )
            await _drain()
            # The next step lands inside the gap.
            clock.now = CLOCK_START + (GAP_MS / 2) / 1000  # half the gap, in seconds
            gate.set()
            await _drain()
            await task

            texts = spoken_texts(recorded)
            # The opener is exempt from the debounce and is heard; only the mid-chain is dropped.
            assert ANSWER_TEXT in texts
            assert not any(text in _bucket_midchains(TOOL_BUCKET) for text in texts)
            opener_bucket = select_opener_bucket(USER_TEXT, ())
            assert texts[0] in _bucket_openers(opener_bucket)


class TestBackToBackStepsCollapse:
    """Two tool steps inside one gap collapse to a single mid-chain fragment."""

    async def test_second_step_within_gap_collapses_to_one_midchain(
        self,
        hass: HomeAssistant,
        mock_config_entry: ConfigEntry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two tool round-trips in quick succession produce exactly one mid-chain delta."""
        with freeze_time("2026-10-10 12:00:00"):
            clock = _EmissionClock()
            _inject_clock(monkeypatch, clock)
            entity = _make_entity(hass, mock_config_entry)
            gate = asyncio.Event()
            _script_stream(
                entity, [_tool_turn(gate=gate), _tool_turn(), _answer_turn()]
            )
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log)
            await _drain()
            # The first tool turn is gated, so the grace window closes and the opener speaks.
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GRACE_MS + 1)
            )
            await _drain()
            # The first mid-chain may speak once the gap since the opener has elapsed.
            clock.now = CLOCK_START + GAP_MS / 1000
            gate.set()
            await _drain()
            await task

            texts = spoken_texts(recorded)
            midchains = [
                text for text in texts if text in _bucket_midchains(TOOL_BUCKET)
            ]
            # The opener is ``texts[0]``; the second, fast tool step lands inside the same gap
            # and is skipped, so exactly one mid-chain follows it.
            assert len(midchains) == 1
            assert texts[1] in _bucket_midchains(TOOL_BUCKET)
            assert texts[-1] == ANSWER_TEXT


class TestMidchainFiresAcrossGap:
    """A second mid-chain fires once the gap since the first has elapsed."""

    async def test_second_step_after_the_gap_fires_again(
        self,
        hass: HomeAssistant,
        mock_config_entry: ConfigEntry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two tool round-trips separated by the gap each speak a mid-chain fragment."""
        with freeze_time("2026-10-10 12:00:00"):
            clock = _EmissionClock()
            _inject_clock(monkeypatch, clock)
            entity = _make_entity(hass, mock_config_entry)
            g1 = asyncio.Event()
            g2 = asyncio.Event()
            _script_stream(
                entity, [_tool_turn(gate=g1), _tool_turn(gate=g2), _answer_turn()]
            )
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log)
            await _drain()
            # The first tool turn is gated, so the grace window closes and the opener speaks.
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GRACE_MS + 1)
            )
            await _drain()
            # Release the first step past the gap since the opener: its mid-chain fires and
            # stamps the shared emission clock.
            clock.now = CLOCK_START + GAP_MS / 1000
            g1.set()
            await _drain()
            # The second step waits out a further gap, so its mid-chain is due too.
            clock.now = CLOCK_START + 2 * GAP_MS / 1000
            g2.set()
            await _drain()
            await task

            texts = spoken_texts(recorded)
            midchains = [
                text for text in texts if text in _bucket_midchains(TOOL_BUCKET)
            ]
            assert len(midchains) == 2
            assert texts[-1] == ANSWER_TEXT


class TestMidchainSlotsFilledFromToolArgs:
    """A mid-chain fragment fills its blank from the tool call's arguments."""

    async def test_list_tool_fills_the_item_slot(
        self,
        hass: HomeAssistant,
        mock_config_entry: ConfigEntry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A ``HassListAddItem`` call renders ``Adding milk...`` from its ``item`` argument."""
        with freeze_time("2026-10-10 12:00:00"):
            clock = _EmissionClock()
            _inject_clock(monkeypatch, clock)
            entity = _make_entity(hass, mock_config_entry)
            # Seed the pool so the list bucket draws its ``{item}`` line first.
            entity._checkin_pool = LinePool(rng=random.Random(1))
            gate = asyncio.Event()
            _script_stream(
                entity,
                [
                    _tool_turn(
                        name=LIST_TOOL_NAME,
                        arguments='{"item": "milk"}',
                        gate=gate,
                    ),
                    _answer_turn(),
                ],
            )
            chat_log = _chat_log(hass, tool_name=LIST_TOOL_NAME)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log)
            await _drain()
            # The gated tool turn lets the grace window close and the opener speak.
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GRACE_MS + 1)
            )
            await _drain()
            # The mid-chain may speak once the gap since the opener has elapsed.
            clock.now = CLOCK_START + GAP_MS / 1000
            gate.set()
            await _drain()
            await task

            texts = spoken_texts(recorded)
            midchain_text, answer = texts[1], texts[2]
            assert answer == ANSWER_TEXT
            assert midchain_text in _bucket_midchains(
                LIST_TOOL_BUCKET, slots={"item": "milk"}
            )
            # The blank is filled, not the slot-free fallback.
            assert midchain_text == "Adding milk..."
