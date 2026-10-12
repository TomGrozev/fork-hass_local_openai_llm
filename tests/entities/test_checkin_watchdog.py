"""End-to-end tests for the capped watchdog reassurance on the conversation entity.

Drives the real ``_async_handle_chat_log`` with a real ``conversation.ChatLog`` and a scripted
LLM stream whose answer chunk the test releases through an ``asyncio.Event``, so the observable
assistant-delta order - what the user would hear - is decided by the test rather than by
wall-clock time. The watchdog's loop clock is advanced on the HA clock with
``async_fire_time_changed``; the dead-air debounce is decided by the test's injected emission
clock. No real time passes.
"""

from __future__ import annotations

import asyncio
import functools
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
    OPENER_LINES,
    WATCHDOG_LINES,
    render,
)
from custom_components.local_openai.checkin.selector import select_opener_bucket
from custom_components.local_openai.const import (
    CONF_CHECK_INS_ENABLED,
    CONF_CHECKIN_GAP_MS,
    CONF_EXPRESSION_TAGS,
    CONF_GRACE_MS,
    CONF_WATCHDOG_MAX_EXTRAS,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity
from custom_components.local_openai.entity import LocalAiEntity
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
GAP_MS = 1500
GRACE_MS = 500
WATCHDOG_MAX_EXTRAS = 2
#: The injected emission clock's starting reading; the debounce only compares deltas.
CLOCK_START = 100.0
#: A clock reading comfortably past an opener's estimated speech (a 60-66 char opener is ~4-5 s
#: of audio), so a later check-in is not blocked by the opener still playing.
PAST_OPENER_S = 10.0
#: A clock step past a gap plus a reassurance's own short speech, so the next one is due under
#: the speech-horizon rule.
GAP_STEP_S = 2 * GAP_MS / 1000


class _EmissionClock:
    """A settable stand-in for the check-in emission clock, advanced by the test."""

    def __init__(self, now: float = CLOCK_START) -> None:
        """Start the clock at ``now``."""
        self.now = now


def _make_entity(
    hass: HomeAssistant,
    entry: ConfigEntry,
    *,
    enabled: bool = True,
    expression_tags: bool = False,
    gap_ms: int = GAP_MS,
    grace_ms: int = GRACE_MS,
    watchdog_max_extras: int = WATCHDOG_MAX_EXTRAS,
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
                CONF_WATCHDOG_MAX_EXTRAS: watchdog_max_extras,
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


def _gated_answer_turn(
    gate: asyncio.Event,
) -> Callable[[], AsyncIterator[ChatCompletionChunk]]:
    """A turn that holds its answer until ``gate`` is set, then streams it and stops."""

    async def scripted() -> AsyncIterator[ChatCompletionChunk]:
        await gate.wait()
        yield _chunk(ChoiceDelta(role="assistant", content=ANSWER_TEXT))
        yield _chunk(ChoiceDelta(), finish_reason="stop")

    return scripted


def _progress_then_gated_answer_turn(
    gate: asyncio.Event,
) -> Callable[[], AsyncIterator[ChatCompletionChunk]]:
    """A turn that emits a role-only chunk (progress), holds, then streams the answer."""

    async def scripted() -> AsyncIterator[ChatCompletionChunk]:
        yield _chunk(ChoiceDelta(role="assistant"))
        await gate.wait()
        yield _chunk(ChoiceDelta(content=ANSWER_TEXT))
        yield _chunk(ChoiceDelta(), finish_reason="stop")

    return scripted


def _answer_then_gated_finish_turn(
    gate: asyncio.Event,
) -> Callable[[], AsyncIterator[ChatCompletionChunk]]:
    """A turn that streams its answer, holds the turn open, then finishes once ``gate`` is set."""

    async def scripted() -> AsyncIterator[ChatCompletionChunk]:
        yield _chunk(ChoiceDelta(role="assistant", content=ANSWER_TEXT))
        await gate.wait()
        yield _chunk(ChoiceDelta(), finish_reason="stop")

    return scripted


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
        """Never invoked: the chat log calls the stubbed LLM API's tool executor."""
        return llm.ToolResult(data={})


def _llm_api(tool_gate: asyncio.Event) -> Any:
    """A stubbed LLM API whose tool execution is held open until ``tool_gate`` is set."""
    api = MagicMock()
    api.tools = [_StubTool(TOOL_NAME)]
    api.custom_serializer = None

    async def call_tool(_tool_input: llm.ToolInput) -> llm.ToolResult:
        await tool_gate.wait()
        return llm.ToolResult(data={})

    api.async_call_tool = call_tool
    return api


def _tool_turn() -> Callable[[], AsyncIterator[ChatCompletionChunk]]:
    """A turn that immediately calls ``TOOL_NAME`` and stops with ``tool_calls``."""

    async def scripted() -> AsyncIterator[ChatCompletionChunk]:
        yield _chunk(
            ChoiceDelta(
                role="assistant",
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="call_1",
                        type="function",
                        function=ChoiceDeltaToolCallFunction(
                            name=TOOL_NAME, arguments='{"name": "kitchen"}'
                        ),
                    )
                ],
            )
        )
        yield _chunk(ChoiceDelta(), finish_reason="tool_calls")

    return scripted


def _answer_turn() -> Callable[[], AsyncIterator[ChatCompletionChunk]]:
    """A turn that answers directly and stops."""

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
    hass: HomeAssistant, *, llm_api: Any | None = None
) -> conversation.ChatLog:
    """Build a chat log already holding the user's utterance, as the handler expects."""
    log = conversation.ChatLog(hass, "test-conversation-id", llm_api=llm_api)
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


def _rendered_watchdogs(*, expression_tags: bool = False) -> set[str]:
    """Every reassurance text the pool can produce for one tag setting."""
    return {
        render(
            line,
            default_mood=DEFAULT_MOOD["watchdog"],
            expression_tags=expression_tags,
        )
        for line in WATCHDOG_LINES
    }


class TestWatchdogCappedReassurances:
    """A long silent wait draws capped reassurances, then the answer lands."""

    async def test_caps_at_max_extras_then_speaks_the_answer(
        self,
        hass: HomeAssistant,
        mock_config_entry: ConfigEntry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The opener speaks, then two watchdogs, then silence, then the released answer."""
        with freeze_time("2026-10-10 12:00:00"):
            clock = _EmissionClock()
            _inject_clock(monkeypatch, clock)
            entity = _make_entity(hass, mock_config_entry)
            gate = asyncio.Event()
            _script_stream(entity, [_gated_answer_turn(gate)])
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log)
            await _drain()
            # The turn is held open with no deltas: nothing is spoken while the opener is still
            # inside its grace window.
            assert spoken_texts(recorded) == []

            # The grace window closes with no answer, so the opener speaks.
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GRACE_MS + 1)
            )
            await _drain()
            texts = spoken_texts(recorded)
            assert len(texts) == 1
            assert texts[0] in _bucket_openers(select_opener_bucket(USER_TEXT, ()))

            # The opener's own audio must finish before any reassurance; then each gap of dead
            # air draws one reassurance, up to the cap of two.
            clock.now = CLOCK_START + PAST_OPENER_S
            for _ in range(3):
                clock.now += GAP_STEP_S
                async_fire_time_changed(
                    hass, dt_util.utcnow() + timedelta(milliseconds=GAP_MS)
                )
                await _drain()

            texts = spoken_texts(recorded)
            watchdogs = [t for t in texts if t in _rendered_watchdogs()]
            # Exactly two extras, in order, after the opener; the third gap is silent.
            assert len(watchdogs) == WATCHDOG_MAX_EXTRAS
            assert texts[1:3] == watchdogs
            assert len(texts) == 3

            # Releasing the gate lets the real answer through; no further reassurance appears.
            gate.set()
            await _drain()
            await task

            texts = spoken_texts(recorded)
            watchdogs = [t for t in texts if t in _rendered_watchdogs()]
            assert len(texts) == 4
            assert texts[0] in _bucket_openers(select_opener_bucket(USER_TEXT, ()))
            assert texts[1:3] == watchdogs
            assert len(watchdogs) == WATCHDOG_MAX_EXTRAS
            assert texts[-1] == ANSWER_TEXT


class TestWatchdogSilentWhenAnswered:
    """The watchdog only covers dead air; a live turn hears nothing extra."""

    async def test_prompt_answer_emits_no_watchdog(
        self,
        hass: HomeAssistant,
        mock_config_entry: ConfigEntry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An answer that streams promptly ends the turn before any reassurance fires."""
        with freeze_time("2026-10-10 12:00:00"):
            clock = _EmissionClock()
            _inject_clock(monkeypatch, clock)
            entity = _make_entity(hass, mock_config_entry)
            gate = asyncio.Event()
            gate.set()
            _script_stream(entity, [_gated_answer_turn(gate)])
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log)
            await _drain()
            await task

            texts = spoken_texts(recorded)
            assert texts == [ANSWER_TEXT]
            assert not any(t in _rendered_watchdogs() for t in texts)
            assert not any(
                t in _bucket_openers(select_opener_bucket(USER_TEXT, ())) for t in texts
            )

    async def test_delta_progress_suppresses_the_watchdog(
        self,
        hass: HomeAssistant,
        mock_config_entry: ConfigEntry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A chunk arriving during the interval marks progress, so the watchdog stays quiet."""
        with freeze_time("2026-10-10 12:00:00"):
            clock = _EmissionClock()
            _inject_clock(monkeypatch, clock)
            entity = _make_entity(hass, mock_config_entry)
            gate = asyncio.Event()
            _script_stream(entity, [_progress_then_gated_answer_turn(gate)])
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log)
            await _drain()
            # The role-only chunk is progress, not visible content, so the opener still speaks.
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GRACE_MS + 1)
            )
            await _drain()
            # A gap elapses, but the model spoke inside it: no reassurance is due.
            clock.now += GAP_MS / 1000
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GAP_MS)
            )
            await _drain()

            texts = spoken_texts(recorded)
            assert len(texts) == 1
            assert texts[0] in _bucket_openers(select_opener_bucket(USER_TEXT, ()))
            assert not any(t in _rendered_watchdogs() for t in texts)

            gate.set()
            await _drain()
            await task

            texts = spoken_texts(recorded)
            assert texts[-1] == ANSWER_TEXT
            assert not any(t in _rendered_watchdogs() for t in texts)


class TestWatchdogStopsOnceAnswerStarts:
    """A stall after the answer has begun must not splice in a reassurance."""

    async def test_stalled_answer_draws_no_reassurance(
        self,
        hass: HomeAssistant,
        mock_config_entry: ConfigEntry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The answer's first visible delta ends the watchdog; later gaps stay silent."""
        with freeze_time("2026-10-10 12:00:00"):
            clock = _EmissionClock()
            _inject_clock(monkeypatch, clock)
            entity = _make_entity(hass, mock_config_entry)
            gate = asyncio.Event()
            _script_stream(entity, [_answer_then_gated_finish_turn(gate)])
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log)
            await _drain()
            # The answer is already streaming - its first visible delta landed - then it stalls.
            assert ANSWER_TEXT in spoken_texts(recorded)

            # Several gaps of silence elapse while the answer is stalled: no reassurance, and
            # the opener was suppressed too, because the answer beat its grace window.
            for _ in range(WATCHDOG_MAX_EXTRAS + 1):
                clock.now += GAP_MS / 1000
                async_fire_time_changed(
                    hass, dt_util.utcnow() + timedelta(milliseconds=GAP_MS)
                )
                await _drain()

            texts = spoken_texts(recorded)
            assert texts == [ANSWER_TEXT]
            assert not any(t in _rendered_watchdogs() for t in texts)
            assert not any(
                t in _bucket_openers(select_opener_bucket(USER_TEXT, ())) for t in texts
            )

            gate.set()
            await _drain()
            await task

            texts = spoken_texts(recorded)
            assert texts == [ANSWER_TEXT]
            assert not any(t in _rendered_watchdogs() for t in texts)


class TestWatchdogNeverJumpsTheOpener:
    """The opener is always spoken; a reassurance must wait for it to resolve."""

    async def test_no_watchdog_before_opener_resolves(
        self,
        hass: HomeAssistant,
        mock_config_entry: ConfigEntry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A gap that elapses while the opener still waits stays silent."""
        with freeze_time("2026-10-10 12:00:00"):
            clock = _EmissionClock()
            _inject_clock(monkeypatch, clock)
            # Gap shorter than grace, so the watchdog wakes while the opener is still pending.
            entity = _make_entity(
                hass, mock_config_entry, gap_ms=200, grace_ms=GRACE_MS * 10
            )
            gate = asyncio.Event()
            _script_stream(entity, [_gated_answer_turn(gate)])
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log)
            await _drain()
            # Many watchdog gaps elapse, but the opener has not resolved: stay silent.
            clock.now += GAP_MS / 1000
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GRACE_MS)
            )
            await _drain()
            assert spoken_texts(recorded) == []

            # Only when the grace window closes does the opener speak.
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GRACE_MS * 10 + 1)
            )
            await _drain()
            texts = spoken_texts(recorded)
            assert len(texts) == 1
            assert texts[0] in _bucket_openers(select_opener_bucket(USER_TEXT, ()))
            assert not any(t in _rendered_watchdogs() for t in texts)

            gate.set()
            await _drain()
            await task


class TestToolCallAdjacency:
    """A check-in that interrupts a pending tool call folds into the tool-calls message."""

    async def test_checkin_between_tool_call_and_result_is_folded(self) -> None:
        """One tool call: the intervening check-in folds in and the tool message follows."""
        messages = [
            {"role": "user", "content": USER_TEXT},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
            {"role": "assistant", "content": "Just a moment more..."},
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        ]

        folded = LocalAiEntity._keep_tool_calls_adjacent(messages)

        assert [m["role"] for m in folded] == ["user", "assistant", "tool"]
        assert folded[1]["content"] == "Just a moment more..."
        assert folded[1]["tool_calls"] == [{"id": "call_1"}]

    async def test_checkin_after_first_of_two_results_is_folded(self) -> None:
        """Two tool calls: a check-in after the first result still folds into the call message."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "a"}, {"id": "b"}],
            },
            {"role": "tool", "tool_call_id": "a", "content": "{}"},
            {"role": "assistant", "content": "Still working..."},
            {"role": "tool", "tool_call_id": "b", "content": "{}"},
        ]

        folded = LocalAiEntity._keep_tool_calls_adjacent(messages)

        assert [m["role"] for m in folded] == ["assistant", "tool", "tool"]
        assert folded[0]["content"] == "Still working..."
        assert [m["tool_call_id"] for m in folded[1:]] == ["a", "b"]

    async def test_no_intervention_leaves_the_list_unchanged(self) -> None:
        """A well-formed tool round is returned in the same order."""
        messages = [
            {"role": "user", "content": USER_TEXT},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
            {"role": "assistant", "content": ANSWER_TEXT},
        ]

        assert LocalAiEntity._keep_tool_calls_adjacent(messages) == messages

    async def test_assistant_after_tool_results_is_left_alone(self) -> None:
        """A content-only assistant message once the calls are answered stays its own message."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
            {"role": "assistant", "content": ANSWER_TEXT},
        ]

        folded = LocalAiEntity._keep_tool_calls_adjacent(messages)

        assert [m["role"] for m in folded] == ["assistant", "tool", "assistant"]
        assert folded[2]["content"] == ANSWER_TEXT


class TestWatchdogDuringToolWait:
    """A reassurance spoken while a tool runs never breaks tool-call adjacency."""

    async def test_checkin_during_tool_wait_keeps_tool_call_adjacent(
        self,
        hass: HomeAssistant,
        mock_config_entry: ConfigEntry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The next turn keeps the tool result right after the tool-calls message."""
        with freeze_time("2026-10-10 12:00:00"):
            clock = _EmissionClock()
            _inject_clock(monkeypatch, clock)
            entity = _make_entity(hass, mock_config_entry)
            tool_gate = asyncio.Event()
            api = _llm_api(tool_gate)
            chat_log = _chat_log(hass, llm_api=api)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)
            # Turn one: tool call, then the answer; turn two: a plain answer.
            _script_stream(entity, [_tool_turn(), _answer_turn(), _answer_turn()])

            task = _start_turn(entity, chat_log)
            await _drain()
            # The grace window closes with the tool held, so the opener speaks.
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GRACE_MS + 1)
            )
            await _drain()
            # The tool-call chunk landed in the first interval, marking progress, so the first
            # gap is silent; the second gap is dead air and draws the reassurance. The opener's
            # own audio has already finished, so it does not block the reassurance.
            clock.now = CLOCK_START + PAST_OPENER_S
            clock.now += GAP_MS / 1000
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GAP_MS)
            )
            await _drain()
            clock.now += GAP_MS / 1000
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GAP_MS)
            )
            await _drain()

            reassurance = next(
                (t for t in spoken_texts(recorded) if t in _rendered_watchdogs()),
                None,
            )
            assert reassurance is not None

            # Releasing the tool lets the result and the answer land, closing turn one.
            tool_gate.set()
            await _drain()
            await task

            # Turn two rebuilds the wire messages from the chat log; the folded reassurance must
            # not sit between the tool-calls message and its tool result.
            second_turn = _start_turn(entity, chat_log)
            await _drain()
            await second_turn

            create = entity.entry.runtime_data.chat.completions.create
            messages = create.call_args_list[2].kwargs["messages"]
            roles = [m["role"] for m in messages]
            tool_calls_index = roles.index("assistant")
            assert messages[tool_calls_index]["tool_calls"]
            assert reassurance in messages[tool_calls_index]["content"]
            assert messages[tool_calls_index + 1]["role"] == "tool"
