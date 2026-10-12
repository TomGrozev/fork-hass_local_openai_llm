"""End-to-end tests for the gated check-in opener on the conversation entity.

Drives the real ``_async_handle_chat_log`` with a real ``conversation.ChatLog`` and a scripted
LLM stream whose first content chunk the test releases through an ``asyncio.Event``, so the
observable assistant-delta order - what the user would hear - is decided by the test rather than
by wall-clock time. The grace deadline is advanced on the injected clock with
``async_fire_time_changed``.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from datetime import timedelta
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock

from freezegun import freeze_time

from tests.conftest import spoken_texts

from custom_components.local_openai.checkin.pool import (
    DEFAULT_MOOD,
    OPENER_BUCKETS,
    OPENER_LENGTH_MAX,
    OPENER_LINES,
    render,
)
from custom_components.local_openai.checkin.selector import select_opener_bucket
from custom_components.local_openai.const import (
    CONF_CHECK_INS_ENABLED,
    CONF_EXPRESSION_TAGS,
    CONF_GRACE_MS,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_MODEL
from homeassistant.core import Context, HomeAssistant
from homeassistant.util import dt as dt_util
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from pytest_homeassistant_custom_component.common import async_fire_time_changed

USER_TEXT = "turn on the light"
ANSWER_TEXT = "Sure, turning the light on now."
GRACE_MS = 500
#: Slowest streaming TTS engine the opener is sized against, characters per second.
SLOWEST_TTS_CHARS_PER_S = 15.0
#: The TTFT the grace window is sized against, seconds.
DESIGN_TTFT_S = 4.0
#: Worst overshoot a spoken opener may add to a direct-answer turn, seconds.
MAX_OPENER_OVERSHOOT_S = 1.0
#: The expression tag a spoken opener may carry (``"[calm] ..."``); stripped before measuring.
_MOOD_TAG = re.compile(r"^\[[a-z]+\]\s*")


def _make_entity(
    hass: HomeAssistant,
    entry: ConfigEntry,
    *,
    enabled: bool,
    expression_tags: bool = False,
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
                CONF_EXPRESSION_TAGS: expression_tags,
            }
        ),
        unique_id=None,
    )
    entity = LocalAiConversationEntity(entry, subentry)
    entity.hass = hass
    entity.entity_id = "conversation.test"
    return entity


def _stub_stream(
    entity: LocalAiConversationEntity,
    release: asyncio.Event,
    finish: asyncio.Event,
) -> None:
    """Script the LLM stream: hold the first content chunk until ``release`` is set.

    A second ``finish`` gate holds the closing chunk, so a test can let the answer start and
    still keep the turn open while the gate observes that first content delta.
    """

    async def scripted() -> AsyncIterator[ChatCompletionChunk]:
        await release.wait()
        yield ChatCompletionChunk(
            id="test",
            created=0,
            model="test",
            object="chat.completion.chunk",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content=ANSWER_TEXT),
                )
            ],
        )
        await finish.wait()
        yield ChatCompletionChunk(
            id="test",
            created=0,
            model="test",
            object="chat.completion.chunk",
            choices=[Choice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
        )

    entity.entry.runtime_data.chat.completions.create = AsyncMock(
        return_value=scripted()
    )


def _chat_log(hass: HomeAssistant) -> conversation.ChatLog:
    """Build a chat log already holding the user's utterance, as the handler expects."""
    log = conversation.ChatLog(hass, "test-conversation-id")
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


def _rendered_openers(*, expression_tags: bool) -> set[str]:
    """Every opener text the pool can produce, across all buckets, for one tag setting."""
    return {
        opener
        for bucket in OPENER_BUCKETS
        for opener in _bucket_openers(bucket, expression_tags=expression_tags)
    }


def _bucket_openers(bucket: str, *, expression_tags: bool) -> set[str]:
    """Every opener text the pool can produce for one bucket."""
    return {
        render(
            line,
            default_mood=DEFAULT_MOOD[f"opener:{bucket}"],
            expression_tags=expression_tags,
        )
        for line in OPENER_LINES[bucket]
    }


class TestOpenerGateTiming:
    """The delta order the user hears depends on whether the answer beats the grace window."""

    async def test_slow_turn_speaks_opener_before_answer(
        self, hass: HomeAssistant, mock_config_entry: ConfigEntry
    ) -> None:
        """A first token that misses the grace window lets the opener out first."""
        with freeze_time("2026-10-10 12:00:00"):
            entity = _make_entity(hass, mock_config_entry, enabled=True)
            release = asyncio.Event()
            finish = asyncio.Event()
            _stub_stream(entity, release, finish)
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log)
            await _drain()
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GRACE_MS + 1)
            )
            await _drain()
            release.set()
            await _drain()
            finish.set()
            await task

            texts = spoken_texts(recorded)
            assert len(texts) == 2
            opener, answer = texts
            assert answer == ANSWER_TEXT

            bucket = select_opener_bucket(USER_TEXT, ())
            assert opener in _bucket_openers(bucket, expression_tags=False)
            assert any(
                isinstance(content, conversation.AssistantContent)
                and content.content == opener
                for content in chat_log.content
            )

    async def test_fast_turn_suppresses_opener(
        self, hass: HomeAssistant, mock_config_entry: ConfigEntry
    ) -> None:
        """An answer that arrives inside the grace window is heard alone."""
        with freeze_time("2026-10-10 12:00:00"):
            entity = _make_entity(hass, mock_config_entry, enabled=True)
            release = asyncio.Event()
            finish = asyncio.Event()
            _stub_stream(entity, release, finish)
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log)
            await _drain()
            release.set()
            # The answer's first visible content reaches the transform, which must set the gate's
            # event. Only now fire the grace deadline: the gate has to see the answer already
            # started and stay silent. This fails if the event is never set -- the opener would
            # slip out mid-answer.
            await _drain()
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GRACE_MS + 1)
            )
            await _drain()
            finish.set()
            await task

            texts = spoken_texts(recorded)
            assert texts == [ANSWER_TEXT]
            assert not any(
                text in _rendered_openers(expression_tags=False) for text in texts
            )
            assert not any(
                isinstance(content, conversation.AssistantContent)
                and content.content in _rendered_openers(expression_tags=False)
                for content in chat_log.content
            )

    async def test_master_toggle_off_never_speaks(
        self, hass: HomeAssistant, mock_config_entry: ConfigEntry
    ) -> None:
        """With ``check_ins_enabled`` off the turn only ever hears the answer."""
        with freeze_time("2026-10-10 12:00:00"):
            entity = _make_entity(hass, mock_config_entry, enabled=False)
            release = asyncio.Event()
            finish = asyncio.Event()
            release.set()
            finish.set()
            _stub_stream(entity, release, finish)
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log)
            await _drain()
            await task

            texts = spoken_texts(recorded)
            assert texts == [ANSWER_TEXT]
            assert not any(
                text in _rendered_openers(expression_tags=False) for text in texts
            )


class TestOpenerOvershootBound:
    """A spoken opener must not overshoot the pinned latency budget."""

    async def test_direct_answer_opener_fits_length_and_overshoot_bounds(
        self, hass: HomeAssistant, mock_config_entry: ConfigEntry
    ) -> None:
        """The emitted no-tool opener is 66 chars or fewer and stays within 1s overshoot."""
        with freeze_time("2026-10-10 12:00:00"):
            entity = _make_entity(
                hass, mock_config_entry, enabled=True, expression_tags=True
            )
            release = asyncio.Event()
            finish = asyncio.Event()
            _stub_stream(entity, release, finish)
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log)
            await _drain()
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GRACE_MS + 1)
            )
            await _drain()
            release.set()
            await _drain()
            finish.set()
            await task

            texts = spoken_texts(recorded)
            assert len(texts) == 2
            opener = texts[0]
            # The tag is spoken-stripped, so the length bound applies to the text it carried.
            spoken = _MOOD_TAG.sub("", opener)
            assert spoken != opener
            assert len(spoken) <= OPENER_LENGTH_MAX
            assert (
                len(spoken) / SLOWEST_TTS_CHARS_PER_S + GRACE_MS / 1000 - DESIGN_TTFT_S
                <= MAX_OPENER_OVERSHOOT_S
            )

            # Worst case over the whole pool, measuring the tag-stripped spoken text.
            worst = max(
                len(_MOOD_TAG.sub("", opener))
                for bucket in OPENER_BUCKETS
                for opener in _bucket_openers(bucket, expression_tags=True)
            )
            assert worst <= OPENER_LENGTH_MAX
            assert (
                worst / SLOWEST_TTS_CHARS_PER_S + GRACE_MS / 1000 - DESIGN_TTFT_S
                <= MAX_OPENER_OVERSHOOT_S
            )
