"""End-to-end tests for the pipeline-language guard on the check-in feature.

The curated pool and keyword tables are English-only, so a non-English pipeline must degrade to
the plain streamed answer: no opener, no mid-chain, no watchdog. These tests drive the real
``_async_handle_chat_log`` with a real ``conversation.ChatLog`` and a scripted LLM stream whose
first content chunk the test releases through an ``asyncio.Event``, mirroring the opener tests so
the observable assistant-delta order is decided by the test, not by wall-clock time.
"""

from __future__ import annotations

import asyncio
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
    OPENER_LINES,
    render,
)
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
    *,
    answer_text: str = ANSWER_TEXT,
) -> None:
    """Script the LLM stream: hold the first content chunk until ``release`` is set."""

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
                    delta=ChoiceDelta(role="assistant", content=answer_text),
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
    *,
    language: str | None,
) -> asyncio.Task[None]:
    """Start ``_async_handle_chat_log`` for ``language`` and return it."""
    user_input = conversation.ConversationInput(
        text=USER_TEXT,
        context=Context(),
        conversation_id=chat_log.conversation_id,
        device_id=None,
        satellite_id=None,
        language=language,
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
        render(
            line,
            default_mood=DEFAULT_MOOD[f"opener:{bucket}"],
            expression_tags=expression_tags,
        )
        for bucket in OPENER_BUCKETS
        for cell in OPENER_LINES[bucket].values()
        for line in cell
    }


class TestNonEnglishSuppression:
    """A non-English pipeline hears only the streamed answer, however slow the turn."""

    async def test_slow_non_english_turn_emits_no_check_in(
        self, hass: HomeAssistant, mock_config_entry: ConfigEntry
    ) -> None:
        """Even past the grace deadline, no opener or watchdog delta is written."""
        with freeze_time("2026-10-10 12:00:00"):
            entity = _make_entity(hass, mock_config_entry, enabled=True)
            release = asyncio.Event()
            finish = asyncio.Event()
            _stub_stream(entity, release, finish)
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log, language="fr")
            await _drain()
            # A live check-in would fire the opener here; suppression must keep it silent.
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GRACE_MS + 1)
            )
            await _drain()
            release.set()
            await _drain()
            finish.set()
            await task

            texts = spoken_texts(recorded)
            assert texts == [ANSWER_TEXT]
            assert not any(
                text in _rendered_openers(expression_tags=False) for text in texts
            )

    async def test_non_english_keeps_expression_tags_in_answer(
        self, hass: HomeAssistant, mock_config_entry: ConfigEntry
    ) -> None:
        """The language guard never touches the answer's own expression tags (ticket 08 path)."""
        tagged = "[excited] Sure, turning the light on now."
        with freeze_time("2026-10-10 12:00:00"):
            entity = _make_entity(
                hass, mock_config_entry, enabled=True, expression_tags=True
            )
            release = asyncio.Event()
            finish = asyncio.Event()
            _stub_stream(entity, release, finish, answer_text=tagged)
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log, language="de")
            await _drain()
            release.set()
            await _drain()
            finish.set()
            await task

            assert spoken_texts(recorded) == [tagged]


class TestEnglishStillSpeaks:
    """The guard leaves English pipelines untouched."""

    async def test_slow_english_turn_still_speaks_opener(
        self, hass: HomeAssistant, mock_config_entry: ConfigEntry
    ) -> None:
        """An English turn past the grace window hears the opener then the answer."""
        with freeze_time("2026-10-10 12:00:00"):
            entity = _make_entity(hass, mock_config_entry, enabled=True)
            release = asyncio.Event()
            finish = asyncio.Event()
            _stub_stream(entity, release, finish)
            chat_log = _chat_log(hass)
            recorded: list[dict[str, Any]] = []
            chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

            task = _start_turn(entity, chat_log, language="en-US")
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
            assert opener in _rendered_openers(expression_tags=False)
