"""End-to-end tests for the expression-tag strip/pass on the streamed answer.

Drives the real ``_async_handle_chat_log`` with a real ``conversation.ChatLog`` and a scripted
LLM stream, so the content deltas the user hears are exactly the ones the transform emitted.
The strip is delivery-side: it governs the raw LLM answer text independently of
``check_ins_enabled``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock

import pytest

from custom_components.local_openai.const import (
    CONF_CHECK_INS_ENABLED,
    CONF_EXPRESSION_TAGS,
)
from custom_components.local_openai.conversation import LocalAiConversationEntity
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_MODEL
from homeassistant.core import Context, HomeAssistant
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

USER_TEXT = "turn on the light"


def _make_entity(
    hass: HomeAssistant,
    entry: ConfigEntry,
    *,
    enabled: bool,
    expression_tags: bool,
) -> LocalAiConversationEntity:
    """Build a conversation entity whose subentry carries the given options."""
    subentry = ConfigSubentry(
        subentry_id="test_conversation_subentry_id",
        subentry_type="conversation",
        title="Conversation Agent",
        data=MappingProxyType(
            {
                CONF_MODEL: "test-model",
                CONF_CHECK_INS_ENABLED: enabled,
                CONF_EXPRESSION_TAGS: expression_tags,
            }
        ),
        unique_id=None,
    )
    entity = LocalAiConversationEntity(entry, subentry)
    entity.hass = hass
    entity.entity_id = "conversation.test"
    return entity


def _chunk(content: str) -> ChatCompletionChunk:
    """One assistant content chunk carrying ``content``."""
    return ChatCompletionChunk(
        id="test",
        created=0,
        model="test",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta(role="assistant", content=content))],
    )


def _finish() -> ChatCompletionChunk:
    """The closing chunk that ends the answer."""
    return ChatCompletionChunk(
        id="test",
        created=0,
        model="test",
        object="chat.completion.chunk",
        choices=[Choice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
    )


def _stub_stream(entity: LocalAiConversationEntity, deltas: list[str]) -> None:
    """Script the LLM stream to emit ``deltas`` in order, then finish."""

    async def scripted() -> AsyncIterator[ChatCompletionChunk]:
        for delta in deltas:
            yield _chunk(delta)
        yield _finish()

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


async def _content_texts(
    entity: LocalAiConversationEntity,
    hass: HomeAssistant,
    deltas: list[str],
) -> list[str]:
    """Run one turn over ``deltas`` and return the non-empty content deltas written."""
    _stub_stream(entity, deltas)
    chat_log = _chat_log(hass)
    recorded: list[dict[str, Any]] = []
    chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

    await _start_turn(entity, chat_log)
    return [delta["content"] for delta in recorded if delta.get("content")]


@pytest.mark.parametrize("enabled", [True, False], ids=["checkins_on", "checkins_off"])
class TestAnswerExpressionTags:
    """Strip-vs-pass on the streamed answer is independent of ``check_ins_enabled``."""

    async def test_tags_stripped_when_off(
        self, hass: HomeAssistant, mock_config_entry: ConfigEntry, enabled: bool
    ) -> None:
        """With expression_tags off, mood tags never reach the emitted answer."""
        entity = _make_entity(
            hass, mock_config_entry, enabled=enabled, expression_tags=False
        )
        texts = await _content_texts(
            entity,
            hass,
            ["[calm] Sure, ", "turning the light on now."],
        )
        assert "".join(texts) == " Sure, turning the light on now."
        assert all("[" not in text for text in texts)

    async def test_tags_passed_when_on(
        self, hass: HomeAssistant, mock_config_entry: ConfigEntry, enabled: bool
    ) -> None:
        """With expression_tags on, mood tags reach the emitted answer verbatim."""
        entity = _make_entity(
            hass, mock_config_entry, enabled=enabled, expression_tags=True
        )
        texts = await _content_texts(
            entity,
            hass,
            ["[calm] Sure, ", "turning the light on now."],
        )
        assert "".join(texts) == "[calm] Sure, turning the light on now."

    async def test_tag_split_across_deltas_is_stripped(
        self, hass: HomeAssistant, mock_config_entry: ConfigEntry, enabled: bool
    ) -> None:
        """A tag split across deltas is carried and stripped once complete."""
        entity = _make_entity(
            hass, mock_config_entry, enabled=enabled, expression_tags=False
        )
        texts = await _content_texts(
            entity,
            hass,
            ["[cal", "m] Sure."],
        )
        assert "".join(texts) == " Sure."

    async def test_tag_split_across_deltas_is_passed(
        self, hass: HomeAssistant, mock_config_entry: ConfigEntry, enabled: bool
    ) -> None:
        """With expression_tags on a split tag is emitted unchanged."""
        entity = _make_entity(
            hass, mock_config_entry, enabled=enabled, expression_tags=True
        )
        texts = await _content_texts(
            entity,
            hass,
            ["[cal", "m] Sure."],
        )
        assert "".join(texts) == "[calm] Sure."

    async def test_non_mood_bracket_is_untouched(
        self, hass: HomeAssistant, mock_config_entry: ConfigEntry, enabled: bool
    ) -> None:
        """A non-mood bracket in the answer survives the strip."""
        entity = _make_entity(
            hass, mock_config_entry, enabled=enabled, expression_tags=False
        )
        texts = await _content_texts(
            entity,
            hass,
            ["Values [1,2] are fine."],
        )
        assert "".join(texts) == "Values [1,2] are fine."
