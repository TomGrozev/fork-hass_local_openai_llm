"""End-to-end tests for the spoken apology on a hard LLM failure.

Drives the real ``_async_handle_chat_log`` with a real ``conversation.ChatLog`` and a client
whose completion call fails outright, so the observable behaviour - a spoken apology delta and a
cleanly ended turn, or the baseline raised error when check-ins are off - is decided by the test.
"""

from __future__ import annotations

import asyncio
import re
from types import MappingProxyType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from tests.conftest import spoken_texts

from custom_components.local_openai.checkin.pool import (
    DEFAULT_MOOD,
    FAILURE_LINES,
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
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

USER_TEXT = "turn on the kitchen light"
GRACE_MS = 500
#: The expression tag a mood-tagged apology carries (``"[sincere] ..."``); stripped to measure.
_MOOD_TAG = re.compile(r"^\[[a-z]+\]\s*")


def _llm_api() -> Any:
    """A stubbed LLM API with no tools: the turn fails before any tool matters."""
    api = MagicMock()
    api.tools = []
    api.custom_serializer = None
    api.async_call_tool = AsyncMock(return_value=llm.ToolResult(data={}))
    return api


def _make_entity(
    hass: HomeAssistant,
    entry: ConfigEntry,
    *,
    enabled: bool,
    expression_tags: bool = False,
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
                CONF_GRACE_MS: GRACE_MS,
                CONF_EXPRESSION_TAGS: expression_tags,
            }
        ),
        unique_id=None,
    )
    entity = LocalAiConversationEntity(entry, subentry)
    entity.hass = hass
    entity.entity_id = "conversation.test"
    return entity


def _fail_stream(entity: LocalAiConversationEntity) -> None:
    """Script the client so its completion call raises a hard ``OpenAIError``."""
    entity.entry.runtime_data.chat.completions.create = AsyncMock(
        side_effect=openai.APIConnectionError(request=MagicMock())
    )


def _chat_log(hass: HomeAssistant) -> conversation.ChatLog:
    """Build a chat log already holding the user's utterance, as the handler expects."""
    log = conversation.ChatLog(hass, "test-conversation-id", llm_api=_llm_api())
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


def _bucket_failures(*, expression_tags: bool = False) -> set[str]:
    """Every apology text the pool can produce."""
    return {
        render(
            line,
            default_mood=DEFAULT_MOOD["failure"],
            expression_tags=expression_tags,
        )
        for line in FAILURE_LINES
    }


class TestFailureApology:
    """A hard LLM failure is answered with a spoken apology instead of a raised error."""

    async def test_openai_error_speaks_mood_tagged_apology_and_ends_turn(
        self,
        hass: HomeAssistant,
        mock_config_entry: ConfigEntry,
    ) -> None:
        """An ``OpenAIError`` emits a mood-tagged apology delta and ends the turn cleanly."""
        entity = _make_entity(
            hass, mock_config_entry, enabled=True, expression_tags=True
        )
        _fail_stream(entity)
        chat_log = _chat_log(hass)
        recorded: list[dict[str, Any]] = []
        chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

        task = _start_turn(entity, chat_log)
        await _drain()
        # No HomeAssistantError: the turn ends cleanly with the apology spoken.
        await task

        texts = spoken_texts(recorded)
        assert len(texts) == 1
        apology = texts[0]
        assert apology in _bucket_failures(expression_tags=True)
        assert _MOOD_TAG.match(apology)
        assert len(_MOOD_TAG.sub("", apology)) >= 60
        assert any(
            isinstance(content, conversation.AssistantContent)
            and content.content == apology
            for content in chat_log.content
        )

    async def test_openai_error_still_raises_when_check_ins_disabled(
        self,
        hass: HomeAssistant,
        mock_config_entry: ConfigEntry,
    ) -> None:
        """With the feature off the baseline is preserved: the error is raised."""
        entity = _make_entity(hass, mock_config_entry, enabled=False)
        _fail_stream(entity)
        chat_log = _chat_log(hass)
        recorded: list[dict[str, Any]] = []
        chat_log.delta_listener = lambda _log, delta: recorded.append(delta)

        task = _start_turn(entity, chat_log)
        await _drain()
        with pytest.raises(
            HomeAssistantError,
            match="API server returned an error. Check the system logs for further details.",
        ):
            await task

        assert spoken_texts(recorded) == []
