"""Tests for the conversation subentry check-in flow.

The conversation subentry flow is driven through the real
``ConversationFlowHandler`` step methods (user/reconfigure -> advanced ->
create/update). Form submissions are run through the schema the flow just
returned, mirroring how Home Assistant's flow manager applies defaults before
calling the step.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import MagicMock

from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
    SOURCE_USER,
    ConfigEntry,
    ConfigSubentry,
)
from homeassistant.const import CONF_LLM_HASS_API, CONF_MODEL, CONF_PROMPT
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.local_openai.config_flow import (
    AITaskDataFlowHandler,
    ConversationFlowHandler,
    LocalAiConfigFlow,
)
from custom_components.local_openai.const import (
    CONF_ALWAYS_CONTINUE_CONVERSATION,
    CONF_CHAT_TEMPLATE_KWARGS,
    CONF_CHAT_TEMPLATE_OPTS,
    CONF_CHECK_INS_ENABLED,
    CONF_CHECK_INS_ENABLED_DEFAULT,
    CONF_CHECKIN_GAP_MS,
    CONF_EXPRESSION_TAGS,
    CONF_EXPRESSION_TAGS_DEFAULT,
    CONF_GRACE_MS,
    CONF_PARALLEL_TOOL_CALLS,
    CONF_RECOMMENDED,
    CONF_REQUEST_BODY_OPTS,
    CONF_REQUEST_BODY_PARAMETERS,
    CONF_STRIP_EMOJIS,
    CONF_TEMPERATURE,
    CONF_WATCHDOG_MAX_EXTRAS,
)

CONVERSATION = "conversation"
AI_TASK_DATA = "ai_task_data"
TRANSLATIONS = (
    Path(__file__).parents[2]
    / "custom_components"
    / "local_openai"
    / "translations"
    / "en.json"
)


def _make_flow(
    hass: HomeAssistant,
    entry: ConfigEntry,
    *,
    source: str,
    subentry: ConfigSubentry | None = None,
) -> ConversationFlowHandler:
    """Build a conversation subentry flow wired to ``entry``."""
    flow = ConversationFlowHandler()
    flow.hass = hass
    flow.handler = (entry.entry_id, CONVERSATION)
    context: dict[str, Any] = {"source": source}
    if subentry is not None:
        context["subentry_id"] = subentry.subentry_id
    flow.context = MappingProxyType(context)
    return flow


def _recommended_input(**overrides: Any) -> dict[str, Any]:
    """A fully-populated recommended-step submission."""
    data: dict[str, Any] = {
        CONF_MODEL: "test-model",
        CONF_PROMPT: "You are a test.",
        CONF_LLM_HASS_API: [],
        CONF_PARALLEL_TOOL_CALLS: True,
        CONF_STRIP_EMOJIS: False,
        CONF_ALWAYS_CONTINUE_CONVERSATION: False,
        CONF_CHAT_TEMPLATE_OPTS: {CONF_CHAT_TEMPLATE_KWARGS: []},
        CONF_REQUEST_BODY_OPTS: {CONF_REQUEST_BODY_PARAMETERS: []},
        CONF_CHECK_INS_ENABLED: CONF_CHECK_INS_ENABLED_DEFAULT,
        CONF_EXPRESSION_TAGS: CONF_EXPRESSION_TAGS_DEFAULT,
        CONF_RECOMMENDED: True,
    }
    data.update(overrides)
    return data


def _model_prompt_only_input(**overrides: Any) -> dict[str, Any]:
    """A recommended-step submission carrying only the model/prompt fields.

    The two ``vol.Required`` sections have no schema default and must be
    present, and the ``llm_hass_api`` default would fail against the empty
    API list of a bare test ``hass``; the check-in booleans are left to
    materialise from the schema.
    """
    data: dict[str, Any] = {
        CONF_MODEL: "test-model",
        CONF_PROMPT: "You are a test.",
        CONF_LLM_HASS_API: [],
        CONF_CHAT_TEMPLATE_OPTS: {CONF_CHAT_TEMPLATE_KWARGS: []},
        CONF_REQUEST_BODY_OPTS: {CONF_REQUEST_BODY_PARAMETERS: []},
    }
    data.update(overrides)
    return data


def _submitted(
    form_result: dict[str, Any], user_input: dict[str, Any]
) -> dict[str, Any]:
    """Validate a submission through the schema the flow just returned.

    Home Assistant's flow manager calls ``data_schema(user_input)`` before the
    step, which is what materialises the ``vol.Optional`` defaults.
    """
    data_schema = form_result.get("data_schema")
    if data_schema is None:
        return user_input
    return data_schema(user_input)


def _marker_for(schema: Any, key: str) -> Any | None:
    """Return the schema marker whose key string equals ``key``."""
    for marker in schema.schema:
        if str(marker) == key:
            return marker
    return None


class TestRecommendedStep:
    """The recommended (default) branch of the user step."""

    async def test_recommended_default_skips_advanced_step(
        self,
        hass: HomeAssistant,
        config_entry_factory: Any,
    ) -> None:
        """Recommended defaults finish straight from the user step."""
        entry = await config_entry_factory(data={}, stub_models=True)
        flow = _make_flow(hass, entry, source=SOURCE_USER)

        result = await flow.async_step_user(
            _submitted(await flow.async_step_user(), _model_prompt_only_input())
        )
        assert result["type"] is FlowResultType.CREATE_ENTRY
        data = result["data"]
        assert data[CONF_CHECK_INS_ENABLED] is True
        assert data[CONF_EXPRESSION_TAGS] is False
        assert data[CONF_RECOMMENDED] is True
        assert CONF_GRACE_MS not in data
        assert CONF_CHECKIN_GAP_MS not in data
        assert CONF_WATCHDOG_MAX_EXTRAS not in data

    async def test_unchecking_recommended_applies_timing_defaults(
        self,
        hass: HomeAssistant,
        config_entry_factory: Any,
    ) -> None:
        """Unchecked recommended reveals the advanced step; omitted timings default."""
        entry = await config_entry_factory(data={}, stub_models=True)
        flow = _make_flow(hass, entry, source=SOURCE_USER)

        result = await flow.async_step_user(
            _submitted(
                await flow.async_step_user(),
                _model_prompt_only_input(**{CONF_RECOMMENDED: False}),
            )
        )
        assert result["type"] is FlowResultType.FORM
        assert result["step_id"] == "advanced"

        result = await flow.async_step_advanced(_submitted(result, {}))
        assert result["type"] is FlowResultType.CREATE_ENTRY
        data = result["data"]
        assert data[CONF_GRACE_MS] == 500
        assert data[CONF_CHECKIN_GAP_MS] == 1500
        assert data[CONF_WATCHDOG_MAX_EXTRAS] == 2
        # Merged with the recommended-step fields.
        assert data[CONF_MODEL] == "test-model"
        assert data[CONF_CHECK_INS_ENABLED] is True
        assert data[CONF_EXPRESSION_TAGS] is False


class TestAdvancedStep:
    """The advanced check-in timing step."""

    async def test_submitted_timing_values_are_stored(
        self,
        hass: HomeAssistant,
        config_entry_factory: Any,
    ) -> None:
        """Submitted timing values survive to the created subentry."""
        entry = await config_entry_factory(data={}, stub_models=True)
        flow = _make_flow(hass, entry, source=SOURCE_USER)

        result = await flow.async_step_user(
            _submitted(
                await flow.async_step_user(),
                _recommended_input(**{CONF_RECOMMENDED: False}),
            )
        )
        assert result["step_id"] == "advanced"

        result = await flow.async_step_advanced(
            _submitted(
                result,
                {
                    CONF_GRACE_MS: 750,
                    CONF_CHECKIN_GAP_MS: 2000,
                    CONF_WATCHDOG_MAX_EXTRAS: 5,
                },
            )
        )
        assert result["type"] is FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_GRACE_MS] == 750
        assert result["data"][CONF_CHECKIN_GAP_MS] == 2000
        assert result["data"][CONF_WATCHDOG_MAX_EXTRAS] == 5

    async def test_submitted_float_timing_values_are_coerced_to_int(
        self,
        hass: HomeAssistant,
        config_entry_factory: Any,
    ) -> None:
        """The NumberSelector's floats are persisted as ints."""
        entry = await config_entry_factory(data={}, stub_models=True)
        flow = _make_flow(hass, entry, source=SOURCE_USER)

        result = await flow.async_step_user(
            _submitted(
                await flow.async_step_user(),
                _recommended_input(**{CONF_RECOMMENDED: False}),
            )
        )
        assert result["step_id"] == "advanced"

        result = await flow.async_step_advanced(
            _submitted(
                result,
                {
                    CONF_GRACE_MS: 750.0,
                    CONF_CHECKIN_GAP_MS: 2000.0,
                    CONF_WATCHDOG_MAX_EXTRAS: 5.0,
                },
            )
        )
        assert result["type"] is FlowResultType.CREATE_ENTRY
        data = result["data"]
        assert isinstance(data[CONF_GRACE_MS], int)
        assert data[CONF_GRACE_MS] == 750
        assert isinstance(data[CONF_CHECKIN_GAP_MS], int)
        assert data[CONF_CHECKIN_GAP_MS] == 2000
        assert isinstance(data[CONF_WATCHDOG_MAX_EXTRAS], int)
        assert data[CONF_WATCHDOG_MAX_EXTRAS] == 5

    async def test_advanced_path_preserves_model_fields(
        self,
        hass: HomeAssistant,
        config_entry_factory: Any,
    ) -> None:
        """Model/prompt/temperature submitted in the recommended step are unchanged."""
        entry = await config_entry_factory(data={}, stub_models=True)
        flow = _make_flow(hass, entry, source=SOURCE_USER)
        recommended = _recommended_input(
            **{
                CONF_PROMPT: "A bespoke prompt.",
                CONF_TEMPERATURE: 0.3,
                CONF_RECOMMENDED: False,
            }
        )

        result = await flow.async_step_user(
            _submitted(await flow.async_step_user(), recommended)
        )
        assert result["step_id"] == "advanced"

        result = await flow.async_step_advanced(
            _submitted(
                result,
                {
                    CONF_GRACE_MS: 900,
                    CONF_CHECKIN_GAP_MS: 1800,
                    CONF_WATCHDOG_MAX_EXTRAS: 3,
                },
            )
        )
        assert result["type"] is FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_MODEL] == "test-model"
        assert result["data"][CONF_PROMPT] == "A bespoke prompt."
        assert result["data"][CONF_TEMPERATURE] == 0.3


class TestReconfigure:
    """The reconfigure branch of the subentry flow."""

    async def test_reconfigure_prefills_and_updates(
        self,
        hass: HomeAssistant,
        config_entry_factory: Any,
    ) -> None:
        """Reconfigure reaches the advanced step and updates the existing subentry."""
        entry = await config_entry_factory(
            data={},
            stub_models=True,
            subentries_data=[
                {
                    "subentry_type": CONVERSATION,
                    "title": "Conversation Agent",
                    "data": {
                        CONF_MODEL: "old-model",
                        CONF_PROMPT: "old prompt",
                        CONF_GRACE_MS: 700,
                    },
                    "unique_id": None,
                }
            ],
        )
        (existing,) = entry.subentries.values()
        flow = _make_flow(hass, entry, source=SOURCE_RECONFIGURE, subentry=existing)

        result = await flow.async_step_reconfigure(
            _submitted(
                await flow.async_step_reconfigure(),
                _recommended_input(
                    **{
                        CONF_MODEL: "old-model",
                        CONF_PROMPT: "old prompt",
                        CONF_RECOMMENDED: False,
                    }
                ),
            )
        )
        assert result["type"] is FlowResultType.FORM
        assert result["step_id"] == "advanced"
        # The advanced form prefills from the existing subentry.
        grace_marker = _marker_for(result["data_schema"], CONF_GRACE_MS)
        assert grace_marker is not None
        assert grace_marker.description == {"suggested_value": 700}

        result = await flow.async_step_advanced(
            _submitted(
                result,
                {
                    CONF_GRACE_MS: 800,
                    CONF_CHECKIN_GAP_MS: 1600,
                    CONF_WATCHDOG_MAX_EXTRAS: 4,
                },
            )
        )
        assert result["type"] is FlowResultType.ABORT
        assert result["reason"] == "reconfigure_successful"

        updated = entry.subentries[existing.subentry_id]
        assert updated.data[CONF_GRACE_MS] == 800
        assert updated.data[CONF_CHECKIN_GAP_MS] == 1600
        assert updated.data[CONF_WATCHDOG_MAX_EXTRAS] == 4
        assert updated.data[CONF_MODEL] == "old-model"

    async def test_subentry_without_checkin_keys_defaults_on_save(
        self,
        hass: HomeAssistant,
        config_entry_factory: Any,
    ) -> None:
        """A subentry predating the check-in keys needs no migration."""
        entry = await config_entry_factory(
            data={},
            stub_models=True,
            subentries_data=[
                {
                    "subentry_type": CONVERSATION,
                    "title": "Legacy Agent",
                    "data": {CONF_MODEL: "legacy-model", CONF_PROMPT: "legacy prompt"},
                    "unique_id": None,
                }
            ],
        )
        (existing,) = entry.subentries.values()
        flow = _make_flow(hass, entry, source=SOURCE_RECONFIGURE, subentry=existing)

        # Opening reconfigure does not raise and leaves the missing keys missing.
        result = await flow.async_step_reconfigure()
        assert result["type"] is FlowResultType.FORM
        assert result["step_id"] == "reconfigure"
        assert CONF_GRACE_MS not in existing.data
        assert CONF_CHECKIN_GAP_MS not in existing.data
        assert CONF_WATCHDOG_MAX_EXTRAS not in existing.data

        # Saving through the advanced path then falls back to the default constants.
        result = await flow.async_step_reconfigure(
            _submitted(result, _recommended_input(**{CONF_RECOMMENDED: False}))
        )
        assert result["step_id"] == "advanced"

        result = await flow.async_step_advanced(_submitted(result, {}))
        assert result["type"] is FlowResultType.ABORT

        updated = entry.subentries[existing.subentry_id]
        assert updated.data[CONF_GRACE_MS] == 500
        assert updated.data[CONF_CHECKIN_GAP_MS] == 1500
        assert updated.data[CONF_WATCHDOG_MAX_EXTRAS] == 2
        assert updated.data[CONF_CHECK_INS_ENABLED] is True
        assert updated.data[CONF_EXPRESSION_TAGS] is False


class TestFlowRegistration:
    """Subentry-flow registration around the conversation flow."""

    async def test_other_subentry_flows_are_unchanged(
        self,
        hass: HomeAssistant,
        config_entry_factory: Any,
    ) -> None:
        """The AI-task subentry flow stays registered alongside the conversation flow."""
        supported = LocalAiConfigFlow.async_get_supported_subentry_types(
            await config_entry_factory()
        )
        assert set(supported) == {CONVERSATION, AI_TASK_DATA}

        handler = AITaskDataFlowHandler()
        handler.hass = hass
        entry = await config_entry_factory()
        handler._get_entry = MagicMock(return_value=entry)  # type: ignore[method-assign]
        assert await handler.get_schema() is not None


class TestTranslations:
    """The conversation subentry translation file."""

    def test_translations_expose_streaming_tts_note_and_advanced_fields(self) -> None:
        """The setup note and the three advanced timing fields are translated."""
        translations = json.loads(TRANSLATIONS.read_text())
        conversation = translations["config_subentries"]["conversation"]
        assert conversation["step"]["user"]["description"]

        advanced = conversation["step"]["advanced"]["data"]
        assert CONF_GRACE_MS in advanced
        assert CONF_CHECKIN_GAP_MS in advanced
        assert CONF_WATCHDOG_MAX_EXTRAS in advanced
