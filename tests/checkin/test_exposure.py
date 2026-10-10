"""Unit tests for the Assist expose-list adapter."""

from __future__ import annotations

from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component

from custom_components.local_openai.checkin.exposure import (
    ASSIST_EXPOSURE_KEY,
    exposed_entities,
)
from custom_components.local_openai.checkin.selector import ExposedEntity


async def _setup(hass: HomeAssistant) -> er.EntityRegistry:
    """Set up the expose store and return the entity registry."""
    await async_setup_component(hass, "homeassistant", {})
    return er.async_get(hass)


class TestExposedEntities:
    """exposed_entities projects the Assist expose list onto the selector's shape."""

    async def test_returns_nothing_when_none_exposed(self, hass: HomeAssistant) -> None:
        """With nothing exposed, the result is an empty tuple."""
        await _setup(hass)
        hass.states.async_set("light.kitchen", "on", {"friendly_name": "Kitchen"})
        async_expose_entity(hass, ASSIST_EXPOSURE_KEY, "light.kitchen", False)

        assert exposed_entities(hass) == ()

    async def test_reports_registry_names_and_domain(self, hass: HomeAssistant) -> None:
        """An exposed registry entity reports its friendly name, aliases and domain."""
        registry = await _setup(hass)
        entry = registry.async_get_or_create(
            "light",
            "demo",
            "unique-kitchen",
            suggested_object_id="kitchen",
            original_name="Kitchen Light",
        )
        registry.async_update_entity(
            entry.entity_id, aliases=["Kitchen Light", "Cooking Light"]
        )
        hass.states.async_set(entry.entity_id, "on", {"friendly_name": "Kitchen Light"})
        async_expose_entity(hass, ASSIST_EXPOSURE_KEY, entry.entity_id, True)

        assert exposed_entities(hass) == (
            ExposedEntity(domain="light", names=("Kitchen Light", "Cooking Light")),
        )

    async def test_excludes_unexposed_entities(self, hass: HomeAssistant) -> None:
        """An entity left unexposed is omitted even when a state exists."""
        registry = await _setup(hass)
        entry = registry.async_get_or_create(
            "light",
            "demo",
            "unique-kitchen",
            suggested_object_id="kitchen",
            original_name="Kitchen Light",
        )
        hass.states.async_set(entry.entity_id, "on", {"friendly_name": "Kitchen Light"})
        async_expose_entity(hass, ASSIST_EXPOSURE_KEY, entry.entity_id, False)

        assert exposed_entities(hass) == ()

    async def test_falls_back_to_state_name_without_registry_entry(
        self, hass: HomeAssistant
    ) -> None:
        """An exposed entity with no registry entry reports its state name."""
        await _setup(hass)
        hass.states.async_set("light.old", "on", {"friendly_name": "Old Lamp"})
        async_expose_entity(hass, ASSIST_EXPOSURE_KEY, "light.old", True)

        assert exposed_entities(hass) == (
            ExposedEntity(domain="light", names=("Old Lamp",)),
        )
