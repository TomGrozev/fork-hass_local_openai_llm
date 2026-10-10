"""
Adapt Home Assistant's Assist expose list to the selector's ``ExposedEntity`` shape.

The selector is pure and I/O-free by design; the caller does the Home Assistant-specific lookup
here (see ``checkin.selector.ExposedEntity``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.helpers import entity_registry, intent

from custom_components.local_openai.checkin.selector import ExposedEntity

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

#: Every Assist conversation agent shares Home Assistant's single ``"conversation"`` exposure
#: key (``KNOWN_ASSISTANTS``), so this agent sees the same set the built-in agent does.
ASSIST_EXPOSURE_KEY: Final = "conversation"


def exposed_entities(hass: HomeAssistant) -> tuple[ExposedEntity, ...]:
    """
    Return the entities exposed to Assist, as the selector's input.

    ``names`` are the entity registry names (friendly name and user aliases); ``domain`` is the
    ``entity_id`` prefix. Only entity names and domains pick the opener bucket — areas describe
    *where*, not *what*, so they are ignored.
    """
    registry = entity_registry.async_get(hass)
    exposed: list[ExposedEntity] = []
    for state in hass.states.async_all():
        if not async_should_expose(hass, ASSIST_EXPOSURE_KEY, state.entity_id):
            continue
        names = tuple(
            intent.async_get_entity_aliases(
                hass,
                registry.async_get(state.entity_id),
                state=state,
            )
        )
        exposed.append(ExposedEntity(domain=state.domain, names=names))
    return tuple(exposed)
