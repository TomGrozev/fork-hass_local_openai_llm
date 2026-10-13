"""Unit tests for the check-before-speak opener gate."""

from __future__ import annotations

import asyncio
import random
from datetime import timedelta
from unittest.mock import create_autospec

from freezegun import freeze_time
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.local_openai.checkin.delivery import CheckinDelivery
from custom_components.local_openai.checkin.gate import OpenerGate
from custom_components.local_openai.checkin.pool import (
    OPENER_BUCKETS,
    LinePool,
    opener_lines,
)

GRACE_MS = 50
SEED = 4321
BUCKET = OPENER_BUCKETS[0]


def _pool() -> LinePool:
    """A pool seeded identically to the gate's, so its first draw is predictable."""
    return LinePool(rng=random.Random(SEED))


def _gate(delivery: CheckinDelivery) -> OpenerGate:
    """A gate over the shared seeded pool and grace window."""
    return OpenerGate(
        grace_s=GRACE_MS / 1000,
        pool=_pool(),
        delivery=delivery,
        bucket=BUCKET,
        intent=None,
        expression_tags=False,
    )


class TestSuppression:
    """An answer that beats the grace window keeps the opener silent."""

    async def test_set_answer_started_returns_false(self) -> None:
        """When the answer already started, run reports no emission."""
        delivery = create_autospec(CheckinDelivery, instance=True)
        gate = _gate(delivery)
        gate.answer_started.set()

        assert await gate.run() is False

    async def test_set_answer_started_emits_nothing(self) -> None:
        """Suppression leaves the delivery untouched."""
        delivery = create_autospec(CheckinDelivery, instance=True)
        gate = _gate(delivery)
        gate.answer_started.set()

        await gate.run()

        delivery.emit.assert_not_awaited()


class TestFire:
    """A slow answer lets the rendered opener through after the grace window."""

    async def test_fires_opener_after_grace(self, hass: HomeAssistant) -> None:
        """Waiting out the grace window emits the pool's opener and returns True."""
        expected = _pool().opener(BUCKET, expression_tags=False)
        delivery = create_autospec(CheckinDelivery, instance=True)
        gate = _gate(delivery)

        with freeze_time(dt_util.utcnow()):
            task = asyncio.create_task(gate.run())
            await asyncio.sleep(0)
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GRACE_MS + 1)
            )
            assert await task is True

        delivery.emit.assert_awaited_once_with(expected)


class TestIntentReachesThePool:
    """The gate hands the intent through, so the opener cell is the intent's, not ``any``."""

    async def test_intent_selects_the_opener_cell(self, hass: HomeAssistant) -> None:
        """An intent-bearing gate speaks a line drawn from that intent's cell."""
        bucket, intent = "media", "play"
        cell = {line.text for line in opener_lines(bucket, intent)}
        assert cell  # the media bucket declares a play cell
        expected = _pool().opener(bucket, intent, expression_tags=False)
        assert expected in cell

        delivery = create_autospec(CheckinDelivery, instance=True)
        gate = OpenerGate(
            grace_s=GRACE_MS / 1000,
            pool=_pool(),
            delivery=delivery,
            bucket=bucket,
            intent=intent,
            expression_tags=False,
        )

        with freeze_time(dt_util.utcnow()):
            task = asyncio.create_task(gate.run())
            await asyncio.sleep(0)
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GRACE_MS + 1)
            )
            assert await task is True

        delivery.emit.assert_awaited_once_with(expected)
