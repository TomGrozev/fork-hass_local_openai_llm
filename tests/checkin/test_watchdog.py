"""Unit tests for the capped watchdog reassurance."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from unittest.mock import create_autospec

from freezegun import freeze_time
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.local_openai.checkin.debounce import DeadAirDebounce
from custom_components.local_openai.checkin.pool import LinePool
from custom_components.local_openai.checkin.watchdog import Watchdog

GAP_MS = 1500


def _watchdog(
    *,
    debounce: DeadAirDebounce,
    max_extras: int,
    opener_task: asyncio.Task[bool] | None = None,
    answer_started: asyncio.Event | None = None,
) -> Watchdog:
    """A watchdog over the given debounce, drawing its lines from a stub pool."""
    pool = create_autospec(LinePool, instance=True)
    pool.watchdog.return_value = "steady"
    return Watchdog(
        gap_s=GAP_MS / 1000,
        max_extras=max_extras,
        pool=pool,
        debounce=debounce,
        expression_tags=False,
        opener_task=opener_task,
        answer_started=(
            answer_started if answer_started is not None else asyncio.Event()
        ),
    )


class TestCappedReassurances:
    """One reassurance per silent gap, up to the cap, then the loop returns."""

    async def test_emits_up_to_max_extras_then_stops(self, hass: HomeAssistant) -> None:
        """Two silent gaps draw two reassurances; the third iteration never runs."""
        debounce = create_autospec(DeadAirDebounce, instance=True)
        debounce.emit_if_due.return_value = True
        watchdog = _watchdog(debounce=debounce, max_extras=2)

        with freeze_time(dt_util.utcnow()):
            task = asyncio.create_task(watchdog.run())
            await asyncio.sleep(0)
            for _ in range(2):
                async_fire_time_changed(
                    hass, dt_util.utcnow() + timedelta(milliseconds=GAP_MS + 1)
                )
                await asyncio.sleep(0)
            await task

        assert debounce.emit_if_due.await_count == 2


class TestProgressSuppressesTheInterval:
    """A chunk arriving during an interval makes it live, not dead air."""

    async def test_note_progress_suppresses_that_interval(
        self, hass: HomeAssistant
    ) -> None:
        """A marked interval is skipped; the next silent one still draws a reassurance."""
        debounce = create_autospec(DeadAirDebounce, instance=True)
        debounce.emit_if_due.return_value = True
        watchdog = _watchdog(debounce=debounce, max_extras=1)

        with freeze_time(dt_util.utcnow()):
            task = asyncio.create_task(watchdog.run())
            await asyncio.sleep(0)
            watchdog.note_progress()
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GAP_MS + 1)
            )
            await asyncio.sleep(0)
            assert debounce.emit_if_due.await_count == 0

            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GAP_MS + 1)
            )
            await asyncio.sleep(0)
            await task

        assert debounce.emit_if_due.await_count == 1


class TestOpenerSuppressesTheAttempt:
    """The opener is always spoken, so a reassurance waits for it to resolve."""

    async def test_unresolved_opener_suppresses_the_attempt(
        self, hass: HomeAssistant
    ) -> None:
        """A gap while the opener still waits emits nothing; it fires once the opener resolves."""
        debounce = create_autospec(DeadAirDebounce, instance=True)
        debounce.emit_if_due.return_value = True
        opener = asyncio.Event()
        opener_task = asyncio.create_task(opener.wait())
        watchdog = _watchdog(debounce=debounce, max_extras=1, opener_task=opener_task)

        with freeze_time(dt_util.utcnow()):
            task = asyncio.create_task(watchdog.run())
            await asyncio.sleep(0)
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GAP_MS + 1)
            )
            await asyncio.sleep(0)
            assert debounce.emit_if_due.await_count == 0

            opener.set()
            await asyncio.sleep(0)
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GAP_MS + 1)
            )
            await asyncio.sleep(0)
            await task

        assert debounce.emit_if_due.await_count == 1


class TestAnswerStartedEndsTheLoop:
    """Once the real answer has started, the watchdog stops emitting and returns."""

    async def test_answer_started_stops_the_loop(self, hass: HomeAssistant) -> None:
        """A gap before the answer emits; the answer starting ends the loop on the next wake."""
        debounce = create_autospec(DeadAirDebounce, instance=True)
        debounce.emit_if_due.return_value = True
        answer_started = asyncio.Event()
        watchdog = _watchdog(
            debounce=debounce, max_extras=3, answer_started=answer_started
        )

        with freeze_time(dt_util.utcnow()):
            task = asyncio.create_task(watchdog.run())
            await asyncio.sleep(0)
            # First gap: the answer has not started, so one reassurance lands.
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GAP_MS + 1)
            )
            await asyncio.sleep(0)
            assert debounce.emit_if_due.await_count == 1

            # The answer starts; the next wake returns before emitting anything more.
            answer_started.set()
            async_fire_time_changed(
                hass, dt_util.utcnow() + timedelta(milliseconds=GAP_MS + 1)
            )
            await asyncio.sleep(0)
            await task

        assert debounce.emit_if_due.await_count == 1
        assert task.done()
