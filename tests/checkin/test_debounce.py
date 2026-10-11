"""Unit tests for the single dead-air check-in debounce."""

from __future__ import annotations

from unittest.mock import create_autospec

from custom_components.local_openai.checkin.debounce import (
    DeadAirDebounce,
    should_emit,
)
from custom_components.local_openai.checkin.delivery import CheckinDelivery

GAP_S = 2.0


def _delivery(*, last_emission: float | None, now: float) -> CheckinDelivery:
    """A spec'd ``CheckinDelivery`` whose emission clock the test controls directly."""
    delivery = create_autospec(CheckinDelivery, instance=True)
    delivery.last_emission = last_emission
    delivery.now = now
    return delivery


class TestShouldEmit:
    """should_emit fires on the first check-in and once the gap has elapsed."""

    def test_no_prior_emission_fires(self) -> None:
        """A turn that has emitted no check-in yet is always due."""
        assert should_emit(last_emission=None, now=100.0, gap_s=GAP_S) is True

    def test_exactly_the_gap_fires(self) -> None:
        """The boundary is inclusive: elapsed exactly ``gap_s`` fires."""
        assert should_emit(last_emission=100.0, now=102.0, gap_s=GAP_S) is True

    def test_just_under_the_gap_skips(self) -> None:
        """Anything short of ``gap_s`` is dead air and is skipped."""
        assert should_emit(last_emission=100.0, now=101.999, gap_s=GAP_S) is False

    def test_well_past_the_gap_fires(self) -> None:
        """A long silence is comfortably due."""
        assert should_emit(last_emission=100.0, now=500.0, gap_s=GAP_S) is True


class TestDeadAirDebounce:
    """emit_if_due consults the delivery's clock and emits only when due."""

    async def test_emits_and_returns_true_when_due(self) -> None:
        """A due check-in is emitted through the delivery and reported as emitted."""
        delivery = _delivery(last_emission=None, now=10.0)
        debounce = DeadAirDebounce(delivery, gap_s=GAP_S)

        assert await debounce.emit_if_due("hello") is True

        delivery.emit.assert_awaited_once_with("hello")

    async def test_skips_without_emitting_when_not_due(self) -> None:
        """An early check-in is dropped, not queued: nothing reaches the delivery."""
        delivery = _delivery(last_emission=10.0, now=10.5)
        debounce = DeadAirDebounce(delivery, gap_s=GAP_S)

        assert await debounce.emit_if_due("hello") is False

        delivery.emit.assert_not_awaited()

    async def test_reads_last_emission_and_now_from_the_delivery(self) -> None:
        """The decision is driven by the delivery's own ``last_emission`` and ``now``."""
        delivery = _delivery(last_emission=100.0, now=100.0 + GAP_S / 2)
        debounce = DeadAirDebounce(delivery, gap_s=GAP_S)

        # Just inside the gap: not due, and nothing reaches the delivery.
        assert await debounce.emit_if_due("hi") is False
        delivery.emit.assert_not_awaited()

        # Past the gap on the same delivery: now due, and it emits.
        delivery.now = 100.0 + GAP_S
        assert await debounce.emit_if_due("hi") is True
        delivery.emit.assert_awaited_once_with("hi")
