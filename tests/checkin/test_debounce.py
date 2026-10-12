"""Unit tests for the single dead-air check-in debounce."""

from __future__ import annotations

from unittest.mock import create_autospec

from custom_components.local_openai.checkin.debounce import (
    DeadAirDebounce,
    should_emit,
)
from custom_components.local_openai.checkin.delivery import CheckinDelivery

GAP_S = 2.0


def _delivery(*, horizon: float | None, now: float) -> CheckinDelivery:
    """A spec'd ``CheckinDelivery`` whose speech horizon the test controls directly."""
    delivery = create_autospec(CheckinDelivery, instance=True)
    delivery.horizon = horizon
    delivery.now = now
    return delivery


class TestShouldEmit:
    """should_emit fires on the first check-in and once the audio has finished for a gap."""

    def test_no_speech_yet_fires(self) -> None:
        """A turn that has spoken nothing yet is always due."""
        assert should_emit(horizon=None, now=100.0, gap_s=GAP_S) is True

    def test_exactly_the_gap_after_speech_fires(self) -> None:
        """The boundary is inclusive: exactly ``gap_s`` of silence after the horizon fires."""
        assert should_emit(horizon=100.0, now=102.0, gap_s=GAP_S) is True

    def test_while_the_audio_is_still_playing_skips(self) -> None:
        """A ``now`` before the horizon means speech is queued: never dead air."""
        assert should_emit(horizon=101.0, now=100.0, gap_s=GAP_S) is False

    def test_just_under_the_gap_after_speech_skips(self) -> None:
        """Less than ``gap_s`` of silence after the horizon is still dead air."""
        assert should_emit(horizon=100.0, now=101.999, gap_s=GAP_S) is False

    def test_well_past_the_horizon_fires(self) -> None:
        """A long silence is comfortably due."""
        assert should_emit(horizon=100.0, now=500.0, gap_s=GAP_S) is True


class TestDeadAirDebounce:
    """emit_if_due consults the delivery's horizon and clock and emits only when due."""

    async def test_emits_and_returns_true_when_due(self) -> None:
        """A due check-in is emitted through the delivery and reported as emitted."""
        delivery = _delivery(horizon=None, now=10.0)
        debounce = DeadAirDebounce(delivery, gap_s=GAP_S)

        assert await debounce.emit_if_due("hello") is True

        delivery.emit.assert_awaited_once_with("hello")

    async def test_skips_without_emitting_while_audio_is_playing(self) -> None:
        """A check-in due during the last one's speech is dropped, not queued."""
        delivery = _delivery(horizon=10.5, now=10.0)
        debounce = DeadAirDebounce(delivery, gap_s=GAP_S)

        assert await debounce.emit_if_due("hello") is False

        delivery.emit.assert_not_awaited()

    async def test_reads_horizon_and_now_from_the_delivery(self) -> None:
        """The decision is driven by the delivery's own ``horizon`` and ``now``."""
        delivery = _delivery(horizon=100.0, now=100.0 + GAP_S / 2)
        debounce = DeadAirDebounce(delivery, gap_s=GAP_S)

        # Just inside the gap after speech ended: not due, and nothing reaches the delivery.
        assert await debounce.emit_if_due("hi") is False
        delivery.emit.assert_not_awaited()

        # Past the gap on the same delivery: now due, and it emits.
        delivery.now = 100.0 + GAP_S
        assert await debounce.emit_if_due("hi") is True
        delivery.emit.assert_awaited_once_with("hi")
