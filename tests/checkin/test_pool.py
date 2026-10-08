"""Unit tests for the curated check-in line pool."""

from __future__ import annotations

import random

import pytest

from custom_components.local_openai.checkin.pool import (
    DEFAULT_MOOD,
    FAILURE_LENGTH_MIN,
    FAILURE_LINES,
    MIDCHAIN_BUCKETS,
    MIDCHAIN_LINES,
    MOODS,
    OPENER_BUCKETS,
    OPENER_LENGTH_MAX,
    OPENER_LENGTH_MIN,
    OPENER_LINES,
    WATCHDOG_LINES,
    Line,
    LinePool,
    render,
    select,
)


def _section_lines() -> list[Line]:
    """Every authored line across all four sections."""
    lines = [line for bucket in OPENER_LINES.values() for line in bucket]
    lines += [line for bucket in MIDCHAIN_LINES.values() for line in bucket]
    lines += list(WATCHDOG_LINES)
    lines += list(FAILURE_LINES)
    return lines


class TestPoolIntegrity:
    """The shipped content stays inside its declared shape and sizing band."""

    def test_opener_buckets_match_declared_keys(self) -> None:
        """OPENER_LINES keys are exactly OPENER_BUCKETS."""
        assert set(OPENER_LINES) == set(OPENER_BUCKETS)

    def test_midchain_buckets_match_declared_keys(self) -> None:
        """MIDCHAIN_LINES keys are exactly MIDCHAIN_BUCKETS."""
        assert set(MIDCHAIN_LINES) == set(MIDCHAIN_BUCKETS)

    @pytest.mark.parametrize("bucket", OPENER_BUCKETS)
    def test_opener_bucket_has_one_to_three_variants(self, bucket: str) -> None:
        """Each opener bucket carries between one and three variants."""
        assert 1 <= len(OPENER_LINES[bucket]) <= 3

    @pytest.mark.parametrize("bucket", OPENER_BUCKETS)
    def test_opener_texts_sit_inside_length_band(self, bucket: str) -> None:
        """Every opener text is within the inclusive 60-66 char band."""
        for line in OPENER_LINES[bucket]:
            assert OPENER_LENGTH_MIN <= len(line.text) <= OPENER_LENGTH_MAX

    def test_failure_lines_meet_length_floor(self) -> None:
        """Every failure line is at least 60 characters long."""
        assert all(len(line.text) >= FAILURE_LENGTH_MIN for line in FAILURE_LINES)

    def test_every_mood_is_declared_or_none(self) -> None:
        """No authored line carries a mood outside MOODS."""
        assert all(line.mood in MOODS or line.mood is None for line in _section_lines())

    @pytest.mark.parametrize("bucket", OPENER_BUCKETS)
    def test_opener_variants_render_distinct_text(self, bucket: str) -> None:
        """No two opener variants in a bucket speak the same line."""
        spoken = [render(line) for line in OPENER_LINES[bucket]]
        assert len(spoken) == len(set(spoken))

    @pytest.mark.parametrize("bucket", MIDCHAIN_BUCKETS)
    def test_midchain_variants_render_distinct_text(self, bucket: str) -> None:
        """No two mid-chain variants in a bucket speak the same line, slots or not."""
        for slots in (None, {"item": "milk"}):
            spoken = [render(line, slots=slots) for line in MIDCHAIN_LINES[bucket]]
            assert len(spoken) == len(set(spoken))


class TestRender:
    """render picks the spoken text and attaches the mood tag."""

    def test_expression_tags_prefix_the_mood(self) -> None:
        """expression_tags=True emits a leading '[mood] ' tag."""
        line = Line("Let me check the weather for you right now.", "calm")
        assert render(line, expression_tags=True).startswith("[calm] ")

    def test_expression_tags_off_has_no_tag(self) -> None:
        """expression_tags=False returns the bare text with no tag."""
        line = Line("Let me check the weather for you right now.", "calm")
        assert render(line) == "Let me check the weather for you right now."

    def test_mood_falls_back_to_default_mood(self) -> None:
        """A mood-less line inherits default_mood for its tag."""
        line = Line("Adding that in now please wait.", None)
        result = render(line, default_mood="playful", expression_tags=True)
        assert result == "[playful] Adding that in now please wait."

    def test_line_mood_wins_over_default_mood(self) -> None:
        """A line's own mood overrides default_mood."""
        line = Line("Adding that in now please wait.", "sincere")
        assert render(line, default_mood="calm", expression_tags=True).startswith(
            "[sincere] "
        )

    def test_no_tag_when_mood_is_absent(self) -> None:
        """With neither line mood nor default_mood, no tag is emitted."""
        line = Line("Adding that in now please wait.")
        assert render(line, expression_tags=True) == "Adding that in now please wait."

    def test_slot_line_fills_from_slots(self) -> None:
        """A slotted line formats its blanks from the supplied mapping."""
        line = Line("Adding {item}...", plain="Adding that in...")
        assert render(line, slots={"item": "milk"}) == "Adding milk..."

    def test_slot_line_falls_back_to_plain_without_slots(self) -> None:
        """A slotted line without slots uses its plain fallback."""
        line = Line("Adding {item}...", plain="Adding that in...")
        assert render(line) == "Adding that in..."

    def test_slot_line_falls_back_when_slots_are_incomplete(self) -> None:
        """A slotted line with incomplete slots uses its plain fallback."""
        line = Line("Adding {item}...", plain="Adding that in...")
        assert render(line, slots={"other": "x"}) == "Adding that in..."

    def test_slot_free_line_is_verbatim(self) -> None:
        """A line with no plain fallback is returned unchanged."""
        line = Line("Let me check that for you now.", "calm")
        assert render(line) == "Let me check that for you now."

    def test_slot_free_line_is_not_formatted_with_slots(self) -> None:
        """A line with no plain fallback is never formatted, braces intact."""
        line = Line("Hello {name}, let me look into that.", "calm")
        assert (
            render(line, slots={"name": "Ada"})
            == "Hello {name}, let me look into that."
        )


class TestSelect:
    """select draws uniformly while avoiding an immediate repeat."""

    def test_avoids_immediate_repeat(self) -> None:
        """Repeated draws from a three-variant bucket never repeat back to back."""
        lines = OPENER_LINES["weather"]
        rng = random.Random(1234)
        last: Line | None = None
        for _ in range(50):
            current = select(lines, last=last, rng=rng)
            assert current is not last
            last = current

    def test_single_variant_returns_that_line(self) -> None:
        """A one-variant bucket always yields its only line."""
        lines = MIDCHAIN_LINES["weather"]
        assert select(lines, rng=random.Random(0)) is lines[0]

    def test_all_candidates_equal_last_still_returns_one(self) -> None:
        """When the only line equals last, it is still returned."""
        line = Line("Only one line here.", "calm")
        assert select((line,), last=line, rng=random.Random(0)) is line

    def test_empty_raises_value_error(self) -> None:
        """An empty candidate set is rejected."""
        with pytest.raises(ValueError, match="empty"):
            select((), rng=random.Random(0))

    def test_injected_rng_makes_the_draw_deterministic(self) -> None:
        """The same seed produces the same sequence of picks."""
        lines = OPENER_LINES["lights"]
        assert _draw_sequence(lines, seed=99) == _draw_sequence(lines, seed=99)


class TestLinePool:
    """The stateful facade renders every context and tracks repeat state."""

    @pytest.mark.parametrize("bucket", OPENER_BUCKETS)
    def test_opener_returns_non_empty(self, bucket: str) -> None:
        """opener yields a non-empty spoken string per bucket."""
        assert LinePool(rng=random.Random(1)).opener(bucket)

    @pytest.mark.parametrize("bucket", MIDCHAIN_BUCKETS)
    def test_midchain_returns_non_empty(self, bucket: str) -> None:
        """midchain yields a non-empty spoken string per bucket."""
        assert LinePool(rng=random.Random(1)).midchain(bucket)

    def test_watchdog_returns_non_empty(self) -> None:
        """watchdog yields a non-empty reassurance."""
        assert LinePool(rng=random.Random(1)).watchdog()

    def test_failure_returns_non_empty(self) -> None:
        """failure yields a non-empty apology."""
        assert LinePool(rng=random.Random(1)).failure()

    def test_consecutive_openers_do_not_repeat(self) -> None:
        """Back-to-back opener calls for one bucket differ."""
        pool = LinePool(rng=random.Random(7))
        first = pool.opener("weather")
        assert pool.opener("weather") != first

    def test_expression_tags_lead_with_a_tag(self) -> None:
        """expression_tags=True gives a leading bracketed mood tag."""
        result = LinePool(rng=random.Random(1)).opener("weather", expression_tags=True)
        assert result.startswith("[")

    def test_without_tags_there_is_no_bracket(self) -> None:
        """expression_tags=False gives clean spoken text with no bracket."""
        result = LinePool(rng=random.Random(1)).opener("weather")
        assert not result.startswith("[")

    def test_midchain_fills_slots(self) -> None:
        """midchain passes slots through to render."""
        pool = LinePool(rng=random.Random(3))
        assert pool.midchain("list", slots={"item": "milk"}) in {
            "Adding milk...",
            "Putting that on your list...",
        }

    def test_midchain_falls_back_without_slots(self) -> None:
        """midchain without slots resolves every variant to slot-free text."""
        pool = LinePool(rng=random.Random(3))
        assert pool.midchain("list") in {
            "Adding that in...",
            "Putting that on your list...",
        }

    @pytest.mark.parametrize(
        ("context", "expected"),
        [
            pytest.param("opener:weather", "calm", id="opener_bucket"),
            pytest.param("midchain:list", "calm", id="midchain_bucket"),
            pytest.param("watchdog", "calm", id="watchdog"),
            pytest.param("failure", "sincere", id="failure"),
        ],
    )
    def test_default_mood_returns_declared_value(
        self, context: str, expected: str
    ) -> None:
        """default_mood reports the declared mood for a context."""
        assert LinePool(rng=random.Random(1)).default_mood(context) == expected
        assert DEFAULT_MOOD[context] == expected

    def test_opener_and_midchain_default_moods_do_not_collide(self) -> None:
        """Same-named opener and mid-chain contexts hold independent default moods."""
        assert "opener:weather" in DEFAULT_MOOD
        assert "midchain:weather" in DEFAULT_MOOD
        assert "weather" not in DEFAULT_MOOD

    def test_unknown_opener_bucket_raises_key_error(self) -> None:
        """An unknown opener bucket surfaces a KeyError."""
        with pytest.raises(KeyError):
            LinePool(rng=random.Random(1)).opener("nope")


def _draw_sequence(lines: tuple[Line, ...], *, seed: int) -> list[Line]:
    """Draw a representative run of picks from one seeded generator."""
    rng = random.Random(seed)
    last: Line | None = None
    drawn: list[Line] = []
    for _ in range(6):
        last = select(lines, last=last, rng=rng)
        drawn.append(last)
    return drawn
