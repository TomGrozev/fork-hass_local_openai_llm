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
    OPENER_INTENTS,
    OPENER_LENGTH_MAX,
    OPENER_LENGTH_MIN,
    OPENER_LINES,
    WATCHDOG_LINES,
    Line,
    LinePool,
    opener_lines,
    render,
    select,
    strip_mood_tags,
    strip_mood_tags_streaming,
)


def _opener_cells() -> list[tuple[str, str, tuple[Line, ...]]]:
    """Every ``(bucket, cell, lines)`` triple across the opener map."""
    return [
        (bucket, cell, lines)
        for bucket, cells in OPENER_LINES.items()
        for cell, lines in cells.items()
    ]


def _cell_ids() -> list[str]:
    """Readable pytest ids for the opener cells."""
    return [f"{bucket}.{cell}" for bucket, cell, _ in _opener_cells()]


def _section_lines() -> list[Line]:
    """Every authored line across all four sections."""
    lines = [line for _, _, cell in _opener_cells() for line in cell]
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
    def test_every_bucket_has_an_any_cell(self, bucket: str) -> None:
        """Every opener bucket carries an ``any`` cell as its fallback."""
        assert "any" in OPENER_LINES[bucket]

    @pytest.mark.parametrize("bucket", OPENER_BUCKETS)
    def test_opener_cell_keys_are_any_or_declared_intents(self, bucket: str) -> None:
        """A bucket's cell keys are ``any`` plus members of OPENER_INTENTS."""
        assert set(OPENER_LINES[bucket]) <= {"any", *OPENER_INTENTS}

    @pytest.mark.parametrize(
        ("bucket", "cell", "lines"), _opener_cells(), ids=_cell_ids()
    )
    def test_opener_texts_sit_inside_length_band(
        self, bucket: str, cell: str, lines: tuple[Line, ...]
    ) -> None:
        """Every opener text is within the inclusive 60-66 char band."""
        for line in lines:
            assert OPENER_LENGTH_MIN <= len(line.text) <= OPENER_LENGTH_MAX

    def test_failure_lines_meet_length_floor(self) -> None:
        """Every failure line is at least 60 characters long."""
        assert all(len(line.text) >= FAILURE_LENGTH_MIN for line in FAILURE_LINES)

    def test_every_mood_is_declared_or_none(self) -> None:
        """No authored line carries a mood outside MOODS."""
        assert all(line.mood in MOODS or line.mood is None for line in _section_lines())

    @pytest.mark.parametrize(
        ("bucket", "cell", "lines"), _opener_cells(), ids=_cell_ids()
    )
    def test_opener_cell_texts_render_distinct(
        self, bucket: str, cell: str, lines: tuple[Line, ...]
    ) -> None:
        """No two openers in a cell speak the same line."""
        spoken = [render(line) for line in lines]
        assert len(spoken) == len(set(spoken))

    @pytest.mark.parametrize("bucket", MIDCHAIN_BUCKETS)
    def test_midchain_variants_render_distinct_text(self, bucket: str) -> None:
        """No two mid-chain variants in a bucket speak the same line, slots or not."""
        for slots in (None, {"item": "milk"}):
            spoken = [render(line, slots=slots) for line in MIDCHAIN_LINES[bucket]]
            assert len(spoken) == len(set(spoken))


class TestOpenerLines:
    """opener_lines resolves a bucket-and-intent pair to a cell, with an ``any`` fallback."""

    @pytest.mark.parametrize("intent", OPENER_INTENTS)
    def test_returns_the_requested_cell_when_present(self, intent: str) -> None:
        """A bucket that carries the requested cell returns that cell's lines."""
        bucket = next(b for b in OPENER_BUCKETS if intent in OPENER_LINES[b])
        assert opener_lines(bucket, intent) is OPENER_LINES[bucket][intent]

    def test_falls_back_to_any_for_a_missing_cell(self) -> None:
        """A bucket with no cell for the intent falls back to ``any``."""
        assert "control" not in OPENER_LINES["weather"]
        assert opener_lines("weather", "control") is OPENER_LINES["weather"]["any"]

    def test_falls_back_to_any_for_none(self) -> None:
        """A ``None`` intent always resolves to the bucket's ``any`` cell."""
        assert opener_lines("media", None) is OPENER_LINES["media"]["any"]

    def test_unknown_bucket_raises_key_error(self) -> None:
        """An unknown bucket surfaces a KeyError."""
        with pytest.raises(KeyError):
            opener_lines("nope", None)


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
        """Repeated draws from a multi-variant cell never repeat back to back."""
        lines = OPENER_LINES["weather"]["any"]
        rng = random.Random(1234)
        last: Line | None = None
        for _ in range(50):
            current = select(lines, last=last, rng=rng)
            assert current is not last
            last = current

    def test_single_variant_returns_that_line(self) -> None:
        """A one-variant sequence always yields its only line."""
        line = Line("Only one line here.", "calm")
        assert select((line,), rng=random.Random(0)) is line

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
        lines = OPENER_LINES["lights"]["any"]
        assert _draw_sequence(lines, seed=99) == _draw_sequence(lines, seed=99)


class TestLinePool:
    """The stateful facade renders every context and tracks repeat state."""

    @pytest.mark.parametrize("bucket", OPENER_BUCKETS)
    def test_opener_returns_non_empty(self, bucket: str) -> None:
        """opener yields a non-empty spoken string per bucket."""
        assert LinePool(rng=random.Random(1)).opener(bucket)

    @pytest.mark.parametrize("intent", OPENER_INTENTS)
    def test_opener_with_intent_draws_from_that_cell(self, intent: str) -> None:
        """opener with an intent draws a line from that intent's cell."""
        bucket = next(b for b in OPENER_BUCKETS if intent in OPENER_LINES[b])
        result = LinePool(rng=random.Random(5)).opener(bucket, intent)
        assert result in {render(line) for line in OPENER_LINES[bucket][intent]}

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
            "Noting that down...",
        }

    def test_midchain_falls_back_without_slots(self) -> None:
        """midchain without slots resolves every variant to slot-free text."""
        pool = LinePool(rng=random.Random(3))
        assert pool.midchain("list") in {
            "Adding that in...",
            "Putting that on your list...",
            "Noting that down...",
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
        """DEFAULT_MOOD declares the expected mood for a context."""
        assert DEFAULT_MOOD[context] == expected

    def test_default_mood_covers_every_context(self) -> None:
        """DEFAULT_MOOD has an entry for every opener and mid-chain bucket."""
        assert all(f"opener:{bucket}" in DEFAULT_MOOD for bucket in OPENER_BUCKETS)
        assert all(f"midchain:{bucket}" in DEFAULT_MOOD for bucket in MIDCHAIN_BUCKETS)

    def test_opener_and_midchain_default_moods_do_not_collide(self) -> None:
        """Same-named opener and mid-chain contexts hold independent default moods."""
        assert "opener:weather" in DEFAULT_MOOD
        assert "midchain:weather" in DEFAULT_MOOD
        assert "weather" not in DEFAULT_MOOD

    def test_unknown_opener_bucket_raises_key_error(self) -> None:
        """An unknown opener bucket surfaces a KeyError."""
        with pytest.raises(KeyError):
            LinePool(rng=random.Random(1)).opener("nope")


class TestStripMoodTags:
    """The delivery-side strip removes ``[mood]`` tags from streamed answer text."""

    def test_whole_tag_is_removed(self) -> None:
        """A complete mood tag is dropped, the answer text around it kept."""
        assert (
            strip_mood_tags("[calm] Let me look into that.")
            == " Let me look into that."
        )

    @pytest.mark.parametrize("mood", sorted(MOODS))
    def test_every_declared_mood_is_stripped(self, mood: str) -> None:
        """Every mood in the fixed set is a strippable tag."""
        assert strip_mood_tags(f"[{mood}] text") == " text"

    def test_multiple_tags_are_all_removed(self) -> None:
        """Every mood tag in the string is removed, not just the first."""
        assert strip_mood_tags("[calm] hi [playful] there") == " hi  there"

    def test_non_mood_bracket_is_untouched(self) -> None:
        """A bracketed word outside MOODS is emitted verbatim."""
        assert strip_mood_tags("a [note] b") == "a [note] b"

    def test_bracket_that_cannot_be_a_mood_tag_is_untouched(self) -> None:
        """A ``[`` that cannot begin a mood tag is emitted normally."""
        assert strip_mood_tags("values [1,2] here") == "values [1,2] here"

    def test_unterminated_prefix_is_kept_when_the_stream_ends(self) -> None:
        """A trailing fragment that never completes is flushed verbatim."""
        assert strip_mood_tags("hello [cal") == "hello [cal"

    def test_split_tag_is_carried_across_chunks(self) -> None:
        """A tag split across deltas is held back, then stripped once complete."""
        emitted, carry = strip_mood_tags_streaming("I am [cal")
        assert emitted == "I am "
        assert carry == "[cal"

        emitted, carry = strip_mood_tags_streaming("m] here", carry)
        assert emitted == " here"
        assert carry == ""

    def test_split_prefix_can_become_a_non_mood_bracket(self) -> None:
        """A carried ``[`` that turns out not to be a mood tag is released verbatim."""
        emitted, carry = strip_mood_tags_streaming("[cal")
        assert emitted == ""
        assert carry == "[cal"

        emitted, carry = strip_mood_tags_streaming("mny] x", carry)
        assert emitted == "[calmny] x"
        assert carry == ""

    def test_lone_bracket_is_carried_not_dropped(self) -> None:
        """A bare ``[`` at a delta boundary could start a tag, so it is carried."""
        emitted, carry = strip_mood_tags_streaming("done [")
        assert emitted == "done "
        assert carry == "["

    def test_carry_prepended_to_next_chunk_strips_across_boundary(self) -> None:
        """A complete tag split across two deltas leaves no tag and no residue."""
        emitted_a, carry = strip_mood_tags_streaming("ok [play")
        emitted_b, carry = strip_mood_tags_streaming("ful] go", carry)
        assert emitted_a + emitted_b == "ok  go"
        assert carry == ""


def _draw_sequence(lines: tuple[Line, ...], *, seed: int) -> list[Line]:
    """Draw a representative run of picks from one seeded generator."""
    rng = random.Random(seed)
    last: Line | None = None
    drawn: list[Line] = []
    for _ in range(6):
        last = select(lines, last=last, rng=rng)
        drawn.append(last)
    return drawn
