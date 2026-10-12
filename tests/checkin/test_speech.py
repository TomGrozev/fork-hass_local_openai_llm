"""Unit tests for the speech-duration estimator."""

from __future__ import annotations

import pytest

from custom_components.local_openai.checkin.speech import estimate_speech_s


class TestEstimateSpeechS:
    """estimate_speech_s is word count over the given rate, on the post-strip spoken text."""

    def test_duration_scales_with_the_speaking_rate(self) -> None:
        """Ten words take four seconds at 150 wpm and two at twice the rate."""
        text = "one two three four five six seven eight nine ten"
        assert estimate_speech_s(text, words_per_minute=150) == pytest.approx(4.0)
        assert estimate_speech_s(text, words_per_minute=300) == pytest.approx(2.0)

    def test_strips_mood_tags_before_counting(self) -> None:
        """A ``[mood]`` tag is not spoken, so it must not count toward the estimate."""
        assert estimate_speech_s(
            "[calm] one two", words_per_minute=150
        ) == pytest.approx(estimate_speech_s("one two", words_per_minute=150))

    def test_extra_whitespace_does_not_count_as_words(self) -> None:
        """Collapsing whitespace leaves the word count unchanged."""
        assert estimate_speech_s(
            "  one   two\nthree  ", words_per_minute=150
        ) == pytest.approx(estimate_speech_s("one two three", words_per_minute=150))

    def test_tag_only_text_estimates_to_zero(self) -> None:
        """Nothing spoken estimates to no time."""
        assert estimate_speech_s("[calm]", words_per_minute=150) == 0.0

    def test_empty_text_estimates_to_zero(self) -> None:
        """An empty delta queues no audio."""
        assert estimate_speech_s("", words_per_minute=150) == 0.0
