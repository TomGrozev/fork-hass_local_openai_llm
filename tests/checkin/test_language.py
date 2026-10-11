"""Unit tests for the pipeline-language check that guards the check-in feature."""

from __future__ import annotations

import pytest

from custom_components.local_openai.checkin.language import is_english


class TestIsEnglish:
    """``is_english`` reads the primary subtag of a pipeline language."""

    @pytest.mark.parametrize(
        "language",
        [None, "en", "en-US", "en_GB", "EN", "en-us"],
    )
    def test_english(self, language: str | None) -> None:
        """A missing language defaults to English; ``en`` and its regional tags pass."""
        assert is_english(language) is True

    @pytest.mark.parametrize("language", ["fr", "de", "es", "fr-FR", "zh-Hans-CN"])
    def test_not_english(self, language: str) -> None:
        """Any non-``en`` primary subtag is suppressed."""
        assert is_english(language) is False
