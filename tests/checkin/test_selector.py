"""Unit tests for the opener contextual selector."""

from __future__ import annotations

import pytest

from custom_components.local_openai.checkin.pool import MIDCHAIN_BUCKETS, OPENER_INTENTS
from custom_components.local_openai.checkin.selector import (
    _KEYWORD_INDEX,
    DOMAIN_BUCKETS,
    ExposedEntity,
    INTENT_KEYWORDS,
    TOOL_BUCKETS,
    _exposed_name_index,
    _normalize,
    bucket_for_tool,
    select_opener_bucket,
    select_opener_intent,
)


class TestNormalize:
    """Normalization lower-cases, strips punctuation, and tokenizes."""

    def test_punctuation_is_dropped_and_apostrophes_join_words(self) -> None:
        """Apostrophes vanish, gluing their halves into one token."""
        assert _normalize("What's the Weather?!").tokens == (
            "whats",
            "the",
            "weather",
        )

    def test_whitespace_runs_collapse_to_single_spaces(self) -> None:
        """Whitespace of any width or kind collapses between tokens."""
        result = _normalize("  turn   on\n the\t light ")
        assert result.tokens == ("turn", "on", "the", "light")

    def test_empty_input_normalizes_to_nothing(self) -> None:
        """An empty utterance yields no text and no tokens."""
        result = _normalize("")
        assert result.tokens == ()
        assert result.text == ""

    def test_punctuation_only_input_normalizes_to_nothing(self) -> None:
        """An utterance of only punctuation yields no text and no tokens."""
        result = _normalize("?!.,")
        assert result.tokens == ()
        assert result.text == ""

    def test_text_field_is_the_space_joined_tokens(self) -> None:
        """The text field mirrors the token tuple joined by single spaces."""
        result = _normalize("Turn OFF the Kitchen Lights")
        assert result.text == " ".join(result.tokens)
        assert result.text == "turn off the kitchen lights"


class TestExposedNameIndex:
    """The exposed-name index maps a name's tokens to its HA domain."""

    def test_friendly_name_maps_its_tokens_to_the_domain(self) -> None:
        """A single friendly name is keyed by its token tuple."""
        index = _exposed_name_index((ExposedEntity("light", ("Kitchen Light",)),))
        assert index[("kitchen", "light")] == "light"

    def test_every_alias_is_indexed(self) -> None:
        """Several aliases for one entity each get an entry."""
        entity = ExposedEntity("media_player", ("Living Room Speaker", "Lounge HiFi"))
        index = _exposed_name_index((entity,))
        assert index[("living", "room", "speaker")] == "media_player"
        assert index[("lounge", "hifi")] == "media_player"

    def test_name_without_alphanumerics_is_skipped(self) -> None:
        """A name that normalizes to nothing contributes no entry."""
        assert _exposed_name_index((ExposedEntity("light", ("!!!",)),)) == {}

    def test_unlisted_domain_maps_to_its_raw_domain(self) -> None:
        """A domain outside the table still indexes as its bare domain string."""
        index = _exposed_name_index((ExposedEntity("switch", ("Coffee Machine",)),))
        assert index[("coffee", "machine")] == "switch"

    def test_equal_exposed_tuples_hit_the_cache(self) -> None:
        """Two calls with equal tuples return the very same cached mapping."""
        exposed = (ExposedEntity("light", ("Kitchen Light",)),)
        assert _exposed_name_index(tuple(exposed)) is _exposed_name_index(
            tuple(exposed)
        )


class TestKeywordIndex:
    """The phrase index is precomputed at import from the keyword table."""

    def test_index_is_populated(self) -> None:
        """The module precompute produced a non-empty index."""
        assert _KEYWORD_INDEX

    def test_single_word_phrase_maps_to_its_bucket(self) -> None:
        """A lone keyword is keyed by its one-token tuple."""
        assert "weather" in _KEYWORD_INDEX[("weather",)]

    def test_multi_word_phrase_is_keyed_by_its_token_tuple(self) -> None:
        """A spaced phrase is keyed by its consecutive tokens."""
        assert "climate" in _KEYWORD_INDEX[("air", "con")]


class TestKeywordSelection:
    """A confident keyword hit selects its bucket."""

    def test_weather_utterance_picks_weather(self) -> None:
        """A weather question lands in the weather bucket."""
        assert select_opener_bucket("what's the weather like", ()) == "weather"

    def test_lights_utterance_picks_lights(self) -> None:
        """A lights command lands in the lights bucket."""
        assert select_opener_bucket("turn off the kitchen lights", ()) == "lights"

    def test_media_utterance_picks_media(self) -> None:
        """A music request lands in the media bucket."""
        assert select_opener_bucket("play some jazz", ()) == "media"

    def test_summed_hits_beat_an_earlier_single_hit(self) -> None:
        """Two hits for one bucket outscore an equal-length single hit earlier in another."""
        assert select_opener_bucket("blind lamp light", ()) == "lights"


class TestWordBoundaryMatching:
    """Keywords match whole tokens only, never as substrings."""

    def test_warm_is_climate_and_not_security(self) -> None:
        """`warm` is a climate keyword and must not hit `arm`."""
        assert select_opener_bucket("warm", ()) == "climate"

    @pytest.mark.parametrize(
        "utterance",
        [
            pytest.param("delight", id="delight-contains-light"),
            pytest.param("delightful", id="delightful-contains-light"),
            pytest.param("discover", id="discover-contains-cover"),
            pytest.param("cheat", id="cheat-contains-heat"),
            pytest.param("prevent", id="prevent-contains-event"),
        ],
    )
    def test_word_containing_a_keyword_does_not_match_it(self, utterance: str) -> None:
        """A word that merely contains a keyword must not select that keyword's bucket."""
        assert select_opener_bucket(utterance, ()) == "generic"


class TestMatchGate:
    """A bucket must clear the confidence gate, else generic wins."""

    def test_no_signal_falls_back_to_generic(self) -> None:
        """No keyword and no exposed name yields the generic bucket."""
        assert select_opener_bucket("hello there friend", ()) == "generic"

    def test_single_word_boundary_hit_clears_the_gate(self) -> None:
        """One whole-token keyword match is enough to win a bucket."""
        assert select_opener_bucket("jazz", ()) == "media"

    def test_punctuation_only_utterance_falls_back_to_generic(self) -> None:
        """An utterance that normalizes to nothing yields generic."""
        assert select_opener_bucket("?!", ()) == "generic"


class TestEntitySignal:
    """An exposed entity name is the high-precision signal."""

    def test_exposed_name_hit_selects_its_bucket(self) -> None:
        """A name match maps through the domain table to its bucket."""
        exposed = (ExposedEntity("light", ("Kitchen Light",)),)
        assert select_opener_bucket("what about the kitchen light", exposed) == "lights"

    @pytest.mark.parametrize(("domain", "bucket"), sorted(DOMAIN_BUCKETS.items()))
    def test_domain_maps_to_its_declared_bucket(self, domain: str, bucket: str) -> None:
        """Each tabled domain resolves to its declared opener bucket."""
        exposed = (ExposedEntity(domain, ("zzq",)),)
        assert select_opener_bucket("zzq", exposed) == bucket

    def test_unlisted_domain_maps_to_generic(self) -> None:
        """A domain outside the table resolves to the generic bucket."""
        exposed = (ExposedEntity("switch", ("zzq",)),)
        assert select_opener_bucket("zzq", exposed) == "generic"

    def test_entity_hit_outweighs_a_keyword_hit_elsewhere(self) -> None:
        """A bare name hit (W_ENTITY) wins alone against another bucket's keyword."""
        exposed = (ExposedEntity("media_player", ("Party Speaker",)),)
        assert (
            select_opener_bucket("party speaker, turn on the light", exposed) == "media"
        )


class TestTieBreaks:
    """Equal scores fall through position, phrase length, then fixed order."""

    def test_earliest_position_wins_a_score_tie(self) -> None:
        """Two equal-length keyword hits tie on score; the earliest wins."""
        assert select_opener_bucket("light blind", ()) == "lights"

    def test_earliest_position_tie_flips_with_order(self) -> None:
        """Swapping the two equal-length hits flips the winner, so position decides."""
        assert select_opener_bucket("blind light", ()) == "covers"

    def test_longer_phrase_wins_at_the_same_position(self) -> None:
        """At the same earliest position, the longer matched name wins."""
        exposed = (
            ExposedEntity("lock", ("front",)),
            ExposedEntity("media_player", ("front speaker",)),
        )
        assert select_opener_bucket("front speaker", exposed) == "media"

    def test_declared_order_wins_a_full_tie(self) -> None:
        """Two equal-length name hits tie on score and length; the earlier lock name wins."""
        exposed = (
            ExposedEntity("lock", ("porch",)),
            ExposedEntity("media_player", ("telly",)),
        )
        assert select_opener_bucket("porch telly", exposed) == "locks"


class TestBucketForTool:
    """bucket_for_tool maps an HA tool name to its mid-chain fragment bucket."""

    @pytest.mark.parametrize(("tool_name", "bucket"), sorted(TOOL_BUCKETS.items()))
    def test_mapped_name_returns_its_bucket(self, tool_name: str, bucket: str) -> None:
        """Each tabled tool name resolves to its declared mid-chain bucket."""
        assert bucket_for_tool(tool_name) == bucket

    def test_unknown_name_falls_back_to_generic(self) -> None:
        """A tool name outside the table resolves to the generic bucket."""
        assert bucket_for_tool("HassNeverHeardOfIt") == "generic"

    def test_namespaced_name_matches_on_its_base_name(self) -> None:
        """A merged-API prefix is stripped before lookup."""
        assert bucket_for_tool("search-services__search_web") == "search"

    def test_unknown_namespaced_name_falls_back_to_generic(self) -> None:
        """An unrecognised base name under a namespace still yields generic."""
        assert bucket_for_tool("search-services__never_heard_of_it") == "generic"

    def test_every_bucket_is_a_midchain_bucket(self) -> None:
        """The table only ever names buckets the pool can render."""
        assert set(TOOL_BUCKETS.values()) <= set(MIDCHAIN_BUCKETS)


class TestVolumeTopic:
    """Volume is its own topic, split out of media."""

    def test_set_volume_utterance_picks_volume(self) -> None:
        """A volume command lands in the volume bucket, not media."""
        assert select_opener_bucket("set your volume to 3", ()) == "volume"

    def test_volume_question_picks_volume(self) -> None:
        """A volume question lands in the volume bucket."""
        assert select_opener_bucket("what's the volume", ()) == "volume"

    @pytest.mark.parametrize("word", ["louder", "quieter", "softer", "mute", "unmute"])
    def test_volume_words_pick_volume(self, word: str) -> None:
        """Each volume keyword resolves to the volume bucket."""
        assert select_opener_bucket(word, ()) == "volume"

    def test_music_utterance_still_picks_media(self) -> None:
        """Removing volume from media left the remaining music keywords on media."""
        assert select_opener_bucket("play some jazz", ()) == "media"


class TestIntentKeywords:
    """The intent table is the single source for the intents the pool can render."""

    def test_keys_are_exactly_the_pool_intents(self) -> None:
        """INTENT_KEYWORDS declares exactly ``pool.OPENER_INTENTS``."""
        assert set(INTENT_KEYWORDS) == set(OPENER_INTENTS)


class TestIntentSelection:
    """select_opener_intent reads what the user wants done, not just the topic."""

    @pytest.mark.parametrize(
        ("utterance", "intent"),
        [
            pytest.param("set your volume to 3", "control", id="set-volume"),
            pytest.param("what's the volume", "query", id="volume-question"),
            pytest.param("play some jazz", "play", id="play-music"),
            pytest.param("pause the music", "control", id="pause"),
            pytest.param("is the front door locked", "query", id="locked-question"),
            pytest.param("can you turn off the kitchen lights", "control", id="polite"),
            pytest.param("turn it up", "control", id="turn-it-up"),
            pytest.param("could you please set the volume", "control", id="please-set"),
            pytest.param("whats the weather like", "query", id="whats-weather"),
            pytest.param("tell me the time", "query", id="tell-me"),
            pytest.param("queue up some music", "play", id="queue"),
            pytest.param("put on the radio", "play", id="put-on"),
        ],
    )
    def test_utterance_picks_its_intent(self, utterance: str, intent: str) -> None:
        """The earliest whole-token intent word decides."""
        assert select_opener_intent(utterance) == intent

    def test_politeness_does_not_read_as_query(self) -> None:
        """Request framing is not an intent word, so the verb decides."""
        assert select_opener_intent("would you turn on the light") == "control"

    def test_no_intent_word_is_none(self) -> None:
        """An utterance with no intent word yields ``None``."""
        assert select_opener_intent("hello there") is None

    def test_empty_utterance_is_none(self) -> None:
        """An utterance that normalizes to nothing yields ``None``."""
        assert select_opener_intent("?!") is None

    def test_word_containing_a_keyword_does_not_match_it(self) -> None:
        """A word that merely contains an intent word must not match it."""
        assert select_opener_intent("player") is None

    def test_earliest_hit_wins_over_a_later_longer_phrase(self) -> None:
        """The earliest whole-token hit wins even when a longer phrase follows."""
        assert select_opener_intent("turn it on and play music") == "control"
