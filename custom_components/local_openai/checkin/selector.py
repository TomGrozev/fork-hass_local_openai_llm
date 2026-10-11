"""
Opener contextual selector: pick one of the 11 opener buckets from data in hand.

Pure - no model, network, or I/O, and sub-millisecond. Two local signals, both computed from
the utterance before the LLM runs:

1. **Exposed-entity-name hit** (high precision). Scan the utterance for the friendly name or
   alias of any currently-exposed Assist entity; map that entity's HA domain to an opener bucket
   via ``DOMAIN_BUCKETS``. Area names are *where*, not *what*, so they are ignored for bucket
   choice in v1.
2. **Keyword/phrase hit** (best effort). Curated per-bucket keywords (``BUCKET_KEYWORDS``)
   matched at **word boundaries only** - never a raw substring - and phrase-weighted by the
   character length of the matched phrase. Multi-word entries match only as consecutive tokens.

The score is ``W_ENTITY * (#entity-name hits) + sum(len(phrase) for each keyword match)`` in
characters; the bucket with the highest score wins; if nothing clears ``MATCH_MIN`` the result is
the ``generic`` bucket. Equal scores break by earliest hit position, then longest single matched
phrase, then ``TIEBREAK_ORDER``.

Everything except the ranking decision - the tables, normalization, and the precomputed index
builders below - is mechanical; ``_select_bucket`` holds the matching + scoring decision.

It also resolves an HA tool name to a mid-chain fragment bucket via ``TOOL_BUCKETS``, so both
signal-to-bucket resolutions (opener signal and tool name) live in this one module.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

#: Weight of one exposed-entity-name hit relative to a phrase-length keyword score.
W_ENTITY: Final = 1000

#: Confidence gate: a winning bucket must score at least this, else ``generic`` is returned.
MATCH_MIN: Final = 1

#: Fixed tie-break order for equal-scoring buckets. ``generic`` is absent: it is only ever the
#: fallback, never a tie-break candidate.
TIEBREAK_ORDER: Final[tuple[str, ...]] = (
    "locks",
    "security",
    "covers",
    "climate",
    "lights",
    "media",
    "scenes",
    "timers",
    "calendar",
    "weather",
)

#: ``TIEBREAK_ORDER`` inverted once at import, so the selector never rebuilds it per call.
TIEBREAK_RANK: Final[Mapping[str, int]] = {
    bucket: index for index, bucket in enumerate(TIEBREAK_ORDER)
}

#: HA entity domain to opener bucket. Any domain not listed maps to ``generic``.
DOMAIN_BUCKETS: Final[Mapping[str, str]] = {
    "weather": "weather",
    "light": "lights",
    "climate": "climate",
    "humidifier": "climate",
    "media_player": "media",
    "cover": "covers",
    "lock": "locks",
    "alarm_control_panel": "security",
    "camera": "security",
    "timer": "timers",
    "scene": "scenes",
    "calendar": "calendar",
}

#: Curated keyword/phrase table - content, not algorithm, so new domains or phrases append here
#: without touching the selector. Boundary-safe and deduped within each bucket.
BUCKET_KEYWORDS: Final[Mapping[str, tuple[str, ...]]] = {
    "weather": (
        "weather",
        "rain",
        "raining",
        "forecast",
        "sunny",
        "snow",
        "snowing",
        "wind",
        "windy",
        "umbrella",
        "cloudy",
        "storm",
    ),
    "lights": (
        "light",
        "lights",
        "lamp",
        "lamps",
        "brightness",
        "brighten",
        "brighter",
        "dim",
        "dimmer",
    ),
    "climate": (
        "thermostat",
        "heating",
        "heat",
        "cooling",
        "cool",
        "warm",
        "warmer",
        "cooler",
        "air con",
        "air conditioning",
        "ac",
        "temperature",
        "degrees",
    ),
    "media": (
        "music",
        "play",
        "song",
        "songs",
        "playlist",
        "volume",
        "mute",
        "pause",
        "resume",
        "skip",
        "spotify",
        "tv",
        "television",
        "podcast",
        "radio",
        "jazz",
    ),
    "covers": (
        "blinds",
        "blind",
        "curtain",
        "curtains",
        "shade",
        "shades",
        "shutter",
        "shutters",
        "garage",
        "cover",
        "roller",
        "awning",
    ),
    "locks": (
        "lock",
        "locks",
        "locked",
        "unlock",
        "unlocked",
        "deadbolt",
        "latch",
        "door",
        "doors",
    ),
    "security": (
        "alarm",
        "arm",
        "armed",
        "disarm",
        "disarmed",
        "security",
        "camera",
        "cameras",
        "motion",
        "siren",
        "intrusion",
    ),
    "timers": ("timer", "timers", "remind", "reminder", "countdown", "stopwatch"),
    "scenes": (
        "scene",
        "scenes",
        "goodnight",
        "good night",
        "good morning",
        "bedtime",
        "movie night",
        "night mode",
        "routine",
        "morning routine",
    ),
    "calendar": (
        "calendar",
        "schedule",
        "appointment",
        "appointments",
        "event",
        "events",
        "meeting",
        "meetings",
        "agenda",
        "what's on",
        "whats on",
        "my day",
        "plans",
    ),
}

#: HA tool name -> mid-chain fragment bucket. Content, not algorithm; any name not listed
#: maps to ``generic``. Values must be members of ``pool.MIDCHAIN_BUCKETS``.
TOOL_BUCKETS: Final[Mapping[str, str]] = {
    "HassGetState": "query",
    "GetLiveContext": "query",
    "HassGetWeather": "weather",
    "HassTurnOn": "control",
    "HassTurnOff": "control",
    "HassLightSet": "control",
    "HassSetPosition": "control",
    "HassClimateSetTemperature": "climate",
    "HassClimateGetTemperature": "climate",
    "HassMediaPlay": "media",
    "HassMediaPause": "media",
    "HassSetVolume": "media",
    "HassListAddItem": "list",
    "HassCalendarGetEvents": "calendar",
    "web_search": "search",
}


def bucket_for_tool(tool_name: str) -> str:
    """Map an HA tool name to its mid-chain fragment bucket (unknown -> ``"generic"``)."""
    return TOOL_BUCKETS.get(tool_name, "generic")


@dataclass(frozen=True, slots=True)
class ExposedEntity:
    """
    The minimum an exposed Assist entity must give the selector.

    ``domain`` is the HA entity domain (the entity_id prefix, e.g. ``light``); ``names`` holds
    its friendly name and aliases. The caller adapts this from HA's expose list, so the selector
    itself stays I/O-free.
    """

    domain: str
    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Normalized:
    """A normalized utterance: its whitespace-collapsed text plus its token list."""

    text: str
    tokens: tuple[str, ...]


def _normalize(text: str) -> _Normalized:
    """
    Lower-case, strip punctuation, collapse whitespace, and tokenize ``text``.

    Returns the token list and the space-joined string, so a caller can look up both single
    words and consecutive-token phrases.
    """
    cleaned = "".join(
        char if char.isalnum() or char.isspace() else "" for char in text.lower()
    )
    tokens = tuple(cleaned.split())
    return _Normalized(text=" ".join(tokens), tokens=tokens)


def _phrase_length(phrase: tuple[str, ...]) -> int:
    """
    Return the matched phrase's length in characters.

    The phrase's token lengths plus one joining space each. Equal to ``len(" ".join(phrase))``
    without the allocation.
    """
    return sum(map(len, phrase)) + len(phrase) - 1


def _window_starts(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> Iterator[int]:
    """Yield every index where ``phrase`` occurs in ``tokens`` as consecutive whole tokens."""
    span = len(phrase)
    for start in range(len(tokens) - span + 1):
        if tokens[start : start + span] == phrase:
            yield start


def _build_keyword_index() -> Mapping[tuple[str, ...], tuple[str, ...]]:
    """
    Build the phrase index once at import: normalized phrase tokens to the buckets holding it.

    Keys are normalized token tuples so lookup is word-boundary by construction; values list each
    bucket that declares the phrase, once.
    """
    index: dict[tuple[str, ...], list[str]] = {}
    for bucket, phrases in BUCKET_KEYWORDS.items():
        for phrase in phrases:
            key = _normalize(phrase).tokens
            buckets = index.setdefault(key, [])
            if bucket not in buckets:
                buckets.append(bucket)
    return {key: tuple(buckets) for key, buckets in index.items()}


#: Precomputed at load; the phrase index never changes after import.
_KEYWORD_INDEX: Final[Mapping[tuple[str, ...], tuple[str, ...]]] = (
    _build_keyword_index()
)


@lru_cache(maxsize=64)
def _exposed_name_index(
    exposed: tuple[ExposedEntity, ...],
) -> Mapping[tuple[str, ...], str]:
    """
    Build (and cache) the exposed-name index for one exposed-entity set.

    Maps each normalized name's tokens to its HA domain. When two exposed entities share a
    normalized name, the later entity's domain wins (plain assignment, deterministic). Cached on
    the set, so a turn that re-selects against the same exposed entities pays for the build once.
    """
    index: dict[tuple[str, ...], str] = {}
    for entity in exposed:
        for name in entity.names:
            tokens = _normalize(name).tokens
            if tokens:
                index[tokens] = entity.domain
    return index


def _select_bucket(
    normalized: _Normalized,
    *,
    keyword_index: Mapping[tuple[str, ...], tuple[str, ...]],
    name_index: Mapping[tuple[str, ...], str],
) -> str:
    """
    Return the opener bucket for a normalized utterance.

    Two signals taken from data already in hand - no model, network or I/O:

    1. **Exposed entity name.** Slide a window of ``name_index`` tokens over the utterance; a
       matched name's HA domain maps through ``DOMAIN_BUCKETS`` (an unlisted domain falls back to
       ``generic``) and scores ``W_ENTITY``. The recorded length is the name's character length.
    2. **Keyword phrase.** Slide a window of ``keyword_index`` tokens; keys are token tuples, so a
       match is whole-token by construction (``("light",)`` never matches ``delight``) and scores
       the phrase's character length.

    Scores sum per bucket; the highest wins, and ``generic`` is returned when nothing clears
    ``MATCH_MIN``. Equal scores break by earliest hit position, then longest matched phrase, then
    ``TIEBREAK_RANK`` - which never lists ``generic``, so it sorts last.

    Worked examples (``W_ENTITY`` = 1000, ``MATCH_MIN`` = 1; no entities unless shown; every
    length is a character length - ``light`` = 5, ``weather`` = 7):

    - ``"what's the weather like"`` -> ``"weather"``; ``"delight"`` -> ``"generic"``, because
      ``light`` is not a whole token of anything in the utterance.
    - ``"light blind"`` -> ``"lights"`` and ``"blind light"`` -> ``"covers"``: two equal-length
      keyword hits, so position decides.
    - ``"what about the kitchen light"``, exposed ``("light", ("Kitchen Light",))`` ->
      ``"lights"``: the name hit (1000) plus the ``light`` keyword (5).
    - ``"party speaker, turn on the light"``, exposed ``("media_player", ("Party Speaker",))`` ->
      ``"media"``: the lone name hit outweighs the ``lights`` keyword hit (5).
    - ``"front speaker"``, exposed lock ``("front",)`` and media_player ``("front speaker",)`` ->
      ``"media"``: both names hit at token 0, so the longer matched phrase wins.
    - ``"porch telly"``, exposed lock ``("porch",)`` and media_player ``("telly",)`` ->
      ``"locks"``: both names are 5 characters, hit at token 0 and score the same, so
      ``TIEBREAK_RANK`` decides and ``locks`` ranks first.
    """
    scores: dict[str, int] = {}
    positions: dict[str, int] = {}
    longest_length: dict[str, int] = {}

    def record(bucket: str, start: int, length: int, weight: int) -> None:
        """Fold one hit into the bucket's running score, earliest position and longest phrase."""
        scores[bucket] = scores.get(bucket, 0) + weight
        positions[bucket] = min(positions.get(bucket, start), start)
        longest_length[bucket] = max(longest_length.get(bucket, 0), length)

    tokens = normalized.tokens

    # Signal 1: exposed entity names, whose domain decides the bucket and whose weight is high.
    for name, domain in name_index.items():
        length = _phrase_length(name)
        for start in _window_starts(tokens, name):
            record(DOMAIN_BUCKETS.get(domain, "generic"), start, length, W_ENTITY)

    # Signal 2: keyword phrases, whole-token windows only, scored by phrase length in characters.
    for phrase, buckets in keyword_index.items():
        length = _phrase_length(phrase)
        for start in _window_starts(tokens, phrase):
            for bucket in buckets:
                record(bucket, start, length, length)

    if not scores:
        return "generic"

    best = max(scores.values())
    if best < MATCH_MIN:
        return "generic"

    return min(
        (bucket for bucket, score in scores.items() if score == best),
        key=lambda bucket: (
            positions[bucket],
            -longest_length[bucket],
            TIEBREAK_RANK.get(bucket, len(TIEBREAK_RANK)),
        ),
    )


def select_opener_bucket(
    utterance: str, exposed_entities: Sequence[ExposedEntity]
) -> str:
    """
    Return the opener bucket for ``utterance`` given the currently-exposed entities.

    Pure: no model, network, or I/O. Falls back to ``"generic"`` when neither signal clears
    ``MATCH_MIN``.
    """
    normalized = _normalize(utterance)
    name_index = _exposed_name_index(tuple(exposed_entities))
    return _select_bucket(
        normalized, keyword_index=_KEYWORD_INDEX, name_index=name_index
    )
