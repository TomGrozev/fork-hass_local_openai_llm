"""
Opener contextual selector: pick one of the 12 opener buckets plus an intent from data in hand.

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

Alongside the topic, a third local signal - a pure keyword **intent axis** (``INTENT_KEYWORDS``,
:func:`select_opener_intent`) - says *what the user wants done*: ``query`` (ask), ``control``
(change), or ``play`` (start media). It reuses the same whole-token windowing and reports the
earliest hit, so request framing ("can you", "please") never reads as ``query``. No hit is
``None``, and the caller falls back to the bucket's ``any`` opener cell.

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
    "volume",
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
    "volume": (
        "volume",
        "louder",
        "quieter",
        "softer",
        "mute",
        "unmute",
        "muted",
        "unmuted",
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

#: Curated intent keyword/phrase table - content, not algorithm. Keys are exactly the members of
#: ``pool.OPENER_INTENTS``. Deliberately excludes request framing ("can you", "could you",
#: "please", "would you"), so a polite command reads as its verb ("turn", "set"), not ``query``.
INTENT_KEYWORDS: Final[Mapping[str, tuple[str, ...]]] = {
    "query": (
        "what",
        "whats",
        "is",
        "are",
        "how",
        "when",
        "which",
        "who",
        "did",
        "does",
        "do",
        "tell me",
        "check",
        "any",
    ),
    "control": (
        "turn",
        "switch",
        "set",
        "make",
        "change",
        "dim",
        "brighten",
        "open",
        "close",
        "lock",
        "unlock",
        "raise",
        "lower",
        "increase",
        "decrease",
        "arm",
        "disarm",
        "start",
        "stop",
        "cancel",
        "pause",
        "resume",
        "skip",
        "mute",
        "unmute",
        "add",
        "louder",
        "quieter",
    ),
    "play": (
        "play",
        "put on",
        "listen to",
        "queue",
    ),
}

#: Fixed order among intents for an exact position-and-length tie. The table shares no phrase
#: between intents, so this only makes a future duplicate resolve deterministically; it follows
#: ``pool.OPENER_INTENTS``.
INTENT_ORDER: Final[tuple[str, ...]] = ("query", "control", "play")

#: ``INTENT_ORDER`` inverted once at import, so the selector never rebuilds it per call.
INTENT_RANK: Final[Mapping[str, int]] = {
    intent: index for index, intent in enumerate(INTENT_ORDER)
}

#: HA tool name -> mid-chain fragment bucket. Content, not algorithm; any name not listed
#: maps to ``generic``. Values must be members of ``pool.MIDCHAIN_BUCKETS``.
#:
#: A merged LLM API namespaces each tool as ``{namespace}__{tool}`` (HA's
#: ``helpers/llm.py`` ``NamespacedTool``, namespace = slugified API name), so a live turn
#: can hand us ``search-services__search_web``. :func:`bucket_for_tool` strips that prefix
#: and matches the base name, so this table lists base names only. The ``llm_intents``
#: entries below are skye-harris' "Tools for Assist" integration.
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
    "HassMediaSearchAndPlay": "media",
    "HassMediaPause": "playback",
    "HassMediaUnpause": "playback",
    "HassMediaNext": "playback",
    "HassMediaPrevious": "playback",
    "HassSetVolume": "volume",
    "HassSetVolumeRelative": "volume",
    "HassMediaPlayerMute": "volume",
    "HassMediaPlayerUnmute": "volume",
    "HassListAddItem": "list",
    "HassCalendarGetEvents": "calendar",
    "web_search": "search",
    # skye-harris/llm_intents ("Tools for Assist") - base names of its registered tools.
    "search_web": "search",
    "search_wikipedia": "search",
    "find_places": "search",
    "get_route": "search",
    "search_youtube": "media",
    "play_video": "media",
    "get_weather_forecast": "weather",
    "calendar_day_info": "calendar",
    "get_device_history_context": "query",
    "calculate": "generic",
    "unit_convert": "generic",
}

#: Merged-API namespace separator in tool names (``{namespace}__{tool}``).
_NAMESPACE_SEP: Final = "__"


def bucket_for_tool(tool_name: str) -> str:
    """
    Map an HA tool name to its mid-chain fragment bucket (unknown -> ``"generic"``).

    A namespaced (merged-API) name matches on its base name, so
    ``search-services__search_web`` resolves like ``search_web``. A name without the
    separator is looked up whole.
    """
    base_name = tool_name.rpartition(_NAMESPACE_SEP)[2]
    return TOOL_BUCKETS.get(base_name, "generic")


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


def _build_intent_index() -> Mapping[tuple[str, ...], tuple[str, ...]]:
    """
    Build the intent phrase index once at import: normalized phrase tokens to their intents.

    Mirrors :func:`_build_keyword_index`; keys are token tuples so lookup is whole-token by
    construction, and values list each intent that declares the phrase, once.
    """
    index: dict[tuple[str, ...], list[str]] = {}
    for intent, phrases in INTENT_KEYWORDS.items():
        for phrase in phrases:
            key = _normalize(phrase).tokens
            intents = index.setdefault(key, [])
            if intent not in intents:
                intents.append(intent)
    return {key: tuple(intents) for key, intents in index.items()}


#: Precomputed at load; the intent phrase index never changes after import.
_INTENT_INDEX: Final[Mapping[tuple[str, ...], tuple[str, ...]]] = _build_intent_index()


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


def _select_intent(
    normalized: _Normalized, *, intent_index: Mapping[tuple[str, ...], tuple[str, ...]]
) -> str | None:
    """
    Return the opener intent for a normalized utterance, or ``None`` when nothing matches.

    Slides a window of ``intent_index`` tokens over the utterance; keys are token tuples, so a
    match is whole-token by construction (``("play",)`` never matches ``player``). Each hit is
    ranked by ``(earliest start, longer matched phrase, ``INTENT_RANK``)`` and the minimum wins:
    the **earliest whole-token hit** takes the intent, and at the same start position the
    **longer phrase** wins. No hit is ``None``.

    Worked examples (no entities, no I/O; every intent word is a whole token):

    - ``"can you turn off the kitchen lights"`` -> ``"control"``: request framing ("can you") is
      not an intent phrase, so ``turn`` at token 2 decides.
    - ``"what's the volume"`` -> ``"query"``: ``what's`` normalizes to ``whats``, a query word.
    - ``"set your volume to 3"`` -> ``"control"``: ``set`` at token 0.
    - ``"play some jazz"`` -> ``"play"``: ``play`` at token 0.
    - ``"pause the music"`` -> ``"control"``: ``pause`` at token 0.
    - ``"is the front door locked"`` -> ``"query"``: ``is`` at token 0.
    - ``"hello there"`` -> ``None``: no intent word appears.
    """
    tokens = normalized.tokens
    best: tuple[int, int, int] | None = None
    best_intent: str | None = None
    for phrase, intents in intent_index.items():
        length = _phrase_length(phrase)
        for start in _window_starts(tokens, phrase):
            for intent in intents:
                candidate = (start, -length, INTENT_RANK.get(intent, len(INTENT_RANK)))
                if best is None or candidate < best:
                    best = candidate
                    best_intent = intent
    return best_intent


def select_opener_intent(utterance: str) -> str | None:
    """
    Return the opener intent for ``utterance``: ``"query"``, ``"control"``, ``"play"`` or ``None``.

    Pure: no model, network, or I/O. The intent is a second, independent axis beside
    :func:`select_opener_bucket`; a ``None`` tells the caller to use the bucket's ``any`` cell.
    """
    return _select_intent(_normalize(utterance), intent_index=_INTENT_INDEX)
