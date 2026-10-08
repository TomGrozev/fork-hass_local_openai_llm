"""
Curated, mood-tagged check-in line pool + pure templating helpers.

The pool is *content*: fixed and shipped in the integration (not user-editable in v1). This
module is the only place check-in lines live, and everything here is deterministic given an
injected ``random.Random`` - no model, network, or I/O.

Structure:

- ``OPENER_LINES`` - 11 domain buckets, each 1-3 variants; every variant's text sits inside the
  ``60 <= n <= 66`` band, long enough to trip Home Assistant's 60-char TTS gate without
  overshooting into noticeable added latency.
- ``MIDCHAIN_LINES`` - 9 tool buckets, short trailing-ellipsis fragments, no length floor.
- ``WATCHDOG_LINES`` - short reassurances, no length floor.
- ``FAILURE_LINES`` - the failure/apology bucket; mood-tagged and 60+ chars.

Templating rules:

1. The mood tag comes first (``"[mood] text"``); the stored text itself carries no tag.
2. ``mood`` is drawn from the fixed ``MOODS`` set.
3. An opener is fixed text. A mid-chain fragment may instead hold a ``{name}`` blank to fill in
   from the tool's arguments; when that value isn't available, ``Line.plain`` gives the same
   line with the blank written out (e.g. ``"Adding {item}..."`` falls back to
   ``"Adding that in..."``).
4. Selection is random within a bucket with no immediate repeat, and each context carries a
   default mood (``DEFAULT_MOOD``) used when a line itself is mood-less.
5. ``render`` attaches the tag only when ``expression_tags`` is on; off, it returns the plain
   text, so a tag is never spoken literally.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: The fixed mood set. ``nervous`` is in the set but no v1 opener draws it.
MOODS: Final[frozenset[str]] = frozenset(
    {"calm", "playful", "sincere", "excited", "nervous"}
)

#: Inclusive text-length band for openers (the stored text carries no mood tag).
OPENER_LENGTH_MIN: Final = 60
OPENER_LENGTH_MAX: Final = 66

#: Minimum length for a failure/apology line.
FAILURE_LENGTH_MIN: Final = 60

#: Opener bucket keys.
OPENER_BUCKETS: Final[tuple[str, ...]] = (
    "weather",
    "lights",
    "climate",
    "media",
    "covers",
    "locks",
    "security",
    "timers",
    "scenes",
    "calendar",
    "generic",
)

#: Mid-chain tool-bucket keys.
MIDCHAIN_BUCKETS: Final[tuple[str, ...]] = (
    "query",
    "control",
    "weather",
    "climate",
    "media",
    "list",
    "calendar",
    "search",
    "generic",
)


@dataclass(frozen=True, slots=True)
class Line:
    """
    One pool entry: its spoken text plus optional mood and slot-free fallback.

    ``text`` may contain ``{name}`` blanks (mid-chain only). When it does, ``plain`` must hold
    the same line with those blanks written out, used when the values aren't available. ``mood``
    is ``None`` for a line that should inherit its context's ``DEFAULT_MOOD``.
    """

    text: str
    mood: str | None = None
    plain: str | None = None


#: Curated openers, keyed by bucket. Every ``text`` must render within the opener band.
OPENER_LINES: Final[dict[str, tuple[Line, ...]]] = {
    "weather": (
        Line(
            "Sure, let me take a look at the weather and see what's happening.", "calm"
        ),
        Line(
            "Ooh, weather - let me peek outside and see what the sky's up to.",
            "playful",
        ),
        Line(
            "Let me check what the weather's doing and pass it right along.", "sincere"
        ),
    ),
    "lights": (
        Line("Okay, let me check the lights and see what's switched on now.", "calm"),
        Line(
            "Let me see what the lights are up to and sort them out for you.", "playful"
        ),
        Line(
            "On it - let me get the lights sorted and see how they're doing.", "excited"
        ),
    ),
    "climate": (
        Line(
            "Let me check the temperature and see how things are set up here.", "calm"
        ),
        Line("Let me take a look at the thermostat and see where it's set.", "playful"),
        Line(
            "Let me check the climate settings and get things comfortable.", "sincere"
        ),
    ),
    "media": (
        Line(
            "Alright, let me cue something up and get that playing for you.", "playful"
        ),
        Line("Let me track that down and get it playing for you right away.", "calm"),
        Line("Ooh, let me line that up and get your music going right now.", "excited"),
    ),
    "covers": (
        Line(
            "Let me check the blinds and see where they're sitting right now.", "calm"
        ),
        Line(
            "Let me peek at the blinds and see what they're up to right now.", "playful"
        ),
        Line("Of course, let me look at the covers and see where they are.", "sincere"),
    ),
    "locks": (
        Line(
            "Of course, let me check the locks and make sure all's secure.", "sincere"
        ),
        Line("Let me have a look at the locks and see how things are sitting.", "calm"),
        Line(
            "Let me quickly check the doors and make sure they're all locked.",
            "playful",
        ),
    ),
    "security": (
        Line("Let me pull up the alarm status and see how things look now.", "calm"),
        Line(
            "Let me check the security now and make sure everything's fine.", "sincere"
        ),
        Line(
            "On it - let me grab the alarm details and check it over for you.",
            "excited",
        ),
    ),
    "timers": (
        Line(
            "Sure thing, let me get that timer going and counting down for you.",
            "playful",
        ),
        Line("Let me set that timer up and get it counting down for you now.", "calm"),
        Line(
            "Of course, let me get that timer started and keep track for you.",
            "sincere",
        ),
    ),
    "scenes": (
        Line(
            "Great choice, let me get that scene set up and running for you.", "excited"
        ),
        Line("Let me get that scene going and set things up over here now.", "calm"),
        Line(
            "Ooh, nice one - let me set that scene up and get it going now.", "playful"
        ),
    ),
    "calendar": (
        Line("Let me look at your calendar and see what's coming up for you.", "calm"),
        Line(
            "Let me have a look at your calendar and see what's on for you.", "sincere"
        ),
        Line(
            "Let me sneak a look at your calendar and see what's coming up.", "playful"
        ),
    ),
    "generic": (
        Line(
            "Okay, let me look into that for you and see what I can find out.", "calm"
        ),
        Line(
            "Let me think that one through properly and look into it for you.",
            "sincere",
        ),
        Line(
            "Hmm, let me dig into that one and see what I can turn up for you.",
            "playful",
        ),
    ),
}

#: Mid-chain fragments, keyed by tool bucket. Short, trailing-ellipsis, no length floor.
MIDCHAIN_LINES: Final[dict[str, tuple[Line, ...]]] = {
    "query": (
        Line("Let me check that...", "calm"),
        Line("Looking that up now...", "calm"),
    ),
    "control": (Line("Setting that up...", "calm"), Line("On it...", "playful")),
    "weather": (Line("Checking the forecast...", "calm"),),
    "climate": (Line("Looking at the thermostat...", "calm"),),
    "media": (Line("Cueing that up...", "playful"),),
    "list": (
        Line("Adding {item}...", "calm", plain="Adding that in..."),
        Line("Putting that on your list...", "calm"),
    ),
    "calendar": (Line("Checking your calendar...", "calm"),),
    "search": (Line("Looking that up...", "calm"),),
    "generic": (Line("One moment...", "calm"),),
}

#: Watchdog reassurances - short, no length floor.
WATCHDOG_LINES: Final[tuple[Line, ...]] = (
    Line("Still on it...", "calm"),
    Line("Almost there...", "sincere"),
    Line("Just a moment more...", "calm"),
)

#: Failure/apology bucket - mood-tagged, every line 60+ chars.
FAILURE_LINES: Final[tuple[Line, ...]] = (
    Line(
        "Sorry, I couldn't reach the assistant just now - please try that again.",
        "sincere",
    ),
    Line(
        "Apologies, something went wrong on my end there. Do give that another go.",
        "calm",
    ),
)

#: Per-context default mood, applied to any line that carries no mood of its own. Keyed by the
#: same context names the pool tracks - ``opener:<bucket>`` / ``midchain:<bucket>`` /
#: ``watchdog`` / ``failure`` - so an opener and a mid-chain bucket of the same name stay
#: independent.
DEFAULT_MOOD: Final[dict[str, str]] = {
    **dict.fromkeys((f"opener:{bucket}" for bucket in OPENER_BUCKETS), "calm"),
    **dict.fromkeys((f"midchain:{bucket}" for bucket in MIDCHAIN_BUCKETS), "calm"),
    "watchdog": "calm",
    "failure": "sincere",
}


def render(
    line: Line,
    *,
    slots: Mapping[str, str] | None = None,
    default_mood: str | None = None,
    expression_tags: bool = False,
) -> str:
    """
    Return the spoken text for ``line``, filling ``slots`` where possible.

    The stored pool text carries no mood tag; the tag is attached here, from ``line.mood`` or
    falling back to ``default_mood``. With ``expression_tags`` on the result is
    ``"[mood] text"``; off it is the plain text, so nothing is ever spoken literally.
    """
    text = line.text
    if line.plain is not None:
        text = line.plain
        if slots is not None:
            try:
                text = line.text.format(**slots)
            except (IndexError, KeyError):
                text = line.plain

    mood = line.mood if line.mood is not None else default_mood
    if expression_tags and mood is not None:
        return f"[{mood}] {text}"
    return text


def select(
    lines: Sequence[Line],
    *,
    last: Line | None = None,
    rng: random.Random | None = None,
) -> Line:
    """Pick a line at random, avoiding an immediate repeat of ``last``."""
    if not lines:
        msg = "select() received an empty sequence"
        raise ValueError(msg)

    candidates = [line for line in lines if line is not last]
    if not candidates:
        candidates = list(lines)

    chooser = rng if rng is not None else random.Random()  # noqa: S311
    return chooser.choice(candidates)


class LinePool:
    """
    Stateful facade over the pool: renders lines and tracks no-immediate-repeat state.

    One instance per conversation agent. ``rng`` is injectable for deterministic tests.
    """

    def __init__(self, *, rng: random.Random | None = None) -> None:
        """Create a pool backed by the shipped curated content."""
        self._rng = rng if rng is not None else random.Random()  # noqa: S311
        self._last: dict[str, Line] = {}

    def _pick(self, key: str, lines: Sequence[Line]) -> Line:
        """Draw the next line for ``key``, avoiding an immediate repeat."""
        line = select(lines, last=self._last.get(key), rng=self._rng)
        self._last[key] = line
        return line

    def opener(self, bucket: str, *, expression_tags: bool = False) -> str:
        """Render an opener line for ``bucket``."""
        key = f"opener:{bucket}"
        line = self._pick(key, OPENER_LINES[bucket])
        return render(
            line,
            default_mood=DEFAULT_MOOD[key],
            expression_tags=expression_tags,
        )

    def midchain(
        self,
        bucket: str,
        *,
        slots: Mapping[str, str] | None = None,
        expression_tags: bool = False,
    ) -> str:
        """Render a mid-chain fragment for ``bucket``, filling ``slots`` when possible."""
        line = self._pick(f"midchain:{bucket}", MIDCHAIN_LINES[bucket])
        return render(
            line,
            slots=slots,
            default_mood=DEFAULT_MOOD[f"midchain:{bucket}"],
            expression_tags=expression_tags,
        )

    def watchdog(self, *, expression_tags: bool = False) -> str:
        """Render a watchdog reassurance."""
        line = self._pick("watchdog", WATCHDOG_LINES)
        return render(
            line,
            default_mood=DEFAULT_MOOD["watchdog"],
            expression_tags=expression_tags,
        )

    def failure(self, *, expression_tags: bool = False) -> str:
        """Render a failure/apology line."""
        line = self._pick("failure", FAILURE_LINES)
        return render(
            line,
            default_mood=DEFAULT_MOOD["failure"],
            expression_tags=expression_tags,
        )

    def default_mood(self, context: str) -> str:
        """Return the default mood for ``context`` (an ``opener:``/``midchain:``-prefixed key)."""
        return DEFAULT_MOOD[context]
