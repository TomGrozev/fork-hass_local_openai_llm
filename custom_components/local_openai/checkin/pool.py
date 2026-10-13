"""
Curated, mood-tagged check-in line pool + pure templating helpers.

The pool is *content*: fixed and shipped in the integration (not user-editable in v1). This
module is the only place check-in lines live, and everything here is deterministic given an
injected ``random.Random`` - no model, network, or I/O.

Structure:

- ``OPENER_LINES`` - 12 domain buckets, each a map of *intent cells*. A cell is keyed by an
  intent from ``OPENER_INTENTS`` (``"query"`` / ``"control"`` / ``"play"``) or by ``"any"``, and
  holds the fixed opener lines for that bucket-and-intent. Every bucket has an ``"any"`` cell;
  the buckets that need it also carry ``"query"`` and/or ``"control"`` (``"media"`` adds
  ``"play"``). Every line's text sits inside the ``60 <= n <= 66`` band, long enough to trip
  Home Assistant's 60-char TTS gate without overshooting into noticeable added latency.
- ``MIDCHAIN_LINES`` - 11 tool buckets, short trailing-ellipsis fragments, no length floor.
- ``WATCHDOG_LINES`` - short reassurances, no length floor.
- ``FAILURE_LINES`` - the failure/apology bucket; mood-tagged and 60+ chars.

Templating rules:

1. The mood tag comes first (``"[mood] text"``); the stored text itself carries no tag.
2. ``mood`` is drawn from the fixed ``MOODS`` set.
3. An opener is fixed text. A mid-chain fragment may instead hold a ``{name}`` blank to fill in
   from the tool's arguments; when that value isn't available, ``Line.plain`` gives the same
   line with the blank written out (e.g. ``"Adding {item}..."`` falls back to
   ``"Adding that in..."``).
4. Opener cells fall back to the bucket's ``"any"`` cell when no intent is given or the bucket
   has no cell for that intent; :func:`opener_lines` is the single place that rule lives.
5. Selection is random within a bucket with no immediate repeat, and each context carries a
   default mood (``DEFAULT_MOOD``) used when a line itself is mood-less.
6. ``render`` attaches the tag only when ``expression_tags`` is on; off, it returns the plain
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

#: The full ``[mood]`` tags the strip helpers recognise (``"[calm]"``, ...).
_MOOD_TAGS: Final[frozenset[str]] = frozenset(f"[{mood}]" for mood in MOODS)

#: Inclusive text-length band for openers (the stored text carries no mood tag).
OPENER_LENGTH_MIN: Final = 60
OPENER_LENGTH_MAX: Final = 66

#: Minimum length for a failure/apology line.
FAILURE_LENGTH_MIN: Final = 60

#: Opener intent-axis keys. ``query`` sounds like finding out, ``control`` like adjusting,
#: ``play`` like getting media going. A bucket may carry a cell for any subset of these.
OPENER_INTENTS: Final[tuple[str, ...]] = ("query", "control", "play")

#: Opener bucket keys.
OPENER_BUCKETS: Final[tuple[str, ...]] = (
    "weather",
    "lights",
    "climate",
    "media",
    "volume",
    "covers",
    "locks",
    "security",
    "timers",
    "scenes",
    "calendar",
    "generic",
)

#: Mid-chain tool-bucket keys. ``playback`` covers transport (pause/resume/next/previous);
#: ``volume`` covers setting, nudging and muting the sound level.
MIDCHAIN_BUCKETS: Final[tuple[str, ...]] = (
    "query",
    "control",
    "weather",
    "climate",
    "media",
    "playback",
    "volume",
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


#: Curated openers, keyed by bucket then by intent cell. Every ``text`` must render within the
#: opener band. Cell keys are ``"any"`` plus any of :data:`OPENER_INTENTS`.
OPENER_LINES: Final[dict[str, dict[str, tuple[Line, ...]]]] = {
    "weather": {
        "any": (
            Line(
                "Let me poke my head outside, so to speak, and check the weather.",
                "playful",
            ),
            Line(
                "Ooh, let me see what the sky's been plotting while we slept.",
                "playful",
            ),
            Line(
                "One moment - I'll lean out and take a proper look at the sky.",
                "calm",
            ),
            Line(
                "The sky's the one in charge. Give me a sec to hear its verdict.",
                "sincere",
            ),
            Line(
                "I'll peer at the clouds and see whether they've got rain in mind.",
                "calm",
            ),
            Line(
                "Weather watch! Let me go and stare at the sky for a little bit.",
                "excited",
            ),
            Line(
                "Give me a moment while I read the clouds and see what's brewing.",
                "sincere",
            ),
            Line(
                "I'll check the great outdoors and tell you what it's doing out.",
                "excited",
            ),
        ),
    },
    "lights": {
        "any": (
            Line(
                "Let me have a proper look at the lights and see how they're set.",
                "calm",
            ),
            Line(
                "Ooh, let me check the lighting and see how it's behaving for you.",
                "playful",
            ),
            Line(
                "Right, I'll pop over and have a quiet word with all the lights.",
                "playful",
            ),
            Line(
                "One sec while I have a proper look at every light in the room.",
                "calm",
            ),
            Line(
                "Give me a moment while I see how the lights are feeling about it.",
                "sincere",
            ),
            Line(
                "Hmm, lights! Let me see what all the bulbs in the house are up to.",
                "excited",
            ),
        ),
        "query": (
            Line(
                "Let me check in with the lights and see what they're saying.",
                "calm",
            ),
            Line(
                "One moment - I'm going to ask the lights how they're feeling.",
                "playful",
            ),
            Line(
                "Give me a sec to read the room and see where the lights stand.",
                "sincere",
            ),
            Line(
                "Hmm, I'll take a look and see what the lights have to say to you.",
                "excited",
            ),
            Line(
                "Just a moment while I quiz the lights about the room we're in.",
                "calm",
            ),
            Line(
                "A quick peek at the lights and I'll tell you what I find there.",
                "playful",
            ),
        ),
        "control": (
            Line(
                "Let me get the lighting just how you like it, wherever you are.",
                "calm",
            ),
            Line(
                "Ooh, let me have a fiddle with the lights and lift the mood.",
                "playful",
            ),
            Line(
                "Right, I'll go and adjust the lights to suit the room nicely.",
                "sincere",
            ),
            Line(
                "Give me a sec to dial the lights in and get them just right.",
                "calm",
            ),
            Line(
                "Hmm, I'll tweak the lights and get them how you like them best.",
                "excited",
            ),
            Line(
                "Just a moment - I'm going to tune the lights for the whole room.",
                "playful",
            ),
        ),
    },
    "climate": {
        "any": (
            Line(
                "Let me have a word with the thermostat and see how things stand.",
                "calm",
            ),
            Line(
                "Ooh, let me check the heating and see what it's been playing at.",
                "playful",
            ),
            Line(
                "Right, I'll go and have a quiet little chat with the thermostat.",
                "playful",
            ),
            Line(
                "One moment while I look at the climate and see where we are.",
                "calm",
            ),
            Line(
                "Give me a sec to see how the house is feeling about the heat.",
                "sincere",
            ),
            Line(
                "Hmm, the thermostat and I need a private moment to talk shop.",
                "excited",
            ),
        ),
        "query": (
            Line(
                "Let me check the temperature and see how the house is doing.",
                "calm",
            ),
            Line(
                "One moment - I'll go and read the thermostat for its verdict.",
                "playful",
            ),
            Line(
                "Give me a sec to see just how warm the place has got inside.",
                "sincere",
            ),
            Line(
                "Hmm, I'll have a good look and see what the climate has to say.",
                "excited",
            ),
            Line(
                "Just a moment while I ask the thermostat how it has been set.",
                "calm",
            ),
            Line(
                "Hmm, the thermostat will know - I'll ask it and report back.",
                "playful",
            ),
        ),
        "control": (
            Line(
                "Let me get the climate just right and make the place comfortable.",
                "calm",
            ),
            Line(
                "Ooh, let me tweak the thermostat until the place feels just right.",
                "playful",
            ),
            Line(
                "Right, I'll adjust the heating and cooling to suit the room.",
                "sincere",
            ),
            Line(
                "Give me a sec to get the temperature to where you'd like it.",
                "excited",
            ),
            Line(
                "Hmm, I'll turn the thermostat a notch and set it just right.",
                "calm",
            ),
            Line(
                "Just a moment - I'm going to dial the climate in for comfort.",
                "playful",
            ),
        ),
    },
    "media": {
        "any": (
            Line(
                "Let me check what the media's been doing and take it from there.",
                "calm",
            ),
            Line(
                "Ooh, let me have a listen to what's playing and sort it out.",
                "playful",
            ),
            Line(
                "Right, I'll check the media and see where everything sits for you.",
                "sincere",
            ),
            Line(
                "One moment while I see what's playing and get it well in hand.",
                "excited",
            ),
            Line(
                "Give me a sec to see what the media's been doing all day long.",
                "calm",
            ),
            Line(
                "Hmm, I'll take a look at what's playing and take it from there.",
                "playful",
            ),
        ),
        "query": (
            Line(
                "Let me see what's playing and tell you what I can find there.",
                "calm",
            ),
            Line(
                "One moment - I'll check what's playing and come back to you.",
                "playful",
            ),
            Line(
                "Give me a sec to see what's playing, then I'll fill you in fully.",
                "sincere",
            ),
            Line(
                "Hmm, I'll have a listen and tell you just what's playing there.",
                "excited",
            ),
            Line(
                "Just a moment while I check just exactly what's playing for you.",
                "calm",
            ),
            Line(
                "I'll have a good listen and let you know just what's playing.",
                "playful",
            ),
        ),
        "control": (
            Line(
                "Let me take charge of the playback and see what it needs from me.",
                "calm",
            ),
            Line(
                "Ooh, playback duty! Let me grab the controls and handle that.",
                "playful",
            ),
            Line(
                "Right, I'll have a word with the player and sort out the playback.",
                "sincere",
            ),
            Line(
                "Give me a sec to get that playback back under control for you.",
                "excited",
            ),
            Line(
                "Hmm, remote in hand - let me deal with the playback as you asked.",
                "calm",
            ),
            Line(
                "Just a moment while I get the playback where you want it set.",
                "playful",
            ),
        ),
        "play": (
            Line(
                "Ooh, a request! Let me rouse the turntable and find that one.",
                "playful",
            ),
            Line(
                "Let me get that one playing for you - just give me a moment.",
                "calm",
            ),
            Line(
                "Right, I'll dig out that record and get it spinning along nicely.",
                "sincere",
            ),
            Line(
                "Give me a sec to cue that record and get it playing for you.",
                "excited",
            ),
            Line(
                "Hmm, let me dig that one out of the crate and set it spinning.",
                "playful",
            ),
            Line(
                "Just a moment - I'm going to find that one and set it playing.",
                "calm",
            ),
        ),
    },
    "volume": {
        "any": (
            Line(
                "Let me check the sound level and see where it's been left set.",
                "calm",
            ),
            Line(
                "Ooh, let me have a proper look at that volume dial of yours.",
                "playful",
            ),
            Line(
                "Right, I'll check the speakers and see how they've been set.",
                "sincere",
            ),
            Line(
                "One moment while I check the volume and see where things stand.",
                "calm",
            ),
            Line(
                "Give me a sec to look at the sound and see where it's been set.",
                "excited",
            ),
            Line(
                "Hmm, I'll check the volume myself and see how it's been doing.",
                "playful",
            ),
        ),
        "query": (
            Line(
                "Let me check the volume and see where it's been set for you.",
                "calm",
            ),
            Line(
                "One moment - I'll read the sound level and tell you what I find.",
                "playful",
            ),
            Line(
                "Give me a sec to check just how loud the whole room has become.",
                "sincere",
            ),
            Line(
                "Hmm, I'll have a look at the volume and let you know the score.",
                "excited",
            ),
            Line(
                "Just a moment while I check where the sound level is sitting.",
                "calm",
            ),
            Line(
                "Hmm, let me read the sound level and tell you where it sits.",
                "playful",
            ),
        ),
        "control": (
            Line(
                "Sound check! Let me find the sweet spot in that volume knob.",
                "playful",
            ),
            Line(
                "Let me have a tinker with the volume and get it to your taste.",
                "calm",
            ),
            Line(
                "Right, I'll adjust the sound level and set it for your ears.",
                "sincere",
            ),
            Line(
                "Give me a sec to dial the volume in to where you like it best.",
                "excited",
            ),
            Line(
                "Hmm, I'll nudge the volume until it suits the whole room nicely.",
                "playful",
            ),
            Line(
                "Just a moment - I'm going to get the sound level to taste for you.",
                "calm",
            ),
        ),
    },
    "covers": {
        "any": (
            Line(
                "Let me check the blinds and see where they have been parked.",
                "calm",
            ),
            Line(
                "Ooh, let me have a look at the covers and see how they're doing.",
                "playful",
            ),
            Line(
                "Right, I'll pop over and see how all the blinds are getting along.",
                "sincere",
            ),
            Line(
                "One moment while I check the covers and see where things stand.",
                "calm",
            ),
            Line(
                "Give me a sec to look at the blinds and see how they're sitting.",
                "excited",
            ),
            Line(
                "Hmm, the blinds and I need a quick chat about all the daylight.",
                "playful",
            ),
        ),
        "query": (
            Line(
                "Let me go and see where those blinds have been left sitting.",
                "calm",
            ),
            Line(
                "One moment - I'll go and read the blinds and tell you the news.",
                "playful",
            ),
            Line(
                "Give me a sec to see how the covers have been left set for you.",
                "sincere",
            ),
            Line(
                "Hmm, I'll take a look and see just where the blinds are sitting.",
                "excited",
            ),
            Line(
                "Just a moment while I check the covers for a straight answer.",
                "calm",
            ),
            Line(
                "Let me go and take a proper look at those blinds for myself.",
                "playful",
            ),
        ),
        "control": (
            Line(
                "Let me adjust the blinds and set them how you like them best.",
                "calm",
            ),
            Line(
                "Ooh, let me have a fiddle with the covers and set them right.",
                "playful",
            ),
            Line(
                "Right, I'll get the blinds where you want them in just a tick.",
                "sincere",
            ),
            Line(
                "Give me a sec to move the covers to suit the room you've got.",
                "excited",
            ),
            Line(
                "Hmm, I'll tweak the blinds and get them right for the whole room.",
                "calm",
            ),
            Line(
                "Just a moment - I'm going to set the covers how you like them.",
                "playful",
            ),
        ),
    },
    "locks": {
        "any": (
            Line(
                "Let me check in with the doors and see how they're all holding.",
                "calm",
            ),
            Line(
                "Ooh, let me rattle the locks and see how they've been holding.",
                "playful",
            ),
            Line(
                "Right, I'll go and have a proper word with the doors and locks.",
                "sincere",
            ),
            Line(
                "One moment while I check the locks and see how things lie there.",
                "calm",
            ),
            Line(
                "Give me a sec to look at the doors and see how they're doing.",
                "excited",
            ),
            Line(
                "Hmm, the locks and I are due a quiet word - one moment, please.",
                "playful",
            ),
        ),
        "query": (
            Line(
                "Safety first - let me go and check the doors and see how they are.",
                "calm",
            ),
            Line(
                "One moment - I'll go and check the locks and see how they hold.",
                "playful",
            ),
            Line(
                "Give me a sec to see how the doors and locks have been faring.",
                "sincere",
            ),
            Line(
                "Hmm, I'll take a look and see where all the locks are sitting.",
                "excited",
            ),
            Line(
                "Just a moment while I check the doors for a bit of peace of mind.",
                "calm",
            ),
            Line(
                "Right, I'll go and give every lock in the house a once-over.",
                "playful",
            ),
        ),
        "control": (
            Line(
                "Let me take care of the locks and set them just as you wish.",
                "calm",
            ),
            Line(
                "Ooh, let me go and sort the doors and set them right for you.",
                "playful",
            ),
            Line(
                "Right, I'll adjust the locks to suit the way you want them set.",
                "sincere",
            ),
            Line(
                "Give me a sec to get the doors sorted and settled in nicely.",
                "excited",
            ),
            Line(
                "Hmm, I'll set the locks the way you like them through the house.",
                "calm",
            ),
            Line(
                "Just a moment - I'm going to get the doors how you want them.",
                "playful",
            ),
        ),
    },
    "security": {
        "any": (
            Line(
                "Let me check the security and see how things are all standing.",
                "calm",
            ),
            Line(
                "Ooh, let me give the alarm a look and see what's what there.",
                "playful",
            ),
            Line(
                "Right, I'll go and see how our alarm system has been keeping.",
                "sincere",
            ),
            Line(
                "One moment while I check the security and see where we stand.",
                "calm",
            ),
            Line(
                "Give me a sec to look at the alarm and see how it's been faring.",
                "excited",
            ),
            Line(
                "Hmm, I'll check in with the alarm and see how it is feeling.",
                "playful",
            ),
        ),
        "query": (
            Line(
                "Let me check the alarm and see how it's been holding for you.",
                "calm",
            ),
            Line(
                "One moment - I'll go and read the security and tell you all.",
                "playful",
            ),
            Line(
                "Give me a sec to see how our own alarm has been faring lately.",
                "sincere",
            ),
            Line(
                "Hmm, I'll take a look and see where the security is standing.",
                "excited",
            ),
            Line(
                "Just a moment while I check the alarm for a bit of peace of mind.",
                "calm",
            ),
            Line(
                "I'll go and have a good look at how the alarm has been doing.",
                "playful",
            ),
        ),
        "control": (
            Line(
                "Let me take care of the security and set it just as you wish.",
                "calm",
            ),
            Line(
                "Ooh, let me go and set the alarm the way you like it the best.",
                "playful",
            ),
            Line(
                "Right, I'll adjust the security to suit the way you want it.",
                "sincere",
            ),
            Line(
                "Give me a sec to get the alarm sorted and settled in nicely.",
                "excited",
            ),
            Line(
                "Hmm, I'll set the security the way you like it in the house.",
                "calm",
            ),
            Line(
                "Just a moment - I'm going to get the alarm how you want it set.",
                "playful",
            ),
        ),
    },
    "timers": {
        "any": (
            Line(
                "Let me check that timer and see how it's getting along there.",
                "calm",
            ),
            Line(
                "Ooh, let me glance at the timer and see where it is sitting.",
                "playful",
            ),
            Line(
                "Right, I'll go and have a little chat with the timer, tick-tock.",
                "sincere",
            ),
            Line(
                "One moment while I check the timer and see how it's ticking.",
                "calm",
            ),
            Line(
                "Give me a sec to look at the timer and see how it's been faring.",
                "excited",
            ),
            Line(
                "Hmm, I'll peek at the timer and see how it has been holding.",
                "playful",
            ),
        ),
        "query": (
            Line(
                "Let me check just how much time that timer still has left to go.",
                "calm",
            ),
            Line(
                "One moment - I'll go and read the timer and tell you the time.",
                "playful",
            ),
            Line(
                "Give me a sec to see how much longer that timer has left to go.",
                "sincere",
            ),
            Line(
                "Hmm, I'll take a good look and see where that timer is standing.",
                "excited",
            ),
            Line(
                "Just a moment while I check the timer for a straight answer.",
                "calm",
            ),
            Line(
                "I'll go and take a peek at the timer and report straight back.",
                "playful",
            ),
        ),
        "control": (
            Line(
                "Let me get that timer exactly the way you asked for it, no fuss.",
                "calm",
            ),
            Line(
                "Ooh, let me have a fiddle with the timer and set it right for you.",
                "playful",
            ),
            Line(
                "Right, I'll adjust the timer to suit just what you need it for.",
                "sincere",
            ),
            Line(
                "Give me a sec to get that timer set the way you like it to run.",
                "excited",
            ),
            Line(
                "Hmm, I'll tweak the timer and get it set the way you like it.",
                "calm",
            ),
            Line(
                "Just a moment - I'm going to set the timer as you'd like it.",
                "playful",
            ),
        ),
    },
    "scenes": {
        "any": (
            Line(
                "Let me set the scene and get the mood just right for the room.",
                "calm",
            ),
            Line(
                "Ooh, let me paint the room a mood with a flick of the lights.",
                "playful",
            ),
            Line(
                "Right, I'll get the whole scene set for the mood you fancy best.",
                "sincere",
            ),
            Line(
                "One moment while I set the scene and pick a mood for the room.",
                "calm",
            ),
            Line(
                "Give me a sec to set the scene and get the room feeling right.",
                "excited",
            ),
            Line(
                "Hmm, let me set the mood and get the whole scene looking lovely.",
                "playful",
            ),
            Line(
                "A moment while I arrange the scene and set the whole tone for it.",
                "sincere",
            ),
            Line(
                "Let me get the scene going and set a lovely mood for the room.",
                "excited",
            ),
        ),
    },
    "calendar": {
        "any": (
            Line(
                "Let me have a look at your calendar and see what's what in it.",
                "calm",
            ),
            Line(
                "Ooh, let me flick through your diary and see what's inside it.",
                "playful",
            ),
            Line(
                "Right, I'll go and check your calendar and see the shape of it.",
                "sincere",
            ),
            Line(
                "One moment while I look through your diary and see what's what.",
                "calm",
            ),
            Line(
                "Give me a sec to see what's waiting for you in your calendar.",
                "excited",
            ),
            Line(
                "Hmm, I'll flick through your calendar and see what's in there.",
                "playful",
            ),
        ),
        "query": (
            Line(
                "Let me flick through your calendar and see what's in the diary.",
                "calm",
            ),
            Line(
                "One moment - I'll read your calendar and tell you all about it.",
                "playful",
            ),
            Line(
                "Give me a sec to see what lies in store for you in your diary.",
                "sincere",
            ),
            Line(
                "Hmm, I'll take a look and see what your calendar has to say.",
                "excited",
            ),
            Line(
                "Just a moment while I check your schedule for the day ahead.",
                "calm",
            ),
            Line(
                "Let me flick through and tell you what the day holds for you.",
                "playful",
            ),
        ),
        "control": (
            Line(
                "Let me take care of your calendar and set it as you like it.",
                "calm",
            ),
            Line(
                "Ooh, let me have a tinker with your diary and tidy it for you.",
                "playful",
            ),
            Line(
                "Right, I'll adjust your calendar to suit what you need from it.",
                "sincere",
            ),
            Line(
                "Give me a sec to get your diary sorted out into proper order.",
                "excited",
            ),
            Line(
                "Hmm, I'll tweak your calendar and get it in good shape for you.",
                "calm",
            ),
            Line(
                "Just a moment - I'm going to get your diary as you'd like it.",
                "playful",
            ),
        ),
    },
    "generic": {
        "any": (
            Line(
                "Good question. Give me a moment to fetch my old thinking cap.",
                "calm",
            ),
            Line(
                "Ooh, let me have a proper think and turn that one over twice.",
                "playful",
            ),
            Line(
                "Right, I'll mull that over and see what can be done about it.",
                "sincere",
            ),
            Line(
                "One moment while I turn that one over in my head a short while.",
                "calm",
            ),
            Line(
                "Give me a sec to chew that over and see what I can find out.",
                "excited",
            ),
            Line(
                "Hmm, let me have a good long think and see what I can find there.",
                "playful",
            ),
            Line(
                "A moment while I gather all my thoughts about that one for you.",
                "sincere",
            ),
            Line(
                "Let me puzzle that one out and see what I can do about that.",
                "excited",
            ),
        ),
        "query": (
            Line(
                "Let me go and dig out just what I can uncover about that one.",
                "calm",
            ),
            Line(
                "One moment - I'll dig into that and see just what I uncover.",
                "playful",
            ),
            Line(
                "Give me a sec to dig into that and find you a cleaner answer.",
                "sincere",
            ),
            Line(
                "Hmm, I'll take a look and see what I can find out about it all.",
                "excited",
            ),
            Line(
                "Just a moment while I dig a little deeper into that one for you.",
                "calm",
            ),
            Line(
                "I'll go and dig around and report back with all that I find.",
                "playful",
            ),
        ),
        "control": (
            Line(
                "Let me take care of that and see what might be done about it.",
                "calm",
            ),
            Line(
                "Ooh, let me have a good go at that and see how it all pans out.",
                "playful",
            ),
            Line(
                "Right, I'll sort that one and see what can be done with it all.",
                "sincere",
            ),
            Line(
                "Give me a sec to take that one in hand and have a good look.",
                "excited",
            ),
            Line(
                "Hmm, I'll get to work at that and see how it all turns out well.",
                "calm",
            ),
            Line(
                "Just a moment while I work out what can be done about that one.",
                "playful",
            ),
        ),
    },
}


def opener_lines(bucket: str, intent: str | None) -> tuple[Line, ...]:
    """
    Return the opener lines for ``bucket`` and ``intent``.

    Falls back to the bucket's ``"any"`` cell when ``intent`` is ``None`` or the bucket carries
    no cell for that intent. This is the single fallback rule callers and tests share.
    """
    cells = OPENER_LINES[bucket]
    if intent is not None and intent in cells:
        return cells[intent]
    return cells["any"]


#: Mid-chain fragments, keyed by tool bucket. Short, trailing-ellipsis, no length floor.
MIDCHAIN_LINES: Final[dict[str, tuple[Line, ...]]] = {
    "query": (
        Line("Let me check that...", "calm"),
        Line("Looking that up...", "playful"),
        Line("Just checking on that...", "sincere"),
        Line("One moment while I look...", "excited"),
    ),
    "control": (
        Line("Setting that up...", "calm"),
        Line("On it...", "playful"),
        Line("Getting that sorted...", "sincere"),
        Line("Sorting that out...", "excited"),
        Line("Working on it...", "calm"),
    ),
    "weather": (
        Line("Checking the forecast...", "calm"),
        Line("Looking at the sky...", "playful"),
        Line("Seeing what's coming...", "sincere"),
        Line("Reading the clouds...", "excited"),
    ),
    "climate": (
        Line("Asking the thermostat...", "calm"),
        Line("Checking the temperature...", "playful"),
        Line("Seeing how it's set...", "sincere"),
        Line("Chatting to the heating...", "excited"),
    ),
    "media": (
        Line("Cueing that up...", "calm"),
        Line("Getting that ready...", "playful"),
        Line("Sorting the media out...", "sincere"),
        Line("Finding that one...", "excited"),
    ),
    "playback": (
        Line("Sorting the playback...", "calm"),
        Line("Reaching for the remote...", "playful"),
        Line("Adjusting that...", "sincere"),
        Line("On the controls...", "excited"),
        Line("Handling the playback...", "calm"),
    ),
    "volume": (
        Line("Adjusting the volume...", "calm"),
        Line("Setting the level...", "playful"),
        Line("Getting the sound right...", "sincere"),
        Line("Nudging the volume...", "excited"),
    ),
    "list": (
        Line("Adding {item}...", "calm", plain="Adding that in..."),
        Line("Putting that on your list...", "playful"),
        Line("Noting that down...", "sincere"),
        Line("Writing that in...", "excited"),
    ),
    "calendar": (
        Line("Checking your calendar...", "calm"),
        Line("Looking at your schedule...", "playful"),
        Line("Flicking through the diary...", "sincere"),
        Line("Seeing what's ahead...", "excited"),
    ),
    "search": (
        Line("Looking that up...", "calm"),
        Line("Searching for that...", "playful"),
        Line("Digging that out...", "sincere"),
        Line("Hunting that down...", "excited"),
    ),
    "generic": (
        Line("One moment...", "calm"),
        Line("Just a second...", "playful"),
        Line("Working on that...", "sincere"),
        Line("Bear with me...", "excited"),
        Line("Thinking it through...", "calm"),
    ),
}

#: Watchdog reassurances - short, no length floor.
WATCHDOG_LINES: Final[tuple[Line, ...]] = (
    Line("Still on it...", "calm"),
    Line("Almost there...", "playful"),
    Line("This one's a thinker...", "sincere"),
    Line("Still rummaging...", "playful"),
    Line("Bear with me...", "calm"),
    Line("Nearly there...", "excited"),
    Line("Thanks for waiting...", "sincere"),
    Line("Still digging...", "calm"),
    Line("Won't be long...", "playful"),
)

#: Failure/apology bucket - mood-tagged, every line 60+ chars.
FAILURE_LINES: Final[tuple[Line, ...]] = (
    Line(
        "Sorry, I couldn't quite reach the assistant - please try that once more.",
        "sincere",
    ),
    Line(
        "Apologies, something went wrong on my end. Do give that another go.",
        "calm",
    ),
    Line(
        "Sorry about that - I hit a snag on my side. Please try again in a moment.",
        "sincere",
    ),
    Line(
        "Ah, that didn't go through properly. Sorry about that - do try again.",
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


def strip_mood_tags_streaming(text: str, carry: str = "") -> tuple[str, str]:
    """
    Strip ``[mood]`` tags from one streamed delta, holding a split tag over.

    The raw LLM answer may carry the same inline ``[mood]`` tags the pool emits, and a
    tokenizer can split one across deltas (``"[cal"`` then ``"m] hi"``). A trailing fragment
    that starts with ``[`` and is still a prefix of some ``[mood]`` tag is returned as
    ``carry`` for the caller to prepend to the next delta; anything else - a non-mood bracket,
    or a ``[`` that can no longer become a mood tag - is emitted verbatim.

    Returns ``(emitted, carry)``. Flush a non-empty ``carry`` at end of stream so a fragment
    that never completes is spoken as written.
    """
    buf = carry + text
    out: list[str] = []
    i = 0
    n = len(buf)
    while i < n:
        char = buf[i]
        if char == "[":
            end = buf.find("]", i)
            if end == -1:
                fragment = buf[i:]
                if any(tag.startswith(fragment) for tag in _MOOD_TAGS):
                    # Could be a mood tag split across deltas: hold it back.
                    return "".join(out), fragment
                out.append(char)
                i += 1
                continue
            if buf[i : end + 1] in _MOOD_TAGS:
                i = end + 1
                continue
            out.append(char)
            i += 1
            continue
        out.append(char)
        i += 1
    return "".join(out), ""


def strip_mood_tags(text: str) -> str:
    """Remove inline ``[mood]`` tags (moods in :data:`MOODS`) from a whole string."""
    stripped, carry = strip_mood_tags_streaming(text)
    return stripped + carry


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

    def _draw(
        self,
        key: str,
        lines: Sequence[Line],
        *,
        slots: Mapping[str, str] | None = None,
        expression_tags: bool = False,
    ) -> str:
        """Pick a line for ``key`` and render it with the context's default mood."""
        return render(
            self._pick(key, lines),
            slots=slots,
            default_mood=DEFAULT_MOOD[key],
            expression_tags=expression_tags,
        )

    def opener(
        self,
        bucket: str,
        intent: str | None = None,
        *,
        expression_tags: bool = False,
    ) -> str:
        """
        Render an opener line for ``bucket`` and ``intent``.

        ``intent`` selects a cell via :func:`opener_lines`; the no-repeat state stays keyed by
        the bucket alone, so lines drawn across a bucket's cells still avoid an immediate
        repeat.
        """
        return self._draw(
            f"opener:{bucket}",
            opener_lines(bucket, intent),
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
        key = f"midchain:{bucket}"
        return self._draw(
            key,
            MIDCHAIN_LINES[bucket],
            slots=slots,
            expression_tags=expression_tags,
        )

    def watchdog(self, *, expression_tags: bool = False) -> str:
        """Render a watchdog reassurance."""
        return self._draw("watchdog", WATCHDOG_LINES, expression_tags=expression_tags)

    def failure(self, *, expression_tags: bool = False) -> str:
        """Render a failure/apology line."""
        return self._draw("failure", FAILURE_LINES, expression_tags=expression_tags)
