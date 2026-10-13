# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Thinking Voice check-ins** — instant spoken check-ins that cover LLM thinking latency on
  voice turns:
  - A contextual spoken **opener** within a few hundred milliseconds of a request, selected from
    a curated, mood-tagged line pool based on the utterance and the entities exposed to Assist.
  - Openers are matched to both the **topic** and the **intent** of the request (asking,
    changing, or playing), with a new **volume** topic — so "set your volume" gets a volume
    line instead of a music one.
  - A larger, more varied line pool per topic and intent.
  - Brief **mid-chain check-ins** before each tool step, so multi-step requests don't fall silent.
  - A capped **watchdog** that offers occasional reassurance during long waits instead of dead
    air, then goes quiet and lets the real answer land whenever it arrives.
  - Optional inline **mood tags** (`[calm]`, `[playful]`, `[sincere]`, `[excited]`) for
    expressive TTS engines, stripped automatically on engines that don't support them.
  - Check-ins are **suppressed when the real answer is fast**, so snappy replies are never
    preceded by a stalling phrase.
  - A spoken **apology** instead of a generic error when the LLM fails outright.
  - Check-ins and the watchdog wait for the **estimated end of speech** already queued — both
    earlier check-ins and the model's own streamed text, estimated from word count at a
    configurable speaking rate — and fire only after `checkin_gap_ms` of estimated silence,
    so they never land in front of an answer still queued.
  - Mid-chain check-in lines follow the tool's **base name**, so API-prefixed tools (e.g.
    `search-services__search_web`) and [skye-harris' Tools for Assist](https://github.com/skye-harris/llm_intents)
    get a matching line instead of a generic one.
  - Mid-chain check-in lines cover **playback** (pause/skip/resume) and **volume** tool
    categories.
  - Configured per conversation agent: `check_ins_enabled` (default `true`),
    `expression_tags` (default `false`), `grace_ms` (default `500`), `checkin_gap_ms`
    (default `1500`), `watchdog_max_extras` (default `2`), `speech_rate_wpm` (default `150`).
  - Early audio requires a **streaming-input TTS engine** (e.g. ElevenLabs, Google, Nabu Casa
    cloud); on non-streaming engines (Piper, edge-tts, …) check-ins play as delayed audio.

### Changed

- This fork is an **in-place feature fork** of
  [`skye-harris/hass_local_openai_llm`](https://github.com/skye-harris/hass_local_openai_llm):
  same `local_openai` domain, no rebrand. Installing it **replaces** any existing upstream
  install — the two cannot coexist. No new domain and no configuration migration is required.
