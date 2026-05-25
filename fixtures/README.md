# `fixtures/` — public test material

Everything in this directory is intentionally tracked. The CI guard at
`.github/workflows/repo-hygiene.yml` will block any file that does not
appear in its allowlist; if you add a new fixture, allowlist it there
and document the licence in this README.

## `clip_10s.wav` + `clip_10s.en.srt`

**Source.** Excerpt of John F. Kennedy's 1961 inaugural address. The
recording is in the **public domain**:

- The speech text is a federal-government work, public domain under
  17 U.S.C. § 105.
- The audio recording was produced by the United States government on
  the day of the inauguration and is in the public domain on the same
  basis. This specific encoding is the one redistributed with OpenAI's
  Whisper repository (`openai/whisper/tests/jfk.flac`), which has been
  used as the de-facto ASR smoke clip across the ML community for years.

The clip here was downloaded from the Whisper repository, trimmed to
exactly ten seconds, and re-encoded as 16 kHz mono 16-bit PCM WAV — the
format Sub-Zero's audio pipeline expects internally.

**Reference transcript.** The `.en.srt` is a single-cue ground truth.
Whisper's actual segmentation will be finer-grained; the test asserts
*content equivalence at the lexical level*, not cue-boundary equality.

**Format.**

```
clip_10s.wav   — 16 kHz mono PCM WAV, 10.000 s exactly
clip_10s.en.srt — single cue covering [00:00.000 → 00:10.000]
```

**Why English not Japanese.** The ROADMAP item originally proposed a
Japanese CC0 source. Public-domain Japanese audio with a known reference
transcript is genuinely hard to source — Common Voice's CC0 subset is
distributed only as a dataset archive, and most Wikimedia Commons
Japanese audio is CC-BY-SA, not CC0. The JFK clip is uncontroversially
PD and demonstrates the same end-to-end pipeline (ASR → MT). To exercise
non-English ASR, swap in a clip + reference SRT with a different source
language and rename the SRT extension (`fixtures/sample.ja.srt`
expresses Japanese, etc.). The CI guard's allowlist will need a one-line
update.

## How tests use these

- `cargo xtask smoke` (the cross-platform smoke runner in `xtask/`)
  detects `fixtures/clip_10s.wav` and runs the ffmpeg + neural-MT
  integration cases against it. When absent, those cases are skipped
  and the rest of the smoke gate still runs.

- The legacy bash equivalent at `scripts/ci/run_integration_smoke.sh`
  hard-codes the path. The xtask runner is the supported entry point
  going forward.
