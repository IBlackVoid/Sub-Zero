# sub-zero

offline-first subtitle translator. nothing leaves your machine.

three things matter:

- the translation never leaves your machine.
- the engine is measurable, not hopeful.
- you can verify every claim it makes with the numbers it emits.

if you do not need those things, you do not need this.

---

## what it does

reads a media file or an existing `.srt`. transcribes when there is no
subtitle track. translates with a local NLLB-200 model. runs a learned
quality gate against the result. emits a sidecar with everything it
saw so you can audit it.

no cloud. no API keys. no telemetry. nothing gets uploaded.

## install

prereqs:

- rust toolchain (`rustup default stable`)
- python 3.10+ for the helper scripts and the fidelity verifier
- ffmpeg on PATH (transcription + media probing)

build:

```
git clone https://github.com/IBlackVoid/Sub-Zero
cd Sub-Zero
cargo build --release
```

binaries land in `target/release/`:

- `sub-zero`      — the engine
- `sub-zero-tui`  — the live dashboard

models (one-time, only if you want neural translation):

- NLLB-200 in CTranslate2 format → `models/nllb/`
- (optional) whisper.cpp binary + ggml model for transcription

without NLLB, the engine falls back to a phrase-table backend for
the language pairs it knows about. enough to smoke-test the pipeline.

## run

```
sub-zero -i clip.mkv --lang en
```

a `clip.en.srt` appears next to the input. a sidecar
`clip.sub-zero.json` next to it carries every gate score, every
recovery event, every chunk timing.

a few of the flags that matter:

```
--source-lang ja           source language (default ja)
--lang en                  target language (default en)
--profile strict           fast | balanced | strict
--parallel --workers 8     chunked transcription, N workers
--gpu                      use CUDA when available
--speaker-aware            learn per-character voice priors
--trace-runtime            emit a per-stage timing sidecar
--verify                   check the output against the audio
```

## the dashboard

```
cargo run -p sub-zero-tui --release
```

a live terminal dashboard. shows the pipeline as it runs, the cues as
they translate, and the per-character voice priors the engine learns.
press `:` for commands. press `:help` for the rest.

three running-screen modes, cycle with `g`:

- **original** — the pre-recorded animation plays.
- **emerge** — the same animation reveals itself cell-by-cell as
  chunks complete. deterministic per input filename — same file, same
  reveal pattern.
- **generative** — a flow-field particle system. fresh artwork per
  run. each chunk completion injects a particle burst.

## audit

every claim this thing makes is checkable. the fidelity bound is the
empirical mutual information between machine and reference, estimated
with a Kraskov k-NN estimator (numpy only, no torch):

```
python scripts/quality_gate/verify_fidelity_bound.py \
  --machine  out.en.srt \
  --baseline baseline.en.srt \
  --reference reference.en.srt \
  --strict
```

on a real Japanese-language corpus (998 aligned cues from a JP gameplay
stream, with a separately-sourced human reference):

| metric                                | value                |
|---------------------------------------|----------------------|
| I(machine ; reference)                | 0.6985 nats          |
| I(per-cue baseline ; reference)       | 0.2282 nats          |
| Δ̂ — fidelity gap vs per-cue baseline | **+0.4703 nats**     |
| name inconsistency ratio (full ep)    | 0.00 %               |
| adjacent repeat ratio                 | 0.37 %               |
| scene low-quality ratio               | 1.13 %               |

three-times the mutual information. real corpus. real reference.

## the bits worth reading

```
src/engine/                          translator + pipeline + DOOM-QLOCK
src/engine/voice_consistency.rs      per-character voice priors
src/engine/character_glossary.rs     persistent name canonicalisation
src/engine/postprocess.rs            reading-rate cap + scene rescue
scripts/quality_gate/                fidelity verifier + learned gate
scripts/tui/braille_convert.py       video → braille animation
tui/src/                             ratatui dashboard
```

## first-run smoke (5 minutes from clean clone)

```
cargo build --release
cargo test
```

both should pass with zero failures. then either:

```
# pipe an existing SRT through the post-process + verifier path:
./target/release/sub-zero -i mysrt.ja.srt --lang en --phrase-table

# or fire up the dashboard:
./target/release/sub-zero-tui
```

inside the dashboard:

```
Enter   pick an input file
r       re-run the most-recent input
p       cycle quality profile
Tab     path completion in the picker
:help   keybind reference
g       cycle running-screen visual mode
s       on Result: save the output to a custom location
```

## status

182 tests pass (151 engine + 31 dashboard). clippy clean. release
build clean. CI on Linux, macOS, Windows.

```
cargo test
cargo clippy --no-deps -- -W clippy::or_fun_call -W clippy::manual_let_else
```

## license

MIT.

---

control is an illusion. determinism isn't.
