# Threat Model

## Scope

System/feature: Sub-Zero CLI engine, TUI companion, local event sidecars,
model/script helpers, and release repository contents.

Reviewed date: 2026-05-19

Reviewer: Codex/IBVoid security-review lens

## Assets

- Secrets/tokens: none required by the core product; user environments may
  still contain secrets and must not be logged or copied.
- User data: media files, subtitle files, generated translations, metadata,
  traces, voice priors, glossary, preferences.
- Privileged operations: local filesystem writes, subprocess execution,
  network listener binding.
- Filesystem/process access: ffmpeg, whisper, Python MT scripts, TUI opener.
- Model/tool permissions: local model files, Python dependencies, optional MT
  daemon.

## Actors

- Local user.
- Malicious media or subtitle file.
- Malicious local web page trying to read event streams.
- Malicious dependency or model/script file.
- Contributor accidentally committing private media, logs, or models.
- CI/release actor with package publishing access.

## Trust Boundaries

| Boundary | Data Crossing | Validation |
| --- | --- | --- |
| CLI args -> runtime config | paths, flags, numeric limits | typed parsing, range checks |
| Media/SRT -> parser | subtitle text, timings, metadata | SRT parser and quality gates |
| Rust -> subprocess | ffmpeg/whisper/Python arguments | `Command::arg`, no shell strings |
| Python MT -> Rust | JSON responses | JSON parse and response mapping |
| Event stream -> local clients | paths, timings, progress | loopback default, no authority |
| TUI save/snapshot -> filesystem | user-provided output path | currently minimal path stripping |
| Repo -> public users | docs, scripts, fixtures | release audit required |

## Entry Points

- CLI positional inputs and flags.
- SRT files and media containers.
- Environment variables such as `SUB_ZERO_*`.
- Python scripts and local model files.
- HTTP `/health` and `/events`.
- WebSocket `/ws`.
- TUI command line, file picker, save and snapshot commands.

## Abuse Cases

| Abuse Case | Impact | Mitigation | Verification |
| --- | --- | --- | --- |
| Remote site reads event stream | Leaks local paths/progress | Loopback-only bind by default; no wildcard CORS; WS origin check | Sidecar tests |
| User binds `0.0.0.0` accidentally | Local metadata exposed on LAN | Require `--allow-remote-events` | CLI and sidecar tests |
| Malicious subtitle triggers parser failure | Denial of service | Parse errors are typed and surfaced | SRT tests |
| Malicious path writes outside expected area | Data overwrite | Document risk; add future path policy for TUI save/snapshot | TUI tests pending |
| Committed private media/models | Legal/privacy risk | Gitignore plus pre-release repository audit | Release gate |
| Compromised Python/model dependency | Execution/integrity risk | Local-only trust model; document provenance and hashes | Release docs pending |
| Shell injection through paths | Command execution | Structured `Command::arg` use | Static grep and tests |

## Findings

| Severity | Surface | Finding | Fix | Status |
| --- | --- | --- | --- | --- |
| High | Public repo contents | Large/private/generated media and model files appear in working tree | Do not delete automatically; run explicit release scrub before publishing | Open |
| Medium | HTTP/WS sidecars | Non-loopback bind exposed event streams too easily | Add explicit remote opt-in and remove wildcard event CORS | In progress |
| Medium | Learned gate | Current model schema was rejected by loader | Accept current schema version and test it | In progress |
| Medium | TUI filesystem writes | Save/snapshot accepts arbitrary local path | Add confirmation/root policy before public release | Open |
| Low | Subprocesses | External tools are necessary trust boundary | Keep structured args; document dependency trust | Ongoing |

## Residual Risk

- Media parsers and ML runtimes are large dependency surfaces. Running them on
  hostile files is inherently risky; sandboxing is future hardening.
- Remote event streaming may be useful later, but it needs token auth and
  explicit documentation before broad use.
- Public release must scrub private samples, local model weights, generated
  benchmark directories, logs, and personal planning files.
