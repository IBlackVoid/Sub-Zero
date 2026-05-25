# Security Policy

Sub-Zero is offline-first software that handles user media, subtitles, local
models, subprocesses, filesystem paths, event sidecars, and encrypted TUI
assets. Security reports are treated seriously because a failure can expose
private files or break the no-cloud privacy contract.

## Supported Versions

Sub-Zero is pre-1.0. Until a stable release policy exists, security support
covers:

| Version | Supported |
| --- | --- |
| `main` | Yes |
| Latest published release or tag, if one exists | Yes |
| Older tags or snapshots | No, unless maintainers agree before triage |

## Reporting a Vulnerability

Do not publish exploit details in a public issue, discussion, pull request, or
chat log.

Use GitHub Private Vulnerability Reporting from the repository Security tab if
it is available. If it is not available, contact a maintainer through the
repository owner profile or open a minimal public issue titled `Security contact
request` that contains no exploit details and asks for a private channel.

Include:

- Affected component, version, commit SHA, and operating system.
- Minimal reproduction steps and inputs, when safe to share privately.
- Expected behavior and observed behavior.
- Impact category, such as code execution, file disclosure, privacy bypass,
  denial of service, supply-chain risk, or encrypted-asset key disclosure.
- Whether the issue is already public or shared with anyone else.

We aim to acknowledge reports within 72 hours and provide an initial triage
decision within 7 days. Complex reports may take longer, but maintainers should
keep the reporter updated.

## In Scope

Reports are in scope when they affect Sub-Zero behavior or amplify risk through
Sub-Zero integration:

- Code execution or command injection through media, subtitle, config, model,
  script, or path input.
- Path traversal or unexpected writes outside the intended output location.
- Privacy bypasses, including network calls that violate the offline contract.
- HTTP or WebSocket sidecar exposure beyond documented loopback behavior.
- Subprocess sandbox failures involving ffmpeg, ASR tools, or Python helpers.
- Secret, token, passphrase, key, or encrypted TUI asset disclosure.
- Supply-chain risks in Rust crates, Python helpers, GitHub Actions, or release
  packaging.
- Parser bugs that cause silent data corruption or unsafe file behavior.

## Out of Scope

Open a normal issue instead for:

- Translation quality regressions without a security impact.
- Crashes on malformed input that fail closed and do not corrupt files or leak
  data.
- Third-party model behavior without a Sub-Zero-specific amplification.
- Issues that require the attacker to already have unrestricted local code
  execution on the victim machine.
- Social engineering, phishing, or attacks against unrelated infrastructure.

## Safe Harbor

Good-faith security research is welcome when it:

- Targets your own clone, test machine, and test data.
- Avoids persistence, stealth, malware, credential theft, and exfiltration.
- Stops after proving impact.
- Keeps details private until maintainers coordinate disclosure.
- Does not attack third-party services, users, or infrastructure.

## Security Design Expectations

Contributions that touch a trust boundary should document and verify the
boundary. Treat these as hostile by default:

- CLI arguments and environment variables.
- Media and subtitle files.
- TUI file picker paths, save paths, snapshots, and preferences.
- Sidecar HTTP and WebSocket clients.
- Python helper arguments, stdin, stdout, stderr, and temporary files.
- Model files, generated sidecars, learned-gate artifacts, and encrypted TUI
  envelopes.

Do not place secrets in source, tests, fixtures, docs, logs, screenshots, shell
history, generated sidecars, or CI output. This includes API keys, private media
paths, easter-egg passphrases, derived keys, salts that identify private assets,
and unreleased encrypted asset plaintext.

## Disclosure Process

Default coordinated disclosure timeline after maintainers confirm a fix:

- Fix lands privately or on `main`, depending on risk.
- A release or patch advisory is prepared.
- Public advisory is published after users have a reasonable update window.

Critical issues, such as remote code execution or key disclosure, may use a
shorter or longer timeline depending on exploitability and reporter
coordination.

## Hall of Credit

Reporters receive credit in the advisory by default unless they ask to remain
anonymous.
