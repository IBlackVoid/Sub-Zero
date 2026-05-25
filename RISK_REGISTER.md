# Risk Register

## Active Risks

| ID | Risk | Impact | Probability | Owner | Mitigation | Status |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | Private/copyrighted media, large models, logs, and generated outputs are present in the working tree | Public release legal/privacy failure | High | Release owner | Explicit release scrub and tracked-file audit before publishing | Open |
| R2 | README/docs overstate current theory/product proof | Credibility loss with reviewers | Medium | Theory/Docs owner | Separate proven facts, empirical results, conjectures, and roadmap claims | Open |
| R3 | Learned quality gate schema mismatch disables a key feature | Quality claims become inaccurate | Medium | Engine owner | Accept current schema version and test it | In progress |
| R4 | Event sidecars expose local paths/progress if bound publicly | Privacy leak | Medium | Security owner | Loopback default and explicit remote opt-in | In progress |
| R5 | Optional model dependencies make first-run setup hard | Adoption drop | High | Product owner | Bootstrap docs, small legal fixture, phrase-table smoke path | Open |
| R6 | Long-run performance depends on local GPU/CPU/model state | Benchmark claims fail to generalize | Medium | Performance owner | Publish representative traces and environment details | Open |
| R7 | TUI save/snapshot writes arbitrary local paths | Accidental overwrite or unsafe UX | Low | TUI owner | Add confirmation/path policy before public release | Open |

## Technical Debt That Matters

| Item | Cost | Trigger to Fix | Owner |
| --- | --- | --- | --- |
| Large `pipeline.rs` and `tui/src/app.rs` files | Harder review and contribution | Before broad contributor onboarding | Engine/TUI owners |
| Mixed research/product language in docs | Confuses users and reviewers | Before public launch | Docs owner |
| Optional Python/model setup is implicit | First-run failure | Before first tagged public release | Release owner |
| Benchmark artifacts mixed with source tree | Repository bloat | Before publishing | Release owner |

## Assumptions

| Assumption | Depends On | How To Falsify |
| --- | --- | --- |
| Users value offline privacy enough to install local models | Target audience and docs | Public beta feedback |
| Quality gates predict useful subtitle output | Dataset representativeness | Human eval and held-out corpus failure |
| Loopback-only event streams fit normal local workflows | TUI and integrations | Users need remote dashboards |
| Phrase-table mode is enough for smoke tests | Known JA->EN fixture coverage | Fresh-clone smoke fails without models |

## Retired Risks

Move closed risks here with the date and reason.
