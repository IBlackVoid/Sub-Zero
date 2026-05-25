---
name: ibvoid-quality-gate
description: Use for code review, final acceptance, release readiness, test planning, verification strategy, or checking work against correctness, clarity, safety, performance, maintainability, and observability standards.
---

# IBVoid Quality Gate

Use this skill before claiming non-trivial work is done. The quality gate turns
engineering standards into explicit checks.

For serious work, consult the shared references when available:
`IBVOID_DOMAIN_DOCTRINE.md`, `IBVOID_EXECUTION_GATES.md`, and
`IBVOID_AGENT_MANDATES.md`. For the full specialist procedure, read
`IBVOID_AGENT_PLAYBOOKS/verifier.md`.

## Review Order

1. Correctness: behavior matches requirements; no silent failures.
2. Safety: hostile inputs, permissions, secrets, data loss, injection, and race
   risks are handled.
3. Clarity: intent is readable without tricks.
4. Scope: unrelated changes are absent.
5. Idiom: the code follows local and language conventions.
6. Tests: meaningful checks exist or a clear reason is stated.
7. Performance: hot paths and data scale are considered.
8. Observability: production-relevant behavior can be inspected.

## Verification Selection

- Config/docs: parse or lint plus targeted grep.
- Single module: focused unit/type/lint check.
- Cross-module: unit plus integration or smoke check.
- UI: browser/screenshot verification when possible.
- Security-sensitive: adversarial cases and dependency/secret scan.
- Performance-sensitive: benchmark, profiler, or concrete measurement plan.

## Output Contract

Lead with blocking findings if reviewing. If verifying completed work, state:

- Checks run.
- Checks not run and why.
- Residual risk.
- Files or areas most affected.
