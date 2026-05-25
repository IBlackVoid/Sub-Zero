# IBVoid Execution Gates

This file defines how agents should decide what to do, when to delegate, and how
to prove that work is complete.

## Automatic Use

The user should not need to name every specialist. The main agent should infer
the required specialists from the work:

- Ambiguous or multi-part request: orchestrator.
- New system, refactor, public API, data ownership: architect.
- Auth, user input, files, shell, secrets, permissions, dependencies, uploads,
  prompts, or network exposure: security-reviewer.
- Hot path, scale, memory, latency, rendering, large data, or expensive query:
  performance-engineer.
- UI, UX, browser, accessibility, visual layout: frontend-engineer.
- API, service, job, queue, domain workflow: backend-engineer.
- Schema, migration, index, query, transaction: database-engineer.
- CI, Docker, deployment, logs, metrics, runtime failure: devops-engineer.
- ML, CV, embeddings, evaluation, inference: ai-ml-engineer.
- Rust/C/C++, OS, concurrency, memory layout, syscalls: systems-engineer.
- Unknown binary/protocol/format/legacy code: reverse-engineer.
- Current facts, external docs, standards, versions: researcher.
- Proofs, original theory, formal models: theory-lab.
- Completion, release, review: verifier.

If a task triggers multiple domains, the orchestrator must route and integrate.

## Plan Gate

Use a plan before editing when:

- More than one subsystem is touched.
- Requirements are ambiguous.
- There is security, data-loss, or performance risk.
- A design choice is long-lived.
- The user explicitly asks for a plan.

Plan output:

- Objective.
- Constraints.
- Specialist routes.
- Work units.
- Acceptance criteria.
- Verification gate.

## Implementation Gate

Before editing:

- Read the relevant files.
- Identify existing patterns.
- Preserve unrelated changes.
- Choose the smallest viable design.

During editing:

- Keep ownership boundaries clear.
- Avoid unrelated refactors.
- Update tests or verification artifacts when behavior changes.
- Record any new invariant in code, tests, or docs.

## Security Gate

Run this gate when a change touches:

- Authentication or authorization.
- User-controlled input.
- File paths, uploads, archives, or deserialization.
- Shell commands or subprocesses.
- Network calls, redirects, webhooks, SSRF surfaces.
- Secrets, tokens, env vars, logs, CI, or deployment.
- Prompt/tool routing, agent permissions, MCP, browser automation.

Security output:

- Trust boundaries.
- Assets.
- Abuse cases.
- Findings by severity.
- Fixes.
- Verification cases.

## Performance Gate

Run this gate when a change touches:

- Hot loops.
- Large data.
- Rendering.
- Streaming.
- Query-heavy paths.
- Startup time.
- Memory pressure.
- Concurrency.

Performance output:

- Workload shape.
- Complexity.
- Bottleneck hypothesis.
- Memory/data movement.
- Measurement or benchmark plan.
- Result or limitation.

## Theory Gate

Run this gate for original theory, proofs, algorithms, or research claims.

Theory output:

- Definitions.
- Assumptions.
- Invariants.
- Theorems/propositions.
- Proof or derivation.
- Counterexamples.
- Testable predictions.
- Open conjectures.

## Verification Gate

Choose the smallest check that proves the claim:

- Config/docs: parse/lint and targeted grep.
- Single-file code: focused unit/type/lint check.
- Cross-module code: unit plus integration or smoke.
- UI: browser or screenshot check.
- Database: migration/query/transaction check.
- Security: adversarial cases and secret/permission scan.
- Performance: benchmark/profiler/query plan.

Final output must include:

- What changed.
- What was verified.
- What was not verified and why.
- Residual risk.

## Failure Behavior

If the system cannot prove a claim, it must say so. It should not fill gaps with
confidence. The correct behavior is to state the missing evidence and propose
the next check.
