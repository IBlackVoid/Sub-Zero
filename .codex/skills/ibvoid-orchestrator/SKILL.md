---
name: ibvoid-orchestrator
description: Use for complex, ambiguous, or multi-domain engineering work that needs planning, task decomposition, specialist routing, integration, acceptance criteria, or progress tracking.
---

# IBVoid Orchestrator

Use this skill when the task is too large or cross-cutting for a single linear
edit. The orchestrator owns decomposition and integration; specialists own
domain judgments.

For serious work, consult the shared references when available:
`IBVOID_DOMAIN_DOCTRINE.md`, `IBVOID_EXECUTION_GATES.md`, and
`IBVOID_AGENT_MANDATES.md`. For the full specialist procedure, read
`IBVOID_AGENT_PLAYBOOKS/orchestrator.md`.

## Workflow

1. Define the objective in one sentence.
2. Identify constraints: repo state, user requirements, safety, time, tooling,
   dependencies, and verification limits.
3. Split the task into independently verifiable units.
4. Route each unit to the smallest useful specialist set.
5. Use the agent mandates to define what each specialist must inspect.
6. Keep one integration owner for coherence.
7. Define acceptance criteria before implementation.
8. After edits, run the verification gates proportional to risk.

## Routing Matrix

- Architecture, public API, ownership boundaries: architect.
- Security, auth, input parsing, secrets, permissions, shell, dependencies:
  security-reviewer.
- Hot paths, large data, algorithms, memory, latency: performance-engineer.
- UI, accessibility, browser behavior: frontend-engineer.
- APIs, jobs, service behavior: backend-engineer.
- Schema, indexes, migrations, transactions: database-engineer.
- CI/CD, Docker, deployment, logs, metrics: devops-engineer.
- ML, CV, evals, inference, embeddings: ai-ml-engineer.
- OS, Rust/C/C++, concurrency, memory, syscalls: systems-engineer.
- Binary/protocol archaeology: reverse-engineer.
- Formal theory, proofs, invariants, novel models: theory-lab.
- Final checks: verifier.

## Output Contract

For plans, produce:

- Objective.
- Work units.
- Owner/specialist for each unit.
- Dependencies between units.
- Acceptance criteria.
- Verification gate.

For implementation work, update the plan as work completes and keep the final
answer focused on changed files, checks run, and remaining risk.
