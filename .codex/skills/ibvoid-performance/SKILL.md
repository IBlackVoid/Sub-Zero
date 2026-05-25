---
name: ibvoid-performance
description: Use for algorithmic complexity, hot-path optimization, large data, latency, throughput, memory allocation, cache behavior, SIMD, rendering performance, database query performance, or benchmark design.
---

# IBVoid Performance

Use this skill when speed, scale, memory, or mechanical sympathy matters.
Optimize from evidence and workload shape, not aesthetic preference.

For serious work, consult the shared references when available:
`IBVOID_DOMAIN_DOCTRINE.md`, `IBVOID_EXECUTION_GATES.md`, and
`IBVOID_AGENT_MANDATES.md`. For the full specialist procedure, read
`IBVOID_AGENT_PLAYBOOKS/performance-engineer.md`.

## Performance Pass

1. Define the workload and target metric.
2. Estimate complexity and identify hot paths.
3. Inspect data movement: allocation, copying, cache locality, serialization,
   syscalls, database round trips, and network calls.
4. Choose the simplest optimization with measurable impact.
5. Verify with benchmark, profiler, trace, query plan, or controlled smoke test.

## Heuristics

- Prefer better algorithms before micro-optimization.
- Keep hot data contiguous when iteration dominates.
- Batch I/O and network calls.
- Avoid hidden allocations in loops.
- Prefer streaming for large data.
- Use indexes and set-based queries for database workloads.
- Keep abstractions zero-cost in hot paths; prove if necessary.

## Output Contract

Return:

- Baseline assumption or measurement.
- Bottleneck hypothesis.
- Proposed change.
- Expected complexity and memory impact.
- Verification method and results, or why measurement was not possible.
