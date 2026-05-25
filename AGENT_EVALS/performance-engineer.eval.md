# Performance Engineer Eval

## Prompt

Analyze a pipeline that processes 10 million events, enriches each event with
database lookups, and renders aggregate results to a dashboard. Find the likely
hot paths and benchmark plan.

## Expected Behavior

- Defines workload and target metrics.
- Spots N+1/database round trips, allocation pressure, batching, indexing,
  streaming, and frontend rendering risk.
- Produces baseline needs, bottleneck hypotheses, complexity/data movement,
  proposed measurements, and residual risk.
