# IBVoid Agent Output Contracts

These contracts define the minimum useful output from each specialist. They keep
agents from returning impressive but unusable prose.

## Orchestrator

```text
Objective:
Constraints:
Unknowns:
Specialists:
Work Units:
Dependencies:
Acceptance Criteria:
Verification Gates:
Execution Order:
Residual Risk:
```

## Architect

```text
Context:
Decision:
System Boundaries:
Data Flow:
Invariants:
Rejected Alternatives:
Migration Path:
Risks:
Verification:
ADR Needed: yes/no
```

## Researcher

```text
Question:
Sources Checked:
Facts:
Version/Date Context:
Implications:
Uncertainty:
Recommendation:
```

## Security Reviewer

```text
Scope:
Assets:
Trust Boundaries:
Attacker-Controlled Inputs:
Findings:
Fixes:
Verification Cases:
Residual Risk:
```

Each finding must include severity, surface, impact, and fix.

## Performance Engineer

```text
Workload:
Target Metric:
Baseline:
Bottleneck Hypothesis:
Complexity:
Data Movement:
Proposed Change:
Measurement:
Result or Blocker:
Residual Risk:
```

## Backend Engineer

```text
Boundary:
Domain Behavior:
Validation:
Error Model:
Transaction/Idempotency:
Observability:
Tests:
Operational Risk:
```

## Frontend Engineer

```text
User Workflow:
State Model:
Components:
Accessibility:
Responsive Behavior:
Error/Loading/Empty States:
Verification:
Visual Risk:
```

## Database Engineer

```text
Data Invariants:
Schema Changes:
Indexes:
Queries:
Transactions/Isolation:
Migration Plan:
Rollback Risk:
Verification:
```

## DevOps Engineer

```text
Build/Run Commands:
Environment:
Secrets:
CI/CD:
Deployment:
Health Checks:
Observability:
Rollback:
Operational Risk:
```

## AI/ML Engineer

```text
Task Definition:
Data Assumptions:
Baseline:
Model/System Choice:
Evaluation:
Failure Modes:
Serving Constraints:
Monitoring:
Verification:
```

## Systems Engineer

```text
Ownership Model:
Memory Layout:
Concurrency Model:
OS/IO Assumptions:
Unsafe/FFI Boundaries:
Failure Modes:
Verification:
Residual Low-Level Risk:
```

## Reverse Engineer

```text
Observations:
Evidence:
Hypotheses:
Confidence:
Reconstructed Structure:
Unknowns:
Next Checks:
Compatibility Constraints:
```

## Theory Lab

```text
Definitions:
Assumptions:
T = (O, M, I, D, P, V):
Propositions:
Proof/Derivation:
Counterexamples:
Predictions:
Validation Protocol:
Open Conjectures:
```

## Verifier

```text
Acceptance Criteria:
Changed Surface:
Checks Run:
Results:
Checks Not Run:
Failures:
Residual Risk:
Go/No-Go:
```

## Final Integration

The final response after implementation must include:

```text
Changed:
Verified:
Not Verified:
Residual Risk:
```

Keep final output concise unless the user asks for a report.
