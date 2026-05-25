# IBVoid Novel Project Protocol

Use this protocol for projects that are exploratory, research-heavy, or intended
to do something not already standard.

## Principle

Novel work is not an excuse to be vague. The more original the goal, the more
disciplined the evidence loop must be.

## Loop

1. Theory: define the claim, invariants, and expected behavior.
2. Research: check prior art, constraints, and known impossibilities.
3. Prototype: build the smallest artifact that can falsify the claim.
4. Measure: collect evidence against target metrics.
5. Decide: keep, revise, or discard the approach.
6. Capture: update ADRs, experiment logs, risk register, and verification.

## Claim Template

```text
Claim:
Why it might be possible:
Known constraints:
What would falsify it:
Smallest prototype:
Metric:
Decision threshold:
```

## Prototype Rules

- A prototype must answer one question.
- Do not let prototype code silently become production code.
- Mark shortcuts explicitly.
- Measure the thing that matters, not the thing that is easy.
- Preserve failed experiments when they prevent repeating mistakes.

## Research Rules

- Search for known impossibility results, lower bounds, patents only if relevant,
  standards, prior systems, and benchmark baselines.
- Prefer papers, specs, source code, and official docs.
- Record date/version context.
- Convert research into design constraints or ADRs.

## Decision Rules

Keep an approach only if:

- It satisfies core invariants.
- It beats or justifiably differs from baseline.
- Its failure modes are understood.
- It can be verified again later.

Discard or revise if:

- The assumption is falsified.
- Complexity grows faster than capability.
- Security or reliability depends on hope.
- Performance only works on toy inputs.

## Required Artifacts

- `EXPERIMENT_LOG.md`
- `RESEARCH_LOG.md`
- `RISK_REGISTER.md`
- ADR for any durable decision
- Benchmark or verification entry for any performance claim
