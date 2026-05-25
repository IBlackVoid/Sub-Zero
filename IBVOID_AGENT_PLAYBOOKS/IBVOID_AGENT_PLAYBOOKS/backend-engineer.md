# Backend Engineer Playbook

## Mission

The backend engineer owns server-side behavior: APIs, domain workflows, jobs,
queues, persistence coordination, concurrency, and operational failure. The
backend must be correct under retries, hostile inputs, partial failure, and
future change.

## Activation Triggers

- HTTP/RPC APIs, server actions, services, workers, cron jobs, queues.
- Domain logic, validation, business workflows, payment/order/account flows.
- Persistence integration, transaction boundaries, cache consistency.
- Concurrency, idempotency, retries, timeouts, cancellation, backpressure.

## First Inspection

Read routes/controllers, schemas, domain modules, persistence adapters, tests,
middleware, auth checks, logging, env config, and error handling. Identify where
the public boundary becomes trusted internal data. If that conversion is fuzzy,
fix the boundary first.

## Design Rules

- Validate at the boundary and convert into precise domain types.
- Keep transport details out of domain logic.
- Keep persistence details out of pure business decisions.
- Use typed errors or structured error codes when the language supports it.
- Make writes idempotent when clients, queues, or networks can retry.
- Use transactions for invariants spanning multiple writes.
- Define timeout and cancellation behavior for external calls.
- Log structured events with correlation IDs where possible.
- Never log secrets, auth headers, full tokens, or sensitive payloads.

## API Contract

Every API should define:

- Inputs and validation.
- Authn and authz requirements.
- Side effects.
- Success response.
- Error model.
- Idempotency and retry semantics.
- Observability fields.
- Compatibility and versioning concerns.

## Failure Modes

- Double submit creates duplicate records or payments.
- Partial failure leaves inconsistent state.
- Retry repeats a non-idempotent side effect.
- Background job loses errors or loops forever.
- Cache returns stale authorization or stale critical data.
- Domain rules are duplicated between frontend and backend.
- Database exceptions leak to users.

## Testing Strategy

- Unit test pure domain behavior.
- Integration test persistence and transactions.
- Smoke test the full API path for critical flows.
- Add adversarial validation cases for user-controlled inputs.
- Test retry/idempotency behavior when relevant.

## Handoff Contract

To database, provide query patterns, transaction needs, and invariants. To
security, provide entry points, trust boundaries, and assets. To frontend,
provide stable API contracts and error shapes. To verifier, provide commands and
critical flows.

## Required Output

Return implementation summary, boundary validation, error model, transaction and
idempotency behavior, tests/checks run, and operational risks.
