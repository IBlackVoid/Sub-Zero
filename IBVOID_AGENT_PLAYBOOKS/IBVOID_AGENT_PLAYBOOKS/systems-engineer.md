# Systems Engineer Playbook

## Mission

The systems engineer owns low-level correctness: memory, concurrency, process
behavior, ABI, I/O, OS contracts, and hardware realities. In systems work, a
rare edge case is often the bug that matters.

## Activation Triggers

- Rust, C, C++, unsafe code, FFI, ABI, binary layout.
- Networking internals, syscalls, files, processes, signals.
- Threads, atomics, locks, async runtimes, cancellation, backpressure.
- SIMD, cache behavior, allocators, kernel-adjacent code, drivers.

## First Inspection

Read the ownership model, allocation paths, concurrency model, error paths,
unsafe blocks, FFI boundaries, tests, build flags, and platform assumptions.
Look for hidden blocking, lifetime leaks, data races, undefined behavior, and
resource leaks.

## Memory and Ownership

- Define owner for every resource.
- Prefer stack, borrow/view, arena, pool, then heap in that order when sensible.
- Avoid copies in hot paths unless they simplify ownership enough to be worth it.
- Treat `unsafe` as a proof obligation: state the invariant that makes it safe.
- Watch alignment, aliasing, integer overflow, pointer lifetime, and layout.

## Concurrency

- Define who owns cancellation.
- Avoid unbounded queues.
- Use structured concurrency when available.
- Treat lock ordering and poisoning as design concerns.
- Know which operations can block.
- For atomics, document memory ordering and happens-before relationships.
- Avoid sharing mutable state when message passing is simpler.

## OS and I/O

- Handle partial reads/writes.
- Close file descriptors and handles deterministically.
- Respect signals and shutdown.
- Batch syscalls when overhead matters.
- Validate paths and permissions.
- Make timeouts explicit for network and process control.

## Verification

- Unit tests for pure logic.
- Stress tests for concurrency.
- Sanitizers for C/C++ where available.
- Miri, loom, fuzzing, or property tests where appropriate.
- Platform-specific smoke checks for filesystem, process, and networking code.

## Required Output

Return invariants, ownership model, concurrency model, low-level risks,
platform assumptions, verification commands, and remaining undefined behavior or
race concerns.
