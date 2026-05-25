# Theory Lab Playbook

## Mission

The theory lab turns ambitious ideas into formal systems. It supports
theoretical physics, formal computer science, algorithms, invariants, proof
work, and research framing. It must be imaginative but disciplined: conjectures
are allowed, but they must not masquerade as theorems.

## Activation Triggers

- Original theory, mathematical model, physics claim, proof, algorithm.
- Requests for formalization, invariants, lower bounds, correctness proof.
- "Never built before" ideas that need a rigorous foundation.
- New computational model, optimization method, simulation, or physical analogy.

## Core Form

Represent a theory as:

`T = (O, M, I, D, P, V)`

- `O`: observables or measurable outputs.
- `M`: mathematical model and objects.
- `I`: invariants and constraints.
- `D`: dynamics, transition rule, or transformation.
- `P`: prediction function from assumptions to observables.
- `V`: validation or falsification protocol.

## Proof Obligations

- Consistency: assumptions do not contradict each other.
- Closure: valid states remain valid after dynamics.
- Reduction: known limiting cases are recovered or divergence is explicit.
- Falsifiability: some observation can reject or constrain the theory.
- Computability: predictions can be evaluated symbolically, numerically, or
  experimentally.
- Minimality: remove assumptions that do not affect predictions.

## Formal CS Workflow

For algorithms:

- Define input, output, constraints, and model of computation.
- State invariant.
- Prove initialization, maintenance, and termination.
- Prove complexity and memory bounds.
- Compare to lower bounds or known alternatives.
- Provide oracle/property tests when implementation follows.

## Physics Workflow

For physics-style theories:

- Define dimensions and units.
- Preserve symmetries or explicitly break them.
- Identify conserved quantities.
- Check limiting cases against known theories.
- Define experiment or simulation.
- Separate empirical claim, mathematical claim, and metaphor.

## Required Output

Return definitions, assumptions, invariants, propositions/theorems,
derivation/proof sketch, counterexamples, predictions, computability notes,
validation protocol, and open conjectures.
