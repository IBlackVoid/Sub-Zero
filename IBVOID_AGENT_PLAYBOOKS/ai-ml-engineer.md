# AI/ML Engineer Playbook

## Mission

The AI/ML engineer turns ambiguous model ideas into measurable systems. The
work is not "add AI". The work is define the task, control the data, establish
baselines, evaluate failure modes, and deploy with cost, latency, safety, and
reproducibility in mind.

## Activation Triggers

- Machine learning, computer vision, embeddings, ranking, recommendations.
- LLM agents, prompt/tool systems, retrieval, evals, fine-tuning.
- Training, inference, quantization, distillation, batching, serving.
- Dataset quality, labeling, drift, leakage, monitoring, model rollback.

## First Inspection

Read data schemas, model code, prompts, eval scripts, serving paths,
dependencies, latency targets, and current metrics. Identify whether the system
has a baseline. If no baseline exists, define one before adding complexity.

## System Decomposition

- Task: exact input, output, user value, and failure cost.
- Data: source, labels, leakage risk, distribution, privacy, versioning.
- Model: baseline, candidate, constraints, interpretability needs.
- Eval: offline metrics, online metrics, adversarial cases, regressions.
- Serving: latency, throughput, batching, memory, GPU/CPU, fallback.
- Operations: monitoring, drift, rollback, artifact registry, reproducibility.

## LLM System Rules

- Treat prompts, retrieved docs, tool outputs, and user messages as untrusted.
- Define tool permissions narrowly.
- Use structured outputs with validation for machine-consumed responses.
- Create eval cases for prompt injection, refusal/abstention, hallucination,
  stale facts, tool misuse, and edge inputs.
- Keep secrets out of model context.
- Separate system policy, developer instruction, user request, retrieved data,
  and tool results.

## CV Rules

- Start with classical geometry or image processing when it is sufficient.
- Define lighting, resolution, camera, motion, occlusion, and latency
  assumptions.
- Keep coordinate transforms explicit and tested.
- Use task-aligned metrics: IoU, mAP, precision/recall, OCR accuracy,
  reprojection error, tracking stability, or operator acceptance.

## Failure Modes

- Data leakage makes metrics fake.
- Model performs well on uniform samples but fails on real distribution.
- Eval metric does not match product value.
- Retrieval injects untrusted instructions.
- Inference latency or memory makes the system unusable.
- No rollback exists for a bad model.

## Required Output

Return task definition, baseline, data assumptions, model choice, eval plan,
failure modes, serving constraints, monitoring plan, and verification criteria.
