# AI Reasoning Research

Experimental work on machine-native cognition: reasoning systems whose internal communication is optimized for machines rather than human readability.

## POC-0

POC-0 tests one narrow hypothesis:

> Can a learned machine-native communication protocol match or exceed text and a human-designed intermediate representation while using less communication and generalizing at least as well?

The experiment deliberately begins without agents, RAG, external LLMs, persistent memory, or a UI. Those components would add confounders before the communication hypothesis is established.

### Current foundation

- canonical task representation hidden from protocol consumers
- deterministic relational-constraint task generator
- deterministic directed-graph task generator
- exact/validated oracle solvers
- deterministic text protocol
- compact human-designed structured protocol
- learned-protocol channel specification (256 symbols, 8 bits/symbol)
- entity-renaming utilities for invariance tests
- 10,000-episode deterministic smoke benchmark
- shared trainable byte encoder for Text and IR
- recurrent workspace controller with fixed parameter count across reasoning cycles
- parameter-free sinusoidal positions for message-length OOD evaluation
- IID-only graph `NEXT_STEP` answer head
- online supervised Text/IR graph training harness with separate evaluation seeds

### Neural baseline

Text and structured IR now use the same neural path:

```text
protocol bytes
    -> shared ByteMessageEncoder architecture
    -> machine-native message states
    -> recurrent workspace controller
    -> workspace state
```

The controller never receives the canonical `Task`, task-family metadata, entity count, oracle state, or any other shortcut. The byte message is the information boundary.

Default neural configuration:

- `d_model=256`
- 2 message-encoder Transformer layers
- 16 workspace slots
- 2 reasoning blocks per cycle
- 8 attention heads
- FFN width 1024
- ~3.76M trainable parameters before answer heads

Reasoning cycles are a runtime compute budget. The same controller blocks are reused on every cycle, so requesting more reasoning does not add parameters.

### Current supervised graph harness

The first trainer targets fixed-size IID graph `NEXT_STEP` problems for both Text and structured IR. It generates tasks online, derives labels from the graph oracle, and evaluates on a separate frozen seed range.

The current fixed-class graph head is intentionally **not** valid for OOD entity-count claims because output classes are tied to the training entity IDs. A candidate/pointer-style output mechanism must replace it before size-scaling evaluation.

A short CPU sanity run established that the full training path executes and gradients update the model, but it did not establish reliable task learning yet. The trainer is therefore a harness, not a benchmark result.

### Run tests

```bash
python -m pytest
```

### Run the 10k smoke benchmark

```bash
python -m ai_reasoning.eval.smoke --episodes 10000
```

The smoke benchmark validates that generated tasks are solvable and that oracle answers satisfy the canonical task definition.

## Experimental controls

Every protocol receives the same generated task. The task is created once from a seed and only the representation changes. Training and evaluation seeds are separated, entity IDs are randomized, relation ordering is randomized, and renaming entities must not change correctness.

Text and structured IR also share the exact same trainable byte-encoder and recurrent-controller architecture. Their input sequences differ, but neither path receives privileged task metadata.

The next implementation milestone is to make both graph baselines learn reliably, add frozen evaluation reporting, and replace the IID-only output head with an OOD-safe candidate mechanism before introducing the learned discrete machine-native protocol.
