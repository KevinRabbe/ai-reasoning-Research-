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

### Run tests

```bash
python -m pytest
```

### Run the 10k smoke benchmark

```bash
python -m ai_reasoning.eval.smoke --episodes 10000
```

The smoke benchmark validates that generated tasks are solvable and that oracle answers satisfy the canonical task definition. It does not train a neural model yet.

## Experimental controls

Every protocol will receive the same generated task. The task is created once from a seed and only the representation changes. Training and evaluation seeds will be separated, entity IDs are randomized, relation ordering is randomized, and renaming entities must not change correctness.

The next implementation milestone is the shared recurrent reasoning controller and the three trainable protocol paths: text, structured IR, and learned discrete machine-native communication.
