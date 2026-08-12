from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ai_reasoning.model import GraphBaselineModel, batch_byte_messages
from ai_reasoning.oracle.graphs import solve_graph
from ai_reasoning.protocols import StructuredProtocol, TextProtocol
from ai_reasoning.tasks.graphs import generate_graph_task


@dataclass(frozen=True, slots=True)
class GraphTrainConfig:
    protocol: str = "structured"
    entity_count: int = 8
    batch_size: int = 32
    train_steps: int = 1000
    reasoning_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    train_seed: int = 0
    eval_seed: int = 1_000_000
    eval_episodes: int = 512


def make_protocol(name: str):
    if name == "text":
        return TextProtocol()
    if name in {"ir", "structured"}:
        return StructuredProtocol()
    raise ValueError("protocol must be 'text' or 'structured'")


def graph_batch(*, protocol, seeds: list[int], entity_count: int, device) -> tuple:
    messages = []
    targets = []
    for seed in seeds:
        task = generate_graph_task(seed=seed, entity_count=entity_count)
        messages.append(protocol.encode(task))
        targets.append(solve_graph(task).answer)
    return (
        batch_byte_messages(messages, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def evaluate_graph(model, *, protocol, config: GraphTrainConfig, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for offset in range(0, config.eval_episodes, config.batch_size):
            size = min(config.batch_size, config.eval_episodes - offset)
            seeds = [config.eval_seed + offset + i for i in range(size)]
            batch, targets = graph_batch(
                protocol=protocol, seeds=seeds, entity_count=config.entity_count, device=device
            )
            logits = model(batch, reasoning_steps=config.reasoning_steps)
            correct += int((logits.argmax(dim=-1) == targets).sum().item())
            total += size
    return correct / total


def train_graph_baseline(
    config: GraphTrainConfig,
    *,
    model_kwargs: dict | None = None,
    device: str | torch.device | None = None,
) -> tuple[GraphBaselineModel, dict[str, float]]:
    if config.train_steps < 1 or config.batch_size < 1:
        raise ValueError("train_steps and batch_size must be positive")
    if not 4 <= config.entity_count <= 64:
        raise ValueError("entity_count must be between 4 and 64")

    torch.manual_seed(config.train_seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    protocol = make_protocol(config.protocol)
    model = GraphBaselineModel(
        entity_count=config.entity_count, **(model_kwargs or {})
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    final_loss = 0.0
    for step in range(config.train_steps):
        base = config.train_seed + step * config.batch_size
        seeds = [base + i for i in range(config.batch_size)]
        batch, targets = graph_batch(
            protocol=protocol,
            seeds=seeds,
            entity_count=config.entity_count,
            device=resolved_device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch, reasoning_steps=config.reasoning_steps)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())

    accuracy = evaluate_graph(
        model, protocol=protocol, config=config, device=resolved_device
    )
    return model, {"final_loss": final_loss, "eval_accuracy": accuracy}
