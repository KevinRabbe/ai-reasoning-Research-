from __future__ import annotations

import random

from .base import QueryType, Relation, RelationType, Task, TaskFamily


def generate_graph_task(
    seed: int,
    entity_count: int = 8,
    edge_count: int | None = None,
    forbidden_count: int = 1,
) -> Task:
    if not 4 <= entity_count <= 64:
        raise ValueError("entity_count must be between 4 and 64")
    if forbidden_count < 0 or forbidden_count > entity_count - 2:
        raise ValueError("invalid forbidden_count")

    rng = random.Random(seed)
    entities = list(range(entity_count))
    start, goal = rng.sample(entities, 2)

    available_mid = [node for node in entities if node not in (start, goal)]
    max_path_edges = min(entity_count - forbidden_count - 1, max(2, entity_count // 2))
    path_edges = rng.randint(2, max_path_edges)
    intermediates = rng.sample(available_mid, path_edges - 1)
    backbone = [start, *intermediates, goal]

    safe_backbone = set(backbone)
    forbidden_candidates = [node for node in entities if node not in safe_backbone]
    forbidden = tuple(sorted(rng.sample(forbidden_candidates, min(forbidden_count, len(forbidden_candidates)))))

    edges: set[tuple[int, int]] = set(zip(backbone, backbone[1:]))
    if edge_count is None:
        edge_count = max(len(edges), entity_count * 2)
    max_edges = entity_count * (entity_count - 1)
    edge_count = min(max(edge_count, len(edges)), max_edges)

    candidates = [
        (a, b)
        for a in entities
        for b in entities
        if a != b and (a, b) not in edges and (a, b) != (start, goal)
    ]
    rng.shuffle(candidates)
    edges.update(candidates[: edge_count - len(edges)])

    relations = [Relation(RelationType.EDGE, a, b) for a, b in edges]
    rng.shuffle(relations)

    return Task(
        family=TaskFamily.GRAPH,
        entity_count=entity_count,
        relations=tuple(relations),
        query_type=QueryType.NEXT_STEP,
        seed=seed,
        start=start,
        goal=goal,
        forbidden=forbidden,
        stats=(
            ("edge_count", len(relations)),
            ("backbone_length", len(backbone) - 1),
            ("forbidden_count", len(forbidden)),
        ),
    )
