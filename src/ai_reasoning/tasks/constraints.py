from __future__ import annotations

import random

from .base import QueryType, Relation, RelationType, Task, TaskFamily


def generate_constraint_task(
    seed: int,
    entity_count: int = 6,
    constraint_count: int | None = None,
) -> Task:
    if not 4 <= entity_count <= 32:
        raise ValueError("entity_count must be between 4 and 32")

    rng = random.Random(seed)
    if constraint_count is None:
        constraint_count = max(entity_count, int(entity_count * 1.5))
    if constraint_count < 3:
        raise ValueError("constraint_count must be at least 3")

    entities = list(range(entity_count))
    hidden_order = entities[:]
    rng.shuffle(hidden_order)
    position = {entity: index for index, entity in enumerate(hidden_order)}

    candidates: dict[RelationType, list[Relation]] = {
        RelationType.BEFORE: [],
        RelationType.NOT_ADJACENT: [],
        RelationType.POSITION_NOT_EQUAL: [],
    }

    for a in entities:
        for b in entities:
            if a == b:
                continue
            if position[a] < position[b]:
                candidates[RelationType.BEFORE].append(Relation(RelationType.BEFORE, a, b))
            if a < b and abs(position[a] - position[b]) != 1:
                candidates[RelationType.NOT_ADJACENT].append(
                    Relation(RelationType.NOT_ADJACENT, a, b)
                )
        for forbidden_position in range(entity_count):
            if position[a] != forbidden_position:
                candidates[RelationType.POSITION_NOT_EQUAL].append(
                    Relation(RelationType.POSITION_NOT_EQUAL, a, forbidden_position)
                )

    selected: list[Relation] = []
    for relation_type in (
        RelationType.BEFORE,
        RelationType.NOT_ADJACENT,
        RelationType.POSITION_NOT_EQUAL,
    ):
        pool = candidates[relation_type]
        if pool and len(selected) < constraint_count:
            selected.append(rng.choice(pool))

    remaining = [
        relation
        for pool in candidates.values()
        for relation in pool
        if relation not in selected
    ]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, constraint_count - len(selected))])
    rng.shuffle(selected)

    before_edges = sum(r.type is RelationType.BEFORE for r in selected)
    return Task(
        family=TaskFamily.CONSTRAINTS,
        entity_count=entity_count,
        relations=tuple(selected),
        query_type=QueryType.VALID_ORDER,
        seed=seed,
        stats=(("constraint_count", len(selected)), ("before_count", before_edges)),
    )
