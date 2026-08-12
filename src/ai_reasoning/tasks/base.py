from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class TaskFamily(IntEnum):
    CONSTRAINTS = 1
    GRAPH = 2


class RelationType(IntEnum):
    BEFORE = 1
    NOT_ADJACENT = 2
    POSITION_NOT_EQUAL = 3
    EDGE = 16


class QueryType(IntEnum):
    VALID_ORDER = 1
    NEXT_STEP = 16


@dataclass(frozen=True, slots=True)
class Relation:
    type: RelationType
    source: int
    target: int


@dataclass(frozen=True, slots=True)
class Task:
    family: TaskFamily
    entity_count: int
    relations: tuple[Relation, ...]
    query_type: QueryType
    seed: int
    start: int | None = None
    goal: int | None = None
    forbidden: tuple[int, ...] = ()
    stats: tuple[tuple[str, int], ...] = ()

    def stat(self, name: str, default: int | None = None) -> int | None:
        for key, value in self.stats:
            if key == name:
                return value
        return default


def rename_entities(task: Task, mapping: tuple[int, ...]) -> Task:
    """Apply a bijection old_id -> new_id while preserving task structure."""
    if len(mapping) != task.entity_count or set(mapping) != set(range(task.entity_count)):
        raise ValueError("mapping must be a permutation of all entity IDs")

    renamed_relations: list[Relation] = []
    for relation in task.relations:
        source = mapping[relation.source]
        if relation.type is RelationType.POSITION_NOT_EQUAL:
            target = relation.target
        else:
            target = mapping[relation.target]
        renamed_relations.append(Relation(relation.type, source, target))

    return Task(
        family=task.family,
        entity_count=task.entity_count,
        relations=tuple(renamed_relations),
        query_type=task.query_type,
        seed=task.seed,
        start=None if task.start is None else mapping[task.start],
        goal=None if task.goal is None else mapping[task.goal],
        forbidden=tuple(mapping[node] for node in task.forbidden),
        stats=task.stats,
    )
