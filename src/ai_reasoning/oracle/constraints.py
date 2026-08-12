from __future__ import annotations

from dataclasses import dataclass

from ai_reasoning.tasks.base import RelationType, Task, TaskFamily


@dataclass(frozen=True, slots=True)
class ConstraintOracleResult:
    answer: tuple[int, ...]
    valid_answers: tuple[tuple[int, ...], ...]
    truncated: bool


def validate_ordering(task: Task, ordering: tuple[int, ...] | list[int]) -> bool:
    if task.family is not TaskFamily.CONSTRAINTS:
        raise ValueError("task is not a constraint task")
    if len(ordering) != task.entity_count or set(ordering) != set(range(task.entity_count)):
        return False

    pos = {entity: index for index, entity in enumerate(ordering)}
    for relation in task.relations:
        if relation.type is RelationType.BEFORE:
            if pos[relation.source] >= pos[relation.target]:
                return False
        elif relation.type is RelationType.NOT_ADJACENT:
            if abs(pos[relation.source] - pos[relation.target]) == 1:
                return False
        elif relation.type is RelationType.POSITION_NOT_EQUAL:
            if pos[relation.source] == relation.target:
                return False
        else:
            raise ValueError(f"unsupported constraint relation: {relation.type}")
    return True


def solve_constraints(task: Task, max_solutions: int = 64) -> ConstraintOracleResult:
    if task.family is not TaskFamily.CONSTRAINTS:
        raise ValueError("task is not a constraint task")
    if max_solutions < 1:
        raise ValueError("max_solutions must be positive")

    before: set[tuple[int, int]] = set()
    non_adjacent: set[frozenset[int]] = set()
    banned_positions: dict[int, set[int]] = {}

    for relation in task.relations:
        if relation.type is RelationType.BEFORE:
            before.add((relation.source, relation.target))
        elif relation.type is RelationType.NOT_ADJACENT:
            non_adjacent.add(frozenset((relation.source, relation.target)))
        elif relation.type is RelationType.POSITION_NOT_EQUAL:
            banned_positions.setdefault(relation.source, set()).add(relation.target)

    predecessors: dict[int, set[int]] = {node: set() for node in range(task.entity_count)}
    for source, target in before:
        predecessors[target].add(source)

    solutions: list[tuple[int, ...]] = []
    truncated = False

    def search(prefix: list[int], remaining: set[int]) -> None:
        nonlocal truncated
        if len(solutions) > max_solutions:
            truncated = True
            return
        if not remaining:
            candidate = tuple(prefix)
            if validate_ordering(task, candidate):
                solutions.append(candidate)
            return

        position = len(prefix)
        previous = prefix[-1] if prefix else None
        for node in sorted(remaining):
            if position in banned_positions.get(node, ()):
                continue
            if not predecessors[node].issubset(prefix):
                continue
            if previous is not None and frozenset((previous, node)) in non_adjacent:
                continue
            search(prefix + [node], remaining - {node})
            if truncated:
                return

    search([], set(range(task.entity_count)))
    if not solutions:
        raise ValueError("constraint task has no solution")

    visible = tuple(solutions[:max_solutions])
    return ConstraintOracleResult(answer=visible[0], valid_answers=visible, truncated=truncated)
