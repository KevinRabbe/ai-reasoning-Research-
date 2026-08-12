from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ai_reasoning.tasks.base import RelationType, Task, TaskFamily


@dataclass(frozen=True, slots=True)
class GraphOracleResult:
    answer: int
    valid_answers: tuple[int, ...]
    optimal_path: tuple[int, ...]
    path_length: int


def _adjacency(task: Task) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    outgoing = {node: [] for node in range(task.entity_count)}
    incoming = {node: [] for node in range(task.entity_count)}
    for relation in task.relations:
        if relation.type is not RelationType.EDGE:
            raise ValueError(f"unsupported graph relation: {relation.type}")
        outgoing[relation.source].append(relation.target)
        incoming[relation.target].append(relation.source)
    for values in outgoing.values():
        values.sort()
    for values in incoming.values():
        values.sort()
    return outgoing, incoming


def solve_graph(task: Task) -> GraphOracleResult:
    if task.family is not TaskFamily.GRAPH:
        raise ValueError("task is not a graph task")
    if task.start is None or task.goal is None:
        raise ValueError("graph task requires start and goal")

    outgoing, incoming = _adjacency(task)
    forbidden = set(task.forbidden)
    if task.start in forbidden or task.goal in forbidden:
        raise ValueError("start and goal must not be forbidden")

    distance = {task.goal: 0}
    queue = deque([task.goal])
    while queue:
        node = queue.popleft()
        for predecessor in incoming[node]:
            if predecessor in forbidden or predecessor in distance:
                continue
            distance[predecessor] = distance[node] + 1
            queue.append(predecessor)

    if task.start not in distance:
        raise ValueError("graph task has no valid path")

    valid_next = tuple(
        node
        for node in outgoing[task.start]
        if node not in forbidden and distance.get(node) == distance[task.start] - 1
    )
    if not valid_next:
        raise ValueError("graph task has no valid next step")

    path = [task.start]
    current = task.start
    while current != task.goal:
        candidates = [
            node
            for node in outgoing[current]
            if node not in forbidden and distance.get(node) == distance[current] - 1
        ]
        if not candidates:
            raise RuntimeError("distance map is inconsistent")
        current = min(candidates)
        path.append(current)

    return GraphOracleResult(
        answer=valid_next[0],
        valid_answers=valid_next,
        optimal_path=tuple(path),
        path_length=len(path) - 1,
    )


def validate_next_step(task: Task, answer: int) -> bool:
    return answer in solve_graph(task).valid_answers
