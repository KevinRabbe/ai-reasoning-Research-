from __future__ import annotations

from ai_reasoning.tasks.base import QueryType, RelationType, Task, TaskFamily

from .base import Protocol, ProtocolMessage


class TextProtocol(Protocol[bytes]):
    """Deterministic English renderer measured as UTF-8 bits."""

    def encode(self, task: Task) -> ProtocolMessage[bytes]:
        labels = [chr(ord("A") + i) if i < 26 else f"E{i}" for i in range(task.entity_count)]
        lines = [f"There are {task.entity_count} entities: {' '.join(labels)}."]

        if task.family is TaskFamily.CONSTRAINTS:
            for relation in task.relations:
                a = labels[relation.source]
                if relation.type is RelationType.BEFORE:
                    lines.append(f"{a} must appear before {labels[relation.target]}.")
                elif relation.type is RelationType.NOT_ADJACENT:
                    lines.append(f"{a} cannot be adjacent to {labels[relation.target]}.")
                elif relation.type is RelationType.POSITION_NOT_EQUAL:
                    lines.append(f"{a} cannot be at position {relation.target}.")
            if task.query_type is QueryType.VALID_ORDER:
                lines.append("Find a valid ordering.")
        elif task.family is TaskFamily.GRAPH:
            assert task.start is not None and task.goal is not None
            for relation in task.relations:
                lines.append(f"There is a directed edge from {labels[relation.source]} to {labels[relation.target]}.")
            if task.forbidden:
                lines.append("Forbidden entities: " + " ".join(labels[i] for i in task.forbidden) + ".")
            lines.append(f"Start at {labels[task.start]}. Goal is {labels[task.goal]}.")
            if task.query_type is QueryType.NEXT_STEP:
                lines.append("Return a next entity on a shortest valid path.")
        else:
            raise ValueError(f"unsupported task family: {task.family}")

        payload = "\n".join(lines).encode("utf-8")
        return ProtocolMessage(payload=payload, bit_cost=len(payload) * 8, symbol_count=len(payload))
