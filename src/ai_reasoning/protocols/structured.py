from __future__ import annotations

from ai_reasoning.tasks.base import Task

from .base import Protocol, ProtocolMessage


class StructuredProtocol(Protocol[bytes]):
    """Human-designed compact byte protocol. Every byte costs exactly 8 bits."""

    VERSION = 1

    def encode(self, task: Task) -> ProtocolMessage[bytes]:
        if task.entity_count > 255:
            raise ValueError("structured v1 supports at most 255 entities")

        data = bytearray(
            [
                self.VERSION,
                int(task.family),
                task.entity_count,
                int(task.query_type),
                255 if task.start is None else task.start,
                255 if task.goal is None else task.goal,
                len(task.forbidden),
            ]
        )
        data.extend(task.forbidden)
        if len(task.relations) > 255:
            raise ValueError("structured v1 supports at most 255 relations")
        data.append(len(task.relations))
        for relation in task.relations:
            if relation.source > 255 or relation.target > 255:
                raise ValueError("structured v1 fields must fit in one byte")
            data.extend((int(relation.type), relation.source, relation.target))

        payload = bytes(data)
        return ProtocolMessage(payload=payload, bit_cost=len(payload) * 8, symbol_count=len(payload))
