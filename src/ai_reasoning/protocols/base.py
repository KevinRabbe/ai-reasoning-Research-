from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from ai_reasoning.tasks.base import Task

PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True, slots=True)
class ProtocolMessage(Generic[PayloadT]):
    payload: PayloadT
    bit_cost: int
    symbol_count: int


class Protocol(ABC, Generic[PayloadT]):
    @abstractmethod
    def encode(self, task: Task) -> ProtocolMessage[PayloadT]:
        raise NotImplementedError
