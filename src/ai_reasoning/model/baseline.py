from __future__ import annotations

from torch import Tensor, nn

from .heads import GraphNextStepHead
from .message import ByteMessageBatch
from .system import ByteReasoningSystem


class GraphBaselineModel(nn.Module):
    def __init__(self, *, entity_count: int = 8, **system_kwargs) -> None:
        super().__init__()
        d_model = int(system_kwargs.get("d_model", 256))
        self.reasoner = ByteReasoningSystem(**system_kwargs)
        self.head = GraphNextStepHead(d_model=d_model, entity_count=entity_count)

    def forward(self, batch: ByteMessageBatch, *, reasoning_steps: int = 8) -> Tensor:
        output = self.reasoner(batch, reasoning_steps=reasoning_steps)
        return self.head(output.workspace)
