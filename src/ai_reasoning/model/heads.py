from __future__ import annotations

from torch import Tensor, nn


class GraphNextStepHead(nn.Module):
    """IID baseline head for a fixed entity-count graph benchmark.

    This head is intentionally not used for OOD entity-count claims because its
    output classes are tied to the training entity IDs. A pointer/candidate head
    will replace it before size-scaling evaluation.
    """

    def __init__(self, *, d_model: int, entity_count: int) -> None:
        super().__init__()
        if entity_count < 2:
            raise ValueError("entity_count must be at least 2")
        self.entity_count = entity_count
        self.slot_scores = nn.Linear(d_model, 1, bias=False)
        self.classifier = nn.Linear(d_model, entity_count)

    def forward(self, workspace: Tensor) -> Tensor:
        if workspace.ndim != 3:
            raise ValueError("workspace must have shape [batch, slots, d_model]")
        weights = self.slot_scores(workspace).squeeze(-1).softmax(dim=-1)
        pooled = (workspace * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(pooled)
