from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ReasoningOutput:
    workspace: Tensor
    intermediate_states: tuple[Tensor, ...] = ()


class ReasoningBlock(nn.Module):
    """One machine-native reasoning transformation.

    The same block instance is reused on every reasoning cycle. No language or
    human-designed cognitive action labels exist in this loop.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        workspace: Tensor,
        message_states: Tensor,
        message_padding_mask: Tensor,
    ) -> Tensor:
        query = self.cross_norm(workspace)
        cross, _ = self.cross_attention(
            query,
            message_states,
            message_states,
            key_padding_mask=message_padding_mask,
            need_weights=False,
        )
        workspace = workspace + self.dropout(cross)

        query = self.self_norm(workspace)
        self_state, _ = self.self_attention(query, query, query, need_weights=False)
        workspace = workspace + self.dropout(self_state)
        workspace = workspace + self.dropout(self.feed_forward(self.ff_norm(workspace)))
        return workspace


class RecurrentReasoningController(nn.Module):
    """Fixed-size reasoning core whose parameters are reused across cycles."""

    def __init__(
        self,
        *,
        d_model: int = 256,
        num_slots: int = 16,
        num_heads: int = 8,
        block_depth: int = 2,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_slots < 1 or block_depth < 1:
            raise ValueError("num_slots and block_depth must be positive")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_slots = num_slots
        self.initial_workspace = nn.Parameter(torch.empty(num_slots, d_model))
        nn.init.normal_(self.initial_workspace, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(
            ReasoningBlock(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(block_depth)
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        message_states: Tensor,
        message_padding_mask: Tensor,
        *,
        reasoning_steps: int = 8,
        return_intermediate: bool = False,
    ) -> ReasoningOutput:
        if reasoning_steps < 1:
            raise ValueError("reasoning_steps must be positive")
        if message_states.ndim != 3:
            raise ValueError("message_states must have shape [batch, symbols, d_model]")
        if message_states.shape[-1] != self.d_model:
            raise ValueError("message d_model does not match controller d_model")
        if message_padding_mask.shape != message_states.shape[:2]:
            raise ValueError("message_padding_mask shape must match message sequence")

        batch_size = message_states.shape[0]
        workspace = self.initial_workspace.unsqueeze(0).expand(batch_size, -1, -1)
        intermediates: list[Tensor] = []

        for _ in range(reasoning_steps):
            for block in self.blocks:
                workspace = block(workspace, message_states, message_padding_mask)
            if return_intermediate:
                intermediates.append(self.final_norm(workspace))

        workspace = self.final_norm(workspace)
        return ReasoningOutput(workspace, tuple(intermediates))
