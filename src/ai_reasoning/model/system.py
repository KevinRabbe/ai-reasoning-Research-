from __future__ import annotations

from torch import nn

from .controller import ReasoningOutput, RecurrentReasoningController
from .encoder import ByteMessageEncoder
from .message import ByteMessageBatch


class ByteReasoningSystem(nn.Module):
    """Text/IR experimental path: bytes -> shared encoder -> recurrent brain."""

    def __init__(
        self,
        *,
        d_model: int = 256,
        encoder_heads: int = 8,
        encoder_layers: int = 2,
        controller_heads: int = 8,
        controller_depth: int = 2,
        num_slots: int = 16,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_symbols: int = 4096,
    ) -> None:
        super().__init__()
        self.message_encoder = ByteMessageEncoder(
            d_model=d_model,
            num_heads=encoder_heads,
            num_layers=encoder_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            max_symbols=max_symbols,
        )
        self.controller = RecurrentReasoningController(
            d_model=d_model,
            num_slots=num_slots,
            num_heads=controller_heads,
            block_depth=controller_depth,
            ff_dim=ff_dim,
            dropout=dropout,
        )

    def forward(
        self,
        batch: ByteMessageBatch,
        *,
        reasoning_steps: int = 8,
        return_intermediate: bool = False,
    ) -> ReasoningOutput:
        message_states = self.message_encoder(batch.tokens, batch.padding_mask)
        return self.controller(
            message_states,
            batch.padding_mask,
            reasoning_steps=reasoning_steps,
            return_intermediate=return_intermediate,
        )
