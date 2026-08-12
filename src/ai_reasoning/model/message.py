from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor

from ai_reasoning.protocols.base import ProtocolMessage

PAD_BYTE = 256
BYTE_VOCAB_SIZE = 257


@dataclass(frozen=True, slots=True)
class ByteMessageBatch:
    """Padded byte messages plus accounting metadata.

    `padding_mask=True` marks positions that must be ignored by attention.
    """

    tokens: Tensor
    padding_mask: Tensor
    bit_costs: Tensor
    symbol_counts: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def max_symbols(self) -> int:
        return int(self.tokens.shape[1])


def batch_byte_messages(
    messages: Sequence[ProtocolMessage[bytes]],
    *,
    device: torch.device | str | None = None,
) -> ByteMessageBatch:
    """Pad byte protocol messages without changing their measured bit cost."""

    if not messages:
        raise ValueError("at least one message is required")
    if any(not isinstance(message.payload, bytes) for message in messages):
        raise TypeError("byte message batching requires bytes payloads")
    if any(len(message.payload) == 0 for message in messages):
        raise ValueError("empty protocol messages are not supported")

    lengths = [len(message.payload) for message in messages]
    max_length = max(lengths)
    tokens = torch.full(
        (len(messages), max_length),
        PAD_BYTE,
        dtype=torch.long,
        device=device,
    )
    padding_mask = torch.ones(
        (len(messages), max_length),
        dtype=torch.bool,
        device=device,
    )

    for row, message in enumerate(messages):
        payload = torch.tensor(list(message.payload), dtype=torch.long, device=device)
        tokens[row, : payload.numel()] = payload
        padding_mask[row, : payload.numel()] = False

    bit_costs = torch.tensor(
        [message.bit_cost for message in messages], dtype=torch.long, device=device
    )
    symbol_counts = torch.tensor(
        [message.symbol_count for message in messages], dtype=torch.long, device=device
    )
    return ByteMessageBatch(tokens, padding_mask, bit_costs, symbol_counts)
