from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .message import BYTE_VOCAB_SIZE, PAD_BYTE


def sinusoidal_positions(length: int, d_model: int, *, device, dtype) -> Tensor:
    """Parameter-free positions so OOD message lengths do not hit untrained embeddings."""

    if length < 1:
        raise ValueError("length must be positive")
    if d_model % 2:
        raise ValueError("d_model must be even for sinusoidal positions")

    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * (-math.log(10_000.0) / d_model)
    )
    encoding = torch.zeros((length, d_model), device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor)
    return encoding


class ByteMessageEncoder(nn.Module):
    """Shared trainable encoder for Text and structured-IR byte streams."""

    def __init__(
        self,
        *,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_symbols: int = 4096,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if max_symbols < 1:
            raise ValueError("max_symbols must be positive")

        self.d_model = d_model
        self.max_symbols = max_symbols
        self.embedding = nn.Embedding(BYTE_VOCAB_SIZE, d_model, padding_idx=PAD_BYTE)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, padding_mask: Tensor) -> Tensor:
        if tokens.ndim != 2 or padding_mask.shape != tokens.shape:
            raise ValueError("tokens and padding_mask must both have shape [batch, symbols]")
        if tokens.shape[1] > self.max_symbols:
            raise ValueError(
                f"message has {tokens.shape[1]} symbols; encoder limit is {self.max_symbols}"
            )
        if padding_mask.dtype is not torch.bool:
            raise TypeError("padding_mask must be boolean")
        if torch.any((tokens < 0) | (tokens >= BYTE_VOCAB_SIZE)):
            raise ValueError("token values must be byte IDs 0..255 or PAD_BYTE=256")
        if torch.any(padding_mask.all(dim=1)):
            raise ValueError("every batch row must contain at least one non-padding symbol")

        x = self.embedding(tokens) * math.sqrt(self.d_model)
        positions = sinusoidal_positions(
            tokens.shape[1], self.d_model, device=x.device, dtype=x.dtype
        )
        x = x + positions.unsqueeze(0)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.final_norm(x)
