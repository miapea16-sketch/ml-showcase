"""A tiny decoder-only GPT built from PyTorch primitives for learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head attention that prevents each position from seeing the future."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.n_embd, config.n_head, dropout=config.dropout, batch_first=True
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.block_size, config.block_size, dtype=torch.bool), diagonal=1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        attended, _ = self.attention(x, x, x, attn_mask=self.causal_mask[:length, :length])
        return attended


class TransformerBlock(nn.Module):
    """Pre-normalized attention + feed-forward block with residual connections."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.norm_feed_forward = nn.LayerNorm(config.n_embd)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_attention(x))
        return x + self.feed_forward(self.norm_feed_forward(x))


class TinyGPT(nn.Module):
    """Small character-level GPT for next-token prediction."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.n_layer)))
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, length = tokens.shape
        if length > self.config.block_size:
            raise ValueError("Input sequence is longer than the configured block size")
        positions = torch.arange(length, device=tokens.device)
        x = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        x = self.final_norm(self.blocks(x))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self, tokens: torch.Tensor, max_new_tokens: int, temperature: float = 0.8
    ) -> torch.Tensor:
        """Sample new token IDs from the model."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eval()
        for _ in range(max_new_tokens):
            context = tokens[:, -self.config.block_size :]
            logits, _ = self(context)
            probabilities = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
