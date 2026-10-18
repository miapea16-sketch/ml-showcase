"""Character-level tokenization and reproducible mini-batch helpers."""

from __future__ import annotations

from pathlib import Path

import torch


class CharTokenizer:
    def __init__(self, text: str) -> None:
        self.vocabulary = sorted(set(text))
        self.stoi = {char: index for index, char in enumerate(self.vocabulary)}
        self.itos = {index: char for char, index in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        unknown = set(text) - set(self.stoi)
        if unknown:
            raise ValueError(f"Text contains characters outside the vocabulary: {unknown}")
        return [self.stoi[char] for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[token] for token in tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def make_splits(
    text: str, tokenizer: CharTokenizer, train_fraction: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_at = int(train_fraction * len(encoded))
    return encoded[:split_at], encoded[split_at:]


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size:
        raise ValueError("Corpus split must be longer than block_size")
    starts = torch.randint(len(data) - block_size, (batch_size,))
    inputs = torch.stack([data[start : start + block_size] for start in starts])
    targets = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return inputs.to(device), targets.to(device)
