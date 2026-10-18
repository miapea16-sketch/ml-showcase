"""Generate text from a trained tiny GPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import GPTConfig, TinyGPT
from .tokenizer import CharTokenizer

DEFAULT_CHECKPOINT = Path(__file__).parent / "tiny_gpt.pt"


def generate(
    prompt: str, checkpoint_path: Path = DEFAULT_CHECKPOINT, max_new_tokens: int = 250
) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tokenizer = CharTokenizer("".join(checkpoint["vocabulary"]))
    config = GPTConfig(**checkpoint["config"])
    model = TinyGPT(config)
    model.load_state_dict(checkpoint["model"])
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    output = model.generate(tokens, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="machine learning ")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-new-tokens", type=int, default=250)
    args = parser.parse_args()
    print(generate(args.prompt, args.checkpoint, args.max_new_tokens))


if __name__ == "__main__":
    main()
