"""Train the tiny GPT on the included original learning corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import GPTConfig, TinyGPT
from .tokenizer import CharTokenizer, get_batch, load_text, make_splits

DATA_PATH = Path(__file__).parent / "data" / "learning_corpus.txt"
CHECKPOINT_PATH = Path(__file__).parent / "tiny_gpt.pt"


def estimate_loss(
    model: TinyGPT,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    config: GPTConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    result = {}
    with torch.no_grad():
        for name, data in (("train", train_data), ("validation", validation_data)):
            losses = []
            for _ in range(3):
                inputs, targets = get_batch(data, config.block_size, 16, device)
                _, loss = model(inputs, targets)
                losses.append(float(loss))
            result[name] = sum(losses) / len(losses)
    model.train()
    return result


def train(max_iters: int = 300, seed: int = 42) -> Path:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = load_text(DATA_PATH)
    tokenizer = CharTokenizer(text)
    train_data, validation_data = make_splits(text, tokenizer)
    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print(f"device={device} characters={len(text)} vocabulary={tokenizer.vocab_size}")
    for step in range(max_iters):
        inputs, targets = get_batch(train_data, config.block_size, 32, device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, max_iters // 5) == 0:
            metrics = estimate_loss(model, train_data, validation_data, config, device)
            print(
                f"step={step + 1:>4} train_loss={metrics['train']:.3f} "
                f"validation_loss={metrics['validation']:.3f}"
            )

    checkpoint = {
        "model": model.cpu().state_dict(),
        "config": config.__dict__,
        "vocabulary": tokenizer.vocabulary,
        "seed": seed,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"saved={CHECKPOINT_PATH}")
    return CHECKPOINT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(max_iters=args.max_iters, seed=args.seed)


if __name__ == "__main__":
    main()
