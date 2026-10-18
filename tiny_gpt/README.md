# Tiny GPT from Scratch

An educational, character-level GPT-style language model built with PyTorch building blocks. It is small enough to train on a laptop and explicit enough to inspect line by line.

This project is inspired by the video you shared and follows the decoder-only Transformer idea: token embeddings, position embeddings, causal multi-head self-attention, layer normalization, residual connections, a feed-forward network, and a next-token prediction head.

## Important expectation

This is **not ChatGPT** and it is not intended to produce reliable general-purpose answers. It learns patterns from the small original corpus in `data/learning_corpus.txt`. The goal is to understand the mechanism, not to train a useful assistant.

## Run it

From the repository root:

```bash
pip install -e ".[tiny-gpt]"
python -m tiny_gpt.train --max-iters 300
python -m tiny_gpt.generate "machine learning " --max-new-tokens 200
```

Training uses CUDA automatically when available and otherwise runs on CPU. A checkpoint is written to `tiny_gpt/tiny_gpt.pt`; that generated file is ignored by Git.

## What to inspect

- `model.py`: the model architecture and causal mask
- `tokenizer.py`: character-to-integer encoding and random training batches
- `train.py`: next-token loss, AdamW optimization, gradient clipping, and checkpointing
- `generate.py`: autoregressive sampling one character at a time

The model learns by receiving a sequence such as `Machine learn` and trying to predict the next character at every position. The target sequence is the same sequence shifted one character to the left.

## Architecture

The default model has 2 Transformer blocks, 4 attention heads, 64-dimensional embeddings, a context window of 64 characters, and a vocabulary made from the included corpus. The causal attention mask makes sure a token cannot use future characters when predicting the next one.

The implementation uses PyTorch for tensors, automatic differentiation, and optimized primitives. “From scratch” here means the Transformer structure is written explicitly rather than calling a pretrained language model. A pure NumPy implementation with manual backpropagation would be a separate, much larger learning exercise.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), the original Transformer paper
- [PyTorch Transformer reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
