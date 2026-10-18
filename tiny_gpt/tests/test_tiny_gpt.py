import torch

from tiny_gpt.model import GPTConfig, TinyGPT
from tiny_gpt.tokenizer import CharTokenizer, get_batch, make_splits


def test_character_tokenizer_round_trip():
    tokenizer = CharTokenizer("abc cab")
    text = "cab"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.vocab_size == 4


def test_model_forward_has_expected_shape_and_loss():
    config = GPTConfig(vocab_size=8, block_size=12, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(config)
    tokens = torch.randint(0, config.vocab_size, (2, config.block_size))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_batch_targets_are_shifted_by_one_character():
    tokenizer = CharTokenizer("abcdefghij" * 5)
    train, _ = make_splits("abcdefghij" * 5, tokenizer)
    inputs, targets = get_batch(train, block_size=5, batch_size=3, device=torch.device("cpu"))
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_generation_extends_the_prompt():
    config = GPTConfig(vocab_size=5, block_size=8, n_embd=16, n_head=4, n_layer=1)
    model = TinyGPT(config)
    prompt = torch.tensor([[1, 2]])
    generated = model.generate(prompt, max_new_tokens=3)
    assert generated.shape == (1, 5)
