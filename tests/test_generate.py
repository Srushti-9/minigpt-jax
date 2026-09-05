import flax.nnx as nnx

from minigpt.config import ModelConfig
from minigpt.model import MiniGPT
from minigpt.generate import generate_text, generate_story
from minigpt import get_tokenizer


def build():
    tokenizer = get_tokenizer()
    cfg = ModelConfig(
        vocab_size=tokenizer.n_vocab,
        maxlen=16,
        embed_dim=32,
        num_heads=4,
        num_transformer_blocks=2,
    )
    return MiniGPT(cfg, rngs=nnx.Rngs(0)), tokenizer


def test_generate_text_returns_string():
    model, tokenizer = build()
    out = generate_text(model, tokenizer, [1, 2, 3], max_new_tokens=5, temperature=1.0)
    assert isinstance(out, str)


def test_generate_story_from_prompt_returns_string():
    model, tokenizer = build()
    out = generate_story(model, tokenizer, "Once upon a time", max_new_tokens=5)
    assert isinstance(out, str)
    assert len(out) > 0


def test_generation_is_deterministic_with_fixed_weights():
    model, tokenizer = build()
    a = generate_text(model, tokenizer, [1, 2, 3], max_new_tokens=8, temperature=1.0)
    b = generate_text(model, tokenizer, [1, 2, 3], max_new_tokens=8, temperature=1.0)
    assert a == b
