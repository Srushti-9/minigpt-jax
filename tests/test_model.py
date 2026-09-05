import jax.numpy as jnp
import flax.nnx as nnx

from minigpt.config import ModelConfig
from minigpt.model import MiniGPT


def small_model():
    cfg = ModelConfig(
        vocab_size=256, maxlen=16, embed_dim=32, num_heads=4, num_transformer_blocks=2
    )
    return MiniGPT(cfg, rngs=nnx.Rngs(0)), cfg


def test_forward_pass_returns_logits_over_vocab():
    model, cfg = small_model()
    batch, seq = 3, 10
    tokens = jnp.zeros((batch, seq), dtype=jnp.int32)
    logits = model(tokens)
    assert logits.shape == (batch, seq, cfg.vocab_size)


def test_causal_mask_is_lower_triangular():
    model, _ = small_model()
    mask = model.causal_attention_mask(4)
    assert mask.shape == (4, 4)
    assert bool((mask == jnp.tril(jnp.ones((4, 4)))).all())


def test_decode_matches_full_sequence_logits():
    model, _ = small_model()
    seq = 8
    tokens = jnp.arange(1, seq + 1, dtype=jnp.int32)[None, :]

    full = model(tokens)

    model.init_decode_cache(batch_size=1)
    step_logits = []
    for t in range(seq):
        out = model(tokens[:, t : t + 1], start_pos=t, decode=True)
        step_logits.append(out)
    model.disable_decode()
    decoded = jnp.concatenate(step_logits, axis=1)

    assert decoded.shape == full.shape
    assert bool(jnp.allclose(full, decoded, atol=1e-4))
