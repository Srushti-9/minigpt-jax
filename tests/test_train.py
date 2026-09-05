import jax.numpy as jnp
import flax.nnx as nnx
import optax

from minigpt.config import ModelConfig
from minigpt.model import MiniGPT
from minigpt.train import loss_fn


def small_model():
    cfg = ModelConfig(
        vocab_size=256, maxlen=16, embed_dim=32, num_heads=4, num_transformer_blocks=2
    )
    return MiniGPT(cfg, rngs=nnx.Rngs(0))


def test_loss_ignores_padding_positions():
    model = small_model()
    inputs = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    targets = jnp.array([[2, 3, 0, 0]], dtype=jnp.int32)

    loss, logits = loss_fn(model, (inputs, targets))

    per_token = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    expected = per_token[0, :2].mean()

    assert bool(jnp.isclose(loss, expected, atol=1e-5))


def test_loss_is_finite_when_all_targets_are_padding():
    model = small_model()
    inputs = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    targets = jnp.zeros((1, 4), dtype=jnp.int32)

    loss, _ = loss_fn(model, (inputs, targets))

    assert bool(jnp.isfinite(loss))
    assert float(loss) == 0.0
