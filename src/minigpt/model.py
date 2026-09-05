import jax.numpy as jnp
import flax.nnx as nnx

from .config import ModelConfig


class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, *, rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, x):
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)


class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, *, rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
            rngs=rngs,
        )

    def __call__(self, x, mask=None):
        attn_out = self.attention(x, mask=mask)
        return x + attn_out


class MiniGPT(nnx.Module):
    def __init__(self, config: ModelConfig = ModelConfig(), *, rngs: nnx.Rngs = nnx.Rngs(0)):
        self.maxlen = config.maxlen
        self.embedding = TokenAndPositionEmbedding(
            config.maxlen, config.vocab_size, config.embed_dim, rngs=rngs
        )
        self.transformer_blocks = [
            TransformerBlock(
                config.embed_dim, config.num_heads, config.feed_forward_dim, rngs=rngs
            )
            for _ in range(config.num_transformer_blocks)
        ]
        self.output_layer = nnx.Linear(
            config.embed_dim, config.vocab_size, use_bias=False, rngs=rngs
        )

    def causal_attention_mask(self, seq_len):
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def __call__(self, token_ids):
        seq_len = token_ids.shape[1]
        mask = self.causal_attention_mask(seq_len)
        x = self.embedding(token_ids)
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
        return self.output_layer(x)
