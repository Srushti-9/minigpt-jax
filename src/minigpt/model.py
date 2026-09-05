import jax.numpy as jnp
import flax.nnx as nnx

from .config import ModelConfig


class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen, vocab_size, embed_dim, *, rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, x, start_pos=0):
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len) + start_pos
        return self.token_emb(x) + self.pos_emb(positions[None, :])


class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, *, rngs):
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
            TransformerBlock(config.embed_dim, config.num_heads, rngs=rngs)
            for _ in range(config.num_transformer_blocks)
        ]
        self.output_layer = nnx.Linear(
            config.embed_dim, config.vocab_size, use_bias=False, rngs=rngs
        )

    def causal_attention_mask(self, seq_len):
        return jnp.tril(jnp.ones((seq_len, seq_len)))

    def init_decode_cache(self, batch_size=1):
        for block in self.transformer_blocks:
            block.attention.decode = True
            block.attention.init_cache((batch_size, self.maxlen, self.embedding.token_emb.features))

    def disable_decode(self):
        for block in self.transformer_blocks:
            block.attention.decode = False

    def __call__(self, token_ids, start_pos=0, decode=False):
        x = self.embedding(token_ids, start_pos=start_pos)
        mask = None if decode else self.causal_attention_mask(token_ids.shape[1])
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
        return self.output_layer(x)
