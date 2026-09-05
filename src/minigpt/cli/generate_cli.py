import argparse

import flax.nnx as nnx

from ..config import ModelConfig
from ..model import MiniGPT
from ..checkpoint import load_checkpoint
from ..generate import generate_story
from .. import get_tokenizer


def build_model(tokenizer, args):
    cfg = ModelConfig(
        vocab_size=tokenizer.n_vocab,
        maxlen=args.maxlen,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_transformer_blocks=args.num_blocks,
    )
    return MiniGPT(cfg, rngs=nnx.Rngs(0))


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained MiniGPT checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40,
                        help="Sample from the top K tokens (0 disables top-k).")
    parser.add_argument("--seed", type=int, default=0, help="Sampling RNG seed.")
    parser.add_argument("--maxlen", type=int, default=ModelConfig.maxlen)
    parser.add_argument("--embed-dim", type=int, default=ModelConfig.embed_dim)
    parser.add_argument("--num-heads", type=int, default=ModelConfig.num_heads)
    parser.add_argument("--num-blocks", type=int, default=ModelConfig.num_transformer_blocks)
    args = parser.parse_args()

    tokenizer = get_tokenizer()
    model = build_model(tokenizer, args)
    load_checkpoint(model, args.checkpoint)

    text = generate_story(
        model, tokenizer, args.prompt,
        temperature=args.temperature, max_new_tokens=args.max_new_tokens,
        top_k=args.top_k, seed=args.seed,
    )
    print("\n--- Generated ---")
    print(text)


if __name__ == "__main__":
    main()
