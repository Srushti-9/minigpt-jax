import argparse

import flax.nnx as nnx

from ..config import ModelConfig, TrainConfig
from ..model import MiniGPT
from ..train import train
from ..checkpoint import save_checkpoint
from .. import get_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train MiniGPT on TinyStories (CPU-friendly).")
    parser.add_argument("--data-path", default=TrainConfig.data_path)
    parser.add_argument("--checkpoint-path", default=TrainConfig.checkpoint_path)
    parser.add_argument("--max-stories", type=int, default=TrainConfig.max_stories)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.num_epochs)
    parser.add_argument("--peak-lr", type=float, default=TrainConfig.peak_lr)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--maxlen", type=int, default=ModelConfig.maxlen)
    parser.add_argument("--embed-dim", type=int, default=ModelConfig.embed_dim)
    parser.add_argument("--num-heads", type=int, default=ModelConfig.num_heads)
    parser.add_argument("--num-blocks", type=int, default=ModelConfig.num_transformer_blocks)
    args = parser.parse_args()

    tokenizer = get_tokenizer()
    model_cfg = ModelConfig(
        vocab_size=tokenizer.n_vocab,
        maxlen=args.maxlen,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_transformer_blocks=args.num_blocks,
    )
    train_cfg = TrainConfig(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        max_stories=args.max_stories,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        peak_lr=args.peak_lr,
        seed=args.seed,
        shuffle=args.shuffle,
    )

    model = MiniGPT(model_cfg, rngs=nnx.Rngs(train_cfg.seed))
    model, _ = train(model, tokenizer, train_cfg)
    save_checkpoint(model, train_cfg.checkpoint_path)


if __name__ == "__main__":
    main()
