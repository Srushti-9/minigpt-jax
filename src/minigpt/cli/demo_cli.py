import argparse

import gradio as gr
import flax.nnx as nnx

from ..config import ModelConfig
from ..model import MiniGPT
from ..checkpoint import load_checkpoint
from ..generate import generate_story
from .. import get_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Launch a Gradio demo for a trained MiniGPT.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--maxlen", type=int, default=ModelConfig.maxlen)
    parser.add_argument("--embed-dim", type=int, default=ModelConfig.embed_dim)
    parser.add_argument("--num-heads", type=int, default=ModelConfig.num_heads)
    parser.add_argument("--num-blocks", type=int, default=ModelConfig.num_transformer_blocks)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    tokenizer = get_tokenizer()
    cfg = ModelConfig(
        vocab_size=tokenizer.n_vocab,
        maxlen=args.maxlen,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_transformer_blocks=args.num_blocks,
    )
    model = MiniGPT(cfg, rngs=nnx.Rngs(0))
    load_checkpoint(model, args.checkpoint)

    def create_story(story_prompt, temperature, max_new_tokens, top_k):
        return generate_story(
            model, tokenizer, story_prompt,
            temperature=temperature, max_new_tokens=int(max_new_tokens),
            top_k=int(top_k),
        )

    demo = gr.Interface(
        fn=create_story,
        inputs=[
            gr.Textbox(label="Story Prompt"),
            gr.Slider(minimum=0.01, maximum=1.5, value=0.8, step=0.01, label="Temperature"),
            gr.Slider(minimum=1, maximum=200, value=30, step=1, label="Max Tokens"),
            gr.Slider(minimum=0, maximum=100, value=40, step=1, label="Top-K (0 = off)"),
        ],
        outputs=["text"],
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
