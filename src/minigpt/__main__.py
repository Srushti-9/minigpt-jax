import argparse
import sys

from .cli import train_cli, generate_cli, demo_cli


def main():
    parser = argparse.ArgumentParser(
        prog="python -m minigpt",
        description="MiniGPT-JAX — train, generate, or demo a tiny CPU GPT.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", add_help=False, help="Train MiniGPT on TinyStories")
    subparsers.add_parser("generate", add_help=False, help="Generate text from a checkpoint")
    subparsers.add_parser("demo", add_help=False, help="Launch the Gradio demo")

    args, rest = parser.parse_known_args()
    sys.argv = [f"minigpt {args.command}"] + rest

    dispatch = {
        "train": train_cli.main,
        "generate": generate_cli.main,
        "demo": demo_cli.main,
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
