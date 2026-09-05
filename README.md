# MiniGPT-JAX

A tiny GPT-style language model built with **JAX** and **Flax NNX**, small enough to
train and run on a laptop **CPU**. It learns from the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
dataset and generates short children's-story text.

This project is a packaged, runnable version of the model developed in the
DeepLearning.AI *"LLMs with JAX"* course — reorganised from teaching notebooks into a
clean, installable Python package with a command-line interface.

## What's inside

```
src/minigpt/
  config.py       # ModelConfig + TrainConfig dataclasses (all hyperparameters)
  model.py        # MiniGPT: token+position embedding, causal attention blocks, output head
  data.py         # TinyStories loader + Grain data pipeline
  train.py        # training loop (AdamW, cosine LR schedule, JIT-compiled step)
  generate.py     # autoregressive text generation
  checkpoint.py   # Orbax save / restore (CPU)
  __main__.py     # `python -m minigpt train|generate|demo` dispatcher
  cli/            # train / generate / demo command implementations
```

### Architecture note

The transformer block here is **attention-only** (a residual connection around
multi-head self-attention), matching the course's introductory model exactly. It does
**not** include a feed-forward network, layer normalisation, or dropout. This keeps it
faithful to the course; it is intentionally minimal, not a production architecture.

## Requirements

- Python **3.12+** (developed and verified on 3.13)
- CPU is sufficient — no GPU required

> **Version note:** dependency versions match the course pins, with one exception:
> `tiktoken` is bumped from `0.4.0` to `0.9.0`, because `0.4.0` ships no wheel for
> Python 3.13 (pip would try to compile it from Rust source and fail). The `gpt2`
> encoding API used here is unchanged between those versions. `grain==0.2.13` provides
> a native Windows wheel for Python 3.13, so **no WSL is needed**.

## Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e .
```

## Usage

All commands run through `python -m minigpt <command>`. This module-based entry
point works everywhere — including locked-down Windows environments where generated
`.exe` console-script shims are blocked by security policy.

## Train

Trains on a small slice of TinyStories and writes a checkpoint. The defaults are tiny
and finish in seconds on a CPU:

```bash
python -m minigpt train --max-stories 100 --epochs 3 --batch-size 24
```

Common flags: `--data-path`, `--checkpoint-path`, `--max-stories`, `--epochs`,
`--batch-size`, `--peak-lr`, `--shuffle`, plus architecture flags (`--embed-dim`,
`--num-heads`, `--num-blocks`, `--maxlen`).

> **Expect gibberish from the tiny run.** With only 100 stories and a handful of steps,
> the loss barely moves and the output is not coherent — this is the same behaviour as
> the course's small demo. Real story-like output requires the full TinyStories corpus
> and many more steps. The point of the small run is to verify the pipeline end-to-end.

## Generate

```bash
python -m minigpt generate --checkpoint minigpt_checkpoint.orbax \
  --prompt "Once upon a time" --max-new-tokens 30 --temperature 0.2
```

> The generator uses greedy `argmax` decoding (matching the course). Because of this,
> `--temperature` does not change the output — it is kept for API compatibility with
> the course code. Sampling-based decoding would be a natural next extension.

## Interactive demo

Launches a local Gradio web UI:

```bash
python -m minigpt demo --checkpoint minigpt_checkpoint.orbax
```

## Data

`data/TinyStories-1000.txt` is a ~1,000-story slice of TinyStories, included so the
project runs with zero setup. It is **not** the full dataset. Stories are separated by
the `<|endoftext|>` token.

## License

MIT — see [LICENSE](LICENSE).
