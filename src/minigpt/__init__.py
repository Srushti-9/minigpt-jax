import tiktoken

from .config import ModelConfig, TrainConfig
from .model import MiniGPT, TransformerBlock, TokenAndPositionEmbedding
from .data import StoryDataset, load_stories_from_file, load_and_preprocess_data
from .generate import generate_text, generate_story
from .train import train
from .checkpoint import save_checkpoint, load_checkpoint


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


__all__ = [
    "ModelConfig",
    "TrainConfig",
    "MiniGPT",
    "TransformerBlock",
    "TokenAndPositionEmbedding",
    "StoryDataset",
    "load_stories_from_file",
    "load_and_preprocess_data",
    "generate_text",
    "generate_story",
    "train",
    "save_checkpoint",
    "load_checkpoint",
    "get_tokenizer",
]
