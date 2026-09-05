from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    maxlen: int = 128
    embed_dim: int = 192
    num_heads: int = 6
    num_transformer_blocks: int = 6
    feed_forward_dim: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "feed_forward_dim", int(2 / 3 * 4 * self.embed_dim))


@dataclass
class TrainConfig:
    data_path: str = "data/TinyStories-1000.txt"
    checkpoint_path: str = "minigpt_checkpoint.orbax"
    max_stories: int = 1000
    batch_size: int = 24
    num_epochs: int = 20
    peak_lr: float = 3e-4
    end_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    shuffle: bool = False
    seed: int = 42
    log_every: int = 2
