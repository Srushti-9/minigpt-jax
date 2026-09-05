from pathlib import Path

import grain.python as pygrain

END_OF_TEXT = "<|endoftext|>"


def load_stories_from_file(file_path, max_stories=None):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"Loading stories from {file_path}...")
    stories = []
    current_story = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if END_OF_TEXT in line:
                parts = line.split(END_OF_TEXT)
                for part in parts[:-1]:
                    current_story.append(part)
                    story_text = "".join(current_story).strip()
                    if story_text:
                        stories.append(story_text + END_OF_TEXT)
                        if max_stories and len(stories) >= max_stories:
                            break
                    current_story = []
                current_story = [parts[-1]] if parts[-1].strip() else []
                if max_stories and len(stories) >= max_stories:
                    break
            else:
                current_story.append(line)

        if current_story and (not max_stories or len(stories) < max_stories):
            story_text = "".join(current_story).strip()
            if story_text:
                stories.append(story_text + END_OF_TEXT)

    print(f"Loaded {len(stories):,} stories")
    return stories


class StoryDataset:
    def __init__(self, stories, maxlen, tokenizer):
        self.stories = stories
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.end_token = tokenizer.encode(
            END_OF_TEXT, allowed_special={END_OF_TEXT}
        )[0]

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        tokens = self.tokenizer.encode(story, allowed_special={END_OF_TEXT})
        if len(tokens) > self.maxlen:
            tokens = tokens[: self.maxlen]
            tokens[-1] = self.end_token
        tokens.extend([0] * (self.maxlen - len(tokens)))
        return tokens


def load_and_preprocess_data(
    file_path,
    tokenizer,
    batch_size,
    maxlen,
    max_stories=100_000,
    num_epochs=1,
    shuffle=False,
    seed=42,
    worker_count=0,
):
    stories = load_stories_from_file(file_path, max_stories=max_stories)
    if not stories:
        raise ValueError("No valid stories found in the dataset")

    estimated_batches_per_epoch = len(stories) // batch_size
    print(f"Estimated batches per epoch: {estimated_batches_per_epoch:,}")

    dataset = StoryDataset(stories, maxlen, tokenizer)
    sampler = pygrain.IndexSampler(
        num_records=len(dataset),
        shuffle=shuffle,
        seed=seed,
        shard_options=pygrain.NoSharding(),
        num_epochs=num_epochs,
    )
    dataloader = pygrain.DataLoader(
        data_source=dataset,
        sampler=sampler,
        operations=[pygrain.Batch(batch_size=batch_size, drop_remainder=True)],
        worker_count=worker_count,
    )
    print(f"Created DataLoader with batch_size={batch_size}, maxlen={maxlen}")
    return dataloader, estimated_batches_per_epoch
