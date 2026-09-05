from minigpt.data import StoryDataset, load_stories_from_file
from minigpt import get_tokenizer

DATA_PATH = "data/TinyStories-1000.txt"


def test_load_stories_respects_max_stories():
    stories = load_stories_from_file(DATA_PATH, max_stories=5)
    assert len(stories) == 5
    assert all(s.endswith("<|endoftext|>") for s in stories)


def test_dataset_item_is_padded_to_maxlen():
    tokenizer = get_tokenizer()
    stories = load_stories_from_file(DATA_PATH, max_stories=3)
    maxlen = 128
    ds = StoryDataset(stories, maxlen, tokenizer)
    item = ds[0]
    assert len(item) == maxlen
    assert len(ds) == 3


def test_short_story_is_right_padded_with_zeros():
    tokenizer = get_tokenizer()
    ds = StoryDataset(["hi<|endoftext|>"], maxlen=32, tokenizer=tokenizer)
    item = ds[0]
    assert len(item) == 32
    assert item[-1] == 0
