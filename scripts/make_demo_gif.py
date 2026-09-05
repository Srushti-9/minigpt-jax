from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT = Path("assets/demo.gif")
W, H = 860, 320
PAD = 24
BG = (15, 23, 42)
BAR = (30, 41, 59)
PROMPT_C = (56, 189, 248)
CMD_C = (226, 232, 240)
DIM = (148, 163, 184)
STORY_C = (167, 243, 208)
DOT_R, DOT_Y, DOT_G = (239, 68, 68), (234, 179, 8), (34, 197, 94)

RUNS = [
    ('python -m minigpt generate --prompt "Once upon a time" --temperature 0.4',
     "Once upon a time, there was a little girl named Tim. She loved to "
     "play named a little, the always. She was a little, the When her mom "
     "and was. She was the One day, she was"),
    ('python -m minigpt generate --prompt "The little cat" --temperature 0.5',
     "The little cat girl named boy Lily her special. She was a going to "
     "play with her. She had she had her she saw her. She was a her she was a"),
    ('python -m minigpt generate --prompt "Once there was a dog" --temperature 0.5',
     "Once there was a dog named Tim named Lily. He was a park to friends "
     "to play with the best a big. One day, day, but he was a big. It was"),
]


def load_font(size):
    for name in ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = load_font(17)
FONT_SM = load_font(13)


def wrap(draw, text, font, max_w):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def base_frame():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 40], fill=BAR)
    for i, c in enumerate((DOT_R, DOT_Y, DOT_G)):
        cx = 20 + i * 22
        d.ellipse([cx, 15, cx + 11, 26], fill=c)
    d.text((W // 2 - 70, 12), "minigpt-jax", font=FONT_SM, fill=DIM)
    return img


def draw_state(cmd, cmd_chars, story, story_chars, show_restore):
    img = base_frame()
    d = ImageDraw.Draw(img)
    y = 60
    d.text((PAD, y), "$", font=FONT, fill=PROMPT_C)
    d.text((PAD + 18, y), cmd[:cmd_chars], font=FONT, fill=CMD_C)
    y += 34
    if show_restore:
        d.text((PAD, y), "Model restored from minigpt_checkpoint.orbax",
               font=FONT_SM, fill=DIM)
        y += 24
        d.text((PAD, y), "--- Generated ---", font=FONT_SM, fill=DIM)
        y += 26
        for line in wrap(d, story[:story_chars], FONT, W - 2 * PAD):
            d.text((PAD, y), line, font=FONT, fill=STORY_C)
            y += 26
    return img


frames, durations = [], []

for run_i, (cmd, story) in enumerate(RUNS):
    for i in range(1, len(cmd) + 1, 3):
        frames.append(draw_state(cmd, i, story, 0, False))
        durations.append(22)
    frames.append(draw_state(cmd, len(cmd), story, 0, False))
    durations.append(420)

    frames.append(draw_state(cmd, len(cmd), story, 0, True))
    durations.append(520)

    for i in range(1, len(story) + 1, 3):
        frames.append(draw_state(cmd, len(cmd), story, i, True))
        durations.append(38)

    frames.append(draw_state(cmd, len(cmd), story, len(story), True))
    durations.append(2000 if run_i == len(RUNS) - 1 else 1100)

frames[0].save(
    OUT, save_all=True, append_images=frames[1:],
    duration=durations, loop=0, optimize=True, disposal=2,
)
print(f"wrote {OUT} ({len(frames)} frames, {OUT.stat().st_size // 1024} KB)")
