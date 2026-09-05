import jax.numpy as jnp

from .data import END_OF_TEXT


def generate_text(model, tokenizer, start_tokens, max_new_tokens=50, temperature=1.0):
    tokens = list(start_tokens)
    end_token = tokenizer.encode(END_OF_TEXT, allowed_special={END_OF_TEXT})[0]

    for _ in range(max_new_tokens):
        context = tokens[-model.maxlen :]
        actual_len = len(context)
        if actual_len < model.maxlen:
            context = context + [0] * (model.maxlen - actual_len)

        context_array = jnp.array(context)[None, :]
        logits = model(context_array)
        next_token_logits = logits[0, actual_len - 1, :] / temperature
        next_token = int(jnp.argmax(next_token_logits))

        if next_token == end_token:
            break
        tokens.append(next_token)

    return tokenizer.decode(tokens)


def generate_story(model, tokenizer, story_prompt, temperature=1.0, max_new_tokens=50):
    start_tokens = tokenizer.encode(story_prompt)[: model.maxlen]
    return generate_text(
        model,
        tokenizer,
        start_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
