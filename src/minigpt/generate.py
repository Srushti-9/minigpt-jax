import jax
import jax.numpy as jnp

from .data import END_OF_TEXT


def _next_token(logits, key, temperature, top_k):
    if temperature <= 0.0:
        return int(jnp.argmax(logits))

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth_value = jnp.sort(logits)[-k]
        logits = jnp.where(logits < kth_value, -jnp.inf, logits)

    return int(jax.random.categorical(key, logits))


def generate_text(
    model,
    tokenizer,
    start_tokens,
    max_new_tokens=50,
    temperature=1.0,
    top_k=40,
    seed=0,
):
    tokens = list(start_tokens)
    end_token = tokenizer.encode(END_OF_TEXT, allowed_special={END_OF_TEXT})[0]
    key = jax.random.key(seed)

    for _ in range(max_new_tokens):
        context = tokens[-model.maxlen :]
        actual_len = len(context)
        if actual_len < model.maxlen:
            context = context + [0] * (model.maxlen - actual_len)

        context_array = jnp.array(context)[None, :]
        logits = model(context_array)[0, actual_len - 1, :]

        key, subkey = jax.random.split(key)
        next_token = _next_token(logits, subkey, temperature, top_k)

        if next_token == end_token:
            break
        tokens.append(next_token)

    return tokenizer.decode(tokens)


def generate_story(
    model,
    tokenizer,
    story_prompt,
    temperature=1.0,
    max_new_tokens=50,
    top_k=40,
    seed=0,
):
    start_tokens = tokenizer.encode(story_prompt)[: model.maxlen]
    return generate_text(
        model,
        tokenizer,
        start_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )
