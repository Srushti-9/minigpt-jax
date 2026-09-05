import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .data import END_OF_TEXT


@nnx.jit
def _decode_step(model, token, pos):
    logits = model(token, start_pos=pos, decode=True)
    return logits[:, -1, :]


def _sample(logits, key, temperature, top_k):
    if temperature <= 0.0:
        return jnp.argmax(logits, axis=-1)

    logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth_value = jnp.sort(logits, axis=-1)[:, -k]
        logits = jnp.where(logits < kth_value[:, None], -jnp.inf, logits)
    return jax.random.categorical(key, logits, axis=-1)


def generate_text(
    model,
    tokenizer,
    start_tokens,
    max_new_tokens=50,
    temperature=1.0,
    top_k=40,
    seed=0,
):
    end_token = tokenizer.encode(END_OF_TEXT, allowed_special={END_OF_TEXT})[0]
    key = jax.random.key(seed)

    prompt = list(start_tokens)[: model.maxlen]
    tokens = list(prompt)

    model.init_decode_cache(batch_size=1)
    try:
        pos = 0
        for tok in prompt:
            logits = _decode_step(model, jnp.array([[tok]]), jnp.array(pos))
            pos += 1

        for _ in range(max_new_tokens):
            if pos >= model.maxlen:
                break
            key, subkey = jax.random.split(key)
            next_token = int(_sample(logits, subkey, temperature, top_k)[0])
            if next_token == end_token:
                break
            tokens.append(next_token)
            logits = _decode_step(model, jnp.array([[next_token]]), jnp.array(pos))
            pos += 1
    finally:
        model.disable_decode()

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
