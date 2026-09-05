import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from .config import ModelConfig, TrainConfig
from .data import load_and_preprocess_data
from .model import MiniGPT


def loss_fn(model, batch):
    inputs, targets = batch
    logits = model(inputs)
    token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    mask = (targets != 0).astype(token_loss.dtype)
    loss = (token_loss * mask).sum() / jnp.clip(mask.sum(), min=1.0)
    return loss, logits


@nnx.jit
def train_step(model, optimizer, metrics, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch[1])
    optimizer.update(grads)


def train(model, tokenizer, cfg: TrainConfig):
    text_dl, batches_per_epoch = load_and_preprocess_data(
        file_path=cfg.data_path,
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        maxlen=model.maxlen,
        max_stories=cfg.max_stories,
        num_epochs=1,
        shuffle=cfg.shuffle,
        seed=cfg.seed,
    )

    total_steps = batches_per_epoch * cfg.num_epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_fraction))
    print(f"Total training steps: {total_steps:,} | warmup: {warmup_steps:,}")

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=cfg.end_lr,
    )
    optimizer = nnx.Optimizer(
        model, optax.adamw(learning_rate=lr_schedule, weight_decay=cfg.weight_decay)
    )
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    metrics_history = {"train_loss": []}

    prep_target_batch = jax.vmap(
        lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0])))
    )

    for epoch in range(cfg.num_epochs):
        step = 0
        for batch in text_dl:
            batch_t = jnp.asarray(batch).T
            input_batch = batch_t.astype(jnp.int32)
            target_batch = prep_target_batch(batch_t).astype(jnp.int32)
            print(".", end="", flush=True)
            train_step(model, optimizer, metrics, (input_batch, target_batch))

            if (step + 1) % cfg.log_every == 0:
                for metric, value in metrics.compute().items():
                    metrics_history[f"train_{metric}"].append(float(value))
                metrics.reset()
                current_lr = lr_schedule(step)
                print(
                    f"\nEpoch: {epoch + 1}, Step {step + 1}, "
                    f"Loss: {metrics_history['train_loss'][-1]:.4f}, "
                    f"LR: {current_lr:.2e}"
                )
            step += 1

    print()
    return model, metrics_history
