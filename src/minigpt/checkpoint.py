from pathlib import Path

import jax
import flax.nnx as nnx
import orbax.checkpoint as ocp
from jax.sharding import SingleDeviceSharding


def _abs(path):
    return Path(path).resolve()


def save_checkpoint(model, path):
    path = _abs(path)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(path, nnx.state(model), force=True)
    print(f"Model saved to {path}")
    return path


def load_checkpoint(model, path):
    path = _abs(path)
    cpu_device = jax.devices("cpu")[0]
    cpu_sharding = SingleDeviceSharding(cpu_device)

    abstract_state = nnx.state(model)
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.ArrayRestoreArgs(sharding=cpu_sharding), abstract_state
    )

    checkpointer = ocp.PyTreeCheckpointer()
    restored_state = checkpointer.restore(
        path, item=abstract_state, restore_args=restore_args
    )
    nnx.update(model, restored_state)
    print(f"Model restored from {path}")
    return model
