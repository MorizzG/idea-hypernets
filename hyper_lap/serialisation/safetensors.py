"""
Save models split into a json file for the hyper parameters neccessary to recreated the static
pytree structure and a safetensors file containing the actual parameters.

inspired by https://github.com/krypticmouse/saferax
"""

from jaxtyping import Array, PyTree
from typing import Any, Optional

import json
from pathlib import Path

import equinox as eqx
import jax.tree as jt
from safetensors import safe_open
from safetensors.flax import save_file


def as_path(path: str | Path):
    if isinstance(path, Path):
        pass
    elif isinstance(path, str):
        path = Path(path)
    else:
        raise ValueError(f"invalid path {path}")

    return path


def load_file(filename: str | Path, strip_prefix: Optional[str] = None):
    result = {}

    with safe_open(filename, framework="flax") as f:
        key: str
        for key in f.keys():
            if strip_prefix:
                if not key.startswith(strip_prefix):
                    continue

                new_key = key.removeprefix(strip_prefix + ".")
            else:
                new_key = key

            result[new_key] = f.get_tensor(key)

    return result


def to_state_dict(tree: PyTree) -> dict[str, Array]:
    # filter out non-arrays
    tree = eqx.filter(tree, eqx.is_array_like)

    paths_and_values = jt.flatten_with_path(tree)[0]

    arrays = {}

    for tree_path, value in paths_and_values:
        path_parts: list[str] = []

        for p in tree_path:
            if hasattr(p, "name"):
                path_parts.append(p.name)
            elif hasattr(p, "idx"):
                path_parts.append(str(p.idx))
            else:
                raise ValueError(f"Don't know how to deal with tree path part {p}")

        path_str = ".".join(path_parts)

        arrays[path_str] = value

    return arrays


def load_state_dict(
    tree: PyTree,
    state_dict: dict[str, Array],
    *,
    match_exact: bool = True,
) -> PyTree:
    # we want to modify state_dict, but not the original at the call site
    state_dict = state_dict.copy()

    tree, static = eqx.partition(tree, eqx.is_array_like)

    paths_and_values, treedef = jt.flatten_with_path(tree)

    new_values = []

    for tree_path, value in paths_and_values:
        path_parts: list[str] = []

        for p in tree_path:
            if hasattr(p, "name"):
                path_parts.append(p.name)
            elif hasattr(p, "idx"):
                path_parts.append(str(p.idx))
            else:
                raise ValueError(f"Don't know how to deal with tree path part {p}")

        path_str = ".".join(path_parts)

        if path_str not in state_dict:
            raise ValueError(f"state dict is missing key {path_str}")

        new_value = state_dict.pop(path_str)

        if not eqx.is_array(value):
            if new_value.shape != ():
                raise ValueError(f"Expected {type(value)}, found {new_value}")

            new_value = new_value.item()
        # elif isinstance(value, float):
        #     if new_value.shape != ():
        #         raise ValueError(f"Expected float, found {new_value}")

        #     new_value = new_value.item()
        # elif isinstance(value, bool):
        #     if new_value.shape != ():
        #         raise ValueError(f"Expected bool, found {new_value}")

        #     new_value = new_value.item()
        elif eqx.is_array(new_value) and new_value.shape != value.shape:
            raise ValueError(
                f"array at path {path_str} have different shapes: "
                f"{value.shape} (old) vs {new_value.shape} (new)"
            )

        new_values.append(new_value)

    if match_exact and state_dict:
        raise ValueError(f"state dict has leftover keys {list(state_dict.keys())}")

    new_tree = jt.unflatten(treedef, new_values)

    new_tree = eqx.combine(new_tree, static)

    return new_tree


def save_pytree(path: str | Path, tree: PyTree):
    path = as_path(path)

    state_dict = to_state_dict(tree)

    save_file(state_dict, path)


def load_pytree[T: PyTree](
    path: str | Path, tree: T, *, strip_prefix: Optional[str] = None, match_exact: bool = True
) -> T:
    path = as_path(path)

    state_dict = load_file(path, strip_prefix=strip_prefix)

    tree = load_state_dict(tree, state_dict, match_exact=match_exact)

    return tree


def save_with_config_safetensors(path: str | Path, config: Any, pytree: PyTree):
    path = as_path(path)

    hyperparams_path = path.with_suffix(".json")
    safetensors_path = path.with_suffix(".safetensors")

    with hyperparams_path.open("w") as f:
        hyper_params_str = json.dumps(config)
        f.write(hyper_params_str)

    save_pytree(safetensors_path, pytree)


def load_config(path: str | Path) -> dict:
    path = as_path(path)

    if not path.exists():
        raise ValueError(f"Path {path} does not exist")

    if not path.suffix == ".json":
        raise ValueError(f"Expected .json file, got {path.suffix} instead")

    with path.open("r") as f:
        config = json.loads(f.read())

    return config
