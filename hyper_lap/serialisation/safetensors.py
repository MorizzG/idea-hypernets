"""
Save models split into a json file for the hyper parameters neccessary to recreated the static
pytree structure and a safetensors file containing the actual parameters.

inspired by https://github.com/krypticmouse/saferax
"""

from jaxtyping import Array, PyTree

import json
from dataclasses import asdict
from pathlib import Path

import jax.tree as jt
from safetensors.flax import load_file, save_file

from hyper_lap.hyper.hypernet import HyperNet, HyperNetConfig
from hyper_lap.models import Unet
from hyper_lap.models.unet import UnetConfig
from hyper_lap.training.utils import HyperParams, make_hypernet

from .utils import to_path


def to_state_dict(tree: PyTree) -> dict[str, Array]:
    paths_and_values = jt.flatten_with_path(tree)[0]

    arrays = {}

    for tree_path, value in paths_and_values:
        path_parts = []

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
    tree: PyTree, state_dict: dict[str, Array], *, match_exact: bool = True
) -> PyTree:
    # we want to modify state_dict, but not the original at the call site
    state_dict = state_dict.copy()

    paths_and_values, treedef = jt.flatten_with_path(tree)

    new_values = []

    for tree_path, value in paths_and_values:
        path_parts = []

        for p in tree_path:
            if hasattr(p, "name"):
                path_parts.append(p.name)
            elif hasattr(p, "idx"):
                path_parts.append(str(p.idx))
            else:
                raise ValueError(f"Don't know how to deal with tree path part {p}")

        path_str = ".".join(path_parts)

        if path_str not in state_dict and match_exact:
            raise ValueError(f"state dict is missing key {path_str}")

        new_value = state_dict.pop(path_str)

        if new_value.shape != value.shape:
            raise ValueError(
                f"array at path {path_str} have different shapes: "
                f"{value.shape} (old) vs {new_value.shape} (new)"
            )

        new_values.append(new_value)

    if match_exact and state_dict:
        raise ValueError(f"state dict has unexpected keys {list(state_dict.keys())}")

    new_tree = jt.unflatten(treedef, new_values)

    return new_tree


def save_hypernet_safetensors(path: str | Path, hyper_params: HyperParams, hypernet: HyperNet):
    path = to_path(path)

    hyperparams_path = path.with_suffix(".json")
    safetensors_path = path.with_suffix(".safetensors")

    hyper_params_dict = asdict(hyper_params)

    hyper_params_dict["unet"] = asdict(hyper_params_dict["unet"])
    hyper_params_dict["hypernet"] = asdict(hyper_params_dict["hypernet"])

    with hyperparams_path.open("wb") as f:
        hyper_params_str = json.dumps(hyper_params_dict)
        f.write((hyper_params_str).encode())

    state_dict = to_state_dict(hypernet)

    save_file(state_dict, safetensors_path)


def load_hypernet_safetensors(path: str | Path) -> tuple[Unet, HyperNet]:
    path = to_path(path)

    hyperparams_path = path.with_suffix(".json")
    safetensors_path = path.with_suffix(".safetensors")

    if not (hyperparams_path.exists()):
        raise ValueError(f"Path {hyperparams_path} does not exist")

    if not (safetensors_path.exists()):
        raise ValueError(f"Path {safetensors_path} does not exist")

    with hyperparams_path.open("rb") as f:
        hyper_params_dict = json.loads(f.read().decode())

    seed = hyper_params_dict.pop("seed")
    unet_params = UnetConfig(**hyper_params_dict.pop("unet"))
    hypernet_params = HyperNetConfig(**hyper_params_dict.pop("hypernet"))

    hyper_params = HyperParams(seed=seed, unet=unet_params, hypernet=hypernet_params)

    unet, hypernet = make_hypernet(hyper_params)

    state_dict = load_file(safetensors_path)

    hypernet = load_state_dict(hypernet, state_dict)

    return unet, hypernet
