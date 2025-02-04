import json
from pathlib import Path

import equinox as eqx

from hyper_lap.hyper.hypernet import HyperNet
from hyper_lap.models import Unet
from hyper_lap.training.utils import make_hypernet

from .utils import as_path


def save_hypernet_eqx(path: str | Path, hyper_params: dict, hypernet: HyperNet):
    path = as_path(path)

    with path.open("wb") as f:
        hyper_params_str = json.dumps(hyper_params)

        f.write((hyper_params_str + "\n").encode())

        eqx.tree_serialise_leaves(f, hypernet)


def load_hypernet_eqx(path: str | Path) -> tuple[Unet, HyperNet]:
    path = as_path(path)

    if not (path.exists()):
        raise ValueError(f"Path {path} does not exist")

    with path.open("rb") as f:
        hyper_params = json.loads(f.readline().decode())

        (unet, hypernet) = make_hypernet(hyper_params)

        return unet, eqx.tree_deserialise_leaves(f, hypernet)
