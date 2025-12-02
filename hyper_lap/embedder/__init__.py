from jaxtyping import Array, Float, Integer, PRNGKeyArray
from typing import Literal

from pathlib import Path

import equinox as eqx

from .clip import ClipEmbedder
from .conv_next import ConvNextEmbedder
from .dino import DinoEmbedder
from .learned import LearnedEmbedding
from .null import ZeroEmbedder
from .resnet import ResNetEmbedder
from .vit import ViTEmbedder


class InputEmbedder(eqx.Module):
    type EmbedderKind = Literal["vit", "convnext", "resnet", "clip", "dino", "learned", "zero"]

    emb_size: int = eqx.field(static=True)

    embedder: (
        ViTEmbedder
        | ResNetEmbedder
        | ConvNextEmbedder
        | ClipEmbedder
        | DinoEmbedder
        | LearnedEmbedding
        | ZeroEmbedder
    )

    def __init__(
        self,
        emb_size: int,
        num_datasets: int,
        *,
        kind: EmbedderKind = "resnet",
        key: PRNGKeyArray,
        **embedder_args,
    ):
        super().__init__()

        self.emb_size = emb_size

        match kind:
            case "vit":
                self.embedder = ViTEmbedder(emb_size, key=key, **embedder_args)
            case "convnext":
                self.embedder = ConvNextEmbedder(emb_size, key=key, **embedder_args)
            case "resnet":
                self.embedder = ResNetEmbedder(emb_size, key=key, **embedder_args)
            case "clip":
                self.embedder = ClipEmbedder(emb_size, key=key, **embedder_args)
            case "dino":
                weights_paths = [
                    "/home/saturn/iwai/iwai104h/models/dinov3-equinox/vitl16.safetensors",
                    "/vol/ideadata/eg94ifeh/models/dinov3-equinox/vitl16.safetensors",
                    "/media/LinuxData/models/dinov3_equinox/vitl16.safetensors",
                ]

                if "weights_path" in embedder_args:
                    weights_path = Path(embedder_args.pop("weights_path"))
                else:
                    weights_path = None

                    for path in weights_paths:
                        weights_path = Path(path)

                        if weights_path.exists():
                            break

                if weights_path is None or not weights_path.exists():
                    raise ValueError("could not find a valid weights_path")

                self.embedder = DinoEmbedder(
                    emb_size, key=key, weights_path=str(weights_path), **embedder_args
                )
            case "learned":
                self.embedder = LearnedEmbedding(emb_size, num_datasets, key=key, **embedder_args)
            case "zero":
                self.embedder = ZeroEmbedder(emb_size)
            case _:
                raise ValueError(f"Unknown embedder: {kind}")

    def __call__(
        self,
        image: Float[Array, "3 h w"],
        label: Integer[Array, "h w"],
        dataset_idx: Integer[Array, ""],
    ) -> Float[Array, " self.emb_size"]:
        assert image.shape[0] == 3

        emb = self.embedder(image, label, dataset_idx)

        return emb
