from jaxtyping import Array, Float, Integer, PRNGKeyArray
from typing import Literal

import equinox as eqx

from hyper_lap.modules.embedder import (
    ClipEmbedder,
    ConvNextEmbedder,
    LearnedEmbedding,
    ResNetEmbedder,
    ViTEmbedder,
)


class InputEmbedder(eqx.Module):
    type EmbedderKind = Literal["vit", "convnext", "resnet", "clip", "learned"]

    emb_size: int = eqx.field(static=True)

    embedder: ViTEmbedder | ResNetEmbedder | ConvNextEmbedder | ClipEmbedder | LearnedEmbedding

    def __init__(
        self,
        emb_size: int,
        num_datasets: int,
        *,
        kind: EmbedderKind = "resnet",
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.emb_size = emb_size

        if kind == "vit":
            self.embedder = ViTEmbedder(emb_size, key=key)
        elif kind == "convnext":
            self.embedder = ConvNextEmbedder(emb_size, key=key)
        elif kind == "resnet":
            self.embedder = ResNetEmbedder(emb_size, key=key)
        elif kind == "clip":
            self.embedder = ClipEmbedder(emb_size, key=key)
        elif kind == "learned":
            self.embedder = LearnedEmbedding(num_datasets, emb_size, key=key)
        else:
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
