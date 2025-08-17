from jaxtyping import Array, Float, Integer, PRNGKeyArray

import equinox as eqx
import equinox.nn as nn


class LearnedEmbedding(eqx.Module):
    emb_size: int = eqx.field(static=True)

    embedding: nn.Embedding

    def __init__(self, emb_size: int, num_datasets: int, *, key: PRNGKeyArray):
        super().__init__()

        self.emb_size = emb_size

        self.embedding = nn.Embedding(num_datasets, emb_size, key=key)

    def __call__(
        self,
        _image: Float[Array, "3 h w"],
        _label: Integer[Array, "h w"],
        dataset_idx: Integer[Array, ""],
    ) -> Array:
        return self.embedding(dataset_idx)
