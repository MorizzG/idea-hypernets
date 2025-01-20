from jaxtyping import Array, Float, PRNGKeyArray

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from chex import assert_shape
from transformers.models.clip import FlaxCLIPVisionModel

from ._util import img_to_imagenet


class ClipEmbedder(eqx.Module):
    num_layers: int = eqx.field(static=True)
    select_layer: int = eqx.field(static=True)

    image_embedder: FlaxCLIPVisionModel = eqx.field(static=True)
    label_embedder: FlaxCLIPVisionModel = eqx.field(static=True)

    projection: nn.Linear

    def __init__(self, emb_size: int, select_layer: int = -2, *, key: PRNGKeyArray):
        super().__init__()

        self.image_embedder = FlaxCLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14-336", from_pt=True
        )  # type: ignore

        self.label_embedder = FlaxCLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14-336", from_pt=True
        )  # type: ignore

        self.num_layers = len(self.image_embedder.params["vision_tower"]["encoder"]["layers"])
        assert (
            len(self.label_embedder.params["vision_tower"]["encoder"]["layers"]) == self.num_layers
        )

        if not -self.num_layers <= select_layer < self.num_layers:
            raise ValueError(
                f"select_layer {select_layer} is out of range for {self.num_layers} layers"
            )

        self.projection = nn.Linear(2 * 1024, emb_size, key=key)

    def __call__(self, x: Float[Array, "3 h w"]) -> Array:
        c, h, w = x.shape

        assert c == 3

        image = x[0, ...]
        # x[1, ...] is background
        label = x[2, ...]

        image = jax.image.resize(image, (1, 336, 336), method="bilinear")
        label = jax.image.resize(label, (1, 336, 336), method="bilinear")

        image = img_to_imagenet(image)
        label = img_to_imagenet(label)

        image_output = self.image_embedder(image, output_hidden_states=True)
        label_output = self.label_embedder(label, output_hidden_states=True)

        image_hidden_states = image_output["hidden_states"]
        label_hidden_states = label_output["hidden_states"]

        assert (
            len(image_hidden_states) == self.num_layers
            and len(label_hidden_states) == self.num_layers
        )

        image_emb = image_hidden_states[self.select_layer]
        label_emb = label_hidden_states[self.select_layer]

        assert_shape([image_emb, label_emb], (1024,))

        concat_emb = jnp.concat([image_emb, label_emb])

        assert_shape(concat_emb, (2 * 1024,))

        emb = self.projection(concat_emb)

        return emb
