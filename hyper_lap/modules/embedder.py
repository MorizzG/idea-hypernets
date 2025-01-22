from jaxtyping import Array, Float, PRNGKeyArray
from typing import Literal

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import transformers
from chex import assert_shape
from jax import lax
from transformers.models.clip import FlaxCLIPVisionModel

from ._util import img_to_imagenet


class ClipEmbedder(eqx.Module):
    num_layers: int = eqx.field(static=True)
    select_layer: int = eqx.field(static=True)
    pool: Literal["cls", "mean"] = eqx.field(static=True)

    # image_embedder: FlaxCLIPVisionModel = eqx.field(static=True)
    # label_embedder: FlaxCLIPVisionModel = eqx.field(static=True)

    clip: FlaxCLIPVisionModel = eqx.field(static=True)

    projection: nn.Linear

    def __init__(
        self,
        emb_size: int,
        select_layer: int = -2,
        pool: Literal["cls", "mean"] = "cls",
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        if pool not in ["cls", "mean"]:
            raise ValueError(f"Invalid pool {pool}")

        self.select_layer = select_layer
        self.pool = pool

        # logging.getLogger("transformers/modeling_flax_utils.py").setLevel(logging.ERROR)

        orig_verbosity = transformers.logging.get_verbosity()

        transformers.logging.set_verbosity_error()

        self.clip = FlaxCLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14-336", from_pt=True
        )  # type: ignore

        transformers.logging.set_verbosity(orig_verbosity)

        self.num_layers = len(self.clip.params["vision_model"]["encoder"]["layers"])
        # self.num_layers = 24

        if not -self.num_layers <= select_layer < self.num_layers:
            raise ValueError(
                f"select_layer {select_layer} is out of range for {self.num_layers} layers"
            )

        self.projection = nn.Linear(2 * 1024, emb_size, key=key)

    def __call__(self, x: Float[Array, "3 h w"]) -> Array:
        c, h, w = x.shape

        assert c == 3

        image = x[0:1, ...]
        # x[1, ...] is background
        label = x[2:3, ...]

        image = jax.image.resize(image, (1, 336, 336), method="bilinear")
        label = jax.image.resize(label, (1, 336, 336), method="bilinear")

        image = img_to_imagenet(image)
        label = img_to_imagenet(label)

        image_output = self.clip(image[None, ...], output_hidden_states=True)
        label_output = self.clip(label[None, ...], output_hidden_states=True)

        # ignore first hidden state, as it's just the input
        image_hidden_states = image_output["hidden_states"][1:]  # type: ignore
        label_hidden_states = label_output["hidden_states"][1:]  # type: ignore

        assert (
            len(image_hidden_states) == self.num_layers
            and len(label_hidden_states) == self.num_layers
        )

        image_hidden_state = image_hidden_states[self.select_layer]
        label_hidden_state = label_hidden_states[self.select_layer]

        assert_shape([image_hidden_state, label_hidden_state], (1, 577, 1024))

        if self.pool == "cls":
            image_emb = image_hidden_state[0, 0, :]
            label_emb = label_hidden_state[0, 0, :]
        elif self.pool == "mean":
            image_emb = image_hidden_state[0].mean(axis=0)
            label_emb = label_hidden_state[0].mean(axis=0)
        else:
            assert False, f"self.pool has unexpected value {self.pool}"

        assert_shape([image_emb, label_emb], (1024,))

        concat_emb = jnp.concat([image_emb, label_emb])

        assert_shape(concat_emb, (2 * 1024,))

        emb = self.projection(concat_emb)

        return lax.stop_gradient(emb)
