from jaxtyping import Array, Float, Integer, PRNGKeyArray
from typing import Literal

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import transformers
from chex import assert_shape
from transformers.models.clip import FlaxCLIPVisionModel


class ClipEmbedder(eqx.Module):
    num_layers: int = eqx.field(static=True)
    select_layer: int = eqx.field(static=True)
    pool: Literal["cls", "mean", "both"] = eqx.field(static=True)

    clip_vision: FlaxCLIPVisionModel

    # tokenizer: CLIPTokenizerFast = eqx.field(static=True)
    # clip_text: FlaxCLIPTextModel = eqx.field(static=True)

    projection: nn.Linear | nn.Identity

    def __init__(
        self,
        emb_size: int,
        select_layer: int = -2,
        pool: Literal["cls", "mean", "both"] = "cls",
        clip: Literal["openai"] = "openai",
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        if pool not in ["cls", "mean", "both"]:
            raise ValueError(f"Invalid pool {pool}")

        self.select_layer = select_layer
        self.pool = pool

        orig_verbosity = transformers.logging.get_verbosity()

        transformers.logging.set_verbosity_error()

        if clip == "openai":
            clip_vision = FlaxCLIPVisionModel.from_pretrained(
                "openai/clip-vit-large-patch14-336", from_pt=True
            )
            assert isinstance(clip_vision, FlaxCLIPVisionModel)
            self.clip_vision = clip_vision

        else:
            raise RuntimeError(f"Unknown clip variant {clip}")

        transformers.logging.set_verbosity(orig_verbosity)

        self.num_layers = len(self.clip_vision.params["vision_model"]["encoder"]["layers"])

        if not -self.num_layers <= select_layer < self.num_layers:
            raise ValueError(
                f"select_layer {select_layer} is out of range for {self.num_layers} layers"
            )

        match self.pool:
            case "cls" | "mean":
                self.projection = nn.Linear(3 * 1024, emb_size, key=key)
            case "both":
                self.projection = nn.Linear(6 * 1024, emb_size, key=key)

    def __call__(
        self,
        image: Float[Array, "3 h w"],
        label: Integer[Array, "h w"],
        _dataset_idx: Integer[Array, ""],
    ) -> Float[Array, " emb_size"]:
        c, h, w = image.shape

        assert c == 3
        assert_shape(label, [h, w])

        pos_masked_image = (label != 0) * image
        neg_masked_image = (label == 0) * image

        input = jnp.stack([image, pos_masked_image, neg_masked_image])

        assert input.shape[:2] == (3, 3)

        if input.shape[-2:] != (336, 336):
            input = jax.image.resize(input, (3, 3, 336, 336), method="bicubic")

        assert_shape(input, (3, 3, 336, 336))

        output = self.clip_vision(input, output_hidden_states=True)

        hidden_states: list[Array] = output["hidden_states"][1:]  # pyright: ignore

        assert len(hidden_states) == self.num_layers

        hidden_state = hidden_states[self.select_layer]

        assert_shape([hidden_state], (3, 577, 1024))

        match self.pool:
            case "cls":
                vision_emb = hidden_state[:, 0, :].ravel()

                assert_shape(vision_emb, (3 * 1024,))
            case "mean":
                vision_emb = hidden_state.mean(axis=1).ravel()

                assert_shape(vision_emb, (3 * 1024,))
            case "both":
                cls_tokens = hidden_state[:, 0, :]
                patch_tokens = hidden_state[:, 1:, :].mean(axis=1)

                vision_emb = jnp.stack([cls_tokens, patch_tokens], axis=1)

                assert_shape(vision_emb, (3, 2, 1024))

                vision_emb = vision_emb.ravel()

        vision_emb = jax.lax.stop_gradient(vision_emb)

        emb = self.projection(vision_emb)

        return emb
