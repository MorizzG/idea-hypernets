from jaxtyping import Array, Float, Integer, PRNGKeyArray
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from chex import assert_equal_shape, assert_shape
from equinox import nn

from hyper_lap.hyper.embedder import InputEmbedder
from hyper_lap.hyper.generator import (
    Conv2dGenerator,
    Conv2dGeneratorABC,
    Conv2dGeneratorNew,
    Conv2dLoraGenerator,
)
from hyper_lap.models import Unet
from hyper_lap.modules.attention import Encoder
from hyper_lap.modules.unet import ConvNormAct, UnetModule


class AttnHyperNet(eqx.Module):
    unet: Unet  #  = eqx.field(static=True)

    kernel_size: int = eqx.field(static=True)
    base_channels: int = eqx.field(static=True)

    block_size: int = eqx.field(static=True)

    emb_size: int = eqx.field(static=True)

    kernel_generator: Conv2dGeneratorABC
    resample_generator: Conv2dGeneratorABC

    transformer: Encoder

    unet_pos_embs: list[Array]
    recomb_pos_embs: list[Array]

    init_kernel: Array
    final_kernel: Array

    @staticmethod
    def flatten_with_def(pos_embs: list[Array]) -> tuple[list[Array], list[tuple[int, int]]]:
        out = []
        out_shapes = []

        for pos_emb in pos_embs:
            n, m, _ = pos_emb.shape

            out_shapes.append((n, m))

            pos_emb = pos_emb.reshape(n * m, -1)

            out += list(pos_emb)

        assert_equal_shape(out)

        return out, out_shapes

    @staticmethod
    def unflatten_from_def(
        flat_pos_embs: list[Array], shapes: list[tuple[int, int]]
    ) -> list[Array]:
        out = []

        assert len(flat_pos_embs) == sum(
            n * m for (n, m) in shapes
        ), f"{len(flat_pos_embs)=}, but {sum(n * m for (n, m) in shapes)=}"

        for n, m in shapes:
            assert len(flat_pos_embs) >= n * m

            flat_pos_emb = flat_pos_embs[: n * m]

            flat_pos_embs = flat_pos_embs[n * m :]

            pos_emb = jnp.stack(flat_pos_emb)

            pos_emb = pos_emb.reshape(n, m, -1)

            out.append(pos_emb)

        return out

    def __init__(
        self,
        unet: Unet,
        block_size: int,
        emb_size: int,
        *,
        kernel_size: int = 3,
        generator_kind: Literal["basic", "lora"] = "basic",
        generator_kw_args: dict[str, Any] | None = None,
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.unet = unet

        self.kernel_size = kernel_size
        self.base_channels = unet.base_channels

        self.block_size = block_size
        self.emb_size = emb_size

        key, kernel_key, resample_key, init_key, final_key = jr.split(key, 5)

        match generator_kind:
            case "basic":
                Gen = Conv2dGenerator
            case "lora":
                Gen = Conv2dLoraGenerator
            case "new":
                Gen = Conv2dGeneratorNew
            case _:
                raise ValueError(f"invalid generator_kind {generator_kind}")

        self.kernel_generator = Gen(
            block_size,
            block_size,
            kernel_size,
            emb_size,
            key=kernel_key,
            **(generator_kw_args or {}),
        )

        self.resample_generator = Gen(
            block_size,
            block_size,
            1,
            emb_size,
            key=resample_key,
            **(generator_kw_args or {}),
        )

        self.init_kernel = jr.normal(init_key, unet.init_conv.conv.weight.shape)
        self.final_kernel = jr.normal(final_key, unet.final_conv.weight.shape)

        # generate positional embeddings for unet module

        transformer_key, unet_pos_embs_key, recomb_pos_embs_key = jr.split(key, 3)

        self.transformer = Encoder(
            depth=6, d_model=emb_size, num_heads=emb_size // 64, d_head=64, key=transformer_key
        )

        self.unet_pos_embs = self.generate_pos_embs(unet.unet, key=unet_pos_embs_key)

        self.recomb_pos_embs = self.generate_pos_embs(unet.recomb, key=recomb_pos_embs_key)

    def generate_pos_embs(self, module: eqx.Module, *, key: PRNGKeyArray) -> list[Array]:
        leaves, _ = jt.flatten(eqx.filter(module, eqx.is_array))

        block_size = self.block_size
        kernel_size = self.kernel_size

        embs = []

        for leaf in leaves:
            assert isinstance(leaf, Array)

            c_out, c_in, k1, k2 = leaf.shape

            assert (
                k1 == k2 == kernel_size or k1 == k2 == 1
            ), f"Array has unexpected shape: {leaf.shape}"
            assert (
                c_out % block_size == 0 and c_in % block_size == 0
            ), f"channels {c_out} {c_in} not divisible by block_size {block_size}"

            b_out = c_out // block_size
            b_in = c_in // block_size

            key, consume = jr.split(key)

            emb = jr.normal(consume, [b_out, b_in, self.emb_size])

            embs.append(emb)

        return embs

    def gen_init_conv(self, init_conv: ConvNormAct) -> ConvNormAct:
        assert_equal_shape([init_conv.conv.weight, self.init_kernel])
        init_conv = eqx.tree_at(
            lambda _init_conv: _init_conv.conv.weight, init_conv, self.init_kernel
        )

        return init_conv

    def gen_final_conv(self, final_conv: nn.Conv2d) -> nn.Conv2d:
        assert_equal_shape([final_conv.weight, self.final_kernel])
        final_conv = eqx.tree_at(
            lambda _final_conv: _final_conv.weight, final_conv, self.final_kernel
        )

        return final_conv

    def gen_weights(self, unet: UnetModule, embs: list[Array]) -> UnetModule:
        model_weights, static_model = eqx.partition(unet, eqx.is_array)

        weights, treedef = jt.flatten(model_weights)

        # vmap over block in positional embeddings

        kernel_generator = self.kernel_generator
        kernel_generator = jax.vmap(kernel_generator)
        kernel_generator = jax.vmap(kernel_generator)

        resample_generator = self.resample_generator
        resample_generator = jax.vmap(resample_generator)
        resample_generator = jax.vmap(resample_generator)

        assert len(weights) == len(
            embs
        ), f"expected {len(embs)} weights, found {len(weights)} instead"

        for i, emb in enumerate(embs):
            b_out, b_in, _ = emb.shape

            c_out, c_in, k1, k2 = weights[i].shape

            assert k1 == k2
            assert b_out == c_out // self.block_size
            assert b_in == c_in // self.block_size

            if k1 == self.kernel_size:
                weight = kernel_generator(emb)
            elif k1 == 1:
                weight = resample_generator(emb)
            else:
                raise RuntimeError(f"weight has unexpected shape {weights[i].shape}")

            assert_shape(weight, [b_out, b_in, self.block_size, self.block_size, k1, k2])

            weight = weight.transpose(0, 2, 1, 3, 4, 5)
            weight = weight.reshape(b_out * self.block_size, b_in * self.block_size, k1, k2)

            assert_equal_shape([weight, weights[i]])

            weights[i] = weight

        model_weights = jt.unflatten(treedef, weights)

        model = eqx.combine(model_weights, static_model)

        return model

    def __call__(self, input_emb: Array) -> Unet:
        dyn_unet, static_unet = eqx.partition(self.unet, eqx.is_array)

        dyn_unet = jax.lax.stop_gradient(dyn_unet)

        init_conv = dyn_unet.init_conv
        unet = dyn_unet.unet
        recomb = dyn_unet.recomb
        final_conv = dyn_unet.final_conv

        init_conv = self.gen_init_conv(init_conv)
        final_conv = self.gen_final_conv(final_conv)

        unet_pos_embs = self.unet_pos_embs
        recomb_pos_embs = self.recomb_pos_embs

        unet_pos_embs_flat, unet_pos_embs_shapes = AttnHyperNet.flatten_with_def(unet_pos_embs)
        recomb_pos_embs_flat, recomb_pos_embs_shapes = AttnHyperNet.flatten_with_def(
            recomb_pos_embs
        )

        all_embs_list = [input_emb] + unet_pos_embs_flat + recomb_pos_embs_flat
        assert_shape(all_embs_list, (self.emb_size,))

        all_embs = jnp.stack(all_embs_list, axis=0)

        all_embs = self.transformer(all_embs)

        unet_embs_flat = list(all_embs[1 : len(unet_pos_embs_flat) + 1])
        recomb_embs_flat = list(all_embs[len(unet_pos_embs_flat) + 1 :])

        unet_embs = AttnHyperNet.unflatten_from_def(unet_embs_flat, unet_pos_embs_shapes)
        recomb_embs = AttnHyperNet.unflatten_from_def(recomb_embs_flat, recomb_pos_embs_shapes)

        assert len(unet_embs) == len(self.unet_pos_embs) and all(
            unet_emb.shape == unet_pos_emb.shape
            for unet_emb, unet_pos_emb in zip(unet_embs, self.unet_pos_embs)
        )
        assert len(recomb_embs) == len(self.recomb_pos_embs) and all(
            recomb_emb.shape == recomb_pos_emb.shape
            for recomb_emb, recomb_pos_emb in zip(recomb_embs, self.recomb_pos_embs)
        )

        unet = self.gen_weights(unet, unet_embs)
        recomb = self.gen_weights(recomb, recomb_embs)

        dyn_unet = eqx.tree_at(
            lambda _model: (_model.init_conv, _model.unet, _model.recomb, _model.final_conv),
            dyn_unet,
            (init_conv, unet, recomb, final_conv),
        )

        model = eqx.combine(dyn_unet, static_unet)

        return model
