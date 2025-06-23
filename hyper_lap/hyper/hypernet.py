from jaxtyping import Array, Float, Integer, PRNGKeyArray
from typing import Literal

import equinox as eqx
import jax
import jax.random as jr
import jax.tree as jt
from chex import assert_equal_shape, assert_shape
from equinox import nn

from hyper_lap.hyper.embedder import InputEmbedder
from hyper_lap.hyper.generator import Conv2dGenerator
from hyper_lap.models import Unet
from hyper_lap.modules.unet import Block, ConvNormAct, UnetModule


class HyperNet(eqx.Module):
    unet: Unet = eqx.field(static=True)

    kernel_size: int = eqx.field(static=True)
    base_channels: int = eqx.field(static=True)

    block_size: int = eqx.field(static=True)
    input_emb_size: int = eqx.field(static=True)
    pos_emb_size: int = eqx.field(static=True)

    input_embedder: InputEmbedder

    kernel_generator: Conv2dGenerator

    unet_pos_embs: list[Array]
    recomb_embs: list[Array]

    init_kernel: Array
    final_kernel: Array

    @staticmethod
    def kernel_shape(
        in_channels: int, out_channels: int, kernel_size: int
    ) -> tuple[int, int, int, int]:
        return out_channels, in_channels, kernel_size, kernel_size

    def __init__(
        self,
        unet: Unet,
        *,
        block_size: int,
        input_emb_size: int,
        pos_emb_size: int,
        kernel_size: int,
        embedder_kind: Literal["vit", "convnext", "resnet", "clip", "learned"],
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.unet = unet

        self.kernel_size = kernel_size
        self.base_channels = unet.base_channels

        self.block_size = block_size
        self.input_emb_size = input_emb_size
        self.pos_emb_size = pos_emb_size

        base_channels = unet.base_channels

        key, kernel_key, emb_key, init_key, final_key = jr.split(key, 5)

        self.input_embedder = InputEmbedder(input_emb_size, kind=embedder_kind, key=emb_key)

        self.kernel_generator = Conv2dGenerator(
            block_size, block_size, kernel_size, input_emb_size, pos_emb_size, key=kernel_key
        )

        self.init_kernel = jr.normal(
            init_key, self.kernel_shape(unet.in_channels, base_channels, 1)
        )
        self.final_kernel = jr.normal(
            final_key, self.kernel_shape(base_channels, unet.out_channels, 1)
        )

        # generate positional embeddings for unet module

        unet_pos_embs_key, recomb_pos_embs_key = jr.split(key)

        self.unet_pos_embs = self.generate_pos_embs(unet.unet, key=unet_pos_embs_key)

        self.recomb_embs = self.generate_pos_embs(unet.recomb, key=recomb_pos_embs_key)

    def generate_pos_embs(self, module: eqx.Module, *, key: PRNGKeyArray) -> list[Array]:
        leaves, _ = jt.flatten(eqx.filter(module, eqx.is_array))

        block_size = self.block_size
        kernel_size = self.kernel_size

        embs = []

        for leaf in leaves:
            assert isinstance(leaf, Array)

            c_out, c_in, k1, k2 = leaf.shape

            assert k1 == k2 == kernel_size, f"Array has unexpected shape: {leaf.shape}"
            assert c_out % block_size == 0 and c_in % block_size == 0, (
                f"channels {c_out} {c_in} not divisible by block_size {block_size}"
            )

            b_out = c_out // block_size
            b_in = c_in // block_size

            key, consume = jr.split(key)

            emb = jr.normal(consume, [b_out, b_in, self.pos_emb_size])

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

    def gen_unet(self, unet: UnetModule, input_emb: Array) -> UnetModule:
        block_size = self.block_size
        kernel_size = self.kernel_size

        model_weights, static_model = eqx.partition(unet, eqx.is_array)

        weights, treedef = jt.flatten(model_weights)

        # vmap over block in positional embeddings
        kernel_generator = self.kernel_generator
        kernel_generator = jax.vmap(kernel_generator, in_axes=(None, 0), out_axes=0)
        kernel_generator = jax.vmap(kernel_generator, in_axes=(None, 0), out_axes=0)

        assert len(weights) == len(self.unet_pos_embs)

        for i, pos_emb in enumerate(self.unet_pos_embs):
            b_out, b_in, _ = pos_emb.shape

            weight = kernel_generator(input_emb, pos_emb)

            assert_shape(weight, [b_out, b_in, block_size, block_size, kernel_size, kernel_size])

            weight = weight.transpose(0, 2, 1, 3, 4, 5)
            weight = weight.reshape(b_out * block_size, b_in * block_size, kernel_size, kernel_size)

            assert weights[i].shape == weight.shape

            weights[i] = weight

        model_weights = jt.unflatten(treedef, weights)

        model = eqx.combine(model_weights, static_model)

        return model

    def gen_recomb(self, recomb: Block, input_emb: Array) -> Block:
        block_size = self.block_size
        kernel_size = self.kernel_size

        model_weights, static_model = eqx.partition(recomb, eqx.is_array)

        weights, treedef = jt.flatten(model_weights)

        # vmap over block in positional embeddings
        kernel_generator = self.kernel_generator
        kernel_generator = jax.vmap(kernel_generator, in_axes=(None, 0), out_axes=0)
        kernel_generator = jax.vmap(kernel_generator, in_axes=(None, 0), out_axes=0)

        assert len(weights) == len(self.recomb_embs)

        for i, pos_emb in enumerate(self.recomb_embs):
            b_out, b_in, _ = pos_emb.shape

            weight = kernel_generator(input_emb, pos_emb)

            assert_shape(weight, [b_out, b_in, block_size, block_size, kernel_size, kernel_size])

            weight = weight.transpose(0, 2, 1, 3, 4, 5)
            weight = weight.reshape(b_out * block_size, b_in * block_size, kernel_size, kernel_size)

            weights[i] = weight

        model_weights = jt.unflatten(treedef, weights)

        model = eqx.combine(model_weights, static_model)

        return model

    def __call__(self, image: Float[Array, "3 h w"], label: Integer[Array, "h w"]) -> Unet:
        input_emb = self.input_embedder(image, label)

        init_conv, unet, recomb, final_conv = (
            self.unet.init_conv,
            self.unet.unet,
            self.unet.recomb,
            self.unet.final_conv,
        )

        init_conv = self.gen_init_conv(init_conv)
        final_conv = self.gen_final_conv(final_conv)

        unet = self.gen_unet(unet, input_emb)
        recomb = self.gen_recomb(recomb, input_emb)

        model = eqx.tree_at(
            lambda _model: (_model.init_conv, _model.unet, _model.recomb, _model.final_conv),
            self.unet,
            (init_conv, unet, recomb, final_conv),
        )

        return model
