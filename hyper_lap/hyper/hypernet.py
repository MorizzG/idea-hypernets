from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar
from typing import Any, Literal, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from chex import assert_equal_shape, assert_shape
from equinox import nn

from hyper_lap.hyper.generator import (
    Conv2dGenerator,
    Conv2dGeneratorABC,
    Conv2dGeneratorNew,
    Conv2dLoraGenerator,
)
from hyper_lap.models import Unet
from hyper_lap.modules.unet import ConvNormAct


class HyperNet(eqx.Module):
    unet: Unet

    filter_spec: PyTree = eqx.field(static=True)

    kernel_size: int = eqx.field(static=True)
    base_channels: int = eqx.field(static=True)

    block_size: int = eqx.field(static=True)

    input_emb_size: int = eqx.field(static=True)
    pos_emb_size: int = eqx.field(static=True)

    res: bool = eqx.field(static=True)

    kernel_generator: Conv2dGeneratorABC
    resample_generator: Conv2dGeneratorABC

    unet_pos_embs: list[Array]
    recomb_pos_embs: list[Array]

    init_kernel: Array
    final_kernel: Array

    @staticmethod
    def init_conv_generator(
        gen: Conv2dGeneratorABC, eps: float, *, key: PRNGKeyArray
    ) -> Conv2dGeneratorABC:
        def init_linear(linear: nn.Linear, *, key: PRNGKeyArray) -> nn.Linear:
            weight_key, bias_key = jr.split(key)

            linear = eqx.tree_at(
                lambda linear: linear.weight,
                linear,
                eps * jr.normal(weight_key, linear.weight.shape),
            )

            if linear.use_bias:
                assert linear.bias is not None

                linear = eqx.tree_at(
                    lambda linear: linear.bias,
                    linear,
                    eps * jr.normal(bias_key, linear.bias.shape),
                )

            return linear

        linears, treedef = jt.flatten(gen, is_leaf=lambda x: isinstance(x, eqx.nn.Linear))

        keys = jr.split(key, len(linears))

        linears = [init_linear(linear, key=key) for linear, key in zip(linears, keys)]

        gen = jt.unflatten(treedef, linears)

        return gen

    def __init__(
        self,
        unet: Unet,
        block_size: int,
        input_emb_size: int,
        pos_emb_size: int,
        *,
        kernel_size: int = 3,
        res: bool,
        filter_spec: PyTree | None = None,
        generator_kind: Literal["basic", "lora", "new"] = "basic",
        generator_kw_args: dict[str, Any] | None = None,
        key: PRNGKeyArray,
    ):
        super().__init__()

        if filter_spec is None:
            filter_spec = jt.map(lambda x: eqx.is_array(x), unet)

        eps = 1e-5

        self.unet = unet

        self.filter_spec = filter_spec

        self.kernel_size = kernel_size
        self.base_channels = unet.base_channels

        self.block_size = block_size
        self.input_emb_size = input_emb_size
        self.pos_emb_size = pos_emb_size

        self.res = res

        total_emb_size = input_emb_size + pos_emb_size

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
            total_emb_size,
            key=kernel_key,
            **(generator_kw_args or {}),
        )

        self.resample_generator = Gen(
            block_size,
            block_size,
            1,
            pos_emb_size,
            key=resample_key,
            **(generator_kw_args or {}),
        )

        # we can reuse keys here since we re-initialize the weights
        self.kernel_generator = self.init_conv_generator(self.kernel_generator, eps, key=kernel_key)
        self.resample_generator = self.init_conv_generator(
            self.resample_generator, eps, key=resample_key
        )

        self.init_kernel = jr.normal(init_key, unet.init_conv.conv.weight.shape)
        self.final_kernel = jr.normal(final_key, unet.final_conv.weight.shape)

        # generate positional embeddings for unet module

        unet_pos_embs_key, recomb_pos_embs_key = jr.split(key)

        self.unet_pos_embs = self.generate_pos_embs(
            unet.unet, self.filter_spec.unet, key=unet_pos_embs_key
        )

        self.recomb_pos_embs = self.generate_pos_embs(
            unet.recomb, self.filter_spec.recomb, key=recomb_pos_embs_key
        )

    def generate_pos_embs(
        self, module: eqx.Module, filter_spec: PyTree, *, key: PRNGKeyArray
    ) -> list[Array]:
        leaves, _ = jt.flatten(eqx.filter(module, filter_spec))

        block_size = self.block_size
        kernel_size = self.kernel_size

        embs = []

        for leaf in leaves:
            assert isinstance(leaf, Array)

            c_out, c_in, k1, k2 = leaf.shape

            assert k1 == k2 == kernel_size or k1 == k2 == 1, (
                f"Array has unexpected shape: {leaf.shape}"
            )
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

        if self.res:
            init_conv = eqx.tree_at(
                lambda _init_conv: _init_conv.conv.weight,
                init_conv,
                init_conv.conv.weight + self.init_kernel,
            )
        else:
            init_conv = eqx.tree_at(
                lambda _init_conv: _init_conv.conv.weight, init_conv, self.init_kernel
            )

        return init_conv

    def gen_final_conv(self, final_conv: nn.Conv2d) -> nn.Conv2d:
        assert_equal_shape([final_conv.weight, self.final_kernel])

        if self.res:
            final_conv = eqx.tree_at(
                lambda _final_conv: _final_conv.weight,
                final_conv,
                final_conv.weight + self.final_kernel,
            )
        else:
            final_conv = eqx.tree_at(
                lambda _final_conv: _final_conv.weight, final_conv, self.final_kernel
            )

        return final_conv

    def gen_weights[T: PyTree](
        self,
        unet: T,
        input_emb: Array,
        pos_embs: list[Array],
        filter_spec: PyTree,
    ) -> tuple[T, Scalar]:
        model_weights, static_model = eqx.partition(unet, filter_spec)

        weights, treedef = jt.flatten(model_weights)

        assert len(weights) == len(pos_embs), (
            f"expected {len(pos_embs)} weights, found {len(weights)} instead"
        )

        # vmap over block in positional embeddings

        # input_emb is same for all weights, so we can capture it
        def kernel_generator(pos_emb):
            emb = jnp.concat([input_emb, pos_emb])

            return self.kernel_generator(emb)

        kernel_generator = jax.vmap(kernel_generator)
        kernel_generator = jax.vmap(kernel_generator)

        resample_generator = self.resample_generator
        resample_generator = jax.vmap(resample_generator)
        resample_generator = jax.vmap(resample_generator)

        reg = jnp.array(0.0)

        for i, pos_emb in enumerate(pos_embs):
            b_out, b_in, _ = pos_emb.shape

            c_out, c_in, k1, k2 = weights[i].shape

            assert k1 == k2
            assert b_out == c_out // self.block_size
            assert b_in == c_in // self.block_size

            if k1 == self.kernel_size:
                weight = kernel_generator(pos_emb)
            elif k1 == 1:
                continue
                # weight = resample_generator(pos_emb)
            else:
                raise RuntimeError(f"weight has unexpected shape {weights[i].shape}")

            assert_shape(weight, [b_out, b_in, self.block_size, self.block_size, k1, k2])

            weight = weight.transpose(0, 2, 1, 3, 4, 5)
            weight = weight.reshape(b_out * self.block_size, b_in * self.block_size, k1, k2)

            assert_equal_shape([weight, weights[i]])

            if self.res:
                weights[i] = jax.lax.stop_gradient(weights[i]) + weight
            else:
                weights[i] = weight

            reg += (weight**2).sum()

        model_weights = jt.unflatten(treedef, weights)

        model = eqx.combine(model_weights, static_model)

        return model, reg

    @overload
    def generate(
        self,
        input_emb: Array,
        *,
        with_aux: Literal[True],
    ) -> tuple[Unet, dict[str, Any]]: ...

    @overload
    def generate(
        self,
        input_emb: Array,
        *,
        with_aux: Literal[False] = False,
    ) -> Unet: ...

    def generate(
        self,
        input_emb: Array,
        *,
        with_aux: bool = False,
    ) -> Unet | tuple[Unet, dict[str, Any]]:
        dyn_unet = self.unet

        init_conv = dyn_unet.init_conv
        unet = dyn_unet.unet
        recomb = dyn_unet.recomb
        final_conv = dyn_unet.final_conv

        init_conv = self.gen_init_conv(init_conv)
        final_conv = self.gen_final_conv(final_conv)

        unet, unet_reg = self.gen_weights(
            unet, input_emb, self.unet_pos_embs, self.filter_spec.unet
        )
        recomb, recomb_reg = self.gen_weights(
            recomb, input_emb, self.recomb_pos_embs, self.filter_spec.recomb
        )

        dyn_unet = eqx.tree_at(
            lambda _model: (_model.init_conv, _model.unet, _model.recomb, _model.final_conv),
            dyn_unet,
            (init_conv, unet, recomb, final_conv),
        )

        # model = eqx.combine(dyn_unet, static_unet)

        model = dyn_unet

        if with_aux:
            aux = {"reg_loss": unet_reg + recomb_reg}

            return model, aux

        return model

    def forward(self, x: Array, input_emb: Array) -> Array:
        unet = self.generate(input_emb)

        logits = unet(x)

        return logits

    def __call__(self, x: Float[Array, "c_in h w"], input_emb: Array | None = None) -> Array:
        if input_emb is None:
            raise ValueError("input_emb can't be None in HyperNet")

        return self.forward(x, input_emb)
