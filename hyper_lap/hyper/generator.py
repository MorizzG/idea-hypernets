from jaxtyping import Array, Float, PRNGKeyArray

from abc import ABC, abstractmethod

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import assert_shape


class Conv2dGeneratorABC(ABC):
    @abstractmethod
    def __call__(self, emb: Float[Array, " emb_size"]) -> Float[Array, "c_out c_in k k"]: ...


class Conv2dGenerator(eqx.Module, Conv2dGeneratorABC):
    # input_emb_size: int = eqx.field(static=True)
    # pos_emb_size: int = eqx.field(static=True)
    emb_size: int = eqx.field(static=True)

    h_size: int = eqx.field(static=True)

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)

    first: nn.Linear

    middle: nn.Linear

    second: nn.Linear

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        # input_emb_size: int,
        # pos_emb_size: int,
        emb_size: int,
        *,
        h_size: int | None = None,
        key: PRNGKeyArray,
    ):
        super().__init__()

        # total_emb_size = input_emb_size + pos_emb_size

        if h_size is None:
            # h_size = input_emb_size + pos_emb_size
            h_size = emb_size

        # self.input_emb_size = input_emb_size
        # self.pos_emb_size = pos_emb_size
        self.emb_size = emb_size

        self.h_size = h_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        first_key, middle_key, second_key = jr.split(key, 3)

        # project from embeddings to hidden_size * in_channels
        self.first = nn.Linear(emb_size, h_size * in_channels, key=first_key)

        self.middle = nn.Linear(h_size, h_size, key=middle_key)

        # project from hidden to kernels
        self.second = nn.Linear(h_size, out_channels * kernel_size**2, key=second_key)

    def __call__(self, emb: Float[Array, " emb_size"]) -> Float[Array, "c_out c_in k k"]:
        # assert input_emb.shape == (self.input_emb_size,)
        # assert pos_emb.shape == (self.pos_emb_size,)

        # emb = jnp.concatenate([input_emb, pos_emb])

        assert_shape(emb, (self.emb_size,))

        x = self.first(emb)

        x = x.reshape(self.in_channels, self.h_size)

        x = jax.nn.swish(x)

        x = jax.vmap(self.middle)(x)

        x = jax.nn.swish(x)

        kernel = jax.vmap(self.second)(x)

        # kernel shape: in_channels, out_channels * kernel_size**2

        # c_in c_out k k
        kernel = kernel.reshape(
            self.in_channels, self.out_channels, self.kernel_size, self.kernel_size
        )

        # swap to c_out c_in k k
        kernel = kernel.swapaxes(0, 1)

        return kernel


class Conv2dGeneratorNew(eqx.Module, Conv2dGeneratorABC):
    # input_emb_size: int = eqx.field(static=True)
    # pos_emb_size: int = eqx.field(static=True)
    emb_size: int = eqx.field(static=True)

    h_size: int = eqx.field(static=True)

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)

    first: nn.Linear

    middle: nn.Linear

    second: nn.Linear

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        emb_size: int,
        *,
        h_size: int | None = None,
        key: PRNGKeyArray,
    ):
        super().__init__()

        if h_size is None:
            h_size = emb_size

        self.emb_size = emb_size

        self.h_size = h_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        first_key, middle_key, second_key = jr.split(key, 3)

        # project from embeddings to hidden_size * in_channels
        self.first = nn.Linear(emb_size, h_size, key=first_key)

        self.middle = nn.Linear(h_size, h_size, key=middle_key)

        # project from hidden to kernels
        self.second = nn.Linear(h_size, in_channels * out_channels * kernel_size**2, key=second_key)

    def __call__(self, emb: Float[Array, " emb_size"]) -> Float[Array, "c_out c_in k k"]:
        # assert input_emb.shape == (self.input_emb_size,)
        # assert pos_emb.shape == (self.pos_emb_size,)

        # emb = jnp.concatenate([input_emb, pos_emb])

        assert_shape(emb, (self.emb_size,))

        x = self.first(emb)

        x = jax.nn.swish(x)

        x = self.middle(x)

        x = jax.nn.swish(x)

        kernel = self.second(x)

        # kernel shape: in_channels, out_channels * kernel_size**2

        # c_in c_out k k
        kernel = kernel.reshape(
            self.in_channels, self.out_channels, self.kernel_size, self.kernel_size
        )

        # swap to c_out c_in k k
        kernel = kernel.swapaxes(0, 1)

        return kernel


class Conv2dLoraGenerator(eqx.Module, Conv2dGeneratorABC):
    # input_emb_size: int = eqx.field(static=True)
    # pos_emb_size: int = eqx.field(static=True)
    emb_size: int = eqx.field(static=True)

    h_size: int = eqx.field(static=True)
    lora_rank: int = eqx.field(static=True)

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)

    first: nn.Linear

    middle: nn.Linear

    second_a: nn.Linear
    second_b: nn.Linear

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        # input_emb_size: int,
        # pos_emb_size: int,
        emb_size: int,
        *,
        h_size: int | None = None,
        lora_rank: int = 5,
        key: PRNGKeyArray,
    ):
        super().__init__()

        # total_emb_size = input_emb_size + pos_emb_size

        if h_size is None:
            # h_size = input_emb_size + pos_emb_size
            h_size = emb_size

        # self.input_emb_size = input_emb_size
        # self.pos_emb_size = pos_emb_size
        self.emb_size = emb_size

        self.h_size = h_size
        self.lora_rank = lora_rank

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        first_key, middle_key, second_key = jr.split(key, 3)

        # project from embeddings to hidden_size * in_channels
        self.first = nn.Linear(emb_size, h_size, key=first_key)

        self.middle = nn.Linear(h_size, h_size, key=middle_key)

        # project from hidden to kernels
        # self.second = nn.Linear(h_size, out_channels * kernel_size**2, key=second_key)

        a_key, b_key = jr.split(second_key)

        self.second_a = nn.Linear(h_size, in_channels * kernel_size * lora_rank, key=a_key)
        self.second_b = nn.Linear(h_size, kernel_size * out_channels * lora_rank, key=b_key)

    def __call__(
        self,
        emb: Float[Array, " emb_size"],
    ) -> Float[Array, "c_out c_in k k"]:
        # assert input_emb.shape == (self.input_emb_size,)
        # assert pos_emb.shape == (self.pos_emb_size,)

        # emb = jnp.concatenate([input_emb, pos_emb])

        assert_shape(emb, (self.emb_size,))

        x = self.first(emb)

        x = jax.nn.swish(x)

        x = self.middle(x)

        x = jax.nn.swish(x)

        a = self.second_a(x)  # in_channels * kernel_size * lora_rank
        b = self.second_b(x)  # kernel_size * out_channels * lora_rank

        a = a.reshape(self.in_channels, self.kernel_size, self.lora_rank)
        b = b.reshape(self.out_channels, self.kernel_size, self.lora_rank)

        kernel = jnp.tensordot(a, b, axes=[2, 2])

        assert_shape(
            kernel, [self.in_channels, self.kernel_size, self.out_channels, self.kernel_size]
        )

        # transpose to c_out c_in k k
        kernel = kernel.transpose(2, 0, 1, 3)

        # kernel shape: in_channels, out_channels * kernel_size**2

        # c_in c_out k k
        # kernel = kernel.reshape(
        #     self.in_channels, self.out_channels, self.kernel_size, self.kernel_size
        # )

        # swap to c_out c_in k k
        # kernel = kernel.swapaxes(0, 1)

        return kernel
