import equinox as eqx
from equinox import nn
import jax
import jax.random as jr
from jaxtyping import Array, Float, Integer, PRNGKeyArray

from hyper_lap.models import Unet
from .embedder import InputEmbedder
from .generator import Conv2dGenerator


class HyperNet(eqx.Module):
    block_size: int

    init_generator: Conv2dGenerator
    final_generator: Conv2dGenerator

    kernel_generator: Conv2dGenerator

    input_embedder: InputEmbedder

    init_emb: Array
    final_emb: Array

    def __init__(
        self, block_size: int, base_channels: int, kernel_size: int = 3, *, key: PRNGKeyArray
    ):
        super().__init__()

        emb_size = 64

        self.block_size = block_size

        kernel_key, init_key, final_key, key = jr.split(key, 4)

        self.init_generator = Conv2dGenerator(1, base_channels, kernel_size, emb_size, key=init_key)
        self.final_generator = Conv2dGenerator(
            base_channels, 1, kernel_size, emb_size, key=final_key
        )

        self.kernel_generator = Conv2dGenerator(
            block_size, block_size, kernel_size, emb_size, key=kernel_key
        )

        self.input_embedder = InputEmbedder(emb_size // 2)

        init_key, final_key = jr.split(key, 2)

        self.init_emb = jr.normal(init_key, [emb_size])
        self.final_emb = jr.normal(final_key, [emb_size])

    def make_init_conv(self, init_conv: nn.Conv2d, input_emb: Array) -> nn.Conv2d:
        init_conv_weight = self.init_generator(input_emb, self.init_emb)

        init_conv = eqx.tree_at(lambda init_conv: init_conv.weight, init_conv, init_conv_weight)

        return init_conv

    def make_final_conv(self, final_conv: nn.Conv2d, input_emb: Array) -> nn.Conv2d:
        final_conv_weight = self.final_generator(input_emb, self.final_emb)

        final_conv = eqx.tree_at(
            lambda final_conv: final_conv.weight, final_conv, final_conv_weight
        )

        return final_conv

    def make_conv(
        self,
        conv: nn.Conv2d,
        input_emb: Float[Array, "emb_dim/2"],
        pos_emb: Float[Array, "k_out k_in emb_dim/2"],
    ) -> nn.Conv2d:
        c_out, c_in, f, _ = conv.weight.shape
        k_out, k_in, _ = pos_emb.shape

        assert c_out % self.block_size == 0 and c_in % self.block_size == 0
        assert c_out // self.block_size == k_out and c_in // self.block_size == k_in

        # vmap over the blocks to generate
        kernel_generator = jax.vmap(self.kernel_generator, in_axes=(None, 0))
        kernel_generator = jax.vmap(kernel_generator, in_axes=(None, 0))

        # should be of shape [k_out k_in block block f f]
        kernel = kernel_generator(input_emb, pos_emb)

        kernel = kernel.transpose(0, 2, 1, 3, 4, 5)
        kernel = kernel.reshape(c_out, c_in, f, f)

        conv = eqx.tree_at(lambda conv: conv.weight, conv, kernel)

        return conv

    def __call__(
        self, model: Unet, image: Float[Array, "c h w"], label: Integer[Array, "h w"]
    ) -> Unet:
        input_emb = self.input_embedder(image, label)

        init_conv = self.make_init_conv(model.init_conv, input_emb)
        unet = model.unet
        recomb = model.recomb
        final_conv = self.make_final_conv(model.final_conv, input_emb)

        model = eqx.tree_at(
            lambda model: (model.init_conv, model.unet, model.recomb, model.final_conv),
            model,
            (init_conv, unet, recomb, final_conv),
        )

        return model
