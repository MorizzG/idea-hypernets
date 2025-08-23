from jaxtyping import Array, Float, PRNGKeyArray

import equinox as eqx
import equinox.nn as nn
import jax.random as jr

from hyper_lap.layers.conv import ConvNormAct
from hyper_lap.layers.film_unet import FilmUnetModule
from hyper_lap.layers.unet import ResBlock
from hyper_lap.layers.vae import VAE
from hyper_lap.serialisation.safetensors import load_pytree


class LatentModel(eqx.Module):
    # vae: FlaxAutoencoderKL = eqx.field(static=True)
    vae: VAE = eqx.field(static=True)

    in_conv: ConvNormAct

    model: FilmUnetModule

    out_conv: ConvNormAct

    recomb: ResBlock

    final_conv: nn.Conv2d

    def __init__(self, model: FilmUnetModule, *, key: PRNGKeyArray):
        super().__init__()

        # self.vae = FlaxAutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

        key, vae_key, embedder_key = jr.split(key, 3)

        vae = VAE(key=vae_key)

        self.vae = load_pytree("models/vae.safetensors", vae)

        self.model = model

        in_key, out_key, recomb_key, final_key = jr.split(key, 4)

        base_channels = model.base_channels

        self.in_conv = ConvNormAct(4, base_channels, groups=base_channels, key=in_key)

        self.out_conv = ConvNormAct(base_channels, 4, groups=4, key=out_key)

        self.recomb = ResBlock(3, 3, groups=3, key=recomb_key)

        self.final_conv = nn.Conv2d(3, 2, 1, use_bias=False, key=final_key)

    def __call__(
        self, x: Float[Array, "3 h w"], cond: Array, key: PRNGKeyArray | None = None
    ) -> Float[Array, "h w"]:
        latent_dist = self.vae.encode(x)

        if key is not None:
            z = latent_dist.sample(key)
        else:
            z = latent_dist.mean

        z = self.in_conv(z)

        z = self.model(z, cond)

        z = self.out_conv(z)

        y = self.vae.decode(z)

        y = self.recomb(y)

        y = self.final_conv(y)

        return y
