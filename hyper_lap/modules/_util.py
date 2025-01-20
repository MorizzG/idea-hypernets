from jaxtyping import Array, Float, Shaped

import equinox as eqx
import jax
import jax.numpy as jnp
from chex import assert_axis_dimension, assert_rank, assert_shape


class ReLU(eqx.nn.Lambda):
    def __init__(self):
        super().__init__(fn=jax.nn.relu)

    # def __call__(self, x: Array) -> Array:
    #     return jax.nn.relu(x)


class SiLU(eqx.nn.Lambda):
    def __init__(self):
        super().__init__(fn=jax.nn.swish)


def _channel_to_spatials2d(x: Shaped[Array, "c h w"]) -> Shaped[Array, "c h w"]:
    c, h, w = x.shape

    assert c % 4 == 0

    x = x.reshape(c // 4, 2, 2, h, w)

    x = x.transpose(0, 3, 1, 4, 2)

    x = x.reshape(c // 4, h * 2, w * 2)

    return x


def _spatials_to_channel2d(x: Shaped[Array, "c h w"]) -> Shaped[Array, "c h w"]:
    c, h, w = x.shape

    assert h % 2 == 0 and w % 2 == 0

    x = x.reshape(c, h // 2, 2, w // 2, 2)

    x = x.transpose(0, 2, 4, 1, 3)

    x = x.reshape(4 * c, h // 2, w // 2)

    return x


def _channel_to_spatials3d(x: Shaped[Array, "c h w d"]) -> Shaped[Array, "c h w d"]:
    c, h, w, d = x.shape

    assert c % 8 == 0

    x = x.reshape(c // 8, 2, 2, 2, h, w, d)

    x = x.transpose(0, 4, 1, 5, 2, 6, 3)

    x = x.reshape(c // 8, h * 2, w * 2, d * 2)

    return x


def _spatials_to_channel3d(x: Shaped[Array, "c h w d"]) -> Shaped[Array, "c h w d"]:
    c, h, w, d = x.shape

    assert h % 2 == 0 and w % 2 == 0 and d % 2 == 0

    x = x.reshape(c, h // 2, 2, w // 2, 2, d // 2, 2)

    x = x.transpose(0, 2, 4, 6, 1, 3, 5)

    x = x.reshape(8 * c, h // 2, w // 2, d // 2)

    return x


def img_to_imagenet(img: Float[Array, "c h w"]):
    assert_rank(img, 3)

    c = img.shape[0]

    if c == 3:
        pass
    elif c == 1:
        img = jnp.repeat(img, 3, 0)
    else:
        raise RuntimeError(f"unexpected number of channels {c}")

    imagenet_mean = jnp.expand_dims(jnp.array([0.48145466, 0.4578275, 0.40821073]), (1, 2))
    imagenet_std = jnp.expand_dims(jnp.array([0.26862954, 0.26130258, 0.27577711]), (1, 2))

    # assert imagenet_mean.shape == (3,1,1) and
    assert_shape((imagenet_mean, imagenet_std), (3, 1, 1))

    mean = img.mean(axis=(1, 2))
    std = img.std(axis=(1, 2))

    img_normed = (img - mean) / (std + 1e-5)

    img_normed = imagenet_std * img_normed + imagenet_mean

    return img_normed
