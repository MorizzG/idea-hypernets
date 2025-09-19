from jaxtyping import Array, Shaped


def channel_to_spatials2d(x: Shaped[Array, "c h w"]) -> Shaped[Array, "c h w"]:
    c, h, w = x.shape

    assert c % 4 == 0

    x = x.reshape(c // 4, 2, 2, h, w)

    x = x.transpose(0, 3, 1, 4, 2)

    x = x.reshape(c // 4, h * 2, w * 2)

    return x


def spatials_to_channel2d(x: Shaped[Array, "c h w"]) -> Shaped[Array, "c h w"]:
    c, h, w = x.shape

    assert h % 2 == 0 and w % 2 == 0

    x = x.reshape(c, h // 2, 2, w // 2, 2)

    x = x.transpose(0, 2, 4, 1, 3)

    x = x.reshape(4 * c, h // 2, w // 2)

    return x


def channel_to_spatials3d(x: Shaped[Array, "c h w d"]) -> Shaped[Array, "c h w d"]:
    c, h, w, d = x.shape

    assert c % 8 == 0

    x = x.reshape(c // 8, 2, 2, 2, h, w, d)

    x = x.transpose(0, 4, 1, 5, 2, 6, 3)

    x = x.reshape(c // 8, h * 2, w * 2, d * 2)

    return x


def spatials_to_channel3d(x: Shaped[Array, "c h w d"]) -> Shaped[Array, "c h w d"]:
    c, h, w, d = x.shape

    assert h % 2 == 0 and w % 2 == 0 and d % 2 == 0

    x = x.reshape(c, h // 2, 2, w // 2, 2, d // 2, 2)

    x = x.transpose(0, 2, 4, 6, 1, 3, 5)

    x = x.reshape(8 * c, h // 2, w // 2, d // 2)

    return x
