from jaxtyping import Array, PRNGKeyArray, PyTree
from typing import Any, Optional, Self, Sequence

import equinox as eqx
import equinox_hessian.nn as nn
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt

from ._util import _channel_to_spatials, _spatials_to_channel


class Downsample2d(nn.HessianMixin, eqx.Module):
    conv: nn.Conv2d

    def __init__(self, in_channels: int, out_channels: int, *, key: PRNGKeyArray):
        super().__init__()

        self.conv = nn.Conv2d(4 * in_channels, out_channels, 1, key=key)

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        x = _spatials_to_channel(x)

        x = self.conv(x)

        return x

    @property
    def hessian_filter_spec(self) -> PyTree:
        spec = eqx.filter(self, eqx.is_array, inverse=True)  # TODO: fixme

        spec = eqx.tree_at(lambda spec: spec.conv1, spec, (self.conv.hessian_filter_spec))

        return spec

    def sample_weights(self, *, key: PRNGKeyArray) -> Self:
        return eqx.tree_at(lambda me: me.conv1, self, self.conv.sample_weights(key=key))

    def forward_with_hessian_state(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, PyTree]:
        x = _spatials_to_channel(x)

        x, conv_state = self.conv.forward_with_hessian_state(x)

        return x, dict(conv_state=conv_state)

    def hessians(self, H: Array, hessian_state: PyTree) -> tuple[Array, PyTree]:
        H, conv_hessians = self.conv.hessians(H, hessian_state["conv_state"])

        H_x = _channel_to_spatials(H)

        hessians = jt.map(lambda _: None, self)
        hessians = eqx.tree_at(lambda hessians: hessians.conv1, hessians, conv_hessians)

        return H_x, hessians


class Upsample2d(nn.HessianMixin, eqx.Module):
    conv: nn.Conv2d

    def __init__(self, in_channels: int, out_channels: int, *, key: PRNGKeyArray):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 4 * out_channels, 1, key=key)

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        x = self.conv(x)

        x = _channel_to_spatials(x)

        return x

    @property
    def hessian_filter_spec(self) -> PyTree:
        spec = eqx.filter(self, eqx.is_array, inverse=True)  # TODO: fixme

        spec = eqx.tree_at(lambda spec: spec.conv1, spec, (self.conv.hessian_filter_spec))

        return spec

    def sample_weights(self, *, key: PRNGKeyArray) -> Self:
        return eqx.tree_at(lambda me: me.conv1, self, self.conv.sample_weights(key=key))

    def forward_with_hessian_state(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, PyTree]:
        x, conv_state = self.conv.forward_with_hessian_state(x)

        x = _channel_to_spatials(x)

        return x, dict(conv_state=conv_state)

    def hessians(self, H: Array, hessian_state: PyTree) -> tuple[Array, PyTree]:
        H = _spatials_to_channel(H)

        H_x, conv_hessians = self.conv.hessians(H, hessian_state["conv_state"])

        hessians = jt.map(lambda _: None, self)
        hessians = eqx.tree_at(lambda hessians: hessians.conv1, hessians, conv_hessians)

        return H_x, hessians


class ConvNormAct(nn.HessianMixin, eqx.Module):
    conv: nn.Conv2d
    # norm: nn.BatchNorm2d
    norm: nn.GroupNorm
    act: nn.ReLU

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups=8,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()  # type: ignore

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding="SAME", use_bias=False, key=key
        )

        # self.norm = nn.BatchNorm2d(out_channels, "batch")
        self.norm = nn.GroupNorm(groups, out_channels)

        self.act = nn.ReLU()

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x

    @property
    def hessian_filter_spec(self) -> PyTree:
        return eqx.tree_at(
            lambda me: [me.conv1, me.norm1, me.act],
            self,
            [m.hessian_filter_spec for m in [self.conv, self.norm, self.act]],
        )

    def sample_weights(self, *, key: PRNGKeyArray) -> Self:
        keys = jr.split(key, 3)

        return eqx.tree_at(
            lambda me: [me.conv1, me.norm1, me.act],
            self,
            [m.sample_weights(key=key) for m, key in zip([self.conv, self.norm, self.act], keys)],
        )

    def forward_with_hessian_state(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, PyTree]:
        x, conv_state = self.conv.forward_with_hessian_state(x)

        x, norm_state = self.norm.forward_with_hessian_state(x)  # type: ignore

        x, act_state = self.act.forward_with_hessian_state(x)

        hessian_state = eqx.tree_at(
            lambda me: [me.conv1, me.norm1, me.act], self, [conv_state, norm_state, act_state]
        )

        return x, hessian_state

    def hessians(self, H: Array, hessian_state: PyTree) -> tuple[Array, PyTree]:
        H, act_hessians = self.act.hessians(H, hessian_state.act)

        H, norm_hessians = self.norm.hessians(H, hessian_state.norm1)

        H, conv_hessians = self.conv.hessians(H, hessian_state.conv1)

        hessians = eqx.tree_at(
            lambda me: [me.conv1, me.norm1, me.act],
            self,
            [conv_hessians, norm_hessians, act_hessians],
        )

        return H, hessians


class Block(nn.Sequential):
    res_conv: Optional[nn.Conv2d]

    use_res: bool = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 8,
        n_convs: int = 2,
        use_res: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        if n_convs < 1:
            raise ValueError("Must have at least one conv")

        self.use_res = use_res

        if use_res and in_channels != out_channels:
            key, consume = jr.split(key)

            self.res_conv = nn.Conv2d(in_channels, out_channels, 1, padding="SAME", key=consume)
        else:
            self.res_conv = None

        keys = jr.split(key)

        layers = [
            ConvNormAct(in_channels, out_channels, kernel_size, groups=groups, key=keys[0])
        ] + [
            ConvNormAct(out_channels, out_channels, kernel_size, groups=groups, key=key)
            for key in keys[1:]
        ]

        super().__init__(layers)

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:  # type: ignore
        res = x

        for layer in self.layers:
            x = layer(x)

        if self.use_res:
            if self.res_conv is not None:
                res = self.res_conv(res)

            x += res

        return x

    def forward_with_hessian_state(  # type: ignore
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, PyTree]:
        res = x

        hessian_states = []

        for layer in self.layers:
            x, h_state = layer.forward_with_hessian_state(x)

            hessian_states.append(h_state)

        hessian_state = eqx.tree_at(lambda me: [layer for layer in me.layers], self, hessian_states)

        if self.use_res:
            assert self.res_conv is not None

            res, h_state = self.res_conv.forward_with_hessian_state(res)

            x = x + res

            hessian_state = eqx.tree_at(lambda state: state.res_conv, hessian_state, h_state)
        else:
            hessian_state = eqx.tree_at(lambda state: state.res_conv, hessian_state, None)

        return x, hessian_state

    def hessians(self, H: Array, hessian_state: PyTree) -> tuple[Array, PyTree]:
        H_x = H

        # since we move back to from collect the hessians in reverse
        hessians_reversed = []

        # walk through layers in reverse
        for layer, h_state in reversed(list(zip(self.layers, hessian_state))):
            assert isinstance(layer, nn.HessianMixin), f"{layer} is not a HessianMixin"

            # update H and save hessians in list
            H_x, hessians = layer.hessians(H_x, h_state)

            hessians_reversed.append(hessians)

        hessians = list(reversed(hessians_reversed))

        # return self, with each module replaced by its hessian
        hessians = eqx.tree_at(lambda me: [module for module in me], self, hessians)

        if self.use_res:
            assert self.res_conv is not None

            H_res, hessians_res = self.res_conv.hessians(H, hessian_state.res_conv)

            H_x = H_x + H_res

            hessians = eqx.tree_at(lambda hess: hess.res_conv, hessians, hessians_res)

        return H_x, hessians


class UnetDown(nn.HessianMixin, eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    blocks: list[Block]
    downs: list[nn.MaxPool2d]

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        *,
        key: PRNGKeyArray,
        block_args: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        if block_args is None:
            block_args = {}

        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)

        channels = base_channels

        self.blocks = []
        self.downs = []

        for channel_mult in channel_mults[1:]:
            new_channels = channel_mult * base_channels

            key, block_key = jr.split(key, 2)

            self.blocks.append(
                Block(channels, new_channels, n_convs=3, key=block_key, **block_args)
            )

            channels = new_channels

            self.downs.append(nn.MaxPool2d(2, 2))

    @property
    def hessian_filter_spec(self) -> PyTree:
        spec = eqx.filter(self, eqx.is_array, inverse=True)  # TODO: fixme

        spec = eqx.tree_at(
            lambda me: me.blocks, spec, ([block.hessian_filter_spec for block in self.blocks])
        )

        spec = eqx.tree_at(
            lambda me: me.downs, spec, ([down.hessian_filter_spec for down in self.downs])
        )

        return spec

    def sample_weights(self, *, key: PRNGKeyArray) -> Self:
        model = self

        key, *block_keys = jr.split(key, len(self.blocks) + 1)

        model = eqx.tree_at(
            lambda me: me.blocks,
            model,
            [block.sample_weights(key=key) for block, key in zip(self.blocks, block_keys)],
        )

        down_keys = jr.split(key, len(self.downs))

        model = eqx.tree_at(
            lambda me: me.downs,
            model,
            [down.sample_weights(key=key) for down, key in zip(self.downs, down_keys)],
        )

        return model

    def __call__(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, list[Array]]:
        skips: list[Array] = []

        for block, down in zip(self.blocks, self.downs):
            x = block(x)

            skips.append(x)

            x = down(x)

        return x, skips

    def forward_with_hessian_state(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[tuple[Array, list[Array]], PyTree]:
        skips: list[Array] = []

        block_h_states = []
        down_h_states = []

        for block, down in zip(self.blocks, self.downs):
            x, h_state = block.forward_with_hessian_state(x)

            block_h_states.append(h_state)

            skips.append(x)

            x, h_state = down.forward_with_hessian_state(x)

            down_h_states.append(h_state)

        hessian_state = dict(downs=down_h_states, blocks=block_h_states)

        return (x, skips), hessian_state

    def hessians(  # type: ignore
        self, H: Array, H_skips: list[Array], hessian_state: PyTree
    ) -> tuple[Array, PyTree]:
        block_hessians_reversed = []
        down_hessians_reversed = []

        H_x = H

        block_hs = hessian_state["blocks"]
        down_hs = hessian_state["downs"]

        # run though block and downs in reverse
        for block, down, H_skip, block_h_state, down_h_state in reversed(
            list(zip(self.blocks, self.downs, H_skips, block_hs, down_hs))
        ):
            H_x, down_hessians = down.hessians(H_x, down_h_state)

            down_hessians_reversed.append(down_hessians)

            # each block's output goes two ways: to the next down/return and to skips
            # by doing the math by hand we can see that the correct way to handle the fork is
            # H' = J.T @ H1 @ J + J.T @ H2 @ J,
            # i.e. add the Hessians *after* pulling them back through the block
            # this is true for both input and parameter hessians

            H_x1, block_hessians1 = block.hessians(H_x, block_h_state)
            H_x2, block_hessians2 = block.hessians(H_skip, block_h_state)

            H_x = H_x1 + H_x2

            block_hessians = jt.map(lambda x, y: x + y, block_hessians1, block_hessians2)

            block_hessians_reversed.append(block_hessians)

        hessians = jt.map(lambda _: None, self)

        hessians = eqx.tree_at(
            lambda me: me.blocks, hessians, list(reversed(block_hessians_reversed))
        )

        hessians = eqx.tree_at(
            lambda me: me.downs, hessians, list(reversed(down_hessians_reversed))
        )

        return H_x, hessians


class UnetUp(nn.HessianMixin, eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    blocks: list[Block]
    ups: list[Upsample2d]

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        *,
        key: PRNGKeyArray,
        block_args: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        if block_args is None:
            block_args = {}

        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)

        channels = base_channels * channel_mults[-1]

        self.blocks = []
        self.ups = []

        for channel_mult in list(reversed(channel_mults))[1:]:
            new_channels = channel_mult * base_channels

            key, block_key, up_key = jr.split(key, 3)

            self.ups.append(Upsample2d(channels, channels, key=up_key))

            self.blocks.append(Block(2 * channels, new_channels, key=block_key, **block_args))

            channels = new_channels

    @property
    def hessian_filter_spec(self) -> PyTree:
        spec = jt.map(lambda _: False, self)

        spec = eqx.tree_at(lambda me: me.ups, spec, ([up.hessian_filter_spec for up in self.ups]))

        spec = eqx.tree_at(
            lambda me: me.blocks, spec, ([block.hessian_filter_spec for block in self.blocks])
        )

        return spec

    def sample_weights(self, *, key: PRNGKeyArray) -> Self:
        model = self

        key, *up_keys = jr.split(key, len(self.ups) + 1)

        model = eqx.tree_at(
            lambda me: me.ups,
            model,
            [up.sample_weights(key=key) for up, key in zip(self.ups, up_keys)],
        )

        block_keys = jr.split(key, len(self.blocks))

        model = eqx.tree_at(
            lambda me: me.blocks,
            model,
            [block.sample_weights(key=key) for block, key in zip(self.blocks, block_keys)],
        )

        return model

    def __call__(
        self, x: Array, skips: list[Array], *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        skips = skips.copy()

        for up, block in zip(self.ups, self.blocks):
            x = up(x)

            skip = skips.pop()

            x = jnp.concat([skip, x], axis=0)

            x = block(x)

        return x

    def forward_with_hessian_state(
        self, x: Array, skips: list[Array], *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, PyTree]:
        skips = skips.copy()

        up_h_states = []
        block_h_states = []

        for up, block in zip(self.ups, self.blocks):
            x, h_state = up.forward_with_hessian_state(x)

            up_h_states.append(h_state)

            skip = skips.pop()

            x = jnp.concat([skip, x], axis=0)

            x, h_state = block.forward_with_hessian_state(x)

            block_h_states.append(h_state)

        hessian_state = dict(ups=up_h_states, blocks=block_h_states)

        return x, hessian_state

    def hessians(self, H: Array, hessian_state: PyTree) -> tuple[Array, list[Array], PyTree]:  # type: ignore
        H_x = H

        H_skips = []

        up_hessians_reversed = []
        block_hessians_reversed = []

        up_hs = hessian_state["ups"]
        block_hs = hessian_state["blocks"]

        for up, block, up_h_state, block_h_state in reversed(
            list(zip(self.ups, self.blocks, up_hs, block_hs))
        ):
            H_x, block_hessian = block.hessians(H_x, block_h_state)

            block_hessians_reversed.append(block_hessian)

            # split H_x into H_skip and H_x from downwards
            channels = H_x.shape[0]
            assert channels % 2 == 0

            H_skip = H_x[: channels // 2]
            H_x = H_x[channels // 2 :]

            H_skips.append(H_skip)

            H_x, up_hessian = up.hessians(H_x, up_h_state)

            up_hessians_reversed.append(up_hessian)

        hessians = jt.map(lambda _: None, self)

        hessians = eqx.tree_at(lambda me: me.ups, hessians, list(reversed(up_hessians_reversed)))

        hessians = eqx.tree_at(
            lambda me: me.blocks, hessians, list(reversed(block_hessians_reversed))
        )

        return H_x, H_skips, hessians


class UnetModule(nn.HessianMixin, eqx.Module):
    down: UnetDown
    middle: Block
    up: UnetUp

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        *,
        key: PRNGKeyArray,
        block_args: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        down_key, middle_key, up_key = jr.split(key, 3)

        self.down = UnetDown(base_channels, channel_mults, key=down_key, block_args=block_args)

        middle_channels = base_channels * channel_mults[-1]

        self.middle = Block(middle_channels, middle_channels, key=middle_key)

        self.up = UnetUp(base_channels, channel_mults, key=up_key, block_args=block_args)

    @property
    def hessian_filter_spec(self) -> PyTree:
        spec = eqx.filter(self, eqx.is_array, inverse=True)  # TODO: fixme

        spec = eqx.tree_at(lambda me: me.down, spec, self.down.hessian_filter_spec)

        spec = eqx.tree_at(lambda me: me.middle, spec, self.middle.hessian_filter_spec)

        spec = eqx.tree_at(lambda me: me.up, spec, self.up.hessian_filter_spec)

        return spec

    def sample_weights(self, *, key: Array) -> Self:
        down_key, middle_key, up_key = jr.split(key, 3)

        model = self

        model = eqx.tree_at(lambda me: me.down, model, self.down.sample_weights(key=down_key))

        model = eqx.tree_at(lambda me: me.middle, model, self.middle.sample_weights(key=middle_key))

        model = eqx.tree_at(lambda me: me.up, model, self.up.sample_weights(key=up_key))

        return model

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        x, skips = self.down(x)

        x = self.middle(x)

        x = self.up(x, skips)

        return x

    def forward_with_hessian_state(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, PyTree]:
        (x, skips), down_state = self.down.forward_with_hessian_state(x)

        x, middle_state = self.middle.forward_with_hessian_state(x)

        x, up_state = self.up.forward_with_hessian_state(x, skips)

        return x, dict(down=down_state, middle=middle_state, up=up_state)

    def hessians(self, H: Array, hessian_state: PyTree) -> tuple[Array, PyTree]:
        H_middle, H_skips, up_hessians = self.up.hessians(H, hessian_state["up"])

        H_middle, middle_hessians = self.middle.hessians(H_middle, hessian_state["middle"])

        H_x, down_hessians = self.down.hessians(H_middle, H_skips, hessian_state["down"])

        hessians = jt.map(lambda _: None, self)

        hessians = eqx.tree_at(lambda me: me.up, hessians, up_hessians)

        hessians = eqx.tree_at(lambda me: me.middle, hessians, middle_hessians)

        hessians = eqx.tree_at(lambda me: me.down, hessians, down_hessians)

        return H_x, hessians
