from jaxtyping import Array, PRNGKeyArray
from typing import Optional

import equinox as eqx
import jax


class ReLU(eqx.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        return jax.nn.relu(x)


class SiLU(eqx.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        return jax.nn.swish(x)
