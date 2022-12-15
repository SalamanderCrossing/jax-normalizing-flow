import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from flax import linen as nn


class SplitFlow(nn.Module):
    def __call__(self, z, ldj, rng, reverse=False):
        if not reverse:
            z, z_split = z.split(2, axis=-1)
            ldj += jax.scipy.stats.norm.logpdf(z_split).sum(axis=(1, 2, 3))
        else:
            z_split = random.normal(rng, z.shape)
            z = jnp.concatenate([z, z_split], axis=-1)
            ldj -= jax.scipy.stats.norm.logpdf(z_split).sum(axis=(1, 2, 3))
        return z, ldj, rng
