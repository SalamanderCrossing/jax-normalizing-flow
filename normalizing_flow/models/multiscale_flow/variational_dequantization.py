from .dequantization import Dequantization
from flax import linen as nn
from jax import random
from jax import numpy as jnp
import numpy as np
from typing import Sequence, Optional


class VariationalDequantization(Dequantization):
    var_flows: Optional[
        Sequence[nn.Module]
    ] = None  # A list of flow transformations to use for modeling q(u|x)

    def dequant(self, z, ldj, rng):
        z = z.astype(jnp.float32)
        img = (
            z / 255.0
        ) * 2 - 1  # We condition the flows on x, i.e. the original image

        # Prior of u is a uniform distribution as before
        # As most flow transformations are defined on [-infinity,+infinity], we apply an inverse sigmoid first.
        rng, uniform_rng = random.split(rng)
        deq_noise = random.uniform(uniform_rng, z.shape)
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=True)
        if self.var_flows is not None:
            for flow in self.var_flows:
                deq_noise, ldj, rng = flow(
                    deq_noise, ldj, rng, reverse=False, orig_img=img
                )
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=False)

        # After the flows, apply u as in standard dequantization
        z = (z + deq_noise) / 256.0
        ldj -= np.log(256.0) * np.prod(z.shape[1:])
        return z, ldj, rng
