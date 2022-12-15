from flax import linen as nn
from jax import numpy as jnp


class ConcatELU(nn.Module):
    """
    Activation function that applies ELU in both direction (inverted and plain).
    Allows non-linearity while providing strong gradients for any input (important for final convolution)
    """

    def __call__(self, x):
        return jnp.concatenate([nn.elu(x), nn.elu(-x)], axis=-1)


class GatedConv(nn.Module):
    """This module applies a two-layer convolutional ResNet block with input gate"""

    c_in: int  # Number of input channels
    c_hidden: int  # Number of hidden dimensions

    @nn.compact
    def __call__(self, x):
        out = nn.Sequential(
            [
                ConcatELU(),
                nn.Conv(self.c_hidden, kernel_size=(3, 3)),
                ConcatELU(),
                nn.Conv(2 * self.c_in, kernel_size=(1, 1)),
            ]
        )(x)
        val, gate = out.split(2, axis=-1)
        return x + val * nn.sigmoid(gate)
