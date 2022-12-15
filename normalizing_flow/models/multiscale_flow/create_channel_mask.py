from jax import numpy as jnp

def create_channel_mask(c_in, invert=False) -> jnp.ndarray:
    mask = jnp.concatenate(
        [
            jnp.ones((c_in // 2,), dtype=jnp.float32),
            jnp.zeros((c_in - c_in // 2,), dtype=jnp.float32),
        ]
    )
    mask = mask.reshape(1, 1, 1, c_in)
    if invert:
        mask = 1 - mask
    return mask
