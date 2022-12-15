from flax import linen as nn


class SqueezeFlow(nn.Module):
    def __call__(self, z, ldj, rng, reverse=False):
        B, H, W, C = z.shape
        if not reverse:
            # Forward direction: H x W x C => H/2 x W/2 x 4C
            z = z.reshape(B, H // 2, 2, W // 2, 2, C)
            z = z.transpose((0, 1, 3, 2, 4, 5))
            z = z.reshape(B, H // 2, W // 2, 4 * C)
        else:
            # Reverse direction: H/2 x W/2 x 4C => H x W x C
            z = z.reshape(B, H, W, 2, 2, C // 4)
            z = z.transpose((0, 1, 3, 2, 4, 5))
            z = z.reshape(B, H * 2, W * 2, C // 4)
        return z, ldj, rng
