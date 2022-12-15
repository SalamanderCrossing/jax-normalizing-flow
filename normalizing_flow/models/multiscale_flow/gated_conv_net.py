from flax import linen as nn
from .gated_conv import GatedConv, ConcatELU


class GatedConvNet(nn.Module):
    c_hidden: int  # Number of hidden dimensions to use within the network
    c_out: int  # Number of output channels
    num_layers: int = 3  # Number of gated ResNet blocks to apply

    def setup(self):
        layers = []
        layers += [nn.Conv(self.c_hidden, kernel_size=(3, 3))]
        for _ in range(self.num_layers):
            layers += [GatedConv(self.c_hidden, self.c_hidden), nn.LayerNorm()]
        layers += [
            ConcatELU(),
            nn.Conv(self.c_out, kernel_size=(3, 3), kernel_init=nn.initializers.zeros),
        ]
        self.nn = nn.Sequential(layers)

    def __call__(self, x):
        return self.nn(x)
