from .coupling_layer import CouplingLayer
from .split_flow import SplitFlow
from .squeeze_flow import SqueezeFlow
from .variational_dequantization import VariationalDequantization
from .gated_conv_net import GatedConvNet
from .create_checkerboard_mask import create_checkerboard_mask
from .create_channel_mask import create_channel_mask


def create_multiscale_flow():
    flow_layers = []

    vardeq_layers = [
        CouplingLayer(
            network=GatedConvNet(c_out=2, c_hidden=16),
            mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
            c_in=1,
        )
        for i in range(4)
    ]
    flow_layers += [VariationalDequantization(var_flows=vardeq_layers)]

    flow_layers += [
        CouplingLayer(
            network=GatedConvNet(c_out=2, c_hidden=32),
            mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
            c_in=1,
        )
        for i in range(2)
    ]
    flow_layers += [SqueezeFlow()]
    for i in range(2):
        flow_layers += [
            CouplingLayer(
                network=GatedConvNet(c_out=8, c_hidden=48),
                mask=create_channel_mask(c_in=4, invert=(i % 2 == 1)),
                c_in=4,
            )
        ]
    flow_layers += [SplitFlow(), SqueezeFlow()]
    for i in range(4):
        flow_layers += [
            CouplingLayer(
                network=GatedConvNet(c_out=16, c_hidden=64),
                mask=create_channel_mask(c_in=8, invert=(i % 2 == 1)),
                c_in=8,
            )
        ]

    flow_model = ImageFlow(flow_layers)
    return flow_model
