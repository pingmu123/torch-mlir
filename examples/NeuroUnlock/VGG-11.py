# https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py


import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

from typing import Any, cast, Dict, List, Optional, Union


class VGG(nn.Module):  # no init_weights
    def __init__(self, features: nn.Module, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    # A: VGG-11
    # B: VGG-13
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, progress: bool, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg11(*, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-11 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG11_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG11_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG11_Weights
        :members:
    """

    return _vgg("A", False, progress, **kwargs)


# def vgg13(*, progress: bool = True, **kwargs: Any) -> VGG:
#     """VGG-11 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

#     Args:
#         weights (:class:`~torchvision.models.VGG11_Weights`, optional): The
#             pretrained weights to use. See
#             :class:`~torchvision.models.VGG11_Weights` below for
#             more details, and possible values. By default, no pre-trained
#             weights are used.
#         progress (bool, optional): If True, displays a progress bar of the
#             download to stderr. Default is True.
#         **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
#             base class. Please refer to the `source code
#             <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
#             for more details about this class.

#     .. autoclass:: torchvision.models.VGG11_Weights
#         :members:
#     """

#     return _vgg("B", False, progress, **kwargs)


net = vgg11()  # no init_weights
net.eval()
print(net)
# compile to torch mlir
# NCHW layout in pytorch
print("================")
print("origin torch mlir")
print("================")
module = torch_mlir.compile(net, torch.ones(1, 3, 224, 224), output_type="torch")
print(module.operation.get_asm(large_elements_limit=10))
