# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
