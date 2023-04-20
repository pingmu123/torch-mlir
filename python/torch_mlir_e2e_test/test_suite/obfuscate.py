import torch
from torch import nn
import torch.nn.functional as F

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(2)
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.train(False)

    @export
    @annotate_args(
        [
            None,
            ([1, 1, 28, 28], torch.float32, True),
        ]
    )
    def forward(self, x):
        # input shape is 1x1x28x28
        # Max pooling over a (2, 2) window, if use default stride, error will happen
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), stride=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), stride=(2, 2))
        # flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@register_test_case(
    module_factory=lambda: LeNet(), 
    passes="func.func(torch-insert-skip{layer=2})"
)
def ObfuscateLeNet_1(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 28, 28))


@register_test_case(
    module_factory=lambda: LeNet(), 
    passes="func.func(torch-widen-conv-layer{layer=1 number=4})"
)
def ObfuscateLeNet_2(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 28, 28))


@register_test_case(
    module_factory=lambda: LeNet(), 
    passes="func.func(torch-insert-sepra-conv-layer{layer=2})"
)
def ObfuscateLeNet_3(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 28, 28))


@register_test_case(
    module_factory=lambda: LeNet(), 
    passes="func.func(torch-branch-layer{layer=2 branch=4})"
)
def ObfuscateLeNet_4(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 28, 28))


@register_test_case(
    module_factory=lambda: LeNet(), 
    passes="func.func(torch-insert-conv)"
)
def ObfuscateLeNet_5(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 28, 28))


@register_test_case(
    module_factory=lambda: LeNet(), 
    passes="func.func(torch-insert-conv{number=5})"
)
def ObfuscateLeNet_6(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 28, 28))


@register_test_case(
    module_factory=lambda: LeNet(), 
    passes="func.func(torch-widen-conv-layer), func.func(torch-insert-conv)"
)
def ObfuscateLeNet_7(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 28, 28))


# error case
# @register_test_case(module_factory=lambda: LeNet(), passes="func.func(torch-insert-skip), func.func(torch-widen-conv-layer)")
# def LeNet_5(module, tu: TestUtils):
#     module.forward(tu.rand(1, 1, 28, 28))

# ==============================================================================


class RNN_scratch(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=18):
        super().__init__()
        torch.manual_seed(2)
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.hidden = torch.zeros(1, self.hidden_size)
        self.train(False)

    @export
    @annotate_args(
        [
            None,
            ([3, 1, 10], torch.float32, True),
        ]
    )
    def forward(self, input):
        hidden = self.hidden
        for i in range(input.size(0)):
            combined = torch.cat((input[i], hidden), 1)
            hidden = torch.sigmoid(self.i2h(combined))
            output = self.i2o(combined)
        return output, hidden


@register_test_case(
    module_factory=lambda: RNN_scratch(), passes="func.func(torch-insert-skip)"
)
def ObfuscateRNN_1(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 10))


@register_test_case(
    module_factory=lambda: RNN_scratch(), passes="func.func(torch-widen-conv-layer)"
)
def ObfuscateRNN_2(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 10))


@register_test_case(
    module_factory=lambda: RNN_scratch(), passes="func.func(torch-insert-conv)"
)
def ObfuscateRNN_3(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 10))


@register_test_case(
    module_factory=lambda: RNN_scratch(),
    passes="func.func(torch-obfuscate-rnn{obfuscation=valueSplit})",
)
def ObfuscateRNN_4(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 10))


@register_test_case(
    module_factory=lambda: RNN_scratch(),
    passes="func.func(torch-obfuscate-rnn{obfuscation=maskSplit})",
)
def ObfuscateRNN_5(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 10))


@register_test_case(
    module_factory=lambda: RNN_scratch(),
    passes="func.func(torch-obfuscate-rnn{obfuscation=valueSplit}), func.func(torch-obfuscate-rnn{obfuscation=valueSplit})",
)
def ObfuscateRNN_6(module, tu: TestUtils):
    module.forward(tu.rand(3, 1, 10))
