from torch import nn
from torch.nn import (AvgPool2d, Conv2d, CrossEntropyLoss, Dropout, Linear,
                      MaxPool2d, MSELoss, ReLU, Sigmoid, Tanh, ZeroPad2d)

from backpack.extensions.backprop_extension import BackpropExtension

from . import (activations, linear)


class Jacobian(BackpropExtension):

    def __init__(self, start: nn.Module=None, end: nn.Module=None, savefield=None):

        if savefield is None:
            savefield = "jacobian"

        supported_modules = (nn.Linear, nn.Conv2d)
        msg = '{} can not be start/end.'
        if start is not None:
            assert isinstance(start, supported_modules), msg.format(start.__class__.__name__)
        if end is not None:
            assert isinstance(end, supported_modules), msg.format(end.__class__.__name__)

        self._start = start
        self._end = end

        if start is None:
            self._started = True
        else:
            self._started = False

        self._ended = False

        super().__init__(savefield=savefield,
                         fail_mode="ERROR",
                         module_exts={
                             Linear: linear.JacobianLinear(),
                             ReLU: activations.JacobianReLU(),
                             Sigmoid: activations.JacobianSigmoid(),
                             Tanh: activations.JacobianTanh(),
                         })

    def apply(self, module, g_inp, g_out):

        if module is self._start:
            self._started = True

        if self._started and not self._ended:
            super().apply(module, g_inp, g_out)

        if module is self._end:
            self._ended = True

