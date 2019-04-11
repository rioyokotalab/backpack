"""Test CVP of reshape operation."""

import torch
from torch.nn import Module
from bpexts.cvp.reshape import CVPReshape
from .cvp_test import set_up_cvp_tests

# hyper-parameters
input_size = (3, 4, 5)
output_size = (6, 5, 2)
atol = 1e-7
rtol = 1e-5
num_hvp = 10


class TorchReshape(Module):
    def forward(self, x):
        return x.reshape(*output_size)


def torch_fn():
    return TorchReshape()


def cvp_fn():
    return CVPReshape(output_size)


for name, test_cls in set_up_cvp_tests(
        torch_fn,
        cvp_fn,
        'CVPReshape',
        input_size=input_size,
        atol=atol,
        rtol=rtol,
        num_hvp=num_hvp):
    exec('{} = test_cls'.format(name))
    del test_cls
