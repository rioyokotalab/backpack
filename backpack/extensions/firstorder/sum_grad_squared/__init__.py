from torch.nn import Conv2d, Linear

from backpack.extensions.backprop_extension import BackpropExtension

from . import conv2d, linear


class SumGradSquared(BackpropExtension):
    """
    The sum of individual-gradients-squared, or second moment of the gradient.
    Is only meaningful is the individual functions are independent (no batchnorm).

    Stores the output in :code:`sum_grad_squared`. Same dimension as the gradient.
    """

    def __init__(self, loss_reduction="none"):

        super().__init__(
            savefield="sum_grad_squared",
            fail_mode="WARNING",
            module_exts={Linear: linear.SGSLinear(loss_reduction),
                         Conv2d: conv2d.SGSConv2d(loss_reduction),},
        )


class EmpKFAC(BackpropExtension):

    def __init__(self, loss_reduction="none"):

        super().__init__(
            savefield="empkfac",
            fail_mode="WARNING",
            module_exts={Linear: linear.EmpKFACLinear(loss_reduction),
                         Conv2d: conv2d.EmpKFACConv2d(loss_reduction),},
        )
