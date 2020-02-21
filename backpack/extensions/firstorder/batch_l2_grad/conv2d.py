from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from backpack.utils import conv as convUtils
from backpack.utils.ein import einsum


class BatchL2Conv2d(FirstOrderModuleExtension):
    def __init__(self):
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        C_axis = 1
        return (einsum("nchw->nc", g_out[0]) ** 2).sum(C_axis)

    def weight(self, ext, module, g_inp, g_out, backproped):
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, g_out[0], module
        )
        return einsum("nml,nkl,nmi,nki->n", (dE_dY, X, dE_dY, X))
