from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from backpack.utils import conv as convUtils
from backpack.utils.ein import einsum


class SGSConv2d(FirstOrderModuleExtension):

    def __init__(self, loss_reduction="none"):
        self._loss_reduction = loss_reduction
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        N_axis = 0
        factor = 1.
        if self._loss_reduction == 'mean':
            factor *= g_out[0].size(N_axis)
        return (einsum("nchw->nc", g_out[0].mul(factor)) ** 2).sum(N_axis)

    def weight(self, ext, module, g_inp, g_out, backproped):
        N_axis = 0
        factor = 1.
        if self._loss_reduction == 'mean':
            factor *= g_out[0].size(N_axis)
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, g_out[0].mul(factor), module
        )
        d1 = einsum("nml,nkl->nmk", (dE_dY, X))
        return (d1 ** 2).sum(N_axis).view_as(module.weight)


class EmpKFACConv2d(FirstOrderModuleExtension):

    def __init__(self, loss_reduction='none'):
        self._loss_reduction = loss_reduction
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        return [self._factor_from_outgrad(g_out[0])]

    def weight(self, ext, module, g_inp, g_out, backproped):
        return [self._factor_from_outgrad(g_out[0]), self._factor_from_input(module)]

    def _factor_from_outgrad(self, g_out):
        n, c_out = g_out.size()[0:2]
        factor = 1.
        if self._loss_reduction == 'mean':
            factor *= n
        g_out = g_out.mul(factor)
        g_out = g_out.view(n, c_out, -1)  # n, c_out, output_size
        m = g_out.transpose(0, 1).flatten(start_dim=1)  # c_out x n(output_size)

        return einsum('ik,jk->ij', m, m).div(m.size(-1))  # c_out x c_out

    def _factor_from_input(self, module):
        X = convUtils.unfold_func(module)(module.input0)  # n x (c_in)(kernel_size) x output_size
        n = X.size(0)
        return einsum('nik,njk->ij', (X, X)).div(n)  # (c_in)(kernel_size) x (c_in)(kernel_size)
