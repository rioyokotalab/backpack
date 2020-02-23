from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from backpack.utils.ein import einsum


class SGSLinear(FirstOrderModuleExtension):

    def __init__(self, loss_reduction='none'):
        self._loss_reduction = loss_reduction
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        N_axis = 0
        factor = 1.
        if self._loss_reduction == 'mean':
            factor *= g_out[0].size(N_axis)
        return (g_out[0].mul(factor) ** 2).sum(N_axis)

    def weight(self, ext, module, g_inp, g_out, backproped):
        factor = 1.
        if self._loss_reduction == 'mean':
            factor *= g_out[0].size(0)
        return einsum("ni,nj->ij", (g_out[0].mul(factor) ** 2, module.input0 ** 2))


class EmpKFACLinear(FirstOrderModuleExtension):

    def __init__(self, loss_reduction='none'):
        self._loss_reduction = loss_reduction
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        return [self._factor_from_outgrad(g_out[0])]

    def weight(self, ext, module, g_inp, g_out, backproped):
        return [self._factor_from_outgrad(g_out[0]), self._factor_from_input(module)]

    def _factor_from_outgrad(self, g_out):
        n = g_out.size(0)
        factor = 1.
        if self._loss_reduction == 'mean':
            factor *= n
        g_out = g_out.mul(factor)
        return einsum('ni,nj->ij', g_out, g_out).div(n)

    def _factor_from_input(self, module):
        inp0 = module.input0
        n = inp0.size(0)
        return einsum('ni,nj->ij', inp0, inp0).div(n)
