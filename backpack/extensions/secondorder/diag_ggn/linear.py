import backpack.utils.linear as LinUtils
from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule
from backpack.utils.ein import einsum


class DiagGGNLinear(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        return LinUtils.extract_bias_diagonal(module, backproped)

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        return LinUtils.extract_weight_diagonal(module, backproped)


class DiagGGNFRLinear(DiagGGNLinear):

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        param = module.bias
        attr = 'last_backproped'
        last_bp = getattr(param, attr, None)
        if last_bp is None:
            setattr(param, attr, backproped)
            return einsum('bic->i', (backproped ** 2,))
        else:
            delattr(param, attr)
            return einsum('bic->i', (backproped.mul(last_bp)))

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        param = module.weight
        attr_bp = 'last_backproped'
        attr_inp0 = 'last_input0'
        last_bp = getattr(param, attr_bp, None)
        last_inp0 = getattr(param, attr_inp0, None)
        if last_bp is None:
            setattr(param, attr_bp, backproped)
            setattr(param, attr_inp0, module.input0)
            return einsum('bic,bj->ij', (backproped ** 2, module.input0 ** 2))
        else:
            delattr(param, attr_bp)
            delattr(param, attr_inp0)
            return einsum('bic,bj->ij', (backproped.mul(last_bp), module.input0.mul(last_inp0)))
