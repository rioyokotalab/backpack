from backpack.core.derivatives.linear import LinearDerivatives, LinearConcatDerivatives
from backpack.utils.utils import einsum
from .diag_ggn_base import DiagGGNBaseModule


class DiagGGNLinear(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=LinearDerivatives(),
            params=["bias", "weight"]
        )

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        return einsum('bic->i', (backproped ** 2,))

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        return einsum('bic,bj->ij', (backproped ** 2, module.input0 ** 2))


class DiagGGNLinearConcat(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=LinearConcatDerivatives(),
            params=["weight"]
        )

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        input = module.homogeneous_input()
        return einsum('bic,bj->ij', (backproped ** 2, input ** 2))


class DiagGGNFRLinear(DiagGGNLinear):

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        attr = 'last_backproped'
        last_bp = getattr(self, attr, None)
        if last_bp is None:
            setattr(self, attr, backproped)
            return einsum('bic->i', (backproped ** 2,))
        else:
            delattr(self, attr)
            return einsum('bic->i', (backproped.mul(last_bp)))

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        attr_bp = 'last_backproped'
        attr_inp0 = 'last_input0'
        last_bp = getattr(self, attr_bp, None)
        last_inp0 = getattr(self, attr_inp0, None)
        if last_bp is None:
            setattr(self, attr_bp, backproped)
            setattr(self, attr_inp0, module.input0)
            return einsum('bic,bj->ij', (backproped ** 2, module.input0 ** 2))
        else:
            delattr(self, attr_bp)
            delattr(self, attr_inp0)
            return einsum('bic,bj->ij', (backproped.mul(last_bp), module.input0.mul(last_inp0)))
