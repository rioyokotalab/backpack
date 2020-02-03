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


class DiagGGNLinearEfficient(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=LinearDerivatives(),
            params=["bias", "weight"]
        )

        self._bias_is_called_before_weight = False
        self._weight_is_called_before_bias = False
        self._attr_sq_bp = 'sq_bp'

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        attr = self._attr_sq_bp

        if not self._weight_is_called_before_bias:
            self._bias_is_called_before_weight = True
            sq_bp = self._get_squared_bp(backproped)
            setattr(self, attr, sq_bp)
        else:
            sq_bp = getattr(self, attr)
            self._weight_is_called_before_bias = False
            delattr(self, attr)

        return einsum('bic->i', (sq_bp,))

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        attr = self._attr_sq_bp

        if not self._bias_is_called_before_weight:
            self._weight_is_called_before_bias = True
            sq_bp = self._get_squared_bp(backproped)
            setattr(self, attr, sq_bp)
        else:
            sq_bp = getattr(self, attr)
            self._bias_is_called_before_weight = False
            delattr(self, attr)

        return einsum('bic,bj->ij', (sq_bp, self._get_squared_inp0(module.input0)))

    def _get_squared_bp(self, backproped):
        return backproped ** 2

    def _get_squared_inp0(self, inp0):
        return inp0 ** 2


class DiagGGNFRLinear(DiagGGNLinearEfficient):

    def _get_squared_bp(self, backproped):
        attr = 'last_bp'
        last_bp = getattr(self, attr, None)
        if last_bp is None:
            setattr(self, attr, backproped)
            return backproped ** 2
        else:
            delattr(self, attr)
            return backproped.mul(last_bp)

    def _get_squared_inp0(self, inp0):
        attr = 'last_inp0'
        last_inp0 = getattr(self, attr, None)
        if last_inp0 is None:
            setattr(self, attr, inp0)
            return inp0 ** 2
        else:
            delattr(self, attr)
            return inp0.mul(last_inp0)
