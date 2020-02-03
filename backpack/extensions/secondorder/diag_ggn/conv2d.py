from backpack.utils import conv as convUtils
from backpack.utils.utils import einsum
from backpack.core.derivatives.conv2d import Conv2DDerivatives, Conv2DConcatDerivatives
from .diag_ggn_base import DiagGGNBaseModule


class DiagGGNConv2d(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DDerivatives(),
            params=["bias", "weight"]
        )

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, backproped)
        return einsum('bijc,bikc->i', (sqrt_ggn, sqrt_ggn))

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        X = convUtils.unfold_func(module)(module.input0)
        weight_diag = convUtils.extract_weight_diagonal(module, X, backproped)
        return weight_diag .view_as(module.weight)


class DiagGGNConv2dConcat(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DConcatDerivatives(),
            params=["weight"]
        )

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        X = convUtils.unfold_func(module)(module.input0)
        if module.has_bias:
            X = module.append_ones(X)

        weight_diag = convUtils.extract_weight_diagonal(module, X, backproped)

        return weight_diag.view_as(module.weight)


class DiagGGNConv2dEfficient(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DDerivatives(),
            params=["bias", "weight"]
        )

        self._bias_is_called_before_weight = False
        self._weight_is_called_before_bias = False
        self._attr_sq_bp = 'sq_bp'

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        attr = self._attr_sq_bp

        if not self._weight_is_called_before_bias:
            self._bias_is_called_before_weight = True
            sq_bp = self._get_squared_bp(module, backproped)
            setattr(self, attr, sq_bp)
        else:
            sq_bp = getattr(self, attr)
            self._weight_is_called_before_bias = False
            delattr(self, attr)

        return einsum('bijc->i', (sq_bp,))

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        attr = self._attr_sq_bp

        if not self._bias_is_called_before_weight:
            self._weight_is_called_before_bias = True
            sq_bp = self._get_squared_bp(module, backproped)
            setattr(self, attr, sq_bp)
        else:
            sq_bp = getattr(self, attr)
            self._bias_is_called_before_weight = False
            delattr(self, attr)

        X = convUtils.unfold_func(module)(module.input0)
        sq_X = self._get_squared_X(X)
        sq_X_bp = einsum('bkl,bmlc->cbkm', (sq_X, sq_bp))
        weight_diag = sq_X_bp.sum([0, 1]).transpose(0, 1)

        return weight_diag.view_as(module.weight)

    def _get_squared_bp(self, module, backproped):
        bp_viewed = convUtils.separate_channels_and_pixels(module, backproped)
        return bp_viewed ** 2

    def _get_squared_X(self, X):
        return X ** 2


class DiagGGNFRConv2d(DiagGGNConv2dEfficient):

    def _get_squared_bp(self, module, backproped):
        attr = 'last_bp_viewed'
        last_bp_viewed = getattr(self, attr, None)
        bp_viewed = convUtils.separate_channels_and_pixels(module, backproped)
        if last_bp_viewed is None:
            setattr(self, attr, bp_viewed)
            return bp_viewed ** 2
        else:
            delattr(self, attr)
            return bp_viewed * last_bp_viewed

    def _get_squared_X(self, X):
        attr = 'last_X'
        last_X = getattr(self, attr, None)
        if last_X is None:
            setattr(self, attr, X)
            return X ** 2
        else:
            delattr(self, attr)
            return X * last_X
