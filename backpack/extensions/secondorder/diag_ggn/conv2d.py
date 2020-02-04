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
        return weight_diag.view_as(module.weight)


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


class DiagGGNFRConv2d(DiagGGNConv2d):

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        attr = 'last_sqrt_ggn'
        last_sqrt_ggn = getattr(module, attr, None)
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, backproped)
        if last_sqrt_ggn is None:
            setattr(module, attr, sqrt_ggn)
            return einsum('bijc,bikc->i', (sqrt_ggn, sqrt_ggn))
        else:
            delattr(module, attr)
            return einsum('bijc,bikc->i', (sqrt_ggn, last_sqrt_ggn))

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        attr_bp = 'last_backproped'
        attr_X = 'last_X'
        last_bp = getattr(module, attr_bp, None)
        X = convUtils.unfold_func(module)(module.input0)
        if last_bp is None:
            setattr(module, attr_bp, backproped)
            setattr(module, attr_X, X)
            weight_diag = convUtils.extract_weight_diagonal(module, X, backproped)
            return weight_diag.view_as(module.weight)
        else:
            delattr(module, attr_bp)
            delattr(module, attr_X)
            last_X = getattr(module, attr_X, None)
            weight_diag = convUtils.extract_weight_diagonal(module, X, backproped, last_X, last_bp)
            return weight_diag.view_as(module.weight)
