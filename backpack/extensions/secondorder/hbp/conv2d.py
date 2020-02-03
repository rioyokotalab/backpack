import warnings

from backpack.core.derivatives.conv2d import (Conv2DConcatDerivatives,
                                              Conv2DDerivatives)
from backpack.utils import conv as convUtils
from backpack.utils.utils import einsum

from .hbp_options import BackpropStrategy, ExpectationApproximation
from .hbpbase import HBPBaseModule


class HBPConv2d(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=Conv2DDerivatives(),
                         params=["weight", "bias"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            return self._weight_for_batch_average(ext, module, backproped)

        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._weight_for_sqrt(ext, module, backproped)

    # TODO: Require tests
    def _weight_for_batch_average(self, ext, module, backproped):
        kron_factors = [self._factor_from_batch_average(module, backproped)]
        kron_factors += self._factors_from_input(ext, module)
        return kron_factors

    def _weight_for_sqrt(self, ext, module, backproped):
        kron_factors = [self._factor_from_sqrt(module, backproped)]
        kron_factors += self._factors_from_input(ext, module)
        return kron_factors

    def _factors_from_input(self, ext, module):
        X = convUtils.unfold_func(module)(module.input0)
        batch = X.size(0)

        ea_strategy = ext.get_ea_strategy()

        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            raise NotImplementedError("Undefined")
        else:
            yield einsum('bik,bjk->ij', (X, X)) / batch

    def _factor_from_sqrt(self, module, backproped):
        sqrt_ggn = backproped
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn)
        sqrt_ggn = einsum('bijc->bic', (sqrt_ggn, ))
        return einsum('bic,blc->il', (sqrt_ggn, sqrt_ggn))

    def bias(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            return self._bias_for_batch_average(module, backproped)
        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._bias_for_sqrt(module, backproped)

    def _bias_for_sqrt(self, module, backproped):
        return [self._factor_from_sqrt(module, backproped)]

    # TODO: Require tests
    def _bias_for_batch_average(self, module, backproped):
        return [self._factor_from_batch_average(module, backproped)]

    def _factor_from_batch_average(self, module, backproped):
        _, out_c, out_x, out_y = module.output.size()
        out_pixels = out_x * out_y
        # sum over spatial coordinates
        result = backproped.view(out_c, out_pixels, out_c,
                                 out_pixels).sum([1, 3])
        return result.contiguous()


class HBPConv2dConcat(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=Conv2DConcatDerivatives(),
                         params=["weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            raise NotImplementedError
        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._weight_for_sqrt(ext, module, backproped)

    def _weight_for_sqrt(self, ext, module, backproped):
        kron_factors = [self._factor_from_sqrt(module, backproped)]
        kron_factors += self._factors_from_input(ext, module)

        return kron_factors

    def _factors_from_input(self, ext, module):
        X = module.homogeneous_unfolded_input()
        batch = X.size(0)

        ea_strategy = ext.get_ea_strategy()

        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            raise NotImplementedError
        else:
            yield einsum('bik,bjk->ij', (X, X)) / batch

    def _factor_from_sqrt(self, module, backproped):
        sqrt_ggn = backproped
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn)
        sqrt_ggn = einsum('bijc->bic', (sqrt_ggn, ))
        return einsum('bic,blc->il', (sqrt_ggn, sqrt_ggn))


class HBPConv2dEfficient(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=Conv2DDerivatives(),
                         params=["weight", "bias"])

        self._weight_kron_factors = []
        self._bias_kron_factors = []
        self._weight_is_called_before_bias = False
        self._bias_is_called_before_weight = False

    def weight(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if not self._bias_is_called_before_weight:
            self._weight_is_called_before_bias = True

            if BackpropStrategy.is_batch_average(bp_strategy):
                self._weight_kron_factors = self._weight_for_batch_average(ext, module, backproped)
            elif BackpropStrategy.is_sqrt(bp_strategy):
                self._weight_kron_factors = self._weight_for_sqrt(ext, module, backproped)
        else:
            self._weight_kron_factors = self._bias_kron_factors
            self._bias_is_called_before_weight = False

        self._weight_kron_factors += self._factors_from_input(ext, module)

        return self._weight_kron_factors

    # TODO: Require tests
    def _weight_for_batch_average(self, ext, module, backproped):
        kron_factors = [self._factor_from_batch_average(module, backproped)]
        return kron_factors

    def _weight_for_sqrt(self, ext, module, backproped):
        kron_factors = [self._factor_from_sqrt(module, backproped)]
        return kron_factors

    def _factors_from_input(self, ext, module):
        X = convUtils.unfold_func(module)(module.input0)
        batch = X.size(0)

        ea_strategy = ext.get_ea_strategy()

        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            raise NotImplementedError("Undefined")
        else:
            yield einsum('bik,bjk->ij', (X, X)) / batch

    def _factor_from_sqrt(self, module, backproped):
        sqrt_ggn = backproped
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn)
        sqrt_ggn = einsum('bijc->bic', (sqrt_ggn, ))
        return einsum('bic,blc->il', (sqrt_ggn, sqrt_ggn))

    def bias(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if not self._weight_is_called_before_bias:
            self._bias_is_called_before_weight = True

            if BackpropStrategy.is_batch_average(bp_strategy):
                self._bias_kron_factors = self._bias_for_batch_average(module, backproped)
            elif BackpropStrategy.is_sqrt(bp_strategy):
                self._bias_kron_factors = self._bias_for_sqrt(module, backproped)
        else:
            self._bias_kron_factors = [self._weight_kron_factors[0]]
            self._weight_is_called_before_bias = False

        return self._bias_kron_factors

    def _bias_for_sqrt(self, module, backproped):
        return [self._factor_from_sqrt(module, backproped)]

    # TODO: Require tests
    def _bias_for_batch_average(self, module, backproped):
        return [self._factor_from_batch_average(module, backproped)]

    def _factor_from_batch_average(self, module, backproped):
        _, out_c, out_x, out_y = module.output.size()
        out_pixels = out_x * out_y
        # sum over spatial coordinates
        result = backproped.view(out_c, out_pixels, out_c,
                                 out_pixels).sum([1, 3])
        return result.contiguous()


class HBPFRConv2d(HBPConv2dEfficient):

    def _weight_for_batch_average(self, ext, module, backproped):
        raise NotImplementedError("Undefined")

    def _bias_for_batch_average(self, module, backproped):
        raise NotImplementedError("Undefined")

    def _factors_from_input(self, ext, module):
        ea_strategy = ext.get_ea_strategy()

        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            raise NotImplementedError("Undefined")
        else:
            attr = 'last_X'
            last_X = getattr(self, attr, None)
            X = convUtils.unfold_func(module)(module.input0)
            batch = X.size(0)
            if last_X is None:
                setattr(self, attr, X)
                yield einsum('bik,bjk->ij', (X, X)) / batch
            else:
                delattr(self, attr)
                yield einsum('bik,bjk->ij', (X, last_X)) / batch

    def _factor_from_sqrt(self, module, backproped):
        attr = 'last_sqrt_ggn'
        last_sqrt_ggn = getattr(self, attr, None)
        sqrt_ggn = backproped
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn)
        sqrt_ggn = einsum('bijc->bic', (sqrt_ggn, ))
        if last_sqrt_ggn is None:
            setattr(self, attr, sqrt_ggn)
            return einsum('bic,blc->il', (sqrt_ggn, sqrt_ggn))
        else:
            delattr(self, attr)
            return einsum('bic,blc->il', (sqrt_ggn, last_sqrt_ggn))


EXTENSIONS = [HBPConv2d(), HBPConv2dConcat()]
