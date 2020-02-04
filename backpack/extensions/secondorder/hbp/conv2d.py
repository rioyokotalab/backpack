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

        self._attr = 'kron_factors_from_sqrt'

    def _set_bias_flag(self, module, value):
        attr = '_bias_is_called_before_weight'
        setattr(module, attr, value)

    def _get_bias_flag(self, module):
        attr = '_bias_is_called_before_weight'
        return getattr(module, attr, False)

    def _set_weight_flag(self, module, value):
        attr = '_weight_is_called_before_weight'
        setattr(module, attr, value)

    def _get_weight_flag(self, module):
        attr = '_weight_is_called_before_weight'
        return getattr(module, attr, False)

    def weight(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()
        attr = self._attr

        kron_factors = None

        if not self._get_bias_flag(module):
            self._set_weight_flag(module, True)

            if BackpropStrategy.is_batch_average(bp_strategy):
                kron_factors = self._weight_for_batch_average(ext, module, backproped)
            elif BackpropStrategy.is_sqrt(bp_strategy):
                kron_factors = self._weight_for_sqrt(ext, module, backproped)

            setattr(module, attr, kron_factors)
        else:
            kron_factors = getattr(module, attr)
            self._set_bias_flag(module, False)
            delattr(module, attr)

        kron_factors += self._factors_from_input(ext, module)

        return kron_factors

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
        attr = self._attr

        kron_factors = None

        if not self._get_weight_flag(module):
            self._set_bias_flag(module, True)

            if BackpropStrategy.is_batch_average(bp_strategy):
                kron_factors = self._bias_for_batch_average(module, backproped)
            elif BackpropStrategy.is_sqrt(bp_strategy):
                kron_factors = self._bias_for_sqrt(module, backproped)

            setattr(module, attr, kron_factors)
        else:
            kron_factors = getattr(module, attr)
            self._set_weight_flag(module, False)
            delattr(module, attr)

        return kron_factors

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
            last_X = getattr(module, attr, None)
            X = convUtils.unfold_func(module)(module.input0)
            batch = X.size(0)
            if last_X is None:
                print('conv2d factors input 1st')
                setattr(module, attr, X)
                yield einsum('bik,bjk->ij', (X, X)) / batch
            else:
                print('conv2d factors input 2nd')
                delattr(module, attr)
                yield einsum('bik,bjk->ij', (X, last_X)) / batch

    def _factor_from_sqrt(self, module, backproped):
        attr = 'last_sqrt_ggn'
        last_sqrt_ggn = getattr(module, attr, None)
        sqrt_ggn = backproped
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn)
        sqrt_ggn = einsum('bijc->bic', (sqrt_ggn, ))
        if last_sqrt_ggn is None:
            print('conv2d factors sqrt 1st')
            setattr(module, attr, sqrt_ggn)
            return einsum('bic,blc->il', (sqrt_ggn, sqrt_ggn))
        else:
            print('conv2d factors sqrt 2nd')
            delattr(module, attr)
            return einsum('bic,blc->il', (sqrt_ggn, last_sqrt_ggn))


EXTENSIONS = [HBPConv2d(), HBPConv2dConcat()]
