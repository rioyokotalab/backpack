from backpack.core.derivatives.linear import LinearDerivatives, LinearConcatDerivatives
from backpack.utils.utils import einsum
from .hbpbase import HBPBaseModule
from .hbp_options import BackpropStrategy, ExpectationApproximation


class HBPLinear(HBPBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=LinearDerivatives(),
            params=["weight", "bias"]
        )

    def weight(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            return self._weight_for_batch_average(ext, module, backproped)

        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._weight_for_sqrt(ext, module, backproped)

    def _weight_for_batch_average(self, ext, module, backproped):
        kron_factors = self._bias_for_batch_average(backproped)
        kron_factors += self._factors_from_input(ext, module)
        return kron_factors

    def _weight_for_sqrt(self, ext, module, backproped):
        kron_factors = self._factor_from_sqrt(backproped)
        kron_factors += self._factors_from_input(ext, module)
        return kron_factors

    def _factors_from_input(self, ext, module):
        ea_strategy = ext.get_ea_strategy()

        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            mean_input = self.__mean_input(module).unsqueeze(-1)
            return [mean_input, mean_input.transpose()]
        else:
            yield self.__mean_input_outer(module)

    def _factor_from_sqrt(self, backproped):
        return [einsum('bic,bjc->ij', (backproped, backproped))]

    def bias(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            return self._bias_for_batch_average(
                backproped
            )
        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._factor_from_sqrt(
                backproped
            )

    def _bias_for_batch_average(self, backproped):
        return [backproped]

    def __mean_input(self, module):
        _, flat_input = self.derivatives.batch_flat(module.input0)
        return flat_input.mean(0)

    def __mean_input_outer(self, module):
        N, flat_input = self.derivatives.batch_flat(module.input0)
        return einsum('bi,bj->ij', (flat_input, flat_input)) / N


class HBPLinearConcat(HBPBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=LinearConcatDerivatives(),
            params=["weight"]
        )

    def weight(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            return self._weight_for_batch_average(ext, module, backproped)
        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._weight_for_sqrt(ext, module, backproped)

    def _weight_for_batch_average(self, ext, module, backproped):
        kron_factors = self._bias_for_batch_average(backproped)
        kron_factors += self._factors_from_input(ext, module)
        return kron_factors

    def _weight_for_sqrt(self, ext, module, backproped):
        kron_factors = self._factor_from_sqrt(backproped)
        kron_factors += self._factors_from_input(ext, module)
        return kron_factors

    def _factors_from_input(self, ext, module):
        ea_strategy = ext.get_ea_strategy()

        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            mean_input = self.__mean_input(module).unsqueeze(-1)
            return [mean_input, mean_input.transpose()]
        else:
            return [self.__mean_input_outer(module)]

    def _factor_from_sqrt(self, backproped):
        return [einsum('bic,bjc->ij', (backproped, backproped))]

    def _bias_for_batch_average(self, backproped):
        return [backproped]

    def __mean_input(self, module):
        _, flat_input = self.derivatives.batch_flat(module.homogeneous_input())
        return flat_input.mean(0)

    def __mean_input_outer(self, module):
        N, flat_input = self.derivatives.batch_flat(module.homogeneous_input())
        return einsum('bi,bj->ij', (flat_input, flat_input)) / N


class HBPLinearEfficient(HBPBaseModule):

    def __init__(self):
        super().__init__(
            derivatives=LinearDerivatives(),
            params=["weight", "bias"]
        )

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

    def _weight_for_batch_average(self, ext, module, backproped):
        kron_factors = self._bias_for_batch_average(backproped)
        return kron_factors

    def _weight_for_sqrt(self, ext, module, backproped):
        kron_factors = self._factor_from_sqrt(backproped, module)
        return kron_factors

    def _factors_from_input(self, ext, module):
        ea_strategy = ext.get_ea_strategy()

        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            mean_input = self.__mean_input(module).unsqueeze(-1)
            return [mean_input, mean_input.transpose()]
        else:
            yield self.__mean_input_outer(module)

    def _factor_from_sqrt(self, backproped, module):
        return [einsum('bic,bjc->ij', (backproped, backproped))]

    def bias(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()
        attr = self._attr

        kron_factors = None

        if not self._get_weight_flag(module):
            self._set_bias_flag(module, True)

            if BackpropStrategy.is_batch_average(bp_strategy):
                kron_factors = self._bias_for_batch_average(backproped)
            elif BackpropStrategy.is_sqrt(bp_strategy):
                kron_factors = self._factor_from_sqrt(backproped, module)

            setattr(module, attr, kron_factors)
        else:
            kron_factors = getattr(module, attr)
            self._set_weight_flag(module, False)
            delattr(module, attr)

        return kron_factors

    def _bias_for_batch_average(self, backproped):
        return [backproped]

    def __mean_input(self, module):
        _, flat_input = self.derivatives.batch_flat(module.input0)
        return flat_input.mean(0)

    def __mean_input_outer(self, module):
        N, flat_input = self.derivatives.batch_flat(module.input0)
        return einsum('bi,bj->ij', (flat_input, flat_input)) / N


class HBPFRLinear(HBPLinearEfficient):

    def _weight_for_batch_average(self, ext, module, backproped):
        raise NotImplementedError("Undefined")

    def _bias_for_batch_average(self, backproped):
        raise NotImplementedError("Undefined")

    def _factors_from_input(self, ext, module):
        ea_strategy = ext.get_ea_strategy()

        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            raise NotImplementedError("Undefined")
        else:
            yield self.__mean_input_outer(module)

    def __mean_input_outer(self, module):
        attr = 'last_flat_input'
        last_fl_inp = getattr(module, attr, None)
        N, flat_input = self.derivatives.batch_flat(module.input0)
        if last_fl_inp is None:
            setattr(module, attr, flat_input)
            return einsum('bi,bj->ij', (flat_input, flat_input)) / N
        else:
            delattr(module, attr)
            return einsum('bi,bj->ij', (flat_input, last_fl_inp)) / N

    def _factor_from_sqrt(self, backproped, module):
        attr = 'last_backproped'
        last_bp = getattr(module, attr, None)
        if last_bp is None:
            setattr(module, attr, backproped)
            return [einsum('bic,bjc->ij', (backproped, backproped))]
        else:
            delattr(module, attr)
            return [einsum('bic,bjc->ij', (backproped, last_bp))]

