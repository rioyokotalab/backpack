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

    def _weight_for_batch_average(self, ext, module, backproped):
        kron_factors = self._bias_for_batch_average(backproped)
        return kron_factors

    def _weight_for_sqrt(self, ext, module, backproped):
        kron_factors = self._factor_from_sqrt(backproped)
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

        if not self._weight_is_called_before_bias:
            self._bias_is_called_before_weight = True

            if BackpropStrategy.is_batch_average(bp_strategy):
                self._bias_kron_factors = self._bias_for_batch_average(backproped)
            elif BackpropStrategy.is_sqrt(bp_strategy):
                self._bias_kron_factors = self._factor_from_sqrt(backproped)
        else:
            self._bias_kron_factors = [self._weight_kron_factors[0]]
            self._weight_is_called_before_bias = False

        return self._bias_kron_factors

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
