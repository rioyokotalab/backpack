from backpack.core.derivatives.linear import LinearDerivatives, LinearConcatDerivatives
from .jacobian_base import JacobianBaseModule


class JacobianLinear(JacobianBaseModule):

    def __init__(self):
        super().__init__(
            derivatives=LinearDerivatives(),
            params=["bias", "weight"]
        )

    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):

        if backproped is None:
            n = grad_out[0].size(0)
            w = self.derivatives.get_weight_data(module)
            return w.T.repeat(n, 1, 1)

        return super().backpropagate(ext, module, grad_inp, grad_out, backproped)

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        return [backproped]

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        return [backproped, module.input0]


