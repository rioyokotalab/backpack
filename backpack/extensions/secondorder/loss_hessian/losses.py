from backpack.core.derivatives.mseloss import MSELossDerivatives
from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from backpack.extensions.secondorder.hbp import LossHessianStrategy

from .loss_hessian_base import LossHessianBaseModule


class LossHessian(LossHessianBaseModule):

    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):

        if ext.loss_hessian_strategy == LossHessianStrategy.EXACT:
            hess_func = self.derivatives.sqrt_hessian
        elif ext.loss_hessian_strategy == LossHessianStrategy.SAMPLING:
            hess_func = self.derivatives.sqrt_hessian_sampled
        else:
            raise ValueError(
                "Unknown hessian strategy {}".format(ext.loss_hessian_strategy)
            )

        hess = hess_func(module, grad_inp, grad_out)
        setattr(ext, ext.savefield, hess)  # save hessian to the BackpropExtension

        return None


class MSELossHessian(LossHessian):
    def __init__(self):
        super().__init__(derivatives=MSELossDerivatives())


class CrossEntropyLossHessian(LossHessian):
    def __init__(self):
        super().__init__(derivatives=CrossEntropyLossDerivatives())
