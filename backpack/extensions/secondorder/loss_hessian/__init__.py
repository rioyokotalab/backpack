from torch.nn import CrossEntropyLoss, MSELoss

from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.secondorder.hbp import LossHessianStrategy

from . import losses


class LossHessian(BackpropExtension):
    VALID_LOSS_HESSIAN_STRATEGIES = [
        LossHessianStrategy.EXACT, LossHessianStrategy.SAMPLING
    ]

    def __init__(self,
                 loss_hessian_strategy=LossHessianStrategy.EXACT,
                 savefield=None):
        if savefield is None:
            savefield = "loss_hessian"
        if loss_hessian_strategy not in self.VALID_LOSS_HESSIAN_STRATEGIES:
            raise ValueError(
                "Unknown hessian strategy: {}".format(loss_hessian_strategy) +
                "Valid strategies: [{}]".format(
                    self.VALID_LOSS_HESSIAN_STRATEGIES))

        self.loss_hessian_strategy = loss_hessian_strategy
        super().__init__(savefield=savefield,
                         fail_mode="SILENT",
                         module_exts={
                             MSELoss: losses.MSELossHessian(),
                             CrossEntropyLoss: losses.CrossEntropyLossHessian(),
                         })

    @property
    def hessian(self):
        return getattr(self, self.savefield, None)


class LossHessianExact(LossHessian):

    def __init__(self):
        super().__init__(loss_hessian_strategy=LossHessianStrategy.EXACT,
                         savefield="loss_hessian_exact")


class LossHessianMC(LossHessian):

    def __init__(self):
        super().__init__(loss_hessian_strategy=LossHessianStrategy.SAMPLING,
                         savefield="loss_hessian_mc")
