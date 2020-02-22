from torch.nn import (
    AvgPool2d,
    Conv2d,
    CrossEntropyLoss,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    MSELoss,
    ReLU,
    Sigmoid,
    Tanh,
    ZeroPad2d,
)

from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.secondorder.hbp import LossHessianStrategy
from backpack.extensions import FAIL_ERROR

from . import activations, conv2d, dropout, flatten, linear, losses, padding, pooling


class DiagGGN(BackpropExtension):
    VALID_LOSS_HESSIAN_STRATEGIES = [
        LossHessianStrategy.EXACT,
        LossHessianStrategy.SAMPLING,
    ]

    def __init__(self, loss_hessian_strategy=LossHessianStrategy.EXACT, savefield=None, fail_mode=FAIL_ERROR):
        if savefield is None:
            savefield = "diag_ggn"
        if loss_hessian_strategy not in self.VALID_LOSS_HESSIAN_STRATEGIES:
            raise ValueError(
                "Unknown hessian strategy: {}".format(loss_hessian_strategy)
                + "Valid strategies: [{}]".format(self.VALID_LOSS_HESSIAN_STRATEGIES)
            )

        self.loss_hessian_strategy = loss_hessian_strategy
        super().__init__(
            savefield=savefield,
            fail_mode=fail_mode,
            module_exts={
                MSELoss: losses.DiagGGNMSELoss(),
                CrossEntropyLoss: losses.DiagGGNCrossEntropyLoss(),
                Linear: linear.DiagGGNLinear(),
                MaxPool2d: pooling.DiagGGNMaxPool2d(),
                AvgPool2d: pooling.DiagGGNAvgPool2d(),
                ZeroPad2d: padding.DiagGGNZeroPad2d(),
                Conv2d: conv2d.DiagGGNConv2d(),
                Dropout: dropout.DiagGGNDropout(),
                Flatten: flatten.DiagGGNFlatten(),
                ReLU: activations.DiagGGNReLU(),
                Sigmoid: activations.DiagGGNSigmoid(),
                Tanh: activations.DiagGGNTanh(),
            },
        )


class DiagGGNExact(DiagGGN):
    """
    Diagonal of the Generalized Gauss-Newton/Fisher. 
    Uses the exact Hessian of the loss w.r.t. the model output.

    Stores the output in :code:`diag_ggn_exact`, 
    has the same dimensions as the gradient.

    For a faster but less precise alternative, 
    see :py:meth:`backpack.extensions.DiagGGNMC`.

    """

    def __init__(self, fail_mode=FAIL_ERROR):
        super().__init__(
            loss_hessian_strategy=LossHessianStrategy.EXACT, savefield="diag_ggn_exact", fail_mode=fail_mode
        )


class DiagGGNMC(DiagGGN):
    """
    Diagonal of the Generalized Gauss-Newton/Fisher.
    Uses a Monte-Carlo approximation of
    the Hessian of the loss w.r.t. the model output.

    Stores the output in :code:`diag_ggn_mc`,
    has the same dimensions as the gradient.

    For a more precise but slower alternative,
    see :py:meth:`backpack.extensions.DiagGGNExact`.

    """

    def __init__(self, mc_samples=1, fail_mode=FAIL_ERROR):
        self._mc_samples = mc_samples
        super().__init__(
            loss_hessian_strategy=LossHessianStrategy.SAMPLING, savefield="diag_ggn_mc", fail_mode=fail_mode
        )

    def get_num_mc_samples(self):
        return self._mc_samples


class DiagGGNFR(BackpropExtension):

    VALID_LOSS_HESSIAN_STRATEGIES = [
        LossHessianStrategy.EXACT, LossHessianStrategy.SAMPLING
    ]

    def __init__(self,
                 loss_hessian_strategy=LossHessianStrategy.EXACT,
                 savefield=None, fail_mode=FAIL_ERROR):
        if savefield is None:
            savefield = "diag_ggn"
        if loss_hessian_strategy not in self.VALID_LOSS_HESSIAN_STRATEGIES:
            raise ValueError(
                "Unknown hessian strategy: {}".format(loss_hessian_strategy) +
                "Valid strategies: [{}]".format(
                    self.VALID_LOSS_HESSIAN_STRATEGIES))

        self.loss_hessian_strategy = loss_hessian_strategy
        super().__init__(savefield=savefield,
                         fail_mode=fail_mode,
                         module_exts={
                             MSELoss: losses.DiagGGNMSELoss(),
                             CrossEntropyLoss:
                                 losses.DiagGGNCrossEntropyLoss(),
                             Linear: linear.DiagGGNFRLinear(),  # for FR
                             MaxPool2d: pooling.DiagGGNMaxPool2d(),
                             AvgPool2d: pooling.DiagGGNAvgPool2d(),
                             ZeroPad2d: padding.DiagGGNZeroPad2d(),
                             Conv2d: conv2d.DiagGGNFRConv2d(),  # for FR
                             Dropout: dropout.DiagGGNDropout(),
                             Flatten: flatten.DiagGGNFlatten(),
                             ReLU: activations.DiagGGNReLU(),
                             Sigmoid: activations.DiagGGNSigmoid(),
                             Tanh: activations.DiagGGNTanh(),
                         })


class DiagGGNExactFR(DiagGGNFR):

    def __init__(self, fail_mode=FAIL_ERROR):
        super().__init__(loss_hessian_strategy=LossHessianStrategy.EXACT,
                         savefield="diag_ggn_exact_fr", fail_mode=fail_mode)

