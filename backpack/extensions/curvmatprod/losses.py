from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from backpack.core.derivatives.mseloss import MSELossDerivatives
from backpack.extensions.curvature import Curvature
from backpack.extensions.curvmatprod.cmpbase import CMPBase


class CMPLoss(CMPBase):
    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        Curvature.check_loss_hessian(
            self.derivatives.hessian_is_psd(), curv_type=ext.get_curv_type()
        )

        CMP = self.derivatives.make_hessian_mat_prod(module, g_inp, g_out)
        return CMP


class CMPMSELoss(CMPLoss):
    def __init__(self):
        super().__init__(derivatives=MSELossDerivatives())


class CMPCrossEntropyLoss(CMPLoss):
    def __init__(self):
        super().__init__(derivatives=CrossEntropyLossDerivatives())
