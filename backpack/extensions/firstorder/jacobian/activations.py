from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.tanh import TanhDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from .jacobian_base import JacobianBaseModule


class JacobianReLU(JacobianBaseModule):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())


class JacobianSigmoid(JacobianBaseModule):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivatives())


class JacobianTanh(JacobianBaseModule):
    def __init__(self):
        super().__init__(derivatives=TanhDerivatives())
