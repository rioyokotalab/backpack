"""
BackPACK Extensions
"""

from .curvmatprod import CMP
from .firstorder import BatchGrad, BatchL2Grad, SumGradSquared, Variance, Jacobian
from .secondorder import (HBP, KFAC, KFLR, KFRA,
                          KFACEfficient, KFLREfficient,
                          KFACFR, KFLRFR,
                          DiagGGN, DiagGGNExact, DiagGGNMC,
                          DiagGGNFR, DiagGGNExactFR, DiagHessian,
                          LossHessian, LossHessianExact, LossHessianMC)
