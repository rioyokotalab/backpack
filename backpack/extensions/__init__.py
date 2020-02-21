"""
BackPACK Extensions
"""

from .curvmatprod import CMP
from .firstorder import BatchGrad, BatchL2Grad, SumGradSquared, Variance, Jacobian
from .secondorder import (
    HBP,
    KFAC,
    KFLR,
    KFRA,
    KFACEfficient,
    KFLREfficient,
    KFACFR,
    KFLRFR,
    DiagGGN,
    DiagGGNExact,
    DiagGGNMC,
    DiagHessian,
    DiagGGNFR, DiagGGNExactFR,
    LossHessian, LossHessianExact, LossHessianMC
)

__all__ = [
    "CMP",
    "BatchL2Grad",
    "BatchGrad",
    "SumGradSquared",
    "Variance",
    "Jacobian",
    "HBP",
    "KFAC",
    "KFLR",
    "KFRA",
    "KFACEfficient",
    "KFLREfficient",
    "KFACFR",
    "KFLRFR",
    "DiagGGN",
    "DiagGGNExact",
    "DiagGGNMC",
    "DiagHessian",
    "LossHessian",
    "LossHessianExact",
    "LossHessianMC",
]
