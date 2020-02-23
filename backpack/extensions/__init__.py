"""
BackPACK Extensions
"""

from .backprop_extension import FAIL_ERROR, FAIL_WARN, FAIL_SILENT
from .curvmatprod import CMP
from .firstorder import BatchGrad, BatchL2Grad, SumGradSquared, EmpKFAC, Variance, Jacobian
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
    "EmpKFAC",
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
    "FAIL_ERROR",
    "FAIL_WARN",
    "FAIL_SILENT",
]
