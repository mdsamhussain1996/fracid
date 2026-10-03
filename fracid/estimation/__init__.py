"""Parameter-estimation back-ends (least squares now; fPINN is added in its own module)."""
from .fpinn import fit_fpinn
from .least_squares import LS_METHODS, FitResult, fit_least_squares, make_result
from .problem import FitProblem, choose_step

__all__ = ["FitProblem", "FitResult", "fit_least_squares", "fit_fpinn", "make_result", "choose_step",
           "LS_METHODS"]
