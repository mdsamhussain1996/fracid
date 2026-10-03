"""Unified entry point for the forward solvers."""
from __future__ import annotations

from .abm import abm_solve
from .gl import gl_solve
from ._common import FDESolution

METHODS = ("abm", "gl", "gl-implicit")


def solve_fde(f, alpha: float, x0, T: float, *, method: str = "abm", **kwargs) -> FDESolution:
    r"""Solve :math:`{}^C D^\alpha x = f(t,x)` with the chosen scheme.

    ``method`` is one of ``"abm"`` (Diethelm ABM, order min(2, 1+α)), ``"gl"``
    (explicit Grünwald–Letnikov, order 1) or ``"gl-implicit"``.
    Remaining keyword arguments (``h``/``n_steps``, ``t0``, ``tau``, ``history``,
    ``memory_steps``) are forwarded.
    """
    if method == "abm":
        return abm_solve(f, alpha, x0, T, **kwargs)
    if method == "gl":
        return gl_solve(f, alpha, x0, T, implicit=False, **kwargs)
    if method == "gl-implicit":
        return gl_solve(f, alpha, x0, T, implicit=True, **kwargs)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
