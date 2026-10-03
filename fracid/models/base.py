r"""Base class for Caputo fractional-order models.

A model represents the commensurate-order system

.. math::
    {}^C D^\alpha \mathbf{x}(t) = \mathbf{f}\bigl(\mathbf{x}(t),\mathbf{x}(t-\tau);\,\theta\bigr),
    \qquad \mathbf{x}(0)=\mathbf{x}_0,\quad 0<\alpha\le 1 ,

with parameter vector :math:`\theta` (a ``dict`` of named scalars) and an optional
constant delay :math:`\tau` (stored as the parameter ``"tau"``).

The right-hand side is written **once**, in :meth:`FractionalModel._rhs_xp`, using only
operations common to NumPy and PyTorch (indexing on the last axis, arithmetic,
``xp.tanh``, ``xp.abs``).  The same code therefore drives

* the NumPy forward solvers (scalar state vectors), and
* the fPINN (batched torch tensors with trainable parameters, autograd-safe).
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from ..solvers import FDESolution, solve_fde


def stack_last(xp, items: Sequence):
    """Stack scalars/arrays along a new last axis for NumPy or PyTorch."""
    if getattr(xp, "__name__", "") == "torch":
        return xp.stack(list(items), dim=-1)
    return xp.stack(list(items), axis=-1)


def fmt(v: float, digits: int = 4) -> str:
    """Compact number formatting for LaTeX output."""
    s = f"{float(v):.{digits}g}"
    if "e" in s:
        m, e = s.split("e")
        s = f"{m}\\times 10^{{{int(e)}}}"
    return s


def term(v: float, body: str, digits: int = 4, first: bool = False) -> str:
    """Return ``' + 3.5\\,body'`` / ``' - 3.5\\,body'`` (sign handled, ``first`` drops leading '+')."""
    sign = "-" if v < 0 else "+"
    a = fmt(abs(v), digits)
    core = body if abs(abs(v) - 1.0) < 1e-12 else f"{a}\\,{body}"
    if first:
        return ("-" if v < 0 else "") + core
    return f" {sign} {core}"


class FractionalModel:
    """Abstract commensurate-order Caputo model.

    Subclasses define the class attributes below and implement :meth:`_rhs_xp` and
    :meth:`latex_equations`.

    Attributes
    ----------
    name : str
        Display name.
    state_names, param_names : tuple of str
    default_params : dict
        Nominal parameter values (used as "true" values in synthetic studies).
    default_x0 : tuple
    default_alpha : float
    default_T, default_h : float
        Suggested simulation horizon and step.
    default_bounds : dict
        Suggested search box for each parameter (used by the UI and estimators).
    """

    name: str = "model"
    state_names: tuple = ()
    param_names: tuple = ()
    default_params: dict = {}
    default_x0: tuple = ()
    default_alpha: float = 0.9
    default_T: float = 10.0
    default_h: float = 0.01
    default_bounds: dict = {}
    default_free: tuple = ()
    delayed: bool = False

    def __init__(self, alpha: float | None = None, params: Mapping[str, float] | None = None,
                 x0: Sequence[float] | None = None):
        self.alpha = float(self.default_alpha if alpha is None else alpha)
        self.params = dict(self.default_params)
        if params:
            unknown = set(params) - set(self.param_names)
            if unknown:
                raise KeyError(f"unknown parameters for {self.name}: {sorted(unknown)}")
            self.params.update({k: float(v) for k, v in params.items()})
        self.x0 = np.asarray(self.default_x0 if x0 is None else x0, dtype=float)
        if self.x0.size != len(self.state_names):
            raise ValueError(f"x0 must have {len(self.state_names)} entries")

    # ------------------------------------------------------------------ dimensions
    @property
    def dim(self) -> int:
        return len(self.state_names)

    # ------------------------------------------------------------------ right-hand side
    def _rhs_xp(self, x, xd, p: Mapping, xp):
        """Backend-agnostic RHS. ``x`` has shape (..., d); returns shape (..., d)."""
        raise NotImplementedError

    def rhs(self, t: float, x, params: Mapping[str, float] | None = None, xd=None) -> np.ndarray:
        r"""Numerical right-hand side :math:`\mathbf{f}(t,\mathbf{x},\mathbf{x}_\tau;\theta)` (NumPy)."""
        p = self.params if params is None else params
        return self._rhs_xp(np.asarray(x, float), None if xd is None else np.asarray(xd, float), p, np)

    # ------------------------------------------------------------------ simulation
    def solver_function(self, params: Mapping[str, float]) -> Callable:
        """Return ``f(t, x[, xd])`` with parameters bound, ready for :func:`solve_fde`."""
        p = dict(params)
        if self.delayed:
            return lambda t, x, xd: self._rhs_xp(x, xd, p, np)
        return lambda t, x: self._rhs_xp(x, None, p, np)

    def simulate(self, T: float | None = None, h: float | None = None, *,
                 n_steps: int | None = None, alpha: float | None = None,
                 params: Mapping[str, float] | None = None, x0=None,
                 method: str = "abm", memory_steps: int | None = None) -> FDESolution:
        r"""Integrate the model on :math:`[0,T]` (defaults come from the instance)."""
        a = self.alpha if alpha is None else alpha
        p = dict(self.params)
        if params:
            p.update(params)
        x0 = self.x0 if x0 is None else np.asarray(x0, float)
        T = self.default_T if T is None else T
        if n_steps is None and h is None:
            h = self.default_h
        kw = dict(method=method, memory_steps=memory_steps)
        if self.delayed:
            kw["tau"] = p["tau"]
        return solve_fde(self.solver_function(p), a, x0, T, h=h, n_steps=n_steps, **kw)

    # ------------------------------------------------------------------ misc
    def with_(self, **overrides) -> "FractionalModel":
        """Copy with ``alpha`` / ``x0`` / parameter overrides (``params={...}``)."""
        params = {**self.params, **overrides.pop("params", {})}
        return type(self)(alpha=overrides.pop("alpha", self.alpha), params=params,
                          x0=overrides.pop("x0", self.x0))

    def latex_equations(self, params: Mapping[str, float], digits: int = 4) -> list[str]:
        """List of right-hand sides in LaTeX, one per state."""
        raise NotImplementedError

    def latex(self, alpha: float | None = None, params: Mapping[str, float] | None = None,
              digits: int = 4, include_ic: bool = True, x0=None) -> str:
        r"""LaTeX ``aligned`` block of the model with the given (e.g. estimated) values.

        Example output::

            \begin{aligned}
            {}^{C}D^{0.9}x &= 35\,(y - x) \\ ...
        """
        a = self.alpha if alpha is None else alpha
        p = {**self.params, **(params or {})}
        rhs = self.latex_equations(p, digits)
        lines = [rf"{{}}^{{C}}\!D^{{{fmt(a, digits)}}} {s} &= {r}"
                 for s, r in zip(self.state_names, rhs)]
        body = " \\\\\n".join(lines)
        if include_ic:
            ic = np.asarray(self.x0 if x0 is None else x0, float)
            ics = ",\\ ".join(f"{s}(0)={fmt(v, digits)}" for s, v in zip(self.state_names, ic))
            body += " \\\\\n" + rf"\text{{with }} {ics} &"
        return "\\begin{aligned}\n" + body + "\n\\end{aligned}"
