r"""Library of Caputo fractional-order models.

Every class below is a :class:`~fracid.models.base.FractionalModel`: it carries its
parameters, initial conditions, the commensurate order :math:`\alpha`, a
right-hand-side function and a LaTeX renderer.  All systems have the form

.. math:: {}^C D^\alpha \mathbf{x} = \mathbf{f}(\mathbf{x}[,\mathbf{x}(t-\tau)];\theta).

========================  =====================================================
Model                     Equations
========================  =====================================================
``FractionalLinear``      :math:`D^\alpha x = A x` (2-D)
``FractionalLorenz``      :math:`\sigma(y-x),\ x(\rho-z)-y,\ xy-\beta z`
``FractionalChen``        :math:`a(y-x),\ (c-a)x-xz+cy,\ xy-bz`
``FractionalChua``        :math:`a(y-x-g(x)),\ x-y+z,\ -by` with piecewise-linear :math:`g`
``FractionalHopfield``    :math:`-Cx + A\tanh(x) + B\tanh(x(t-\tau)) + I` (2 neurons)
``FractionalSIR``         :math:`-\beta SI/N,\ \beta SI/N-\gamma I,\ \gamma I`
========================  =====================================================

.. note::
   Fractional chaos is order-dependent: e.g. the Lorenz system is chaotic only for
   :math:`\alpha\gtrsim 0.99`; the Chen system for :math:`\alpha\gtrsim 0.82`.
   References: Li & Peng (2004) *Chaos Solitons Fractals* 22; Hartley et al. (1995)
   *IEEE TCAS-I* 42; Lu (2002) *Phys. Lett. A* 298 (delayed Hopfield).
"""
from __future__ import annotations

from .base import FractionalModel, fmt, stack_last


def _sum(items: list[tuple[float, str]], digits: int) -> str:
    """Join ``(coefficient, symbol)`` pairs into a LaTeX sum with correct signs."""
    out = ""
    for v, body in items:
        if abs(v) < 1e-12:
            continue
        mag = fmt(abs(v), digits)
        core = mag if body == "1" else (body if abs(abs(v) - 1) < 1e-12 else f"{mag}\\,{body}")
        out += (("-" if v < 0 else "") + core) if not out else (f" {'-' if v < 0 else '+'} {core}")
    return out or "0"


# =============================================================================
class FractionalLinear(FractionalModel):
    r"""Fractional linear system :math:`{}^C D^\alpha \mathbf{x} = A\mathbf{x}`, :math:`A\in\mathbb{R}^{2\times 2}`.

    Exact solution :math:`\mathbf{x}(t)=E_{\alpha,1}(At^\alpha)\mathbf{x}_0` (Mittag-Leffler).
    """

    name = "Fractional linear (2-D)"
    state_names = ("x", "y")
    param_names = ("a11", "a12", "a21", "a22")
    default_params = dict(a11=-1.0, a12=2.0, a21=-2.0, a22=-1.0)
    default_x0 = (1.0, 0.5)
    default_alpha = 0.9
    default_T, default_h = 10.0, 0.02
    default_bounds = dict(a11=(-4.0, 4.0), a12=(-4.0, 4.0), a21=(-4.0, 4.0), a22=(-4.0, 4.0))
    default_free = param_names

    def _rhs_xp(self, x, xd, p, xp):
        return stack_last(xp, [p["a11"] * x[..., 0] + p["a12"] * x[..., 1],
                               p["a21"] * x[..., 0] + p["a22"] * x[..., 1]])

    def matrix(self, params=None):
        import numpy as np
        p = {**self.params, **(params or {})}
        return np.array([[p["a11"], p["a12"]], [p["a21"], p["a22"]]])

    def latex_equations(self, p, digits=4):
        return [_sum([(p["a11"], "x"), (p["a12"], "y")], digits),
                _sum([(p["a21"], "x"), (p["a22"], "y")], digits)]


# =============================================================================
class FractionalLorenz(FractionalModel):
    r"""Fractional Lorenz system.

    .. math::
        D^\alpha x = \sigma(y-x),\quad D^\alpha y = x(\rho - z) - y,\quad D^\alpha z = xy-\beta z .
    """

    name = "Fractional Lorenz"
    state_names = ("x", "y", "z")
    param_names = ("sigma", "rho", "beta")
    default_params = dict(sigma=10.0, rho=28.0, beta=8.0 / 3.0)
    default_x0 = (1.0, 1.0, 1.0)
    default_alpha = 0.99
    default_T, default_h = 3.0, 0.005
    default_bounds = dict(sigma=(5.0, 15.0), rho=(15.0, 40.0), beta=(1.0, 5.0))
    default_free = param_names

    def _rhs_xp(self, x, xd, p, xp):
        X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
        return stack_last(xp, [p["sigma"] * (Y - X), X * (p["rho"] - Z) - Y, X * Y - p["beta"] * Z])

    def latex_equations(self, p, digits=4):
        s, r, b = (fmt(p[k], digits) for k in ("sigma", "rho", "beta"))
        return [f"{s}\\,(y - x)", f"x\\,({r} - z) - y", f"xy - {b}\\,z"]


class FractionalChen(FractionalModel):
    r"""Fractional Chen system (Li & Peng, 2004).

    .. math::
        D^\alpha x = a(y-x),\quad D^\alpha y=(c-a)x - xz + cy,\quad D^\alpha z = xy - bz ,

    chaotic for :math:`(a,b,c)=(35,3,28)` and :math:`\alpha\gtrsim 0.82`.
    """

    name = "Fractional Chen"
    state_names = ("x", "y", "z")
    param_names = ("a", "b", "c")
    default_params = dict(a=35.0, b=3.0, c=28.0)
    default_x0 = (-9.0, -5.0, 14.0)
    default_alpha = 0.9
    default_T, default_h = 3.0, 0.005
    default_bounds = dict(a=(20.0, 50.0), b=(1.0, 6.0), c=(15.0, 40.0))
    default_free = param_names

    def _rhs_xp(self, x, xd, p, xp):
        X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
        a, b, c = p["a"], p["b"], p["c"]
        return stack_last(xp, [a * (Y - X), (c - a) * X - X * Z + c * Y, X * Y - b * Z])

    def latex_equations(self, p, digits=4):
        a, b, c = (fmt(p[k], digits) for k in ("a", "b", "c"))
        ca = fmt(p["c"] - p["a"], digits)
        return [f"{a}\\,(y - x)", f"({ca})\\,x - xz + {c}\\,y", f"xy - {b}\\,z"]


class FractionalChua(FractionalModel):
    r"""Fractional Chua circuit (dimensionless form, Hartley et al. 1995).

    .. math::
        D^\alpha x = a\,(y - x - g(x)),\quad D^\alpha y = x - y + z,\quad D^\alpha z = -b\,y,

    .. math::
        g(x) = m_1 x + \tfrac12 (m_0-m_1)\bigl(|x+1|-|x-1|\bigr).
    """

    name = "Fractional Chua"
    state_names = ("x", "y", "z")
    param_names = ("a", "b", "m0", "m1")
    default_params = dict(a=15.6, b=28.0, m0=-1.143, m1=-0.714)
    default_x0 = (0.2, -0.1, 0.1)
    default_alpha = 0.95
    default_T, default_h = 10.0, 0.01
    default_bounds = dict(a=(8.0, 20.0), b=(15.0, 40.0), m0=(-2.0, -0.5), m1=(-1.5, -0.2))
    default_free = param_names

    def _rhs_xp(self, x, xd, p, xp):
        X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
        g = p["m1"] * X + 0.5 * (p["m0"] - p["m1"]) * (xp.abs(X + 1.0) - xp.abs(X - 1.0))
        return stack_last(xp, [p["a"] * (Y - X - g), X - Y + Z, -p["b"] * Y])

    def latex_equations(self, p, digits=4):
        a, b, m0, m1 = (fmt(p[k], digits) for k in ("a", "b", "m0", "m1"))
        return [f"{a}\\,\\bigl(y - x - g(x)\\bigr),\\quad g(x)={m1}\\,x + \\tfrac12({fmt(p['m0'] - p['m1'], digits)})"
                f"\\bigl(|x+1|-|x-1|\\bigr)",
                "x - y + z", f"-{b}\\,y"]


# =============================================================================
class FractionalHopfield(FractionalModel):
    r"""Two-neuron fractional Hopfield-type neural network with constant delay.

    .. math::
        D^\alpha x_i = -c_i x_i + \sum_j a_{ij}\tanh x_j(t) + \sum_j b_{ij}\tanh x_j(t-\tau) + I_i .

    The history on :math:`[-\tau,0]` is the constant :math:`\mathbf{x}_0`. Default values
    are the classical delayed Hopfield example of Lu (2002) (:math:`\tau=1`).
    ``tau`` is a *structural* parameter: it is kept fixed during estimation.
    """

    name = "Fractional Hopfield (delayed)"
    state_names = ("x1", "x2")
    param_names = ("c1", "c2", "a11", "a12", "a21", "a22", "b11", "b12", "b21", "b22",
                   "I1", "I2", "tau")
    default_params = dict(c1=1.0, c2=1.0, a11=2.0, a12=-0.1, a21=-5.0, a22=3.0,
                          b11=-1.5, b12=-0.1, b21=-0.2, b22=-2.5, I1=0.0, I2=0.0, tau=1.0)
    default_x0 = (0.4, 0.6)
    default_alpha = 0.9
    default_T, default_h = 20.0, 0.025
    default_bounds = dict(c1=(0.2, 3.0), c2=(0.2, 3.0), a11=(0.0, 4.0), a12=(-1.0, 1.0),
                          a21=(-8.0, -2.0), a22=(1.0, 5.0), b11=(-3.0, 0.0), b12=(-1.0, 1.0),
                          b21=(-1.0, 1.0), b22=(-4.0, -1.0), I1=(-1.0, 1.0), I2=(-1.0, 1.0))
    default_free = ("c1", "c2", "a11", "a22")
    delayed = True

    def _rhs_xp(self, x, xd, p, xp):
        t1, t2 = xp.tanh(x[..., 0]), xp.tanh(x[..., 1])
        d1, d2 = xp.tanh(xd[..., 0]), xp.tanh(xd[..., 1])
        return stack_last(xp, [
            -p["c1"] * x[..., 0] + p["a11"] * t1 + p["a12"] * t2 + p["b11"] * d1 + p["b12"] * d2 + p["I1"],
            -p["c2"] * x[..., 1] + p["a21"] * t1 + p["a22"] * t2 + p["b21"] * d1 + p["b22"] * d2 + p["I2"]])

    def latex_equations(self, p, digits=4):
        out = []
        for i in (1, 2):
            items = [(-p[f"c{i}"], f"x_{i}"),
                     (p[f"a{i}1"], r"\tanh x_1"), (p[f"a{i}2"], r"\tanh x_2"),
                     (p[f"b{i}1"], r"\tanh x_1(t-\tau)"), (p[f"b{i}2"], r"\tanh x_2(t-\tau)"),
                     (p[f"I{i}"], "1")]
            out.append(_sum(items, digits))
        out[-1] += rf",\quad \tau={fmt(p['tau'], digits)}"
        return out


# =============================================================================
class FractionalSIR(FractionalModel):
    r"""Fractional SIR epidemic model with memory (total population :math:`N` fixed).

    .. math::
        D^\alpha S = -\beta \frac{SI}{N},\quad
        D^\alpha I = \beta \frac{SI}{N} - \gamma I,\quad
        D^\alpha R = \gamma I,

    with :math:`S+I+R=N` conserved. The basic reproduction number is
    :math:`R_0=\beta/\gamma`. ``N`` is structural (fixed during estimation).
    """

    name = "Fractional SIR"
    state_names = ("S", "I", "R")
    param_names = ("beta", "gamma", "N")
    default_params = dict(beta=0.5, gamma=0.2, N=1000.0)
    default_x0 = (990.0, 10.0, 0.0)
    default_alpha = 0.85
    default_T, default_h = 60.0, 0.1
    default_bounds = dict(beta=(0.05, 2.0), gamma=(0.02, 1.0))
    default_free = ("beta", "gamma")

    def _rhs_xp(self, x, xd, p, xp):
        S, I = x[..., 0], x[..., 1]
        inf = p["beta"] * S * I / p["N"]
        rec = p["gamma"] * I
        return stack_last(xp, [-inf, inf - rec, rec])

    def latex_equations(self, p, digits=4):
        b, g, N = (fmt(p[k], digits) for k in ("beta", "gamma", "N"))
        return [rf"-{b}\,\frac{{S I}}{{{N}}}", rf"{b}\,\frac{{S I}}{{{N}}} - {g}\,I", f"{g}\\,I"]


# =============================================================================
MODELS = {cls.name: cls for cls in (FractionalLinear, FractionalLorenz, FractionalChen,
                                    FractionalChua, FractionalHopfield, FractionalSIR)}
_ALIASES = {"linear": FractionalLinear, "lorenz": FractionalLorenz, "chen": FractionalChen,
            "chua": FractionalChua, "hopfield": FractionalHopfield, "sir": FractionalSIR}


def get_model(name: str, **kwargs) -> FractionalModel:
    """Instantiate a model by display name or short alias (``'chen'``, ``'sir'``, ...)."""
    cls = MODELS.get(name) or _ALIASES.get(name.lower())
    if cls is None:
        raise KeyError(f"unknown model {name!r}; options: {list(MODELS) + list(_ALIASES)}")
    return cls(**kwargs)
