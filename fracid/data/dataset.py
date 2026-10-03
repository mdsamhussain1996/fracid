r"""Time-series container and CSV input/output.

A :class:`Dataset` holds (possibly noisy, subsampled, partially observed) measurements

.. math:: y_k(t_i) = x_{c_k}(t_i) + \varepsilon_{ik},

where :math:`c_k` are the *observed* state indices.  If the data came from the
synthetic generator the ground truth (:math:`\alpha`, :math:`\theta`, clean data)
is kept in ``meta`` so that estimation errors can be reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """Measured trajectories.

    Attributes
    ----------
    t : ndarray (M,)
        Strictly increasing sample times.
    y : ndarray (M, k)
        Observations.
    observed : tuple of int
        Index into the model state vector for each column of ``y`` (``None`` entries are
        allowed for CSV data whose columns are mapped later).
    names : tuple of str
        Column names.
    clean : ndarray (M, k), optional
        Noise-free values (synthetic data only).
    meta : dict
        Free-form provenance: ``true_alpha``, ``true_params``, ``snr_db``, ``model`` ...
    """

    t: np.ndarray
    y: np.ndarray
    observed: tuple = ()
    names: tuple = ()
    clean: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float).ravel()
        self.y = np.asarray(self.y, dtype=float)
        if self.y.ndim == 1:
            self.y = self.y[:, None]
        if self.y.shape[0] != self.t.size:
            raise ValueError("t and y must have the same number of rows")
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("time stamps must be strictly increasing")
        if not self.observed:
            self.observed = tuple(range(self.y.shape[1]))
        if not self.names:
            self.names = tuple(f"y{i}" for i in range(self.y.shape[1]))

    # ------------------------------------------------------------------ convenience
    @property
    def n(self) -> int:
        return self.t.size

    @property
    def is_synthetic(self) -> bool:
        return "true_alpha" in self.meta

    @property
    def is_uniform(self) -> bool:
        dt = np.diff(self.t)
        return bool(np.allclose(dt, dt[0], rtol=1e-6, atol=1e-12))

    def shifted(self) -> "Dataset":
        """Copy with the time axis shifted so that the first sample is at :math:`t=0`."""
        d = Dataset(self.t - self.t[0], self.y.copy(), self.observed, self.names,
                    None if self.clean is None else self.clean.copy(), dict(self.meta))
        return d

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.y, columns=list(self.names))
        df.insert(0, "t", self.t)
        return df

    def to_csv(self, path_or_buf=None) -> str | None:
        return self.to_frame().to_csv(path_or_buf, index=False)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_frame(cls, df: pd.DataFrame, time_col: str | None = None,
                   value_cols: Sequence[str] | None = None,
                   observed: Sequence[int] | None = None) -> "Dataset":
        """Build a dataset from a DataFrame (non-numeric / NaN rows are dropped).

        ``time_col`` defaults to the first column; ``value_cols`` to all other numeric columns.
        """
        time_col = time_col or df.columns[0]
        if value_cols is None:
            value_cols = [c for c in df.columns if c != time_col]
        sub = df[[time_col, *value_cols]].apply(pd.to_numeric, errors="coerce").dropna()
        sub = sub.sort_values(time_col).drop_duplicates(time_col)
        if len(sub) < 5:
            raise ValueError("need at least 5 valid rows")
        ds = cls(sub[time_col].to_numpy(), sub[list(value_cols)].to_numpy(),
                 tuple(observed) if observed is not None else (), tuple(value_cols))
        return ds

    @classmethod
    def from_csv(cls, path_or_buf, **kw) -> "Dataset":
        return cls.from_frame(pd.read_csv(path_or_buf), **kw)
