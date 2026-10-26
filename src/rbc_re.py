"""Shared code: data loading, models and metrics for Re(Ra, Pr) in 2D Rayleigh-Benard convection.

Setup: closed square box (Gamma = 1), no-slip walls, isothermal plates, adiabatic sidewalls.
Re = sqrt(Ra/Pr) * u_rms,  u_rms = sqrt(<ux^2 + uz^2>_{A,t}).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
RNG = 0
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rbc2d_re_dataset.csv"
FEATS = ["x_Ra", "x_Pr"]


def load(path=DATA):
    """Return (full table, training table). Training = use_in_training==1 and adiabatic sidewalls."""
    df = pd.read_csv(path)
    tr = df[(df.use_in_training == 1) & (df.sidewalls == "adiabatic")].copy()
    tr["x_Ra"] = np.log10(tr.Ra)
    tr["x_Pr"] = np.log10(tr.Pr)
    tr["y"] = np.log10(tr.Re)
    return df, tr.reset_index(drop=True)


class Hybrid(BaseEstimator, RegressorMixin):
    """Power law in (Ra, Pr) + non-parametric model of the residual."""

    def __init__(self, residual_model=None):
        self.residual_model = residual_model

    def fit(self, X, y):
        self.base_ = LinearRegression().fit(X, y)
        self.res_ = clone(self.residual_model).fit(X, y - self.base_.predict(X))
        return self

    def predict(self, X):
        return self.base_.predict(X) + self.res_.predict(X)


def _gp():
    k = C(0.05, (1e-4, 1)) * RBF([1.5, 1.5], (0.3, 20)) + WhiteKernel(1e-3, (1e-6, 1e-1))
    return make_pipeline(StandardScaler(),
                         GaussianProcessRegressor(k, n_restarts_optimizer=3, random_state=RNG))


def models():
    return {
        "PowerLaw (log-linear)": LinearRegression(),
        "Poly2 + Ridge": make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=0.05)),
        "RandomForest (direct)": RandomForestRegressor(300, random_state=RNG, n_jobs=-1),
        "GradBoost (direct)": GradientBoostingRegressor(n_estimators=300, max_depth=2,
                                                        learning_rate=0.05, random_state=RNG),
        "PowerLaw + RF residual": Hybrid(RandomForestRegressor(300, min_samples_leaf=2,
                                                               random_state=RNG, n_jobs=-1)),
        "PowerLaw + GP residual": Hybrid(_gp()),
    }


def metrics(y_true_log, y_pred_log):
    """Metrics on log10(Re) (R2_log) and on Re itself (R2_Re, APE in %)."""
    t, p = 10 ** np.asarray(y_true_log), 10 ** np.asarray(y_pred_log)
    ape = np.abs(p - t) / t * 100
    r2l = r2_score(y_true_log, y_pred_log) if len(t) > 1 else np.nan
    r2r = r2_score(t, p) if len(t) > 1 else np.nan
    return dict(R2_log=r2l, R2_Re=r2r, MedAPE=np.median(ape), MeanAPE=ape.mean(), MaxAPE=ape.max())


def fit_predict(model, d, tr_idx, te_idx):
    m = clone(model).fit(d.loc[tr_idx, FEATS].values, d.loc[tr_idx, "y"].values)
    return m.predict(d.loc[te_idx, FEATS].values)
