"""Sprint-shock rollup: fit sigma by Gauss-Hermite marginal likelihood, then Monte Carlo %done."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp

from sprint_forecast.model import P_CLIP, ItemModel, predict_proba, train_item_model

GH_NODES = 32
SIGMA_MAX = 3.0
N_DRAWS = 10_000
FULL_TOL = 1e-9


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), P_CLIP, 1 - P_CLIP)
    return np.log(p / (1 - p))


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def _gauss_hermite() -> tuple[np.ndarray, np.ndarray]:
    """Nodes x and weights w with sum_k w_k f(x_k) ~= E[f(Z)] for Z ~ N(0, 1)."""
    nodes, weights = np.polynomial.hermite.hermgauss(GH_NODES)
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


def shifted_logit(p: np.ndarray, sigma: float, max_iter: int = 50) -> np.ndarray:
    """Per-item logit eta with E[sigmoid(eta + z)] = p for z ~ N(0, sigma^2): the shared sprint shock spreads
    outcomes without moving any item's calibrated probability (plain logit(p) + z pulls it toward 0.5)."""
    target = np.clip(np.asarray(p, dtype=float), P_CLIP, 1 - P_CLIP)
    eta = _logit(target)
    if sigma <= 0 or len(eta) == 0:
        return eta
    x, w = _gauss_hermite()
    z = sigma * x
    eta = eta * np.sqrt(1.0 + np.pi * sigma**2 / 8.0)  # logistic-normal approximation as the start
    for _ in range(max_iter):
        s = expit(eta[:, None] + z[None, :])
        step = (s @ w - target) / np.maximum((s * (1.0 - s)) @ w, 1e-12)
        eta = eta - np.clip(step, -2.0, 2.0)
        if np.max(np.abs(step)) < 1e-10:
            break
    return eta


def sprint_log_likelihood(sigma: float, p: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """sum over sprints of log  integral prod_i Bern(y_i | sigmoid(eta_i + z)) N(z; 0, sigma^2) dz,
    with eta_i = shifted_logit(p_i, sigma) so each item's marginal probability stays p_i."""
    x, w = _gauss_hermite()
    z = sigma * x
    eta = shifted_logit(p, sigma)[:, None] + z[None, :]
    y = np.asarray(y, dtype=float)[:, None]
    item_ll = y * _log_sigmoid(eta) + (1.0 - y) * _log_sigmoid(-eta)
    codes, uniques = pd.factorize(pd.Series(groups))
    per_sprint = np.zeros((len(uniques), GH_NODES))
    np.add.at(per_sprint, codes, item_ll)
    log_w = np.log(w)
    return float(logsumexp(per_sprint + log_w[None, :], axis=1).sum())


def fit_sigma(p: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Maximum-likelihood sprint shock sigma in [0, 3]; 0 when there are fewer than two sprints."""
    if len(p) == 0 or pd.Series(groups).nunique() < 2:
        return 0.0
    res = minimize_scalar(
        lambda s: -sprint_log_likelihood(s, p, y, groups), bounds=(0.0, SIGMA_MAX), method="bounded",
    )
    best = float(res.x)
    if sprint_log_likelihood(0.0, p, y, groups) >= sprint_log_likelihood(best, p, y, groups):
        return 0.0
    return best


def simulate_pct_done(
    p: np.ndarray, points: np.ndarray, sigma: float, n_draws: int = N_DRAWS, seed: int = 0,
) -> np.ndarray:
    """Samples of sum(points * done) / sum(points), with one shared shock z ~ N(0, sigma^2) per draw;
    each item's mean done rate stays p (see shifted_logit)."""
    p = np.asarray(p, dtype=float)
    w = np.asarray(points, dtype=float)
    if len(p) == 0 or w.sum() <= 0:
        return np.full(n_draws, np.nan)
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, sigma, size=(n_draws, 1)) if sigma > 0 else np.zeros((n_draws, 1))
    prob = expit(shifted_logit(p, sigma)[None, :] + z)
    done = rng.random((n_draws, len(p))) < prob
    return done @ w / w.sum()


def summarize(samples: np.ndarray) -> dict[str, float]:
    s = np.asarray(samples, dtype=float)
    q10, q50, q90 = np.quantile(s, [0.1, 0.5, 0.9])
    return {
        "p_full": float(np.mean(s >= 1.0 - FULL_TOL)),
        "p_80": float(np.mean(s >= 0.8 - FULL_TOL)),
        "expected": float(np.mean(s)),
        "p10": float(q10),
        "p50": float(q50),
        "p90": float(q90),
    }


@dataclass
class Forecaster:
    item_model: ItemModel
    sigma: float


def fit_forecaster(frame: pd.DataFrame, seed: int = 0) -> Forecaster:
    """Train model A and fit sigma on its calibrated predictions for the calibration sprints."""
    model = train_item_model(frame, seed=seed)
    cf = model.calib_frame
    sigma = fit_sigma(cf["p"].to_numpy(), cf["y"].to_numpy(), cf["sprint_id"].to_numpy()) if len(cf) else 0.0
    return Forecaster(model, sigma)


def forecast_sprint(
    fc: Forecaster, items: pd.DataFrame, n_draws: int = N_DRAWS, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Item probabilities and %done samples for one sprint's feature rows."""
    p = predict_proba(fc.item_model, items)
    return p, simulate_pct_done(p, items["points"].to_numpy(float), fc.sigma, n_draws, seed)
