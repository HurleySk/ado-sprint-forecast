import numpy as np
import pytest
from scipy.special import expit

from sprint_forecast.rollup import (
    _gauss_hermite,
    fit_forecaster,
    fit_sigma,
    forecast_sprint,
    simulate_pct_done,
    sprint_log_likelihood,
    summarize,
)

P = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.6])
W = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 3.0])


def test_sigma_zero_matches_independent_poisson_binomial():
    samples = simulate_pct_done(P, W, sigma=0.0, n_draws=10_000, seed=0)
    s = summarize(samples)
    assert s["expected"] == pytest.approx(float((P * W).sum() / W.sum()), abs=0.01)
    assert s["p_full"] == pytest.approx(float(np.prod(P)), abs=0.01)
    assert samples.min() >= 0.0 and samples.max() <= 1.0


def test_shock_widens_the_band():
    narrow = summarize(simulate_pct_done(P, W, sigma=0.0, seed=0))
    wide = summarize(simulate_pct_done(P, W, sigma=1.5, seed=0))
    assert wide["p90"] - wide["p10"] > narrow["p90"] - narrow["p10"]


def test_simulation_is_deterministic_and_handles_empty():
    a = simulate_pct_done(P, W, sigma=0.7, n_draws=500, seed=3)
    assert np.array_equal(a, simulate_pct_done(P, W, sigma=0.7, n_draws=500, seed=3))
    assert np.isnan(simulate_pct_done(np.array([]), np.array([]), 0.5, n_draws=10)).all()


def _planted(sigma, n_sprints=400, n_items=15, seed=0):
    """Outcomes with a planted sprint shock; p is each item's marginal (calibrated) done probability."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.8, 1.0, size=(n_sprints, n_items))
    z = rng.normal(0.0, sigma, size=(n_sprints, 1))
    y = (rng.random((n_sprints, n_items)) < expit(base + z)).astype(float)
    groups = np.repeat(np.arange(n_sprints), n_items)
    x, w = _gauss_hermite()
    return expit(base.ravel()[:, None] + sigma * x[None, :]) @ w, y.ravel(), groups


def test_sigma_fit_recovers_planted_sigma():
    p, y, groups = _planted(0.8, seed=0)
    assert fit_sigma(p, y, groups) == pytest.approx(0.8, abs=0.15)  # seed 0 gives 0.83; seeds 0-9 give 0.76-0.89


def test_sigma_fit_near_zero_without_shock():
    p, y, groups = _planted(0.0, seed=0)
    assert fit_sigma(p, y, groups) < 0.35  # seed 0 gives 0.26; seeds 0-9 give 0.0-0.26


def test_log_likelihood_at_zero_is_independent_bernoulli():
    p, y, groups = _planted(0.0, n_sprints=5, n_items=4, seed=1)
    direct = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    assert sprint_log_likelihood(0.0, p, y, groups) == pytest.approx(direct, rel=1e-9)


def test_fit_sigma_degenerate_inputs():
    assert fit_sigma(np.array([]), np.array([]), np.array([])) == 0.0
    assert fit_sigma(np.array([0.5, 0.5]), np.array([1.0, 0.0]), np.array(["a", "a"])) == 0.0


def test_forecaster_on_synth(synth40):
    frame = synth40.ckpt
    prog = synth40.progress.set_index(["sprint_id", "checkpoint"])
    last = frame["start"].max()
    fc = fit_forecaster(frame[frame["end"] < last], seed=0)
    assert 0.0 <= fc.sigma <= 3.0
    target = frame[(frame["start"] == last) & (frame["checkpoint"] == 0.5)]
    rows = target[target["sprint_id"] == target["sprint_id"].iloc[0]]
    done, total = prog.loc[(rows["sprint_id"].iloc[0], 0.5), ["done_points", "committed_points"]]
    p, samples = forecast_sprint(fc, rows, n_draws=2000, seed=0, done_points=done, total_points=total)
    s = summarize(samples)
    assert done / total <= s["p10"] <= s["p50"] <= s["p90"] <= 1.0
    assert 0.0 <= s["p_full"] <= s["p_80"] <= 1.0
    assert len(p) == len(rows)


def test_shock_preserves_each_items_calibrated_probability():
    samples = simulate_pct_done(np.full(20, 0.85), np.ones(20), sigma=1.0, n_draws=200_000, seed=0)
    assert samples.mean() == pytest.approx(0.85, abs=0.005)
    low = simulate_pct_done(np.full(20, 0.1), np.ones(20), sigma=2.0, n_draws=200_000, seed=0)
    assert low.mean() == pytest.approx(0.1, abs=0.005)


def test_rollup_adds_the_points_already_done():
    all_done = simulate_pct_done(np.ones(3), np.array([1.0, 2.0, 3.0]), sigma=0.5, n_draws=2000, seed=0,
                                 done_points=4.0, total_points=10.0)
    assert all_done.mean() == pytest.approx(1.0, abs=1e-3)
    nothing_open = simulate_pct_done(np.array([]), np.array([]), 0.5, n_draws=10, done_points=3.0, total_points=8.0)
    assert np.all(nothing_open == 3 / 8)
    assert np.isnan(simulate_pct_done(np.array([0.5]), np.array([1.0]), 0.5, n_draws=10, total_points=0.0)).all()
