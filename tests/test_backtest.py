import numpy as np
import pandas as pd
import pytest

from sprint_forecast.backtest import (
    METRICS,
    LeakageError,
    calibration_table,
    check_no_leakage,
    crps_samples,
    format_report,
    pinball,
    run_backtest,
    select_targets,
)


def test_crps_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    x = rng.random(200)
    y = 0.4
    pairwise = np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))
    assert crps_samples(x, y) == pytest.approx(pairwise, rel=1e-12)
    assert crps_samples(np.array([0.7]), 0.4) == pytest.approx(0.3)


def test_pinball_loss():
    assert pinball(0.5, 0.8, 0.9) == pytest.approx(0.27)
    assert pinball(0.5, 0.2, 0.9) == pytest.approx(0.03)


def test_leakage_check_raises_on_overlap():
    start = pd.Timestamp("2024-03-04T05:00:00Z")
    ok = pd.DataFrame({"sprint_id": ["s0"], "end": [start - pd.Timedelta(milliseconds=1)]})
    check_no_leakage(ok, start)
    bad = pd.DataFrame({"sprint_id": ["s0", "s1"], "end": [start - pd.Timedelta(days=1), start]})
    with pytest.raises(LeakageError, match="s1"):
        check_no_leakage(bad, start)


def test_select_targets_needs_min_history(synth16):
    targets = select_targets(synth16.sprints.sprints, min_history=8)
    assert set(targets["team"]) == {"Team Red", "Team Blue", "Team Green"}
    assert targets.groupby("team").size().max() <= 8
    assert select_targets(synth16.sprints.sprints, min_history=8, team="Team Red")["team"].eq("Team Red").all()


def test_calibration_table_bins():
    table = calibration_table(np.array([0, 1, 1, 0]), np.array([0.05, 0.95, 1.0, 0.12]))
    assert list(table["range"]) == ["0.0-0.1", "0.1-0.2", "0.9-1.0"]
    assert list(table["n"]) == [1, 1, 2]


@pytest.fixture(scope="module")
def result(synth16):
    return run_backtest(synth16.sprints, synth16.frame, min_history=8, retrain_every=4, n_draws=2000, seed=0)


def test_backtest_has_no_leakage(result):
    a = result.sprint_rows[result.sprint_rows["model"] == "a"]
    assert len(a) > 0
    assert (a["train_end_max"] < a["start"]).all()


def test_backtest_metrics_are_finite(result):
    overall = result.summary[result.summary["team_key"] == "(all)"].set_index("model")
    assert set(overall.index) == {"a", "c", "team_mean"}
    assert np.isfinite(overall[METRICS].to_numpy(dtype=float)).all()
    assert set(result.summary["team_key"]) == {"(all)", "Alpha/Team Red", "Alpha/Team Blue", "Beta/Team Green"}
    assert np.isfinite(result.item_summary[["brier", "log_loss", "auc"]].to_numpy(dtype=float)).all()
    assert result.calibration["n"].sum() == len(result.item_rows)
    assert len(result.loto) == 3 and np.isfinite(result.loto[["crps", "mae"]].to_numpy(dtype=float)).all()
    text = format_report(result)
    assert "baseline C on CRPS" in text and "Leave-one-team-out" in text


def test_backtest_model_filter_and_team_filter(synth16):
    res = run_backtest(synth16.sprints, synth16.frame, models=("c", "team_mean"), team="Team Green", n_draws=500)
    assert set(res.sprint_rows["model"]) == {"c", "team_mean"}
    assert set(res.sprint_rows["team"]) == {"Team Green"}
    assert res.item_rows.empty and res.loto.empty


def test_backtest_with_no_targets(synth16):
    res = run_backtest(synth16.sprints, synth16.frame, min_history=99, n_draws=100)
    assert res.sprint_rows.empty
    assert "No sprints qualified" in format_report(res)
