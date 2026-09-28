import numpy as np
import pandas as pd

from sprint_forecast.baseline import team_mean_samples, velocity_bootstrap

T0 = pd.Timestamp("2024-01-08T05:00:00Z")
DAY = pd.Timedelta(days=1)


def _sprints(team_key, project, done_points, pct, first=0):
    rows = []
    for k, (dp, pc) in enumerate(zip(done_points, pct), start=first):
        start = T0 + 14 * k * DAY
        rows.append({
            "sprint_id": f"{team_key}|S{k}", "team_key": team_key, "project": project,
            "start": start, "end": start + 14 * DAY - pd.Timedelta(milliseconds=1),
            "done_points": float(dp), "pct_done": float(pc),
        })
    return pd.DataFrame(rows, columns=["sprint_id", "team_key", "project", "start", "end", "done_points", "pct_done"])


def _target(k, team_key="Alpha/Team Red", project="Alpha", committed=40.0):
    return pd.Series({"team_key": team_key, "project": project, "start": T0 + 14 * k * DAY, "committed_points": committed})


def test_draws_from_last_eight_team_velocities_capped_at_one():
    hist = _sprints("Alpha/Team Red", "Alpha", [1, 2, 10, 20, 30, 40, 50, 60, 70, 80], [0.5] * 10)
    samples = velocity_bootstrap(hist, _target(10), n_draws=5000, seed=0)
    assert samples.shape == (5000,)
    assert set(np.round(samples, 6)) == {0.25, 0.5, 0.75, 1.0}
    assert np.isclose(samples, 0.25).any() and np.isclose(samples, 1.0).mean() > 0.4


def test_ignores_sprints_not_ended_before_start():
    hist = _sprints("Alpha/Team Red", "Alpha", [10, 10, 10, 999, 999], [0.5] * 5)
    samples = velocity_bootstrap(hist, _target(3), n_draws=1000, seed=0)
    assert np.allclose(samples, 0.25)


def test_falls_back_to_project_completion_with_few_team_sprints():
    red = _sprints("Alpha/Team Red", "Alpha", [10, 10], [0.9, 0.9], first=23)
    blue = _sprints("Alpha/Team Blue", "Alpha", [5] * 25, [0.1] * 5 + [0.6] * 20)
    other = _sprints("Beta/Team Green", "Beta", [5] * 25, [0.0] * 25)
    hist = pd.concat([red, blue, other], ignore_index=True)
    samples = velocity_bootstrap(hist, _target(25), n_draws=2000, seed=0)
    assert set(np.round(samples, 6)) <= {0.6, 0.9}  # last 20 Alpha sprints only; Blue's early 0.1 are too old
    assert (samples == 0.9).any()


def test_no_history_returns_none_and_is_deterministic():
    empty = _sprints("Alpha/Team Red", "Alpha", [], [])
    assert velocity_bootstrap(empty, _target(0), seed=0) is None
    hist = _sprints("Alpha/Team Red", "Alpha", [10, 20, 30, 40], [0.5] * 4)
    assert np.array_equal(velocity_bootstrap(hist, _target(4), seed=7), velocity_bootstrap(hist, _target(4), seed=7))


def test_team_mean_is_a_point_mass():
    assert team_mean_samples(0.72).tolist() == [0.72]
    assert team_mean_samples(float("nan")) is None
