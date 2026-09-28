"""Model C (velocity bootstrap) and the team-mean reference. Both return samples of %done, or None."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

VELOCITY_DRAWS_FROM = 8
MIN_TEAM_SPRINTS = 3
PROJECT_WINDOW = 20


def velocity_bootstrap(
    history_sprints: pd.DataFrame,
    target: pd.Series,
    n_draws: int = 10_000,
    seed: int = 0,
) -> np.ndarray | None:
    """%done = min(1, V / committed) with V drawn from the team's last 8 delivered velocities (end < start).
    With fewer than 3 prior team sprints, draw %done from the project's last 20 sprints instead.
    `target` needs team_key, project, start, committed_points. Returns None when there is no history."""
    prior = history_sprints[(history_sprints["end"] < target["start"]) & history_sprints["done_points"].notna()]
    prior = prior.sort_values(["end", "sprint_id"], kind="mergesort")
    rng = np.random.default_rng(seed)
    team = prior[prior["team_key"] == target["team_key"]].tail(VELOCITY_DRAWS_FROM)
    committed = float(target["committed_points"])
    if len(team) >= MIN_TEAM_SPRINTS and committed > 0:
        v = rng.choice(team["done_points"].to_numpy(float), size=n_draws, replace=True)
        return np.minimum(1.0, v / committed)
    ratios = prior[prior["project"] == target["project"]].tail(PROJECT_WINDOW)["pct_done"].dropna()
    if ratios.empty:
        return None
    return rng.choice(ratios.to_numpy(float), size=n_draws, replace=True)


def team_mean_samples(team_trailing_completion: float) -> np.ndarray | None:
    """Reference forecast: a point mass at the team's trailing mean %done (None without history)."""
    if team_trailing_completion is None or math.isnan(team_trailing_completion):
        return None
    return np.array([float(team_trailing_completion)])
