"""Score an iteration at any time t with a trained forecaster (shared by `predict` and `export`)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sprint_forecast.cache import CacheData
from sprint_forecast.checkpoints import done_points_at, open_rows
from sprint_forecast.features import FEATURE_VERSION, build_checkpoint_features, build_features_for, team_history
from sprint_forecast.rollup import Forecaster, forecast_sprint, simulate_pct_done, summarize
from sprint_forecast.sprints import SprintData, horizon, iteration_group, scope_at, sprint_calendar
from sprint_forecast.timeline import to_utc

OLD_MODEL = "model was trained by an older version; run `sprint-forecast train`"


@dataclass
class ScoredSprint:
    sprint: pd.Series  # one summarize_sprints row of the committed scope; outcome columns empty while it runs
    items: pd.DataFrame  # open rows at t (checkpoint features) plus "p", each item's calibrated probability of done
    summary: dict[str, float]  # summarize() of the simulated share of committed points done by the sprint's end
    velocity: float  # team trailing velocity in points; NaN without history
    done_points: float  # committed points in the iteration and done at t; 0 before the cutoff
    t: pd.Timestamp  # the instant scored: the requested t, or the sprint's end once that has passed
    committed: pd.DataFrame  # the committed items (SprintData.items rows) the summary is a share of


def check_bundle(bundle: dict) -> None:
    """Raise ValueError unless `bundle` was trained on this version's features."""
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError(OLD_MODEL)


def score_iteration(
    fc: Forecaster,
    cache: CacheData,
    history: SprintData,
    *,
    iteration: str,
    t,
    work_item_types: list[str],
    done_categories: list[str],
    commit_grace_days: float,
    team: str | None = None,
) -> list[ScoredSprint]:
    """Forecast each team sprint of `iteration` as it stands at `t` (its end, once that has passed; the last
    extract, when the sprint ended after it). Committed items done by t count as done, committed items still open
    are scored and simulated, and items added after the cutoff are scored but stay out of the summary. Before the
    cutoff the committed scope is the iteration's open items at t. Raises ValueError for an iteration no team runs,
    or a team that does not run it."""
    group = iteration_group(sprint_calendar(cache, commit_grace_days), iteration, team)
    cutoff, end = group["cutoff"].iloc[0], group["end"].iloc[0]
    t = min(to_utc(t), end)
    seen = horizon(cache, group["project"]).iloc[0]
    if t >= end and seen < end:  # the outcome is not in the data yet
        t = seen
    if t < cutoff:
        sprints, items = scope_at(
            cache, history, iteration=iteration, cutoff=t, work_item_types=work_item_types,
            done_categories=done_categories, commit_grace_days=commit_grace_days, team=team,
        )
        sprints, items = sprints.assign(cutoff=cutoff), items.assign(cutoff=cutoff)
    else:
        sprints = history.sprints[history.sprints["sprint_id"].isin(group["sprint_id"])]
        items = history.items[history.items["sprint_id"].isin(sprints["sprint_id"])]
    if items.empty:
        return []
    day1 = build_features_for(sprints, items, history.sprints, history.items)
    windows = group[group["sprint_id"].isin(sprints["sprint_id"])]
    rows = open_rows(
        cache, history, windows, pd.Series(t, index=windows.index),
        work_item_types=work_item_types, done_categories=done_categories,
    )
    feats = build_checkpoint_features(rows, day1, history.sprints)
    done = pd.Series(dtype="float64")
    if t >= cutoff:
        done = done_points_at(cache, items, pd.Series(t, index=sprints["sprint_id"].to_numpy()), done_categories)
    velocity = team_history(sprints, history.sprints).set_index("sprint_id")["trailing_velocity"]
    scored = []
    for _, sprint in sprints.iterrows():
        sid = sprint["sprint_id"]
        mine = feats[feats["sprint_id"] == sid]
        done_points = float(done.get(sid, 0.0))
        total = float(sprint["committed_points"])
        if t >= end:  # over: whatever is still open did not finish
            p = np.zeros(len(mine))
            samples = simulate_pct_done(p[:0], p[:0], fc.sigma, done_points=done_points, total_points=total)
        else:
            p, samples = forecast_sprint(fc, mine, seed=0, done_points=done_points, total_points=total)
        scored.append(ScoredSprint(
            sprint, mine.assign(p=p), summarize(samples), float(velocity.get(sid, math.nan)), done_points, t,
            items[items["sprint_id"] == sid],
        ))
    return scored
