"""Score an iteration's scope with a trained forecaster (shared by `predict` and `export`)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from sprint_forecast.cache import CacheData
from sprint_forecast.features import build_features_for, team_history
from sprint_forecast.rollup import Forecaster, forecast_sprint, summarize
from sprint_forecast.sprints import SprintData, scope_at


@dataclass
class ScoredSprint:
    sprint: pd.Series  # one summarize_sprints row; outcome columns are empty
    items: pd.DataFrame  # feature rows plus "p", each item's calibrated probability of reaching done
    summary: dict[str, float]  # summarize() of the simulated share of committed points done
    velocity: float  # team trailing velocity in points; NaN without history


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
    """Forecast each team sprint of `iteration` on its scope as of `t`. Raises ValueError for an iteration no
    team runs, or a team that does not run it."""
    sprints, items = scope_at(
        cache, history, iteration=iteration, cutoff=t, work_item_types=work_item_types,
        done_categories=done_categories, commit_grace_days=commit_grace_days, team=team,
    )
    if items.empty:
        return []
    feats = build_features_for(sprints, items, history.sprints, history.items)
    velocity = team_history(sprints, history.sprints).set_index("sprint_id")["trailing_velocity"]
    by_id = sprints.set_index("sprint_id", drop=False)
    scored = []
    for sid, rows in feats.groupby("sprint_id", sort=False):
        p, samples = forecast_sprint(fc, rows, seed=0)
        scored.append(ScoredSprint(by_id.loc[sid], rows.assign(p=p), summarize(samples), float(velocity.get(sid, math.nan))))
    return scored
