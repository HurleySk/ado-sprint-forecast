"""Cycle time: how long each finished item took from its first active state to done, and the time in each state."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sprint_forecast.cache import CacheData
from sprint_forecast.sprints import match_team, raw_points, sprint_calendar

ACTIVE_CATEGORIES = ("InProgress", "Resolved")
CYCLE_COLUMNS = [
    "item_id", "project", "team", "team_key", "type", "iteration", "assigned_to_sk",
    "points_at_start", "points", "re_estimated", "started", "closed", "days",
]
CYCLE_STATE_COLUMNS = ["item_id", "state", "state_category", "days"]


@dataclass
class CycleData:
    items: pd.DataFrame   # one row per finished item
    states: pd.DataFrame  # one row per finished item and state it spent time in between start and done


def business_days(start: pd.Series, end: pd.Series) -> np.ndarray:
    """Weekday time from start to end in days, UTC; Saturdays and Sundays don't count."""
    s = start.dt.tz_convert("UTC").dt.tz_localize(None)
    e = end.dt.tz_convert("UTC").dt.tz_localize(None)
    d0, d1 = s.dt.normalize(), e.dt.normalize()
    day0, day1 = d0.to_numpy().astype("datetime64[D]"), d1.to_numpy().astype("datetime64[D]")
    whole = np.busday_count(day0, day1).astype("float64")
    frac0 = ((s - d0) / pd.Timedelta(days=1)).to_numpy()
    frac1 = ((e - d1) / pd.Timedelta(days=1)).to_numpy()
    return whole - frac0 * np.is_busday(day0) + frac1 * np.is_busday(day1)


def _teams(cache: CacheData, items: pd.DataFrame) -> pd.Series:
    """The team running the item's iteration that owns its area; else any team in the project by area. Projects
    with no team subscriptions count as one team named after the project, as in sprint_calendar."""
    cal = sprint_calendar(cache)
    by_iteration = cal.groupby("iteration")["team"].apply(list).to_dict()
    fallback = set(cal.loc[cal["is_fallback"], "project"])
    merged = cache.teams.merge(cache.team_areas, on="team_sk")
    areas: dict[str, dict[str, list[str]]] = {}
    for row in merged.itertuples(index=False):
        areas.setdefault(row.project, {}).setdefault(row.name, []).append(row.area_path)
    teams = []
    for area, it, project in zip(items["area"], items["iteration"], items["project"]):
        if project in fallback:
            teams.append(project)
            continue
        project_areas = areas.get(project, {})
        team = match_team(area, by_iteration.get(it, []), project_areas) if it in by_iteration else None
        teams.append(team if team is not None else match_team(area, sorted(project_areas), project_areas))
    return pd.Series(teams, index=items.index, dtype=object)


def build_cycle(cache: CacheData, *, work_item_types: list[str], done_categories=("Completed",)) -> CycleData:
    """Items of the counted types that reached a done category. An item's cycle runs from its first active state
    (InProgress, or Resolved when that isn't done) to the first time it was done; reopening afterwards is ignored.
    Items that went straight to done have no cycle (started and days empty) and no state rows."""
    done = set(done_categories)
    active = set(ACTIVE_CATEGORIES) - done
    revs = cache.revisions.sort_values(["item_id", "rev"], kind="mergesort").reset_index(drop=True)
    first_done = revs[revs["state_category"].isin(done)].groupby("item_id")["rev"].min().rename("done_rev")
    revs = revs.merge(first_done, left_on="item_id", right_index=True)
    revs = revs[revs["rev"] <= revs["done_rev"]]
    closing = revs[revs["rev"] == revs["done_rev"]]
    closing = closing[closing["type"].isin(work_item_types)].set_index("item_id")
    if closing.empty:
        return CycleData(pd.DataFrame(columns=CYCLE_COLUMNS), pd.DataFrame(columns=CYCLE_STATE_COLUMNS))
    revs = revs[revs["item_id"].isin(closing.index)]
    started_rev = (revs[(revs["rev"] < revs["done_rev"]) & revs["state_category"].isin(active)]
                   .groupby("item_id")["rev"].min().rename("start_rev"))
    starting = revs.merge(started_rev, left_on="item_id", right_index=True)
    starting = starting[starting["rev"] == starting["start_rev"]].set_index("item_id")

    items = pd.DataFrame(index=closing.index)
    items["project"] = closing["project"]
    items["type"] = closing["type"]
    items["iteration"] = closing["iteration"]
    items["area"] = closing["area"]
    items["assigned_to_sk"] = closing["assigned_to_sk"]
    items["points"] = raw_points(closing)
    items["points_at_start"] = raw_points(starting).reindex(items.index)
    items["re_estimated"] = (items["points_at_start"].notna() & items["points"].notna()
                             & (items["points_at_start"] != items["points"]))
    items["started"] = starting["changed"].reindex(items.index)
    items["closed"] = closing["changed"]
    has_start = items["started"].notna()
    items["days"] = np.nan
    if has_start.any():
        items.loc[has_start, "days"] = business_days(items.loc[has_start, "started"], items.loc[has_start, "closed"])
    items["team"] = _teams(cache, items)
    items["team_key"] = (items["project"] + "/" + items["team"]).where(items["team"].notna())
    items = items.reset_index().sort_values(["closed", "item_id"], kind="mergesort")

    # each revision's state holds until the next revision; sum that time per state between start and done
    seg = revs.merge(started_rev, left_on="item_id", right_index=True)
    seg = seg[seg["rev"] >= seg["start_rev"]].copy()
    seg["until"] = seg.groupby("item_id")["changed"].shift(-1)
    seg = seg[seg["until"].notna()]
    seg["days"] = business_days(seg["changed"], seg["until"]) if len(seg) else pd.Series(dtype="float64")
    states = (seg.groupby(["item_id", "state", "state_category"], sort=False, dropna=False)["days"].sum()
              .reset_index().sort_values(["item_id", "state"], kind="mergesort"))
    return CycleData(items[CYCLE_COLUMNS].reset_index(drop=True), states[CYCLE_STATE_COLUMNS].reset_index(drop=True))
