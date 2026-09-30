"""Go-live forecast: the chance that every open child of a parent item (a feature, say) is done by each coming
sprint's end. Children in any running sprint are drawn with the item model; each later sprint draws one of
the parent's own recent sprints: the points of its children finished in it, less the points of children linked to it
then (after its first child started). The parent is done the first sprint every child linked by then is done. The
`_no_growth` columns leave the linking out, so they are the earliest likely dates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sprint_forecast.cache import CacheData
from sprint_forecast.cycle import ACTIVE_CATEGORIES, _teams
from sprint_forecast.rollup import simulate_pct_done
from sprint_forecast.sprints import REMOVED, SprintData, raw_points, sprint_calendar
from sprint_forecast.timeline import to_utc

PARENT_COLUMNS = [
    "parent_id", "project", "team", "team_key", "n_children", "n_open", "open_points", "n_unestimated_open",
    "in_sprint_points", "done_points", "burn_sprints", "mean_burn", "mean_added", "status", "p_this_sprint", "p50_end",
    "p85_end", "p50_end_no_growth", "p85_end_no_growth",
]
CURVE_COLUMNS = ["parent_id", "k", "iteration", "end", "p_done_by", "p_done_by_no_growth"]
STATUSES = ("forecast", "not enough history", "no recent progress", "no team")
BURN_WINDOW = 8     # most recent ended sprints of burn and growth to draw from
MIN_BURN_SPRINTS = 2
HORIZON = 12        # sprints after the running one
ACTIVE_DAYS = 112   # a parent with nothing in a current or coming sprint counts if a child finished this recently
N_DRAWS = 4000
DEFAULT_P = 0.5     # a child in the running sprint with no item-model probability
DONE_TOL = 1e-6


def _fill_points(kids: pd.DataFrame, history: SprintData, now: pd.Timestamp) -> pd.Series:
    """Points of each child, unestimated ones taking the median of the team's earlier sprint items (else of every
    team's, else 1)."""
    known = history.items[history.items["raw_points"].notna() & (history.items["end"] < now)]
    by_team = known.groupby("team_key")["raw_points"].median()
    overall = float(known["raw_points"].median()) if len(known) else 1.0
    fill = kids["team_key"].map(by_team).fillna(overall)
    return kids["raw_points"].fillna(fill).astype("float64")


def _parent_team(ch: pd.DataFrame) -> str | None:
    """The team holding most of the open children's points; all the children's when no open child has a team."""
    for rows in (ch[~ch["is_done"]], ch):
        weight = rows.dropna(subset=["team_key"]).groupby("team_key")["points"].sum()
        if len(weight):
            return sorted(weight[weight == weight.max()].index)[0]
    return None


def _steps(tcal: pd.DataFrame, now: pd.Timestamp, horizon_sprints: int) -> list[tuple[int, str | None, pd.Timestamp]]:
    """(k, iteration, end) of the running sprint (k = 0, when there is one) and the next `horizon_sprints`, dated
    past the calendar on the team's usual cadence (its median sprint length, else 14 days, when sprints share a
    start). A calendar that ran out before `now` is stepped on to the first end after it."""
    steps = []
    running = tcal[(tcal["start"] <= now) & (tcal["end"] > now)].head(1)
    for r in running.itertuples(index=False):
        steps.append((0, r.iteration, r.end))
    future = tcal[tcal["start"] > now].head(horizon_sprints)
    for k, r in enumerate(future.itertuples(index=False), start=1):
        steps.append((k, r.iteration, r.end))
    starts = tcal["start"].sort_values()
    cadence = starts.diff().dropna().median() if len(starts) > 1 else pd.NaT
    for fallback in ((tcal["end"] - tcal["start"]).median(), pd.Timedelta(days=14)):
        if pd.isna(cadence) or cadence <= pd.Timedelta(0):
            cadence = fallback
    last = steps[-1][2] if steps else tcal["end"].max()
    if last + cadence <= now:
        last = last + cadence * ((now - last) // cadence)
    for k in range(len(future) + 1, horizon_sprints + 1):
        last = last + cadence
        steps.append((k, None, last))
    return steps


def golive(
    cache: CacheData,
    history: SprintData,
    *,
    now,
    work_item_types,
    done_categories,
    p_open: dict[int, float] | None = None,
    sigma: float = 0.0,
    n_draws: int = N_DRAWS,
    horizon_sprints: int = HORIZON,
    active_days: float = ACTIVE_DAYS,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per parent with open children of the counted types that is active (a child open in a running or
    coming sprint, or finished in the last `active_days`), and its curve: P(every child done) by the end of each
    sprint k, with the scope linked to it since work on it began drawn back in from the sprint after the running one
    (`p_done_by_no_growth` without it). A child finished before it was linked is neither burn nor growth. `p_open`
    holds the item model's probability for items open in running sprints; `sigma` is the sprint shock those are
    drawn with. Parents with under two ended sprints since their first child started under them, or no burn in them,
    get no curve."""
    now = to_utc(now)
    done = set(done_categories)
    p_open = p_open or {}
    parents_out, curve_out = [], []
    revs = cache.revisions[cache.revisions["changed"] <= now]
    latest = revs.sort_values(["item_id", "changed", "rev"], kind="mergesort").groupby("item_id").tail(1)
    kids = latest[
        latest["parent_id"].notna() & latest["type"].isin(work_item_types) & (latest["state_category"] != REMOVED)
    ].copy()
    if kids.empty:
        return pd.DataFrame(columns=PARENT_COLUMNS), pd.DataFrame(columns=CURVE_COLUMNS)
    kids["parent_id"] = kids["parent_id"].astype("int64")
    kids["is_done"] = kids["state_category"].isin(done)
    theirs = revs[revs["item_id"].isin(kids["item_id"])].sort_values(["changed", "rev"], kind="mergesort")
    in_done = theirs["state_category"].isin(done)
    entered = in_done & ~in_done.groupby(theirs["item_id"]).shift(fill_value=False)
    last_done = theirs[entered].groupby("item_id").tail(1).set_index("item_id")  # a reopened child: its last close
    def per_kid(values: pd.Series) -> pd.Series:
        return values.reindex(kids["item_id"]).set_axis(kids.index)

    kids["done_at"] = per_kid(last_done["changed"]).where(kids["is_done"])
    started = theirs[theirs["state_category"].isin(set(ACTIVE_CATEGORIES) | done)].groupby("item_id")["changed"].min()
    linked = theirs[theirs["parent_id"] == theirs["item_id"].map(kids.set_index("item_id")["parent_id"])]
    kids["linked_at"] = per_kid(linked.groupby("item_id")["changed"].min())
    kids["done_elsewhere"] = kids["done_at"] < kids["linked_at"]  # finished, then moved under this parent
    started = per_kid(started)
    kids["started"] = started.where(started.isna() | (started >= kids["linked_at"]), kids["linked_at"]).where(
        ~kids["done_elsewhere"])  # its start under this parent
    kids["raw_points"] = raw_points(kids).where(~kids["is_done"], per_kid(raw_points(last_done)))
    kids["raw_points"] = kids["raw_points"].fillna(raw_points(kids))
    team = _teams(cache, kids)
    kids["team_key"] = (kids["project"] + "/" + team).where(team.notna())
    kids["points"] = _fill_points(kids, history, now)

    cal = sprint_calendar(cache)
    live = set(cal.loc[cal["end"] > now, "iteration"])
    running_its = set(cal.loc[(cal["start"] <= now) & (cal["end"] > now), "iteration"])
    recent = now - pd.Timedelta(days=active_days)
    for pid, ch in kids.groupby("parent_id", sort=True):
        open_ = ch[~ch["is_done"]]
        finished = ch[ch["is_done"]]
        if open_.empty or not (open_["iteration"].isin(live).any() or (finished["done_at"] >= recent).any()):
            continue
        team_key = _parent_team(ch)
        tcal = cal[cal["team_key"] == team_key].sort_values("start", kind="mergesort")
        row = {
            "parent_id": pid, "project": ch["project"].mode().iloc[0],
            "team": team_key.split("/", 1)[1] if team_key else None, "team_key": team_key,
            "n_children": len(ch), "n_open": len(open_), "open_points": float(open_["points"].sum()),
            "n_unestimated_open": int(open_["raw_points"].isna().sum()), "in_sprint_points": 0.0,
            "done_points": float(finished["points"].sum()), "burn_sprints": 0, "mean_burn": np.nan,
            "mean_added": np.nan, "p_this_sprint": np.nan, "p50_end": pd.NaT, "p85_end": pd.NaT,
            "p50_end_no_growth": pd.NaT, "p85_end_no_growth": pd.NaT,
        }
        in_sprint = open_[open_["iteration"].isin(running_its) | open_["item_id"].isin(set(p_open))]
        row["in_sprint_points"] = float(in_sprint["points"].sum())
        if tcal.empty:
            parents_out.append({**row, "status": "no team"})
            continue
        past = tcal[tcal["end"] <= now]
        first_start = ch["started"].min()
        windows = past[past["end"] > first_start].tail(BURN_WINDOW)
        here = ch[~ch["done_elsewhere"]]
        burned, grown = here[here["is_done"]], here[here["linked_at"] > first_start]
        burn, added = (np.array([
            rows.loc[(rows[at] >= w.start) & (rows[at] <= w.end), "points"].sum()
            for w in windows.itertuples(index=False)
        ], dtype=float) for rows, at in ((burned, "done_at"), (grown, "linked_at")))
        row["burn_sprints"] = len(burn)
        row["mean_burn"] = float(burn.mean()) if len(burn) else np.nan
        row["mean_added"] = float(added.mean()) if len(added) else np.nan
        if len(burn) < MIN_BURN_SPRINTS:
            parents_out.append({**row, "status": "not enough history"})
            continue
        if not burn.sum() > 0:
            parents_out.append({**row, "status": "no recent progress"})
            continue
        rng = np.random.default_rng([seed, pid])
        remaining = np.full(n_draws, row["open_points"])
        floor = remaining.copy()
        all_done = np.zeros(n_draws, dtype=bool)
        curve = []
        for k, it, end in _steps(tcal, now, horizon_sprints):
            if k == 0:
                if len(in_sprint):
                    p = np.array([p_open.get(int(i), DEFAULT_P) for i in in_sprint["item_id"]], dtype=float)
                    w = in_sprint["points"].to_numpy(dtype=float)
                    share = simulate_pct_done(p, w, sigma, n_draws, seed=int(rng.integers(2**31)))
                    remaining = remaining - share * w.sum()
                    floor = remaining.copy()
            else:
                j = rng.integers(len(burn), size=n_draws)
                remaining = remaining + added[j] - burn[j]
                floor = floor - burn[j]
            all_done |= remaining <= DONE_TOL
            curve.append({"parent_id": pid, "k": k, "iteration": it, "end": end, "p_done_by": float(all_done.mean()),
                          "p_done_by_no_growth": float(np.mean(floor <= DONE_TOL))})
        c = pd.DataFrame(curve)
        row["p_this_sprint"] = float(c.loc[c["k"] == 0, "p_done_by"].iloc[0]) if (c["k"] == 0).any() else np.nan
        for suffix, by in (("", "p_done_by"), ("_no_growth", "p_done_by_no_growth")):
            for col, level in (("p50_end", 0.5), ("p85_end", 0.85)):
                hit = c.loc[c[by] >= level, "end"]
                row[col + suffix] = hit.iloc[0] if len(hit) else pd.NaT
        parents_out.append({**row, "status": "forecast"})
        curve_out.append(c)
    parents = pd.DataFrame(parents_out, columns=PARENT_COLUMNS)
    curves = pd.concat(curve_out, ignore_index=True)[CURVE_COLUMNS] if curve_out else pd.DataFrame(columns=CURVE_COLUMNS)
    return parents, curves
