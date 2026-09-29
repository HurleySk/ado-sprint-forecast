"""Checkpoint rows: every item open in a sprint at an instant t, described as of t, with its outcome at the end.

An open item is in the sprint's iteration, of a configured type, neither done nor removed, and in the sprint's team
by area, all as of t. Past the cutoff a row knows whether the item was in the sprint's day-1 committed scope and how
many committed points were done by t; before the cutoff the open items are the committed scope."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sprint_forecast.cache import CacheData
from sprint_forecast.sprints import (
    CALENDAR_COLUMNS,
    REMOVED,
    Assigner,
    SprintData,
    carryover,
    horizon,
    impute_points,
    iteration_end_map,
    pairs_for,
    raw_points,
    sprint_calendar,
)
from sprint_forecast.timeline import as_of_many

CHECKPOINTS = (0.0, 0.25, 0.5, 0.75)
OPEN_ROW_COLUMNS = CALENDAR_COLUMNS + [
    "t", "item_id", "type", "state_category", "state_category_at_commit", "points_at_commit", "area",
    "assigned_to_sk", "parent_id", "created", "last_changed", "state_changed", "revisions_so_far", "n_state_changes",
    "carryover_count", "raw_points", "points", "is_unestimated", "is_added", "reassigned", "committed_points",
    "done_points_at_t", "y",
]
CHECKPOINT_ROW_COLUMNS = ["checkpoint"] + OPEN_ROW_COLUMNS
PROGRESS_COLUMNS = ["sprint_id", "checkpoint", "t", "done_points", "committed_points"]


def checkpoint_time(cutoff: pd.Series, end: pd.Series, f: float) -> pd.Series:
    """The instant a share f of the way from each sprint's commit cutoff to its end."""
    return cutoff + (end - cutoff) * f


def state_changes(revisions: pd.DataFrame) -> pd.DataFrame:
    """Each item's first revision (first=True) and every later revision whose state differs from the one before."""
    r = revisions.sort_values(["item_id", "changed", "rev"], kind="mergesort").reset_index(drop=True)
    state = r["state"].fillna("")
    prev = state.groupby(r["item_id"], sort=False).shift()
    first = prev.isna()
    keep = first | (state != prev)
    return r.loc[keep, ["item_id", "changed"]].assign(first=first[keep]).reset_index(drop=True)


def _last_state_change(events: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """When each row's item last changed state at or before the row's t (index aligned with `rows`)."""
    left = pd.DataFrame({"item_id": rows["item_id"].astype("int64"), "t": rows["t"]}).reset_index(drop=True)
    left["_row"] = np.arange(len(left))
    left["t"] = left["t"].astype(events["changed"].dtype)
    right = events[["item_id", "changed"]].assign(item_id=events["item_id"].astype("int64"))
    out = pd.merge_asof(
        left.sort_values("t", kind="mergesort"), right.sort_values("changed", kind="mergesort"),
        left_on="t", right_on="changed", by="item_id", direction="backward",
    )
    return out.sort_values("_row")["changed"].set_axis(rows.index)


def _state_changes_between(events: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """State changes of each row's item in (start, t], not counting its first revision (index aligned with `rows`)."""
    left = pd.DataFrame({
        "item_id": rows["item_id"].astype("int64"), "start": rows["start"], "t": rows["t"],
    }).reset_index(drop=True)
    left["_row"] = np.arange(len(left))
    later = events.loc[~events["first"], ["item_id", "changed"]]
    hist = left.merge(later.assign(item_id=later["item_id"].astype("int64")), on="item_id")
    hist = hist[(hist["changed"] > hist["start"]) & (hist["changed"] <= hist["t"])]
    counts = left["_row"].map(hist.groupby("_row").size()).fillna(0).astype("int64")
    return counts.set_axis(rows.index)


def done_points_at(cache: CacheData, items: pd.DataFrame, t: pd.Series, done_categories) -> pd.Series:
    """Per sprint: points of `items` (sprint_id, item_id, iteration, points) that are in their sprint's iteration and
    in a done category as of that sprint's t. `t` is indexed by sprint_id and covers every item's sprint."""
    if items.empty:
        return pd.Series(dtype="float64")
    it = items.reset_index(drop=True)
    keys = pd.DataFrame({"item_id": it["item_id"].astype("int64"), "t": it["sprint_id"].map(t)})
    at = as_of_many(cache.revisions, keys, "t")
    done = (at["iteration"] == it["iteration"]) & at["state_category"].isin(set(done_categories))
    return (it["points"].astype("float64") * done).groupby(it["sprint_id"]).sum()


def open_rows(
    cache: CacheData,
    sd: SprintData,
    sprints: pd.DataFrame,
    t: pd.Series,
    *,
    work_item_types,
    done_categories,
) -> pd.DataFrame:
    """Items open in each sprint row at its own t (`t` index-aligned with `sprints`, one row per sprint_id), with
    their state as of t, commit-time state and points (NaN for items added after the cutoff), the sprint's committed
    points and the committed points done by t, and y: in the iteration and done at the sprint's end (NaN while the
    sprint is open). Before a sprint's cutoff its open items are its committed scope. Past the cutoff, sprints with
    no committed scope in `sd` are dropped."""
    if sprints["sprint_id"].duplicated().any():
        raise ValueError("open_rows needs one row per sprint")
    empty = pd.DataFrame(columns=OPEN_ROW_COLUMNS)
    revs = cache.revisions
    types, done = set(work_item_types), set(done_categories)
    windows = sprints[CALENDAR_COLUMNS].assign(t=t).reset_index(drop=True).rename(columns={"iteration": "target"})
    pairs = pairs_for(revs, windows)
    if pairs.empty:
        return empty
    at_t = as_of_many(revs, pairs[["item_id", "t"]], "t")
    is_open = (
        (at_t["iteration"] == pairs["target"])
        & at_t["type"].isin(types)
        & ~at_t["state_category"].isin(done | {REMOVED})
    )
    pairs, at_t = pairs[is_open], at_t[is_open]
    team = Assigner(cache, sprint_calendar(cache))(at_t["area"], pairs["target"], pairs["project"])
    mine = team == pairs["team"]
    rows = pairs[mine].rename(columns={"target": "iteration"}).reset_index(drop=True)
    st = at_t[mine].reset_index(drop=True)
    if rows.empty:
        return empty
    rows["item_id"] = rows["item_id"].astype("int64")
    for col in ("type", "state_category", "area", "assigned_to_sk", "parent_id", "created"):
        rows[col] = st[col]
    rows["last_changed"] = st["changed"]
    rows["revisions_so_far"] = st["revisions_so_far"].astype("int64")
    rows["raw_points"] = raw_points(st)
    rows = impute_points(rows, sd.items)
    rows["carryover_count"] = carryover(revs, rows.assign(cutoff=rows["t"]), iteration_end_map(cache))

    until_cutoff = rows["t"].where(rows["t"] < rows["cutoff"], rows["cutoff"])
    at_c = as_of_many(revs, pd.DataFrame({"item_id": rows["item_id"], "t": until_cutoff}), "t")
    rows["reassigned"] = (at_c["assigned_to_sk"].fillna("") != rows["assigned_to_sk"].fillna("")).astype("int64")
    events = state_changes(revs[revs["item_id"].isin(set(rows["item_id"]))])
    rows["state_changed"] = _last_state_change(events, rows)
    rows["n_state_changes"] = _state_changes_between(events, rows)

    commit = sd.items[["sprint_id", "item_id", "state_category_at_commit", "points"]].rename(
        columns={"points": "points_at_commit"})
    commit = commit.assign(item_id=commit["item_id"].astype("int64"))
    rows = rows.merge(commit, on=["sprint_id", "item_id"], how="left")
    committed = sd.sprints.set_index("sprint_id")["committed_points"]
    after = rows["t"] >= rows["cutoff"]
    rows = rows[~after | rows["sprint_id"].isin(committed.index)].reset_index(drop=True)
    if rows.empty:
        return empty
    after = rows["t"] >= rows["cutoff"]
    rows["is_added"] = (after & rows["points_at_commit"].isna()).astype("int64")
    rows.loc[~after, "state_category_at_commit"] = rows.loc[~after, "state_category"]
    rows.loc[~after, "points_at_commit"] = rows.loc[~after, "points"]
    open_points = rows.groupby("sprint_id")["points"].sum()
    rows["committed_points"] = (
        rows["sprint_id"].map(committed).where(after, rows["sprint_id"].map(open_points)).astype("float64")
    )
    t_by_sprint = rows.drop_duplicates("sprint_id").set_index("sprint_id")["t"]
    scope = sd.items[sd.items["sprint_id"].isin(set(rows.loc[after, "sprint_id"]))]
    done_points = done_points_at(cache, scope, t_by_sprint, done_categories)
    rows["done_points_at_t"] = rows["sprint_id"].map(done_points).fillna(0.0).where(after, 0.0)

    at_end = as_of_many(revs, pd.DataFrame({"item_id": rows["item_id"], "end": rows["end"]}), "end")
    ended_done = (at_end["iteration"] == rows["iteration"]) & at_end["state_category"].isin(done)
    rows["y"] = ended_done.astype("float64").where(rows["end"] <= horizon(cache, rows["project"]))
    out = rows[OPEN_ROW_COLUMNS].sort_values(["start", "sprint_id", "item_id"], kind="mergesort")
    return out.reset_index(drop=True)


def build_checkpoint_rows(
    cache: CacheData,
    sd: SprintData,
    *,
    work_item_types,
    done_categories,
    checkpoints=CHECKPOINTS,
) -> pd.DataFrame:
    """open_rows for every sprint of `sd` at each checkpoint (a share of the way from its cutoff to its end)."""
    s = sd.sprints
    frames = []
    for f in checkpoints:
        rows = open_rows(
            cache, sd, s, checkpoint_time(s["cutoff"], s["end"], f),
            work_item_types=work_item_types, done_categories=done_categories,
        )
        if len(rows):
            frames.append(rows.assign(checkpoint=float(f)))
    if not frames:
        return pd.DataFrame(columns=CHECKPOINT_ROW_COLUMNS)
    out = pd.concat(frames, ignore_index=True)[CHECKPOINT_ROW_COLUMNS]
    return out.sort_values(["start", "sprint_id", "checkpoint", "item_id"], kind="mergesort").reset_index(drop=True)


def checkpoint_progress(cache: CacheData, sd: SprintData, *, done_categories, checkpoints=CHECKPOINTS) -> pd.DataFrame:
    """Per sprint and checkpoint: t, the sprint's committed points in its iteration and done as of t, and its
    committed points. Sprints with no open item at a checkpoint still get a row here."""
    s = sd.sprints
    if s.empty:
        return pd.DataFrame(columns=PROGRESS_COLUMNS)
    frames = []
    for f in checkpoints:
        t = checkpoint_time(s["cutoff"], s["end"], f)
        done = done_points_at(cache, sd.items, t.set_axis(s["sprint_id"].to_numpy()), done_categories)
        frames.append(pd.DataFrame({
            "sprint_id": s["sprint_id"],
            "checkpoint": float(f),
            "t": t,
            "done_points": s["sprint_id"].map(done).fillna(0.0).astype("float64"),
            "committed_points": s["committed_points"].astype("float64"),
        }))
    return pd.concat(frames, ignore_index=True)[PROGRESS_COLUMNS]
