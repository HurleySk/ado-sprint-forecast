"""Feature matrices. The day-1 frame describes committed items at the commit cutoff; the checkpoint frame describes
items open at any time t in a sprint. Team history uses only sprints with end < this sprint's start."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sprint_forecast.cache import CacheData
from sprint_forecast.checkpoints import CHECKPOINTS, build_checkpoint_rows
from sprint_forecast.sprints import SprintData

MISSING = "(missing)"
CATEGORICAL = ["type", "state_category"]
ITEM_FEATURES = [
    "type", "state_category", "points", "points_rel", "is_unestimated", "carryover_count",
    "age_days", "days_since_change", "revisions_so_far", "has_parent", "is_unassigned",
]
SPRINT_FEATURES = [
    "load_ratio", "n_items", "bug_share", "carryover_share", "unestimated_share",
    "team_trailing_completion", "sprint_length_days", "team_sprint_index",
]
ASSIGNEE_FEATURES = ["assignee_load_ratio"]
PROGRESS_FEATURES = ["elapsed", "days_left", "done_share", "days_in_state", "n_state_changes", "is_added", "reassigned"]
FEATURES = ITEM_FEATURES + SPRINT_FEATURES + ASSIGNEE_FEATURES + PROGRESS_FEATURES
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
# The day-1 frame's features (model A before 0.2.0); it still supplies sprint context and assignee load.
DAY1_FEATURES = [
    "type", "state_category_at_commit", "points", "points_rel", "is_unestimated", "carryover_count",
    "age_days", "days_since_change", "revisions_so_far", "has_parent",
] + SPRINT_FEATURES + ["assignee_load_ratio", "is_unassigned"]
ID_COLUMNS = ["sprint_id", "project", "team", "team_key", "iteration", "start", "end", "cutoff", "item_id"]
FEATURE_FRAME_COLUMNS = ID_COLUMNS + DAY1_FEATURES + ["y"]
CHECKPOINT_FRAME_COLUMNS = (
    ID_COLUMNS + ["checkpoint", "t", "state_category_at_commit", "points_at_commit"] + FEATURES + ["y"]
)
FEATURE_VERSION = 2  # stored in the model bundle; bump when FEATURES or their meaning change

VELOCITY_WINDOW = 3
COMPLETION_WINDOW = 5
ASSIGNEE_WINDOW = 3
DAY = pd.Timedelta(days=1)


def _ns(series: pd.Series) -> np.ndarray:
    return series.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()


def _ts_ns(ts) -> int:
    return pd.Timestamp(ts).tz_convert("UTC").as_unit("ns").value


def _trailing_mean(values: np.ndarray, k: int, window: int) -> float:
    return float(values[max(0, k - window):k].mean()) if k > 0 else float("nan")


def team_history(target_sprints: pd.DataFrame, history_sprints: pd.DataFrame) -> pd.DataFrame:
    """Per target sprint: trailing velocity (mean delivered points of last 3), trailing completion (last 5),
    and the number of earlier team sprints, all from team sprints with end < target start."""
    hist = history_sprints.dropna(subset=["done_points"]).sort_values("end", kind="mergesort")
    by_team = {
        tk: (_ns(g["end"]), g["done_points"].to_numpy(float), g["pct_done"].to_numpy(float))
        for tk, g in hist.groupby("team_key")
    }
    rows = []
    for sprint_id, team_key, start in zip(target_sprints["sprint_id"], target_sprints["team_key"], target_sprints["start"]):
        ends, dp, pct = by_team.get(team_key, (np.array([], dtype="int64"), np.array([]), np.array([])))
        k = int(np.searchsorted(ends, _ts_ns(start), side="left"))
        rows.append({
            "sprint_id": sprint_id,
            "trailing_velocity": _trailing_mean(dp, k, VELOCITY_WINDOW),
            "team_trailing_completion": _trailing_mean(pct, k, COMPLETION_WINDOW),
            "team_sprint_index": k,
        })
    return pd.DataFrame(rows, columns=["sprint_id", "trailing_velocity", "team_trailing_completion", "team_sprint_index"])


def assignee_load(target_items: pd.DataFrame, history_items: pd.DataFrame) -> pd.Series:
    """Per target item: its assignee's committed points this sprint / their mean delivered points over the
    last 3 earlier sprints (end < start) they appear in. NaN when unassigned or no history."""
    h = history_items[history_items["assigned_to_sk"].notna() & history_items["done"].notna()]
    h = h.assign(_delivered=h["points"] * h["done"].astype("float64"))
    per_sprint = h.groupby(["assigned_to_sk", "sprint_id"], as_index=False).agg(
        end=("end", "first"), delivered=("_delivered", "sum")
    ).sort_values("end", kind="mergesort")
    by_person = {a: (_ns(g["end"]), g["delivered"].to_numpy(float)) for a, g in per_sprint.groupby("assigned_to_sk")}
    t = target_items[target_items["assigned_to_sk"].notna()]
    committed = t.groupby(["sprint_id", "assigned_to_sk"])["points"].sum()
    starts = t.groupby("sprint_id")["start"].first()
    ratio = {}
    for (sprint_id, person), pts in committed.items():
        ends, delivered = by_person.get(person, (np.array([], dtype="int64"), np.array([])))
        k = int(np.searchsorted(ends, _ts_ns(starts[sprint_id]), side="left"))
        mean = _trailing_mean(delivered, k, ASSIGNEE_WINDOW)
        ratio[(sprint_id, person)] = pts / mean if mean > 0 else float("nan")
    keys = list(zip(target_items["sprint_id"], target_items["assigned_to_sk"]))
    return pd.Series([ratio.get(k, float("nan")) for k in keys], index=target_items.index, dtype="float64")


def build_features_for(
    target_sprints: pd.DataFrame,
    target_items: pd.DataFrame,
    history_sprints: pd.DataFrame,
    history_items: pd.DataFrame,
) -> pd.DataFrame:
    items = target_items.reset_index(drop=True)
    if items.empty:
        return pd.DataFrame(columns=FEATURE_FRAME_COLUMNS)
    hist = team_history(target_sprints, history_sprints).set_index("sprint_id")
    agg = items.assign(
        _bug=(items["type"] == "Bug").astype(float),
        _carry=(items["carryover_count"] > 0).astype(float),
        _unest=items["is_unestimated"].astype(float),
    ).groupby("sprint_id").agg(
        n_items=("item_id", "size"), committed_points=("points", "sum"),
        bug_share=("_bug", "mean"), carryover_share=("_carry", "mean"), unestimated_share=("_unest", "mean"),
    )
    sid = items["sprint_id"]
    velocity = sid.map(hist["trailing_velocity"])
    velocity = velocity.where(velocity > 0)
    out = items[ID_COLUMNS].copy()
    out["type"] = items["type"].fillna(MISSING).astype(str)
    out["state_category_at_commit"] = items["state_category_at_commit"].fillna(MISSING).astype(str)
    out["points"] = items["points"].astype(float)
    out["points_rel"] = out["points"] / velocity
    out["is_unestimated"] = items["is_unestimated"].astype(float)
    out["carryover_count"] = items["carryover_count"].astype(float)
    out["age_days"] = (items["cutoff"] - items["created"]) / DAY
    out["days_since_change"] = (items["cutoff"] - items["last_changed"]) / DAY
    out["revisions_so_far"] = items["revisions_so_far"].astype(float)
    out["has_parent"] = items["parent_id"].notna().astype(float)
    out["load_ratio"] = sid.map(agg["committed_points"]) / velocity
    for col in ("n_items", "bug_share", "carryover_share", "unestimated_share"):
        out[col] = sid.map(agg[col]).astype(float)
    out["team_trailing_completion"] = sid.map(hist["team_trailing_completion"])
    out["sprint_length_days"] = (items["end"] - items["start"]) / DAY
    out["team_sprint_index"] = sid.map(hist["team_sprint_index"]).astype(float)
    out["assignee_load_ratio"] = assignee_load(items, history_items)
    out["is_unassigned"] = items["assigned_to_sk"].isna().astype(float)
    out["y"] = items["done"].astype("float64")
    return out[FEATURE_FRAME_COLUMNS]


def build_features(sd: SprintData) -> pd.DataFrame:
    return build_features_for(sd.sprints, sd.items, sd.sprints, sd.items)


def build_checkpoint_features(rows: pd.DataFrame, day1: pd.DataFrame, history_sprints: pd.DataFrame) -> pd.DataFrame:
    """Model features of open rows (checkpoints.open_rows or build_checkpoint_rows): the item as of each row's t,
    its sprint's plan and its assignee's load at the cutoff (from `day1`, the day-1 frame of the same sprints; NaN
    for items added after the cutoff), and the sprint's progress at t. Row order follows `rows`."""
    rows = rows.reset_index(drop=True)
    if rows.empty:
        return pd.DataFrame(columns=CHECKPOINT_FRAME_COLUMNS)
    targets = rows.drop_duplicates("sprint_id")[["sprint_id", "team_key", "start"]]
    hist = team_history(targets, history_sprints).set_index("sprint_id")
    sid, t, cutoff = rows["sprint_id"], rows["t"], rows["cutoff"]
    velocity = sid.map(hist["trailing_velocity"])
    velocity = velocity.where(velocity > 0)
    out = rows[ID_COLUMNS].copy()
    out["checkpoint"] = rows["checkpoint"].astype("float64") if "checkpoint" in rows else np.nan
    out["t"] = t
    out["state_category_at_commit"] = rows["state_category_at_commit"]
    out["points_at_commit"] = rows["points_at_commit"].astype("float64")
    out["type"] = rows["type"].fillna(MISSING).astype(str)
    out["state_category"] = rows["state_category"].fillna(MISSING).astype(str)
    out["points"] = rows["points"].astype(float)
    out["points_rel"] = out["points"] / velocity
    out["is_unestimated"] = rows["is_unestimated"].astype(float)
    out["carryover_count"] = rows["carryover_count"].astype(float)
    out["age_days"] = (t - rows["created"]) / DAY
    out["days_since_change"] = (t - rows["last_changed"]) / DAY
    out["revisions_so_far"] = rows["revisions_so_far"].astype(float)
    out["has_parent"] = rows["parent_id"].notna().astype(float)
    out["is_unassigned"] = rows["assigned_to_sk"].isna().astype(float)
    plan = day1.drop_duplicates("sprint_id").set_index("sprint_id")
    for col in SPRINT_FEATURES:
        out[col] = sid.map(plan[col]).astype(float)
    load = day1.assign(item_id=day1["item_id"].astype("int64")).set_index(["sprint_id", "item_id"])
    keys = pd.MultiIndex.from_arrays([sid, rows["item_id"].astype("int64")])
    out["assignee_load_ratio"] = load["assignee_load_ratio"].reindex(keys).to_numpy(dtype=float)
    span = rows["end"] - cutoff
    elapsed = (t - cutoff) / span.where(span > pd.Timedelta(0))
    out["elapsed"] = elapsed.clip(0.0, 1.0).fillna((t >= cutoff).astype(float))
    out["days_left"] = ((rows["end"] - t.where(t > cutoff, cutoff)) / DAY).clip(lower=0.0)
    committed = rows["committed_points"].astype(float)
    out["done_share"] = (rows["done_points_at_t"].astype(float) / committed.where(committed > 0)).fillna(0.0)
    out["days_in_state"] = (t - rows["state_changed"]) / DAY
    for col in ("n_state_changes", "is_added", "reassigned"):
        out[col] = rows[col].astype(float)
    out["y"] = rows["y"].astype("float64")
    return out[CHECKPOINT_FRAME_COLUMNS]


def build_checkpoint_frame(
    cache: CacheData,
    sd: SprintData,
    *,
    work_item_types,
    done_categories,
    checkpoints=CHECKPOINTS,
) -> pd.DataFrame:
    """Checkpoint features of every sprint in `sd` at each checkpoint: the frame model A trains and backtests on."""
    rows = build_checkpoint_rows(
        cache, sd, work_item_types=work_item_types, done_categories=done_categories, checkpoints=checkpoints,
    )
    return build_checkpoint_features(rows, build_features(sd), sd.sprints)
