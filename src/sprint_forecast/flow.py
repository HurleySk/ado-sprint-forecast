"""How work moved: where each sprint item went, state changes within a sprint, and sprints in a row without one."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sprint_forecast.cache import CacheData
from sprint_forecast.checkpoints import _state_changes_between, state_changes
from sprint_forecast.sprints import REMOVED, SprintData
from sprint_forecast.timeline import as_of_many

# done: done at the end. done_in_grace: closed within the close grace after it. closed_later: closed afterwards
# without leaving the sprint's iteration. carried: moved to a later sprint. backlog: moved anywhere else (an undated
# or earlier iteration). removed: set to a Removed state. open: still in the iteration and not done.
FATES = ("done", "done_in_grace", "closed_later", "carried", "backlog", "removed", "open")
FATE_COLUMNS = ["sprint_id", "item_id", "fate"]


def _destination(cache: CacheData, iterations: pd.Series, starts: pd.Series) -> np.ndarray:
    """carried when each iteration is a dated one starting after the sprint's start, else backlog."""
    its = cache.iterations.dropna(subset=["start_date"]).drop_duplicates("path").set_index("path")["start_date"]
    later = iterations.map(its).to_numpy() > starts.to_numpy()
    return np.where(pd.Series(later).fillna(False).astype(bool), "carried", "backlog")


def item_fates(cache: CacheData, sd: SprintData, done_categories=("Completed",)) -> pd.DataFrame:
    """One row per ended sprint item (committed or added): FATES code of where it went."""
    parts = [f[f["done"].notna()] for f in (sd.items, sd.added) if len(f)]
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["done"])
    if rows.empty:
        return pd.DataFrame(columns=FATE_COLUMNS)
    done = set(done_categories)
    rows = rows.assign(item_id=rows["item_id"].astype("int64")).reset_index(drop=True)
    strict = rows["done_strict"].astype("float64").fillna(rows["done"].astype("float64"))
    fate = np.where(strict == 1, "done", np.where(rows["done"] == 1, "done_in_grace", ""))
    fate = pd.Series(fate, dtype=object)

    at_e = as_of_many(cache.revisions, rows[["item_id", "end"]], "end")
    todo = fate == ""
    removed = todo & (at_e["state_category"] == REMOVED)
    fate[removed] = "removed"
    moved = todo & ~removed & (at_e["iteration"] != rows["iteration"])
    fate[moved] = _destination(cache, at_e.loc[moved, "iteration"], rows.loc[moved, "start"])

    rest = rows[fate == ""].assign(_row=lambda d: d.index)
    if len(rest):
        later = rest[["_row", "item_id", "iteration", "end", "start"]].merge(
            cache.revisions[["item_id", "changed", "rev", "iteration", "state_category"]].rename(
                columns={"iteration": "now_iteration"}), on="item_id")
        later = later[later["changed"] > later["end"]]
        event = (later["now_iteration"] != later["iteration"]) | later["state_category"].isin(done | {REMOVED})
        first = later[event].sort_values(["changed", "rev"], kind="mergesort").drop_duplicates("_row")
        kind = pd.Series("open", index=rest["_row"].to_numpy(), dtype=object)
        moved_later = first["now_iteration"] != first["iteration"]
        kind[first.loc[moved_later, "_row"].to_numpy()] = _destination(
            cache, first.loc[moved_later, "now_iteration"], first.loc[moved_later, "start"])
        stay = first[~moved_later]
        kind[stay["_row"].to_numpy()] = np.where(stay["state_category"] == REMOVED, "removed", "closed_later")
        fate[kind.index] = kind.to_numpy()
    return pd.DataFrame({"sprint_id": rows["sprint_id"], "item_id": rows["item_id"], "fate": fate})[FATE_COLUMNS]


def sprint_state_changes(cache: CacheData, items: pd.DataFrame) -> pd.Series:
    """State changes of each row's item between its sprint's start and end (index aligned with `items`)."""
    if items.empty:
        return pd.Series(dtype="int64")
    ids = set(items["item_id"].astype("int64"))
    events = state_changes(cache.revisions[cache.revisions["item_id"].isin(ids)])
    return _state_changes_between(events, items.assign(t=items["end"]))


def idle_sprints(cache: CacheData, rows: pd.DataFrame) -> pd.Series:
    """Per row (item_id, start of its current sprint, t): sprints in a row, the current one included, the item has
    been in that started after its last state change up to t; 0 when its state changed in the current sprint. An
    item whose state never changed counts every dated iteration it has been in."""
    if rows.empty:
        return pd.Series(dtype="int64")
    left = rows[["item_id", "start", "t"]].reset_index(drop=True).assign(_row=lambda d: d.index)
    left["item_id"] = left["item_id"].astype("int64")
    revs = cache.revisions[cache.revisions["item_id"].isin(set(left["item_id"]))]
    events = state_changes(revs)
    events = events[~events["first"]].merge(left, on="item_id")
    last = events[events["changed"] <= events["t"]].groupby("_row")["changed"].max()
    left = left.merge(last.rename("last"), left_on="_row", right_index=True, how="left")
    ends = cache.iterations.dropna(subset=["end_date"]).drop_duplicates("path").set_index("path")["end_date"]
    starts = cache.iterations.dropna(subset=["start_date"]).drop_duplicates("path").set_index("path")["start_date"]
    been = left.merge(revs[["item_id", "changed", "iteration"]], on="item_id")
    been = been[been["changed"] <= been["t"]].drop_duplicates(["_row", "iteration"])
    been = been.assign(_end=been["iteration"].map(ends), _start=been["iteration"].map(starts)).dropna(subset=["_end"])
    been = been[(been["_start"] <= been["t"]) & (been["last"].isna() | (been["_start"] > been["last"]))]
    counts = left["_row"].map(been.groupby("_row").size()).fillna(0).astype("int64")
    counts[left["last"].notna() & (left["last"] >= left["start"])] = 0
    return counts.set_axis(rows.index)
