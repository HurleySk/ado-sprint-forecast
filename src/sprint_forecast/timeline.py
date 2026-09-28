"""As-of reconstruction: an item's state at instant t is its last revision with changed <= t."""
from __future__ import annotations

import numpy as np
import pandas as pd

STATE_COLUMNS = [
    "rev", "changed", "created", "type", "state", "state_category", "iteration", "area",
    "assigned_to_sk", "story_points", "effort", "parent_id",
]


def to_utc(t) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def as_of(revisions: pd.DataFrame, t) -> pd.DataFrame:
    """One row per item: its last revision with changed <= t. Items with no revision by t are absent."""
    t = to_utc(t)
    sub = revisions[revisions["changed"] <= t]
    sub = sub.sort_values(["item_id", "changed", "rev"], kind="mergesort")
    return sub.drop_duplicates("item_id", keep="last").reset_index(drop=True)


def as_of_many(revisions: pd.DataFrame, keys: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """For each row of `keys` (columns item_id, time_col, plus any non-revision columns), attach the item's
    last revision with changed <= keys[time_col] (STATE_COLUMNS) and `revisions_so_far`, the number of
    revisions up to that instant. Rows with no revision yet get NaN/NaT. Output keeps the order of `keys`."""
    right = revisions.sort_values(["changed", "rev"], kind="mergesort")
    right = right.assign(revisions_so_far=right.groupby("item_id").cumcount() + 1)
    right = right[["item_id", "revisions_so_far"] + STATE_COLUMNS]
    left = keys.reset_index(drop=True).assign(_row=np.arange(len(keys)))
    left[time_col] = left[time_col].astype(right["changed"].dtype)
    left = left.sort_values(time_col, kind="mergesort")
    out = pd.merge_asof(
        left, right, left_on=time_col, right_on="changed", by="item_id",
        direction="backward", allow_exact_matches=True,
    )
    return out.sort_values("_row").drop(columns="_row").reset_index(drop=True)
