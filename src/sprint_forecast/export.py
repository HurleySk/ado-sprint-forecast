"""Power BI-ready CSVs: forecasts for running and upcoming sprints (one file per run) plus sprint outcomes."""
from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sprint_forecast.cache import CacheData
from sprint_forecast.cycle import CYCLE_STATE_COLUMNS, build_cycle
from sprint_forecast.forecast import ScoredSprint, score_iteration
from sprint_forecast.model import contributions, describe_drivers
from sprint_forecast.sprints import SPRINT_COLUMNS, build_sprints, sprint_calendar
from sprint_forecast.timeline import as_of_many, to_utc

SPRINT_FORECASTS_DIR = "sprint_forecasts"
ITEM_FORECASTS_DIR = "item_forecasts"
SPRINTS_FILE = "sprints.csv"
ITEM_HISTORY_FILE = "items.csv"
BACKTEST_FILE = "backtest.csv"
N_RISK_FACTORS = 3
ISO_UTC = "%Y-%m-%dT%H:%M:%SZ"

SPRINT_FORECAST_COLUMNS = [
    "run_id", "run_at", "forecast_key", "sprint_id", "project", "team", "team_key", "iteration",
    "start", "end", "cutoff", "status", "basis", "scored_as_of", "elapsed_share",
    "n_items", "committed_points", "n_unestimated", "trailing_velocity", "load_ratio",
    "p_full", "p_80", "expected", "p10", "p50", "p90",
    "items_done_so_far", "points_done_so_far", "pct_done_so_far", "data_as_of", "model_trained_at",
]
RISK_FACTOR_COLUMNS = [f"risk_factor_{k}" for k in range(1, N_RISK_FACTORS + 1)]
ITEM_FORECAST_COLUMNS = [
    "run_id", "forecast_key", "sprint_id", "item_id", "type", "state_category_at_commit", "points",
    "is_unestimated", "carryover_count", "p_done", *RISK_FACTOR_COLUMNS,
    "state_category_now", "in_sprint_now", "done_now",
]
ITEM_HISTORY_COLUMNS = [
    "sprint_id", "item_id", "type", "points", "is_unestimated", "carryover_count", "added_mid", "assignee", "done",
    "state_category_at_end",
]
CYCLE_FILE = "cycle.csv"
CYCLE_STATES_FILE = "cycle_states.csv"
CYCLE_FILE_COLUMNS = [
    "item_id", "project", "team", "team_key", "type", "iteration", "assignee",
    "points_at_start", "points", "re_estimated", "started", "closed", "days",
]
UNKNOWN_USER = "Unknown user"


@dataclass
class ExportResult:
    run_id: str
    out: Path
    sprints: pd.DataFrame  # the sprint forecast rows of this run
    files: list[Path] = field(default_factory=list)


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """UTF-8 CSV with UTC timestamps as 2024-06-10T05:00:00Z, written to a temp file and renamed into place so a
    reader (Power BI, a sync client) never sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            values = out[col] if out[col].dt.tz is not None else out[col].dt.tz_localize("UTC")
            out[col] = values.dt.tz_convert("UTC").dt.strftime(ISO_UTC)
    tmp = path.with_name(path.name + ".tmp")
    out.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(tmp, path)
    return path


def _copy(src: Path, dst: Path) -> Path:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return dst


def select_sprints(cal: pd.DataFrame, now) -> pd.DataFrame:
    """Team sprints running at `now` plus each team's next sprint, with status 'running' or 'upcoming'."""
    now = to_utc(now)
    running = cal[(cal["start"] <= now) & (cal["end"] > now)].assign(status="running")
    upcoming = cal[cal["start"] > now].sort_values(["start", "sprint_id"], kind="mergesort")
    upcoming = upcoming.groupby("team_key", sort=False).head(1).assign(status="upcoming")
    chosen = pd.concat([running, upcoming], ignore_index=True)
    return chosen.sort_values(["start", "sprint_id"], kind="mergesort").reset_index(drop=True)


def progress_at(cache: CacheData, items: pd.DataFrame, t, done_categories: list[str]) -> pd.DataFrame:
    """Where each committed item stands at `t` (columns item_id, iteration): its state category, whether it is
    still in the sprint's iteration, and whether it counts as done (still in the sprint and in a done category)."""
    keys = pd.DataFrame({"item_id": items["item_id"].to_numpy(), "t": to_utc(t)})
    now = as_of_many(cache.revisions, keys, "t")
    in_sprint = now["iteration"].to_numpy() == items["iteration"].to_numpy()
    done = in_sprint & now["state_category"].isin(done_categories).to_numpy()
    return pd.DataFrame(
        {"state_category_now": now["state_category"].to_numpy(), "in_sprint_now": in_sprint, "done_now": done},
        index=items.index,
    )


def assignee_names(keys: pd.Series, users: pd.DataFrame) -> pd.Series:
    """Display name for each Analytics user key: None when unassigned, UNKNOWN_USER for a key with no name."""
    names = keys.map(dict(zip(users["user_sk"], users["name"])))
    names = names.where(names.notna() | keys.isna(), UNKNOWN_USER)
    return names.astype(object).where(names.notna(), None)


def item_history(history, users: pd.DataFrame) -> pd.DataFrame:
    """Every item in a reconstructed sprint: committed at the cutoff, or added after it and still there at the end
    (added_mid). Each with who held it at the sprint's end and whether it was done by then (empty until it ends)."""
    def rows(items: pd.DataFrame, holder: str, added: bool) -> pd.DataFrame:
        return pd.DataFrame({
            "start": items["start"],
            "sprint_id": items["sprint_id"],
            "item_id": items["item_id"].astype("int64"),
            "type": items["type"],
            "points": items["points"],
            "is_unestimated": items["is_unestimated"].astype(bool),
            "carryover_count": items["carryover_count"].astype("int64"),
            "added_mid": added,
            "assignee": assignee_names(items[holder], users),
            "done": items["done"].astype("float64"),
            "state_category_at_end": items["state_category_at_end"],
        })
    frames = [rows(history.items, "assigned_at_end_sk", False), rows(history.added, "assigned_to_sk", True)]
    out = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else frames[0]
    out = out.sort_values(["start", "sprint_id", "added_mid", "item_id"], kind="mergesort")
    return out[ITEM_HISTORY_COLUMNS].reset_index(drop=True)


def cycle_files(cache: CacheData, users: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every finished item's cycle time (first active state to done) and its business days in each state."""
    cy = build_cycle(cache, work_item_types=settings["work_item_types"], done_categories=settings["done_categories"])
    items = cy.items.assign(
        item_id=cy.items["item_id"].astype("int64"),
        assignee=assignee_names(cy.items["assigned_to_sk"], users),
        re_estimated=cy.items["re_estimated"].astype(bool),
    )
    return items[CYCLE_FILE_COLUMNS], cy.states[CYCLE_STATE_COLUMNS]


def _item_rows(fc, scored: ScoredSprint, progress: pd.DataFrame, run_id: str, key: str) -> pd.DataFrame:
    rows = scored.items
    contrib = contributions(fc.item_model, rows)
    factors = [describe_drivers(contrib.loc[i], rows.loc[i], k=N_RISK_FACTORS) for i in rows.index]
    frame = pd.DataFrame({
        "run_id": run_id,
        "forecast_key": key,
        "sprint_id": rows["sprint_id"],
        "item_id": rows["item_id"].astype("int64"),
        "type": rows["type"],
        "state_category_at_commit": rows["state_category_at_commit"],
        "points": rows["points"],
        "is_unestimated": rows["is_unestimated"].astype(bool),
        "carryover_count": rows["carryover_count"].astype("int64"),
        "p_done": rows["p"],
    }, index=rows.index)
    for k, col in enumerate(RISK_FACTOR_COLUMNS):
        frame[col] = [f[k] if k < len(f) else None for f in factors]
    return frame.join(progress)[ITEM_FORECAST_COLUMNS]


def _sprint_row(scored: ScoredSprint, items: pd.DataFrame, *, run_id, now, key, status, basis, t, data_as_of, trained_at):
    s = scored.sprint
    committed = float(s["committed_points"])
    done_points = float((items["points"] * items["done_now"]).sum())
    v = scored.velocity
    length = (s["end"] - s["start"]) / pd.Timedelta(days=1)
    elapsed = (now - s["start"]) / pd.Timedelta(days=1)
    return {
        "run_id": run_id, "run_at": now, "forecast_key": key,
        **{c: s[c] for c in ("sprint_id", "project", "team", "team_key", "iteration", "start", "end", "cutoff")},
        "status": status, "basis": basis, "scored_as_of": t,
        "elapsed_share": float(np.clip(elapsed / length, 0.0, 1.0)) if length > 0 else math.nan,
        "n_items": int(s["n_items"]), "committed_points": committed, "n_unestimated": int(s["n_unestimated"]),
        "trailing_velocity": v, "load_ratio": committed / v if v > 0 else math.nan,
        **scored.summary,
        "items_done_so_far": int(items["done_now"].sum()), "points_done_so_far": done_points,
        "pct_done_so_far": done_points / committed if committed > 0 else math.nan,
        "data_as_of": data_as_of, "model_trained_at": trained_at,
    }


def export_forecasts(
    cache: CacheData, bundle: dict, out: Path, *, now=None, backtest_csv: Path | None = None,
) -> ExportResult:
    """Forecast every running team sprint (scored at its commit cutoff) and each team's next sprint (scored on
    its scope now). Appends sprint_forecasts/<run_id>.csv and item_forecasts/<run_id>.csv; replaces sprints.csv
    (every reconstructed sprint and its outcome), items.csv (every committed or added item, who held it at the end
    and its outcome), cycle.csv and cycle_states.csv (every finished item's cycle time and its time per state) and,
    when `backtest_csv` exists, backtest.csv."""
    out = Path(out)
    now = pd.Timestamp.now(tz="UTC").floor("s") if now is None else to_utc(now)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    fc = bundle["forecaster"]
    settings = {k: bundle[k] for k in ("work_item_types", "done_categories", "commit_grace_days")}
    trained_at = to_utc(bundle["trained_at"]) if bundle.get("trained_at") else pd.NaT
    history = build_sprints(cache, **settings)
    chosen = select_sprints(sprint_calendar(cache, settings["commit_grace_days"]), now)
    sprint_rows, item_frames = [], []
    for iteration, group in chosen.groupby("iteration", sort=False):
        cutoff = group["cutoff"].iloc[0]
        t, basis = (cutoff, "commit cutoff") if now >= cutoff else (now, "now")
        status = group.set_index("sprint_id")["status"]
        for scored in score_iteration(fc, cache, history, iteration=iteration, t=t, **settings):
            sid = scored.sprint["sprint_id"]
            if sid not in status.index:
                continue
            key = f"{run_id}|{sid}"
            progress = progress_at(cache, scored.items, now, settings["done_categories"])
            items = _item_rows(fc, scored, progress, run_id, key)
            item_frames.append(items)
            sprint_rows.append(_sprint_row(
                scored, items, run_id=run_id, now=now, key=key, status=status[sid], basis=basis, t=t,
                data_as_of=cache.extracted_at.get(scored.sprint["project"], pd.NaT), trained_at=trained_at,
            ))
    sprints = pd.DataFrame(sprint_rows, columns=SPRINT_FORECAST_COLUMNS)
    items = pd.concat(item_frames, ignore_index=True) if item_frames else pd.DataFrame(columns=ITEM_FORECAST_COLUMNS)
    files = [
        write_csv(sprints, out / SPRINT_FORECASTS_DIR / f"{run_id}.csv"),
        write_csv(items, out / ITEM_FORECASTS_DIR / f"{run_id}.csv"),
        write_csv(history.sprints[SPRINT_COLUMNS], out / SPRINTS_FILE),
        write_csv(item_history(history, cache.users), out / ITEM_HISTORY_FILE),
    ]
    cycle, cycle_states = cycle_files(cache, cache.users, settings)
    files += [write_csv(cycle, out / CYCLE_FILE), write_csv(cycle_states, out / CYCLE_STATES_FILE)]
    if backtest_csv is not None and Path(backtest_csv).exists():
        files.append(_copy(Path(backtest_csv), out / BACKTEST_FILE))
    return ExportResult(run_id, out, sprints, files)
