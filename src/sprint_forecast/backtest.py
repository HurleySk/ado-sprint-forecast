"""Expanding-window backtest at each checkpoint of a sprint: model A (plus LR item metrics), the progress reference,
model C and the team-mean reference."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from sprint_forecast.baseline import team_mean_samples, velocity_bootstrap
from sprint_forecast.features import team_history
from sprint_forecast.model import P_CLIP, predict_proba, predict_proba_lr
from sprint_forecast.rollup import FULL_TOL, N_DRAWS, fit_forecaster, forecast_sprint, simulate_pct_done, summarize
from sprint_forecast.sprints import SprintData

SPRINT_MODELS = ("a", "progress", "c", "team_mean")
METRICS = ["crps", "pinball_10", "pinball_50", "pinball_90", "brier_full", "brier_80", "coverage", "mae"]
TARGET_COLUMNS = ["sprint_id", "project", "team", "team_key", "iteration", "start", "n_items", "committed_points"]
ITEM_ROW_COLUMNS = ["sprint_id", "team_key", "checkpoint", "item_id", "is_added", "y", "p_a", "p_lr"]
SUMMARY_COLUMNS = ["model", "checkpoint", "team_key", "n_sprints"] + METRICS
ITEM_SUMMARY_COLUMNS = ["model", "checkpoint", "scope", "n_items", "brier", "log_loss", "auc"]
CALIBRATION_COLUMNS = ["range", "n", "mean_p", "frac_done"]
LOTO_COLUMNS = ["team_key", "n_sprints", "crps", "mae", "coverage", "item_auc"]


class LeakageError(RuntimeError):
    """A training sprint did not end before the target sprint started."""


@dataclass
class BacktestResult:
    sprint_rows: pd.DataFrame  # one row per target sprint, model and checkpoint
    item_rows: pd.DataFrame  # ITEM_ROW_COLUMNS: model A's and LR's item probabilities at each checkpoint
    summary: pd.DataFrame  # SUMMARY_COLUMNS: mean metrics by model and checkpoint, overall ("(all)") and per team
    item_summary: pd.DataFrame  # ITEM_SUMMARY_COLUMNS: by checkpoint, committed and added items apart
    calibration: pd.DataFrame  # model A at the commit cutoff
    calibration_pooled: pd.DataFrame  # model A over every checkpoint
    loto: pd.DataFrame
    skipped: dict = field(default_factory=dict)


def crps_samples(samples: np.ndarray, y: float) -> float:
    """CRPS = E|X - y| - 0.5 E|X - X'|, using the sorted-sample identity for the second term."""
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    i = np.arange(1, n + 1)
    return float(np.mean(np.abs(x - y)) - np.sum((2 * i - n - 1) * x) / n**2)


def pinball(q: float, y: float, tau: float) -> float:
    d = y - q
    return float(max(tau * d, (tau - 1) * d))


def sprint_metrics(samples: np.ndarray, actual: float) -> dict:
    s = summarize(samples)
    full = float(actual >= 1.0 - FULL_TOL)
    at_80 = float(actual >= 0.8 - FULL_TOL)
    return {
        **s,
        "actual": float(actual),
        "crps": crps_samples(samples, actual),
        "pinball_10": pinball(s["p10"], actual, 0.1),
        "pinball_50": pinball(s["p50"], actual, 0.5),
        "pinball_90": pinball(s["p90"], actual, 0.9),
        "brier_full": (s["p_full"] - full) ** 2,
        "brier_80": (s["p_80"] - at_80) ** 2,
        "coverage": float(s["p10"] <= actual <= s["p90"]),
        "mae": abs(s["expected"] - actual),
    }


def check_no_leakage(train: pd.DataFrame, target_start: pd.Timestamp) -> None:
    if len(train) and (train["end"] >= target_start).any():
        late = train.loc[train["end"] >= target_start, "sprint_id"].iloc[0]
        raise LeakageError(f"training sprint {late!r} did not end before the target started at {target_start}")


def select_targets(
    sprints: pd.DataFrame, min_history: int, project: str | None = None, team: str | None = None,
) -> pd.DataFrame:
    """Sprints with a known outcome and at least `min_history` earlier sprints (end < start) of their team."""
    s = sprints[sprints["pct_done"].notna()].sort_values(["start", "sprint_id"], kind="mergesort")
    prior = team_history(s, s).set_index("sprint_id")["team_sprint_index"]
    s = s[s["sprint_id"].map(prior) >= min_history]
    if project is not None:
        s = s[s["project"] == project]
    if team is not None:
        s = s[s["team"] == team]
    return s.reset_index(drop=True)


def _item_block(fc, items: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "sprint_id": items["sprint_id"].to_numpy(),
        "team_key": items["team_key"].to_numpy(),
        "checkpoint": items["checkpoint"].to_numpy(dtype=float),
        "item_id": items["item_id"].to_numpy(),
        "is_added": items["is_added"].to_numpy(dtype=int),
        "y": items["y"].to_numpy(),
        "p_a": p,
        "p_lr": predict_proba_lr(fc.item_model, items),
    })[ITEM_ROW_COLUMNS]


def _summaries(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    keys = ["model", "checkpoint"]
    overall = rows.groupby(keys)[METRICS].mean().assign(team_key="(all)")
    overall["n_sprints"] = rows.groupby(keys).size()
    per_team = rows.groupby(keys + ["team_key"])[METRICS].mean()
    per_team["n_sprints"] = rows.groupby(keys + ["team_key"]).size()
    out = pd.concat([overall.reset_index(), per_team.reset_index()], ignore_index=True)
    return out[SUMMARY_COLUMNS]


def item_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), P_CLIP, 1 - P_CLIP)
    both = len(np.unique(y)) == 2
    return {
        "n_items": len(y),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else float("nan"),
        "auc": float(roc_auc_score(y, p)) if both else float("nan"),
    }


def _item_summaries(item_rows: pd.DataFrame) -> pd.DataFrame:
    """Item metrics of model A and LR per checkpoint, for committed items and items added after the cutoff apart."""
    out = []
    for (f, added), block in item_rows.groupby(["checkpoint", "is_added"], sort=True):
        for name, col in (("a", "p_a"), ("lr", "p_lr")):
            out.append({
                "model": name, "checkpoint": f, "scope": "added" if added else "committed",
                **item_metrics(block["y"].to_numpy(), block[col].to_numpy()),
            })
    return pd.DataFrame(out, columns=ITEM_SUMMARY_COLUMNS)


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    frame["bin"] = np.clip((frame["p"] * bins).astype(int), 0, bins - 1)
    table = frame.groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"), frac_done=("y", "mean")).reset_index()
    table["range"] = [f"{b / bins:.1f}-{(b + 1) / bins:.1f}" for b in table["bin"]]
    return table[CALIBRATION_COLUMNS]


def leave_one_team_out(
    sprints: pd.DataFrame, frame: pd.DataFrame, held_out: list[str], n_draws: int, seed: int,
) -> pd.DataFrame:
    """Cross-team generalization check (not time-ordered): train on the other teams' checkpoint rows, predict the
    held-out team's sprints at the commit cutoff."""
    rows = []
    known = sprints[sprints["pct_done"].notna()]
    for team_key in held_out:
        try:
            fc = fit_forecaster(frame[frame["team_key"] != team_key], seed=seed)
        except ValueError:
            continue
        ys, ps, metrics = [], [], []
        for t in known[known["team_key"] == team_key].itertuples(index=False):
            items = frame[(frame["sprint_id"] == t.sprint_id) & (frame["checkpoint"] == 0.0)]
            p, samples = forecast_sprint(fc, items, n_draws, seed)
            metrics.append(sprint_metrics(samples, t.pct_done))
            ys.append(items["y"].to_numpy())
            ps.append(p)
        if not metrics:
            continue
        m = pd.DataFrame(metrics)
        rows.append({
            "team_key": team_key, "n_sprints": len(m), "crps": m["crps"].mean(), "mae": m["mae"].mean(),
            "coverage": m["coverage"].mean(), "item_auc": item_metrics(np.concatenate(ys), np.concatenate(ps))["auc"],
        })
    return pd.DataFrame(rows, columns=LOTO_COLUMNS)


def run_backtest(
    sd: SprintData,
    frame: pd.DataFrame,
    progress: pd.DataFrame,
    *,
    models: tuple[str, ...] = SPRINT_MODELS,
    min_history: int = 8,
    retrain_every: int = 4,
    n_draws: int = N_DRAWS,
    seed: int = 0,
    project: str | None = None,
    team: str | None = None,
    loto: bool = True,
) -> BacktestResult:
    """Score each target sprint at every checkpoint of `frame` (checkpoint rows; `progress` holds each sprint's
    committed points done by each checkpoint): model A at every checkpoint, the progress reference after the cutoff,
    model C and the team mean at the cutoff. Model A trains on the rows of sprints that ended before the target
    started."""
    if "progress" in models and "a" not in models:
        raise ValueError("the progress reference needs model a")
    frame = frame[frame["y"].notna()]
    done_by = progress.set_index(["sprint_id", "checkpoint"])["done_points"]
    checkpoints = sorted(frame["checkpoint"].unique())
    targets = select_targets(sd.sprints, min_history, project, team)
    sprint_rows, item_blocks = [], []
    skipped = {m: 0 for m in SPRINT_MODELS}
    fc, train = None, None
    for i, t in enumerate(targets.itertuples(index=False)):
        rows = frame[frame["sprint_id"] == t.sprint_id]
        day1 = rows[rows["checkpoint"] == 0.0]
        base = {c: getattr(t, c) for c in TARGET_COLUMNS}
        total = float(t.committed_points)
        if "a" in models:
            if i % max(1, retrain_every) == 0:
                train = frame[frame["end"] < t.start]
                try:
                    fc = fit_forecaster(train, seed=seed)
                except ValueError:
                    fc = None
            if fc is None:
                skipped["a"] += 1
                skipped["progress"] += int("progress" in models)
            else:
                check_no_leakage(train, t.start)
                p0 = pd.Series(predict_proba(fc.item_model, day1), index=day1["item_id"].to_numpy())
                for f in checkpoints:
                    items = rows[rows["checkpoint"] == f]
                    done = float(done_by.get((t.sprint_id, f), 0.0))
                    p, samples = forecast_sprint(fc, items, n_draws, seed, done_points=done, total_points=total)
                    sprint_rows.append({
                        **base, "checkpoint": f, "model": "a", **sprint_metrics(samples, t.pct_done),
                        "sigma": fc.sigma, "train_sprints": train["sprint_id"].nunique(),
                        "train_end_max": train["end"].max(),
                    })
                    if len(items):
                        item_blocks.append(_item_block(fc, items, p))
                    if "progress" in models and f > 0:
                        still = items[items["is_added"] == 0]
                        samples = simulate_pct_done(
                            still["item_id"].map(p0).to_numpy(dtype=float),
                            still["points_at_commit"].to_numpy(dtype=float),
                            fc.sigma, n_draws, seed, done_points=done, total_points=total,
                        )
                        sprint_rows.append({
                            **base, "checkpoint": f, "model": "progress", **sprint_metrics(samples, t.pct_done),
                        })
        if "c" in models:
            samples = velocity_bootstrap(sd.sprints, pd.Series(t._asdict()), n_draws, seed)
            if samples is None:
                skipped["c"] += 1
            else:
                sprint_rows.append({**base, "checkpoint": 0.0, "model": "c", **sprint_metrics(samples, t.pct_done)})
        if "team_mean" in models:
            samples = team_mean_samples(float(day1["team_trailing_completion"].iloc[0]))
            if samples is None:
                skipped["team_mean"] += 1
            else:
                sprint_rows.append({
                    **base, "checkpoint": 0.0, "model": "team_mean", **sprint_metrics(samples, t.pct_done),
                })
    rows = pd.DataFrame(sprint_rows)
    item_rows = pd.concat(item_blocks, ignore_index=True) if item_blocks else pd.DataFrame(columns=ITEM_ROW_COLUMNS)
    at_cutoff = item_rows[item_rows["checkpoint"] == 0.0]
    empty_cal = pd.DataFrame(columns=CALIBRATION_COLUMNS)
    calibration = calibration_table(at_cutoff["y"], at_cutoff["p_a"]) if len(at_cutoff) else empty_cal
    pooled = calibration_table(item_rows["y"], item_rows["p_a"]) if len(item_rows) else empty_cal
    held_out = sorted(targets["team_key"].unique())
    loto_rows = pd.DataFrame(columns=LOTO_COLUMNS)
    if loto and "a" in models and sd.sprints["team_key"].nunique() >= 2 and held_out:
        loto_rows = leave_one_team_out(sd.sprints, frame, held_out, n_draws, seed)
    return BacktestResult(
        rows, item_rows, _summaries(rows), _item_summaries(item_rows), calibration, pooled, loto_rows, skipped,
    )


def format_report(result: BacktestResult) -> str:
    """Plain-text report: sprint metrics by model and checkpoint (overall, then per team), the verdicts, item metrics
    by checkpoint, calibration at the cutoff and pooled, LOTO."""
    if result.sprint_rows.empty:
        return "No sprints qualified for the backtest (try a smaller --min-history)."

    def fmt(v):
        return f"{v:.3f}"

    lines = [
        "Sprint-level metrics by checkpoint (share of the way from the commit cutoff to the sprint's end; "
        "lower is better, except coverage: target 0.80)",
        result.summary.to_string(index=False, float_format=fmt),
    ]
    overall = result.summary[result.summary["team_key"] == "(all)"].set_index(["model", "checkpoint"])
    if ("a", 0.0) in overall.index and ("c", 0.0) in overall.index:
        a, c = overall.loc[("a", 0.0), "crps"], overall.loc[("c", 0.0), "crps"]
        verdict = "beats" if a < c else "does NOT beat"
        lines.append(f"\nAt the commit cutoff, model A {verdict} baseline C on CRPS ({a:.3f} vs {c:.3f}).")
    later = sorted(f for m, f in overall.index if m == "progress")
    if later:
        lines.append("")
    for f in later:
        a, p = overall.loc[("a", f)], overall.loc[("progress", f)]
        lines.append(
            f"At {f:.0%} of the sprint, model A vs progress + day-1 odds: "
            f"MAE {a['mae']:.3f} vs {p['mae']:.3f}, CRPS {a['crps']:.3f} vs {p['crps']:.3f}."
        )
    if len(result.item_summary):
        lines.append("\nItem-level metrics by checkpoint (committed items, and items added after the cutoff)")
        lines.append(result.item_summary.to_string(index=False, float_format=fmt))
        lines.append("\nCalibration of model A at the commit cutoff (10 bins)")
        lines.append(result.calibration.to_string(index=False, float_format=fmt))
        lines.append("\nCalibration of model A over every checkpoint (10 bins)")
        lines.append(result.calibration_pooled.to_string(index=False, float_format=fmt))
    if len(result.loto):
        lines.append("\nLeave-one-team-out at the commit cutoff (cross-team generalization check, not time-ordered)")
        lines.append(result.loto.to_string(index=False, float_format=fmt))
    if any(result.skipped.values()):
        lines.append(f"\nSkipped targets per model: {result.skipped}")
    return "\n".join(lines)
