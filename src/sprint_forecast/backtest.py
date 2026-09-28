"""Expanding-window backtest of model A (plus LR item metrics), model C and the team-mean reference."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from sprint_forecast.baseline import team_mean_samples, velocity_bootstrap
from sprint_forecast.features import team_history
from sprint_forecast.model import P_CLIP, predict_proba_lr
from sprint_forecast.rollup import FULL_TOL, N_DRAWS, fit_forecaster, forecast_sprint, summarize
from sprint_forecast.sprints import SprintData

SPRINT_MODELS = ("a", "c", "team_mean")
METRICS = ["crps", "pinball_10", "pinball_50", "pinball_90", "brier_full", "brier_80", "coverage", "mae"]
TARGET_COLUMNS = ["sprint_id", "project", "team", "team_key", "iteration", "start", "n_items", "committed_points"]


class LeakageError(RuntimeError):
    """A training sprint did not end before the target sprint started."""


@dataclass
class BacktestResult:
    sprint_rows: pd.DataFrame
    item_rows: pd.DataFrame
    summary: pd.DataFrame
    item_summary: pd.DataFrame
    calibration: pd.DataFrame
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
        "item_id": items["item_id"].to_numpy(),
        "y": items["y"].to_numpy(),
        "p_a": p,
        "p_lr": predict_proba_lr(fc.item_model, items),
    })


def _summaries(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["model", "team_key", "n_sprints"] + METRICS)
    overall = rows.groupby("model")[METRICS].mean().assign(team_key="(all)")
    overall["n_sprints"] = rows.groupby("model").size()
    per_team = rows.groupby(["model", "team_key"])[METRICS].mean()
    per_team["n_sprints"] = rows.groupby(["model", "team_key"]).size()
    out = pd.concat([overall.reset_index(), per_team.reset_index()], ignore_index=True)
    return out[["model", "team_key", "n_sprints"] + METRICS]


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


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    frame["bin"] = np.clip((frame["p"] * bins).astype(int), 0, bins - 1)
    table = frame.groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"), frac_done=("y", "mean")).reset_index()
    table["range"] = [f"{b / bins:.1f}-{(b + 1) / bins:.1f}" for b in table["bin"]]
    return table[["range", "n", "mean_p", "frac_done"]]


def leave_one_team_out(
    sprints: pd.DataFrame, frame: pd.DataFrame, held_out: list[str], n_draws: int, seed: int,
) -> pd.DataFrame:
    """Cross-team generalization check (not time-ordered): train on the other teams, predict the held-out team."""
    rows = []
    known = sprints[sprints["pct_done"].notna()]
    for team_key in held_out:
        try:
            fc = fit_forecaster(frame[frame["team_key"] != team_key], seed=seed)
        except ValueError:
            continue
        ys, ps, metrics = [], [], []
        for t in known[known["team_key"] == team_key].itertuples(index=False):
            items = frame[frame["sprint_id"] == t.sprint_id]
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
    return pd.DataFrame(rows, columns=["team_key", "n_sprints", "crps", "mae", "coverage", "item_auc"])


def run_backtest(
    sd: SprintData,
    frame: pd.DataFrame,
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
    frame = frame[frame["y"].notna()]
    targets = select_targets(sd.sprints, min_history, project, team)
    sprint_rows, item_blocks = [], []
    skipped = {"a": 0, "c": 0, "team_mean": 0}
    fc, train = None, None
    for i, t in enumerate(targets.itertuples(index=False)):
        items = frame[frame["sprint_id"] == t.sprint_id]
        base = {c: getattr(t, c) for c in TARGET_COLUMNS}
        if "a" in models:
            if i % max(1, retrain_every) == 0:
                train = frame[frame["end"] < t.start]
                try:
                    fc = fit_forecaster(train, seed=seed)
                except ValueError:
                    fc = None
            if fc is None:
                skipped["a"] += 1
            else:
                check_no_leakage(train, t.start)
                p, samples = forecast_sprint(fc, items, n_draws, seed)
                sprint_rows.append({
                    **base, "model": "a", **sprint_metrics(samples, t.pct_done), "sigma": fc.sigma,
                    "train_sprints": train["sprint_id"].nunique(), "train_end_max": train["end"].max(),
                })
                item_blocks.append(_item_block(fc, items, p))
        if "c" in models:
            samples = velocity_bootstrap(sd.sprints, pd.Series(t._asdict()), n_draws, seed)
            if samples is None:
                skipped["c"] += 1
            else:
                sprint_rows.append({**base, "model": "c", **sprint_metrics(samples, t.pct_done)})
        if "team_mean" in models:
            samples = team_mean_samples(float(items["team_trailing_completion"].iloc[0]))
            if samples is None:
                skipped["team_mean"] += 1
            else:
                sprint_rows.append({**base, "model": "team_mean", **sprint_metrics(samples, t.pct_done)})
    rows = pd.DataFrame(sprint_rows)
    item_rows = pd.concat(item_blocks, ignore_index=True) if item_blocks else pd.DataFrame(
        columns=["sprint_id", "team_key", "item_id", "y", "p_a", "p_lr"])
    item_summary = pd.DataFrame([
        {"model": name, **item_metrics(item_rows["y"].to_numpy(), item_rows[col].to_numpy())}
        for name, col in (("a", "p_a"), ("lr", "p_lr"))
    ]) if len(item_rows) else pd.DataFrame(columns=["model", "n_items", "brier", "log_loss", "auc"])
    calibration = calibration_table(item_rows["y"], item_rows["p_a"]) if len(item_rows) else pd.DataFrame(
        columns=["range", "n", "mean_p", "frac_done"])
    held_out = sorted(targets["team_key"].unique())
    loto_rows = pd.DataFrame(columns=["team_key", "n_sprints", "crps", "mae", "coverage", "item_auc"])
    if loto and "a" in models and sd.sprints["team_key"].nunique() >= 2 and held_out:
        loto_rows = leave_one_team_out(sd.sprints, frame, held_out, n_draws, seed)
    return BacktestResult(rows, item_rows, _summaries(rows), item_summary, calibration, loto_rows, skipped)


def format_report(result: BacktestResult) -> str:
    """Plain-text report: sprint metrics per model (overall, then per team), item metrics, calibration, LOTO."""
    if result.sprint_rows.empty:
        return "No sprints qualified for the backtest (try a smaller --min-history)."
    lines = ["Sprint-level metrics (lower is better, except coverage: target 0.80)"]
    lines.append(result.summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    overall = result.summary[result.summary["team_key"] == "(all)"].set_index("model")
    if {"a", "c"} <= set(overall.index):
        a, c = overall.loc["a", "crps"], overall.loc["c", "crps"]
        verdict = "beats" if a < c else "does NOT beat"
        lines.append(f"\nModel A {verdict} baseline C on CRPS ({a:.3f} vs {c:.3f}).")
    if len(result.item_summary):
        lines.append("\nItem-level metrics")
        lines.append(result.item_summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        lines.append("\nCalibration of model A (10 bins)")
        lines.append(result.calibration.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(result.loto):
        lines.append("\nLeave-one-team-out (cross-team generalization check, not time-ordered)")
        lines.append(result.loto.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if any(result.skipped.values()):
        lines.append(f"\nSkipped targets per model: {result.skipped}")
    return "\n".join(lines)
