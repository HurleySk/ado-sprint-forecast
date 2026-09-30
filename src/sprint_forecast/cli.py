"""Click CLI: init, extract, data, backtest, train, predict, export, demo."""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import click
import joblib
import pandas as pd

from sprint_forecast import __version__
from sprint_forecast.analytics import fetch_titles, make_fetch_json
from sprint_forecast.backtest import SPRINT_MODELS, format_report, run_backtest
from sprint_forecast.cache import CacheData, connect, load_cache
from sprint_forecast.checkpoints import checkpoint_progress
from sprint_forecast.config import (
    AUTH_METHODS,
    DEFAULT_DONE,
    DEFAULT_TYPES,
    Config,
    config_path,
    data_dir,
    load_config,
    parse_done_categories,
    save_config,
)
from sprint_forecast.export import export_forecasts, write_csv
from sprint_forecast.extract import PROJECT_ERRORS, describe_error, extract_all
from sprint_forecast.features import FEATURE_VERSION, build_checkpoint_frame
from sprint_forecast.forecast import check_bundle, score_iteration
from sprint_forecast.model import contributions, describe_drivers
from sprint_forecast.rollup import fit_forecaster
from sprint_forecast.sprints import SprintData, build_sprints, data_report, sprint_calendar
from sprint_forecast.synth import generate

CACHE_FILE = "cache.db"
MODEL_FILE = "model.joblib"
BACKTEST_FILE = "backtest.csv"
EXPORT_DIR = "export"
DEMO_DIR = "demo"
TITLE_WIDTH = 50


@dataclass
class Settings:
    work_item_types: list[str]
    done_categories: list[str]
    commit_grace_days: float = 1.0
    close_grace_hours: float = 0.0


def _settings(root: Path, done_categories: str | None) -> Settings:
    path = config_path(root)
    cfg = load_config(path) if path.exists() else None
    try:
        done = parse_done_categories(done_categories) if done_categories else None
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="--done-categories") from None
    return Settings(
        work_item_types=cfg.work_item_types if cfg else list(DEFAULT_TYPES),
        done_categories=done or (cfg.done_categories if cfg else list(DEFAULT_DONE)),
        commit_grace_days=cfg.commit_grace_days if cfg else 1.0,
        close_grace_hours=cfg.close_grace_hours if cfg else 0.0,
    )


def _require_config(root: Path) -> Config:
    try:
        return load_config(config_path(root))
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e)) from None


def _load_cache(workdir: Path) -> CacheData:
    path = workdir / CACHE_FILE
    if not path.exists():
        raise click.ClickException(f"no cache at {path}; run `sprint-forecast extract` (or try `sprint-forecast demo`)")
    conn = connect(path)
    try:
        return load_cache(conn)
    finally:
        conn.close()


def _build(cache: CacheData, s: Settings) -> SprintData:
    return build_sprints(
        cache, work_item_types=s.work_item_types, done_categories=s.done_categories,
        commit_grace_days=s.commit_grace_days, close_grace_hours=s.close_grace_hours,
    )


def _checkpoint_frame(cache: CacheData, sd: SprintData, s: Settings) -> pd.DataFrame:
    return build_checkpoint_frame(cache, sd, work_item_types=s.work_item_types, done_categories=s.done_categories)


def _pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f}%"


def format_data_report(report: dict) -> str:
    grace = report.get("close_grace_hours", 0.0)
    lines = ["Projects"]
    for p in report["projects"]:
        fallback = "  (project-level sprints: no team subscribes to dated iterations)" if p["fallback"] else ""
        lines.append(
            f"  {p['project']}: {p['teams']} teams ({p['teams_running_sprints']} running sprints), "
            f"{p['dated_iterations']} dated / {p['undated_iterations']} undated iterations, "
            f"{p['sprints']} sprints, {p['committed_items']} committed items{fallback}"
        )
    lines += [
        f"Sprints reconstructed: {report['n_sprints']} ({report['dropped_empty_sprints']} empty sprints dropped)",
        f"Sprints not ended at extraction (outcome unknown, not used for training): {report['open_sprints']}",
        f"Committed items: {report['n_committed_items']}; unestimated share {_pct(report['unestimated_share'])}",
        f"Unassigned items (no unique team match, excluded): {report['unassigned_items']}",
        f"Items added mid-sprint (after the commit cutoff; scored per item, not part of committed scope): "
        f"{report['added_mid_sprint']}",
        "Done-state mix at sprint end: "
        + (", ".join(f"{k} {v}" for k, v in sorted(report["end_state_mix"].items())) or "n/a"),
        f"Resolved share of Resolved+Completed at end: {_pct(report['resolved_share_of_resolved_or_completed'])}"
        " (high values suggest --done-categories Resolved,Completed)",
        f"Items left out by title: {report.get('excluded_items', 0)}",
    ]
    if grace > 0:
        lines.append(
            f"Done also counts items closed up to {grace:g} hours after the end while still in the sprint: "
            f"{_pct(report.get('graced_share', float('nan')))} of done committed points"
        )
    return "\n".join(lines)


def _data(workdir: Path, s: Settings) -> None:
    cache = _load_cache(workdir)
    report = data_report(cache, _build(cache, s))
    report["excluded_items"] = len(cache.excluded_items)
    click.echo(format_data_report(report))


def _backtest(workdir: Path, s: Settings, **kwargs) -> None:
    cache = _load_cache(workdir)
    sd = _build(cache, s)
    progress = checkpoint_progress(cache, sd, done_categories=s.done_categories)
    result = run_backtest(sd, _checkpoint_frame(cache, sd, s), progress, **kwargs)
    click.echo(format_report(result))
    out = workdir / BACKTEST_FILE
    write_csv(result.sprint_rows, out)
    click.echo(f"\nPer-sprint rows written to {out}")


def _train(workdir: Path, s: Settings) -> dict:
    cache = _load_cache(workdir)
    sd = _build(cache, s)
    frame = _checkpoint_frame(cache, sd, s)
    try:
        fc = fit_forecaster(frame, seed=0)
    except ValueError as e:
        raise click.ClickException(f"cannot train: {e}") from None
    known = frame[frame["y"].notna()]
    bundle = {
        "forecaster": fc,
        "feature_version": FEATURE_VERSION,
        "work_item_types": s.work_item_types,
        "done_categories": s.done_categories,
        "commit_grace_days": s.commit_grace_days,
        "close_grace_hours": s.close_grace_hours,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_sprints": int(len(sd.sprints)),
        "n_items": int(len(known.drop_duplicates(["sprint_id", "item_id"]))),
        "n_rows": int(len(known)),
        "version": __version__,
    }
    path = workdir / MODEL_FILE
    joblib.dump(bundle, path)
    click.echo(
        f"Trained on {bundle['n_sprints']} sprints / {bundle['n_items']} items / {bundle['n_rows']} checkpoint rows "
        f"(calibration: {fc.item_model.calibration}, sprint shock sigma = {fc.sigma:.2f}) -> {path}"
    )
    per_checkpoint = known.groupby("checkpoint").size()
    click.echo("  rows per checkpoint: " + ", ".join(f"{f:.0%} {n}" for f, n in per_checkpoint.items()))
    return bundle


def _load_model(workdir: Path) -> dict:
    path = workdir / MODEL_FILE
    if not path.exists():
        raise click.ClickException(f"no model at {path}; run `sprint-forecast train` first")
    bundle = joblib.load(path)
    try:
        check_bundle(bundle)
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    return bundle


def _title_lookup(root: Path) -> Callable[[list[int]], dict[int, str]] | None:
    """Title fetcher for display (never cached); None when there is no config to authenticate with."""
    path = config_path(root)
    if not path.exists():
        return None
    cfg = load_config(path)

    def lookup(ids: list[int]) -> dict[int, str]:
        return fetch_titles(make_fetch_json(cfg.auth_method, cfg.pat), cfg.org_url, ids)

    return lookup


def _truncate(text: str, width: int = TITLE_WIDTH) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def _predict(
    workdir: Path,
    *,
    iteration: str,
    team: str | None,
    as_of: str,
    top: int,
    title_lookup: Callable[[list[int]], dict[int, str]] | None,
    now: pd.Timestamp | None = None,
) -> list[dict]:
    bundle = _load_model(workdir)
    fc = bundle["forecaster"]
    s = Settings(bundle["work_item_types"], bundle["done_categories"], bundle["commit_grace_days"],
                 bundle.get("close_grace_hours", 0.0))
    cache = _load_cache(workdir)
    history = _build(cache, s)
    cal = sprint_calendar(cache, s.commit_grace_days)
    group = cal[cal["iteration"] == iteration]
    if group.empty:
        raise click.ClickException(f"no team runs a dated iteration with path {iteration!r}")
    cutoff, end = group["cutoff"].iloc[0], group["end"].iloc[0]
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    t = cutoff if as_of == "commit" else now
    kwargs = dict(
        iteration=iteration, work_item_types=s.work_item_types, done_categories=s.done_categories,
        commit_grace_days=s.commit_grace_days, team=team,
    )
    try:
        scored = score_iteration(fc, cache, history, t=t, **kwargs)
        day1 = {}
        if as_of == "now" and now >= cutoff:
            day1 = {
                sc.sprint["sprint_id"]: sc.summary for sc in score_iteration(fc, cache, history, t=cutoff, **kwargs)
            }
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    t = min(t, end)
    if not scored:
        label = "commit cutoff" if as_of == "commit" else ("sprint end" if now >= end else "now")
        click.echo(f"No committed items in {iteration} as of {label} ({t:%Y-%m-%d %H:%M} UTC).")
        return []
    titles: dict[int, str] = {}
    if title_lookup is not None and top > 0:
        try:
            titles = title_lookup(sorted(int(i) for sc in scored for i in sc.items["item_id"]))
        except Exception:  # titles are cosmetic; never fail a forecast over them
            titles = {}
    results = []
    for sc in scored:
        srow, summary, v = sc.sprint, sc.summary, sc.velocity
        committed = float(srow["committed_points"])
        if as_of == "commit":
            label = "commit cutoff"
        else:  # score_iteration stops at the sprint's end, or at the last extract when the end is not in it yet
            label = "sprint end" if sc.t >= end else ("now" if sc.t >= now else "last extract")
        click.echo(f"\n{srow['team_key']} | {iteration}")
        click.echo(
            f"  window {srow['start']:%Y-%m-%d} .. {srow['end']:%Y-%m-%d} UTC; "
            f"scored as of {label} ({sc.t:%Y-%m-%d %H:%M} UTC)"
        )
        click.echo(
            f"  committed: {int(srow['n_items'])} items, {committed:.1f} points "
            f"({int(srow['n_unestimated'])} unestimated)"
        )
        if sc.t >= cutoff:
            share = sc.done_points / committed if committed > 0 else math.nan
            click.echo(f"  done so far: {sc.done_points:.1f} of {committed:.1f} points ({_pct(share)})")
        if v > 0:
            click.echo(f"  load: {committed / v:.2f}x trailing velocity ({v:.1f} points)")
        else:
            click.echo("  load: n/a (no velocity history for this team)")
        click.echo(
            f"  P(full) {_pct(summary['p_full'])}   P(>=80%) {_pct(summary['p_80'])}   "
            f"expected {_pct(summary['expected'])}   p10/p50/p90 "
            f"{_pct(summary['p10'])} / {_pct(summary['p50'])} / {_pct(summary['p90'])}"
        )
        d1 = day1.get(srow["sprint_id"])
        if d1:
            click.echo(
                f"  day-1 forecast: expected {_pct(d1['expected'])}, p10/p50/p90 "
                f"{_pct(d1['p10'])} / {_pct(d1['p50'])} / {_pct(d1['p90'])}"
            )
        risky = sc.items.sort_values("p", kind="mergesort").head(top)
        if len(risky):
            click.echo("  riskiest items:")
            contrib = contributions(fc.item_model, risky)
            for idx, r in risky.iterrows():
                drivers = describe_drivers(contrib.loc[idx], r)
                title = _truncate(titles.get(int(r["item_id"]), "")) if titles else ""
                added = "  added" if r["is_added"] == 1 else ""
                click.echo(f"    #{int(r['item_id'])}  p={r['p']:.2f}  {r['points']:.1f} pts{added}  {title}".rstrip())
                if drivers:
                    click.echo(f"        why: {'; '.join(drivers)}")
        results.append({"sprint_id": srow["sprint_id"], **summary})
    return results


def _export(workdir: Path, out: Path, now: pd.Timestamp | None = None):
    bundle = _load_model(workdir)
    result = export_forecasts(_load_cache(workdir), bundle, out, now=now, backtest_csv=workdir / BACKTEST_FILE)
    if result.sprints.empty:
        click.echo("No running or upcoming team sprints with committed scope; wrote sprint outcomes only.")
    for r in result.sprints.itertuples(index=False):
        click.echo(
            f"{r.team_key} | {r.iteration}: {r.status}, scored as of {r.basis}; "
            f"P(full) {_pct(r.p_full)}, expected {_pct(r.expected)}, done so far {_pct(r.pct_done_so_far)}"
        )
    click.echo(f"Wrote {len(result.files)} files to {out}")
    return result


done_option = click.option(
    "--done-categories", "done_categories", default=None,
    help="Comma-separated StateCategory values that count as done, e.g. Resolved,Completed "
         "(default: from config, else Completed).",
)


@click.group()
@click.version_option(__version__, prog_name="sprint-forecast")
@click.option(
    "--root", type=click.Path(file_okay=False, path_type=Path), default=".", show_default=True,
    help="Directory that holds .sprint-forecast/.",
)
@click.pass_context
def main(ctx: click.Context, root: Path) -> None:
    """Forecast how much of a sprint's day-1 committed scope an Azure DevOps team will deliver."""
    ctx.obj = root


@main.command()
@click.option("--org", required=True, help="Organization URL, e.g. https://dev.azure.com/contoso")
@click.option("--projects", required=True, help='Comma-separated project names, or "*" for every project.')
@click.option("--auth", type=click.Choice(AUTH_METHODS), default="pat", show_default=True)
@click.option("--force", is_flag=True, help="Overwrite an existing config.")
@click.pass_obj
def init(root: Path, org: str, projects: str, auth: str, force: bool) -> None:
    """Write .sprint-forecast/config.toml."""
    path = config_path(root)
    if path.exists() and not force:
        raise click.ClickException(f"{path} already exists; pass --force to overwrite")
    names = ["*"] if projects.strip() == "*" else [p.strip() for p in projects.split(",") if p.strip()]
    if not names:
        raise click.BadParameter("no project names given", param_hint="--projects")
    save_config(Config(org_url=org.strip().rstrip("/"), projects=names, auth_method=auth), path)
    click.echo(f"Wrote {path}")
    if auth == "pat":
        click.echo("Set ADO_PAT (a PAT with Analytics read scope) before running `sprint-forecast extract`.")


@main.command()
@click.option("--project", "projects", multiple=True, help="Only extract this project (repeatable).")
@click.option("--full", is_flag=True, help="Drop and re-download the selected projects' revisions.")
@click.pass_obj
def extract(root: Path, projects: tuple[str, ...], full: bool) -> None:
    """Pull revisions, iterations and teams into .sprint-forecast/cache.db."""
    cfg = _require_config(root)
    try:
        fetch_json = make_fetch_json(cfg.auth_method, cfg.pat)
    except (ValueError, RuntimeError) as e:
        raise click.ClickException(str(e)) from None
    conn = connect(data_dir(root) / CACHE_FILE)
    try:
        results = extract_all(
            fetch_json, conn, cfg.org_url, list(projects) or cfg.projects,
            work_item_types=cfg.work_item_types, full=full, exclude_title_pattern=cfg.exclude_title_pattern,
            echo=click.echo,
        )
    except PROJECT_ERRORS as e:
        raise click.ClickException(f"cannot list projects: {describe_error(e)}") from None
    finally:
        conn.close()
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if r.failed)
    click.echo(f"Done: {len(results) - skipped - failed} projects extracted, {skipped} skipped, {failed} failed.")
    if failed:
        raise click.exceptions.Exit(1)


@main.command()
@done_option
@click.pass_obj
def data(root: Path, done_categories: str | None) -> None:
    """Data-quality report for the cached data."""
    _data(data_dir(root), _settings(root, done_categories))


@main.command()
@click.option("--project", default=None, help="Only score this project's sprints.")
@click.option("--team", default=None, help="Only score this team's sprints.")
@click.option("--model", "model_choice", type=click.Choice(["a", "c", "all"]), default="all", show_default=True)
@click.option("--min-history", type=click.IntRange(min=1), default=8, show_default=True)
@click.option("--retrain-every", type=click.IntRange(min=1), default=4, show_default=True)
@done_option
@click.pass_obj
def backtest(root, project, team, model_choice, min_history, retrain_every, done_categories) -> None:
    """Expanding-window backtest at each checkpoint of a sprint; writes .sprint-forecast/backtest.csv."""
    models = {"a": ("a", "progress", "team_mean"), "c": ("c", "team_mean"), "all": SPRINT_MODELS}[model_choice]
    _backtest(
        data_dir(root), _settings(root, done_categories), models=models, min_history=min_history,
        retrain_every=retrain_every, project=project, team=team,
    )


@main.command()
@done_option
@click.pass_obj
def train(root: Path, done_categories: str | None) -> None:
    """Train model A on every completed sprint; writes .sprint-forecast/model.joblib."""
    _train(data_dir(root), _settings(root, done_categories))


@main.command()
@click.option("--iteration", required=True, help="Iteration path, e.g. 'Alpha\\Sprint 12'.")
@click.option("--team", default=None, help="Only this team (default: every team running the iteration).")
@click.option(
    "--as-of", "as_of", type=click.Choice(["now", "commit"]), default="now", show_default=True,
    help="now: the sprint as it stands (done so far plus the forecast for what is still open; its end once over). "
         "commit: the day-1 view at the commit cutoff.",
)
@click.option("--top", type=click.IntRange(min=0), default=5, show_default=True, help="Riskiest items to list.")
@click.option("--titles/--no-titles", default=True, help="Fetch titles from ADO for display (never cached).")
@click.pass_obj
def predict(root: Path, iteration: str, team: str | None, as_of: str, top: int, titles: bool) -> None:
    """Forecast one iteration's committed scope."""
    lookup = _title_lookup(root) if titles else None
    _predict(data_dir(root), iteration=iteration, team=team, as_of=as_of, top=top, title_lookup=lookup)


@main.command()
@click.option(
    "--out", type=click.Path(file_okay=False, path_type=Path), default=None,
    help="Folder for the CSVs (default: .sprint-forecast/export). Point it at a synced SharePoint or OneDrive "
         "folder to feed Power BI.",
)
@click.pass_obj
def export(root: Path, out: Path | None) -> None:
    """Forecast running sprints and each team's next sprint; write Power BI-ready CSVs."""
    workdir = data_dir(root)
    _export(workdir, out or workdir / EXPORT_DIR)


@main.command()
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--n-sprints", type=click.IntRange(min=12), default=40, show_default=True)
@click.pass_obj
def demo(root: Path, seed: int, n_sprints: int) -> None:
    """Run the whole pipeline on synthetic data (no ADO needed) under .sprint-forecast/demo/."""
    workdir = data_dir(root) / DEMO_DIR
    generate(workdir / CACHE_FILE, seed=seed, n_sprints=n_sprints)
    s = Settings(list(DEFAULT_TYPES), list(DEFAULT_DONE))
    click.echo(f"Synthetic cache written to {workdir / CACHE_FILE}\n\n== data ==")
    _data(workdir, s)
    click.echo("\n== backtest ==")
    _backtest(workdir, s)
    click.echo("\n== train ==")
    _train(workdir, s)
    history = _build(_load_cache(workdir), s)
    red = history.sprints[history.sprints["team"] == "Team Red"]
    latest = red.sort_values("start").iloc[-1]["iteration"]
    click.echo("\n== predict ==")
    _predict(workdir, iteration=latest, team="Team Red", as_of="commit", top=3, title_lookup=None)


def run() -> None:
    """Console entry point. Keeps "*" and other wildcards literal on Windows (click would glob them against
    the current directory) and writes '?' for characters the terminal's encoding cannot show."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
    main(windows_expand_args=False)


if __name__ == "__main__":
    run()
