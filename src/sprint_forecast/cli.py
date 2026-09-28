"""Click CLI: init, extract, data, backtest, train, predict, demo."""
from __future__ import annotations

import math
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
from sprint_forecast.extract import extract_all
from sprint_forecast.features import build_features, build_features_for, team_history
from sprint_forecast.model import contributions, describe_drivers
from sprint_forecast.rollup import fit_forecaster, forecast_sprint, summarize
from sprint_forecast.sprints import SprintData, build_sprints, data_report, scope_at, sprint_calendar
from sprint_forecast.synth import generate

CACHE_FILE = "cache.db"
MODEL_FILE = "model.joblib"
BACKTEST_FILE = "backtest.csv"
DEMO_DIR = "demo"
TITLE_WIDTH = 50


@dataclass
class Settings:
    work_item_types: list[str]
    done_categories: list[str]
    commit_grace_days: float = 1.0


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
        commit_grace_days=s.commit_grace_days,
    )


def _pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f}%"


def format_data_report(report: dict) -> str:
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
        f"Committed items: {report['n_committed_items']}; unestimated share {_pct(report['unestimated_share'])}",
        f"Unassigned items (no unique team match, excluded): {report['unassigned_items']}",
        f"Items added mid-sprint (after the commit cutoff, excluded): {report['added_mid_sprint']}",
        "Done-state mix at sprint end: "
        + (", ".join(f"{k} {v}" for k, v in sorted(report["end_state_mix"].items())) or "n/a"),
        f"Resolved share of Resolved+Completed at end: {_pct(report['resolved_share_of_resolved_or_completed'])}"
        " (high values suggest --done-categories Resolved,Completed)",
    ]
    return "\n".join(lines)


def _data(workdir: Path, s: Settings) -> None:
    cache = _load_cache(workdir)
    click.echo(format_data_report(data_report(cache, _build(cache, s))))


def _backtest(workdir: Path, s: Settings, **kwargs) -> None:
    sd = _build(_load_cache(workdir), s)
    result = run_backtest(sd, build_features(sd), **kwargs)
    click.echo(format_report(result))
    out = workdir / BACKTEST_FILE
    result.sprint_rows.to_csv(out, index=False)
    click.echo(f"\nPer-sprint rows written to {out}")


def _train(workdir: Path, s: Settings) -> dict:
    sd = _build(_load_cache(workdir), s)
    frame = build_features(sd)
    try:
        fc = fit_forecaster(frame, seed=0)
    except ValueError as e:
        raise click.ClickException(f"cannot train: {e}") from None
    bundle = {
        "forecaster": fc,
        "work_item_types": s.work_item_types,
        "done_categories": s.done_categories,
        "commit_grace_days": s.commit_grace_days,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_sprints": int(len(sd.sprints)),
        "n_items": int(frame["y"].notna().sum()),
        "version": __version__,
    }
    path = workdir / MODEL_FILE
    joblib.dump(bundle, path)
    click.echo(
        f"Trained on {bundle['n_sprints']} sprints / {bundle['n_items']} items "
        f"(calibration: {fc.item_model.calibration}, sprint shock sigma = {fc.sigma:.2f}) -> {path}"
    )
    return bundle


def _load_model(workdir: Path) -> dict:
    path = workdir / MODEL_FILE
    if not path.exists():
        raise click.ClickException(f"no model at {path}; run `sprint-forecast train` first")
    return joblib.load(path)


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
    s = Settings(bundle["work_item_types"], bundle["done_categories"], bundle["commit_grace_days"])
    cache = _load_cache(workdir)
    history = _build(cache, s)
    cal = sprint_calendar(cache, s.commit_grace_days)
    group = cal[cal["iteration"] == iteration]
    if group.empty:
        raise click.ClickException(f"no team runs a dated iteration with path {iteration!r}")
    cutoff = group["cutoff"].iloc[0]
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    if as_of == "now" or (as_of == "auto" and now < cutoff):
        t, label = now, "now"
    else:
        t, label = cutoff, "commit cutoff"
    try:
        sprints, items = scope_at(
            cache, history, iteration=iteration, cutoff=t, work_item_types=s.work_item_types,
            done_categories=s.done_categories, commit_grace_days=s.commit_grace_days, team=team,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    if items.empty:
        click.echo(f"No committed items in {iteration} as of {label} ({t:%Y-%m-%d %H:%M} UTC).")
        return []
    feats = build_features_for(sprints, items, history.sprints, history.items)
    velocity = team_history(sprints, history.sprints).set_index("sprint_id")["trailing_velocity"]
    titles: dict[int, str] = {}
    if title_lookup is not None and top > 0:
        try:
            titles = title_lookup(sorted(int(i) for i in feats["item_id"]))
        except Exception:  # titles are cosmetic; never fail a forecast over them
            titles = {}
    results = []
    for sid, rows in feats.groupby("sprint_id", sort=False):
        srow = sprints.set_index("sprint_id").loc[sid]
        p, samples = forecast_sprint(fc, rows, seed=0)
        summary = summarize(samples)
        v = velocity.get(sid, float("nan"))
        click.echo(f"\n{srow['team_key']} | {iteration}")
        click.echo(
            f"  window {srow['start']:%Y-%m-%d} .. {srow['end']:%Y-%m-%d} UTC; "
            f"scored as of {label} ({t:%Y-%m-%d %H:%M} UTC)"
        )
        click.echo(
            f"  committed: {int(srow['n_items'])} items, {srow['committed_points']:.1f} points "
            f"({int(srow['n_unestimated'])} unestimated)"
        )
        if v > 0:
            click.echo(f"  load: {srow['committed_points'] / v:.2f}x trailing velocity ({v:.1f} points)")
        else:
            click.echo("  load: n/a (no velocity history for this team)")
        click.echo(
            f"  P(full) {_pct(summary['p_full'])}   P(>=80%) {_pct(summary['p_80'])}   "
            f"expected {_pct(summary['expected'])}   p10/p50/p90 "
            f"{_pct(summary['p10'])} / {_pct(summary['p50'])} / {_pct(summary['p90'])}"
        )
        risky = rows.assign(p=p).sort_values("p", kind="mergesort").head(top)
        if len(risky):
            click.echo("  riskiest items:")
            contrib = contributions(fc.item_model, risky)
            for idx, r in risky.iterrows():
                drivers = describe_drivers(contrib.loc[idx], r)
                title = _truncate(titles.get(int(r["item_id"]), "")) if titles else ""
                click.echo(f"    #{int(r['item_id'])}  p={r['p']:.2f}  {r['points']:.1f} pts  {title}".rstrip())
                if drivers:
                    click.echo(f"        why: {'; '.join(drivers)}")
        results.append({"sprint_id": sid, **summary})
    return results


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
            work_item_types=cfg.work_item_types, full=full, echo=click.echo,
        )
    finally:
        conn.close()
    skipped = sum(1 for r in results if r.skipped)
    click.echo(f"Done: {len(results) - skipped} projects extracted, {skipped} skipped.")


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
    """Expanding-window backtest; writes .sprint-forecast/backtest.csv."""
    models = {"a": ("a", "team_mean"), "c": ("c", "team_mean"), "all": SPRINT_MODELS}[model_choice]
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
    "--as-of", "as_of", type=click.Choice(["auto", "now", "commit"]), default="auto", show_default=True,
    help="auto: the commit cutoff once it has passed, else now. now: current contents (done items drop out).",
)
@click.option("--top", type=click.IntRange(min=0), default=5, show_default=True, help="Riskiest items to list.")
@click.option("--titles/--no-titles", default=True, help="Fetch titles from ADO for display (never cached).")
@click.pass_obj
def predict(root: Path, iteration: str, team: str | None, as_of: str, top: int, titles: bool) -> None:
    """Forecast one iteration's committed scope."""
    lookup = _title_lookup(root) if titles else None
    _predict(data_dir(root), iteration=iteration, team=team, as_of=as_of, top=top, title_lookup=lookup)


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


if __name__ == "__main__":
    main()
