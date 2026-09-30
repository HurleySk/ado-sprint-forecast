"""Sprint reconstruction: (project, team, iteration) -> committed scope at the cutoff and outcome at the end."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sprint_forecast.cache import CacheData
from sprint_forecast.timeline import as_of_many, to_utc

REMOVED = "Removed"

CALENDAR_COLUMNS = ["sprint_id", "project", "team", "team_key", "iteration", "start", "end", "cutoff"]
SPRINT_COLUMNS = CALENDAR_COLUMNS + [
    "n_items", "committed_points", "done_points", "pct_done", "pct_done_count", "n_unestimated", "n_added_mid",
    "done_points_strict", "pct_done_strict",
]
ITEM_COLUMNS = CALENDAR_COLUMNS + [
    "item_id", "type", "state_category_at_commit", "area", "assigned_to_sk", "assigned_at_end_sk", "parent_id",
    "created", "last_changed", "revisions_so_far", "carryover_count", "raw_points", "points", "is_unestimated",
    "done", "state_category_at_end", "done_strict",
]
# Items that joined a sprint after its cutoff and were still in it at the end; as of the end, not the cutoff.
ADDED_COLUMNS = CALENDAR_COLUMNS + [
    "item_id", "type", "area", "assigned_to_sk", "carryover_count", "raw_points", "points", "is_unestimated",
    "done", "state_category_at_end", "done_strict",
]


@dataclass
class SprintData:
    sprints: pd.DataFrame
    items: pd.DataFrame
    report: dict = field(default_factory=dict)
    added: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=ADDED_COLUMNS))
    close_grace_hours: float = 0.0  # done also counts items closed this long after the end, still in the sprint


def raw_points(frame: pd.DataFrame) -> pd.Series:
    """Story points, falling back to effort; non-positive values count as missing."""
    sp = frame["story_points"].where(frame["story_points"] > 0)
    ef = frame["effort"].where(frame["effort"] > 0)
    return sp.fillna(ef).astype("float64")


def match_team(area: str | None, candidates: list[str], team_areas: dict[str, list[str]]) -> str | None:
    """Exact area match, else longest area-path prefix, else the only candidate; ties -> None."""
    target = (area if isinstance(area, str) else "").casefold()
    best_len, best = -1, []
    for team in candidates:
        for team_area in team_areas.get(team, []):
            ta = team_area.casefold()
            if target == ta or target.startswith(ta + "\\"):
                if len(ta) > best_len:
                    best_len, best = len(ta), [team]
                elif len(ta) == best_len and team not in best:
                    best.append(team)
    if len(best) == 1:
        return best[0]
    if not best and len(candidates) == 1:
        return candidates[0]
    return None


def sprint_calendar(cache: CacheData, commit_grace_days: float = 1.0) -> pd.DataFrame:
    """One row per (project, team, dated iteration); project-level rows when no team in a project subscribes."""
    its = cache.iterations
    ok = its["start_date"].notna() & its["end_date"].notna() & (its["end_date"] > its["start_date"])
    dated = its.loc[ok, ["project", "path", "start_date", "end_date"]]
    subs = cache.teams.rename(columns={"name": "team"}).merge(cache.team_iterations, on="team_sk")
    team_rows = subs.merge(dated, left_on=["project", "iteration_path"], right_on=["project", "path"])
    fallback = sorted(set(dated["project"]) - set(team_rows["project"]))
    team_rows = team_rows.assign(is_fallback=False)
    proj_rows = dated[dated["project"].isin(fallback)].assign(team=lambda d: d["project"], is_fallback=True)
    cols = ["project", "team", "path", "start_date", "end_date", "is_fallback"]
    cal = pd.concat([team_rows[cols], proj_rows[cols]], ignore_index=True)
    cal = cal.rename(columns={"path": "iteration", "start_date": "start", "end_date": "end"})
    cal["is_fallback"] = cal["is_fallback"].astype(bool)
    cal["team_key"] = cal["project"] + "/" + cal["team"]
    cal["sprint_id"] = cal["team_key"] + "|" + cal["iteration"]
    cutoff = cal["start"] + pd.Timedelta(days=commit_grace_days)
    cal["cutoff"] = cutoff.where(cutoff < cal["end"], cal["end"])
    cal = cal.drop_duplicates("sprint_id").sort_values(["start", "sprint_id"], kind="mergesort")
    return cal[CALENDAR_COLUMNS + ["is_fallback"]].reset_index(drop=True)


def iteration_group(cal: pd.DataFrame, iteration: str, team: str | None = None) -> pd.DataFrame:
    """The calendar rows (one per team) of `iteration`, only `team`'s when given. Raises ValueError for an iteration
    no team runs, or a team that does not run it."""
    group = cal[cal["iteration"] == iteration]
    if group.empty:
        raise ValueError(f"no team runs a dated iteration with path {iteration!r}")
    if team is not None and team not in set(group["team"]):
        raise ValueError(f"team {team!r} does not run {iteration!r}; teams: {', '.join(group['team'])}")
    return group if team is None else group[group["team"] == team]


class Assigner:
    """Assign (area, iteration, project) rows to one of the teams running that iteration."""

    def __init__(self, cache: CacheData, cal: pd.DataFrame):
        self.candidates = cal.groupby("iteration")["team"].apply(list).to_dict()
        self.fallback = set(cal.loc[cal["is_fallback"], "project"])
        merged = cache.teams.merge(cache.team_areas, on="team_sk")
        self.areas: dict[str, dict[str, list[str]]] = {}
        for row in merged.itertuples(index=False):
            self.areas.setdefault(row.project, {}).setdefault(row.name, []).append(row.area_path)

    def __call__(self, areas: pd.Series, iterations: pd.Series, projects: pd.Series) -> pd.Series:
        teams = [
            match_team(a, self.candidates.get(it, []), {} if p in self.fallback else self.areas.get(p, {}))
            for a, it, p in zip(areas, iterations, projects)
        ]
        return pd.Series(teams, index=areas.index, dtype=object)


def iteration_end_map(cache: CacheData) -> pd.Series:
    its = cache.iterations.dropna(subset=["end_date"]).drop_duplicates("path")
    return its.set_index("path")["end_date"]


def horizon(cache: CacheData, projects: pd.Series) -> pd.Series:
    """When each row's project was extracted; now for projects with no recorded extraction."""
    now = pd.Timestamp.now(tz="UTC")
    return pd.Series([cache.extracted_at.get(p, now) for p in projects], index=projects.index, dtype="datetime64[ns, UTC]")


def pairs_for(revisions: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """(item_id, target) for every item ever in a window's iteration, joined to the window's dates."""
    seen = revisions.loc[revisions["iteration"].isin(windows["target"]), ["item_id", "iteration"]]
    seen = seen.drop_duplicates().rename(columns={"iteration": "target"})
    return seen.merge(windows, on="target").reset_index(drop=True)


def carryover(revisions: pd.DataFrame, rows: pd.DataFrame, iteration_end: pd.Series) -> np.ndarray:
    """Distinct earlier iterations each row's item was in (revisions <= cutoff) whose iteration ended before start."""
    left = rows[["item_id", "cutoff", "start"]].reset_index(drop=True)
    left["_row"] = np.arange(len(left))
    hist = left.merge(revisions[["item_id", "changed", "iteration"]], on="item_id")
    hist = hist[hist["changed"] <= hist["cutoff"]]
    if hist.empty or iteration_end.empty:  # mapping onto nothing yields float NaN, which cannot compare to dates
        return np.zeros(len(left), dtype="int64")
    hist = hist[hist["iteration"].map(iteration_end) < hist["start"]]
    counts = hist.groupby("_row")["iteration"].nunique()
    return left["_row"].map(counts).fillna(0).astype("int64").to_numpy()


def done_at_end(
    cache: CacheData, item_ids: pd.Series, iterations: pd.Series, ends: pd.Series, projects: pd.Series,
    done_categories, close_grace_hours: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """(done, done_strict) per row: in its iteration and in a done category at its end (strict), or also at any
    point up to close_grace_hours after it (as far as the last extract has seen), still in that iteration."""
    done = set(done_categories)
    keys = pd.DataFrame({"item_id": item_ids.to_numpy(), "t": ends.to_numpy()})
    at_e = as_of_many(cache.revisions, keys, "t")
    strict = (at_e["iteration"].to_numpy() == iterations.to_numpy()) & at_e["state_category"].isin(done).to_numpy()
    if close_grace_hours <= 0 or len(keys) == 0:
        return strict, strict
    seen = horizon(cache, projects.reset_index(drop=True))
    until = (keys["t"] + pd.Timedelta(hours=close_grace_hours)).clip(upper=seen)
    later = cache.revisions[["item_id", "changed", "iteration", "state_category"]].merge(
        keys.assign(_row=np.arange(len(keys)), _it=iterations.to_numpy(), _until=until), on="item_id")
    later = later[(later["changed"] > later["t"]) & (later["changed"] <= later["_until"])
                  & (later["iteration"] == later["_it"]) & later["state_category"].isin(done)]
    graced = np.zeros(len(keys), dtype=bool)
    graced[later["_row"].unique()] = True
    return strict | graced, strict


def _committed(
    cache: CacheData,
    pairs: pd.DataFrame,
    at_c: pd.DataFrame,
    assign: Assigner,
    types: set[str],
    excluded: set[str],
) -> tuple[pd.DataFrame, int]:
    """Committed rows (index aligned with `pairs`) with team assigned; also returns the unassigned count."""
    mask = (
        (at_c["iteration"].to_numpy() == pairs["target"].to_numpy())
        & at_c["type"].isin(types).to_numpy()
        & ~at_c["state_category"].isin(excluded).to_numpy()
    )
    st = at_c.loc[mask]
    c = pairs.loc[mask, ["item_id", "target", "project", "start", "end", "cutoff"]].copy()
    c["type"] = st["type"].to_numpy()
    c["state_category_at_commit"] = st["state_category"].to_numpy()
    c["area"] = st["area"].to_numpy()
    c["assigned_to_sk"] = st["assigned_to_sk"].to_numpy()
    c["parent_id"] = st["parent_id"].to_numpy()
    c["created"] = st["created"].to_numpy()
    c["last_changed"] = st["changed"].to_numpy()
    c["revisions_so_far"] = st["revisions_so_far"].astype("int64").to_numpy()
    c["raw_points"] = raw_points(st).to_numpy()
    c["team"] = assign(c["area"], c["target"], c["project"])
    n_unassigned = int(c["team"].isna().sum())
    c = c[c["team"].notna()].rename(columns={"target": "iteration"})
    c["team_key"] = c["project"] + "/" + c["team"]
    c["sprint_id"] = c["team_key"] + "|" + c["iteration"]
    c["carryover_count"] = carryover(cache.revisions, c, iteration_end_map(cache))
    return c, n_unassigned


def impute_points(items: pd.DataFrame, history_items: pd.DataFrame) -> pd.DataFrame:
    """Fill missing raw_points with the team's prior median item size, then the global prior median, then 1."""
    items = items.copy()
    items["is_unestimated"] = items["raw_points"].isna()
    items["points"] = items["raw_points"].astype("float64")
    known = history_items[history_items["raw_points"].notna()]
    need = items[items["is_unestimated"]]
    for (team_key, start), idx in need.groupby(["team_key", "start"]).groups.items():
        prior = known[known["end"] < start]
        team_prior = prior[prior["team_key"] == team_key]
        if not team_prior.empty:
            fill = float(team_prior["raw_points"].median())
        elif not prior.empty:
            fill = float(prior["raw_points"].median())
        else:
            fill = 1.0
        items.loc[idx, "points"] = fill
    return items


def summarize_sprints(items: pd.DataFrame, n_added: dict[str, int] | None = None) -> pd.DataFrame:
    if items.empty:
        return pd.DataFrame(columns=SPRINT_COLUMNS)
    done = items["done"].astype("float64")
    strict = items["done_strict"].astype("float64") if "done_strict" in items else done
    work = items.assign(_done=done, _done_pts=items["points"] * done, _strict_pts=items["points"] * strict,
                        _unest=items["is_unestimated"].astype("int64"))
    g = work.groupby("sprint_id", sort=False)
    out = g[CALENDAR_COLUMNS[1:]].first()
    out["n_items"] = g.size()
    out["committed_points"] = g["points"].sum()
    out["done_points"] = g["_done_pts"].sum(min_count=1)
    out["pct_done"] = out["done_points"] / out["committed_points"].where(out["committed_points"] > 0)
    out["pct_done_count"] = g["_done"].mean()
    out["n_unestimated"] = g["_unest"].sum()
    out["done_points_strict"] = g["_strict_pts"].sum(min_count=1)
    out["pct_done_strict"] = out["done_points_strict"] / out["committed_points"].where(out["committed_points"] > 0)
    out = out.reset_index()
    out["n_added_mid"] = out["sprint_id"].map(n_added or {}).fillna(0).astype("int64")
    return out[SPRINT_COLUMNS].sort_values(["start", "sprint_id"], kind="mergesort").reset_index(drop=True)


def _finish_items(c: pd.DataFrame, history_items: pd.DataFrame | None) -> pd.DataFrame:
    items = impute_points(c, c if history_items is None else history_items)
    items = items[ITEM_COLUMNS].sort_values(["start", "sprint_id", "item_id"], kind="mergesort")
    return items.reset_index(drop=True)


def build_sprints(
    cache: CacheData,
    *,
    work_item_types: list[str],
    done_categories: list[str] = ("Completed",),
    commit_grace_days: float = 1.0,
    close_grace_hours: float = 0.0,
) -> SprintData:
    revs = cache.revisions
    cal = sprint_calendar(cache, commit_grace_days)
    types, done = set(work_item_types), set(done_categories)
    assign = Assigner(cache, cal)
    windows = cal.drop_duplicates("iteration")[["project", "iteration", "start", "end", "cutoff"]]
    pairs = pairs_for(revs, windows.rename(columns={"iteration": "target"}))
    at_c = as_of_many(revs, pairs[["item_id", "cutoff"]], "cutoff")
    at_e = as_of_many(revs, pairs[["item_id", "end"]], "end")
    c, n_unassigned = _committed(cache, pairs, at_c, assign, types, done | {REMOVED})
    e = at_e.loc[c.index]
    is_open = (c["end"] > horizon(cache, c["project"])).to_numpy()  # outcome not observed yet
    ended_done, strict = done_at_end(
        cache, c["item_id"], c["iteration"], c["end"], c["project"], done, close_grace_hours)
    c["done"] = np.where(is_open, np.nan, ended_done.astype("float64"))
    c["done_strict"] = np.where(is_open, np.nan, strict.astype("float64"))
    c["state_category_at_end"] = np.where(is_open, None, e["state_category"].to_numpy())
    c["assigned_at_end_sk"] = e["assigned_to_sk"].to_numpy()

    target = pairs["target"].to_numpy()
    added_mask = (
        (at_c["iteration"].to_numpy() != target)
        & (at_e["iteration"].to_numpy() == target)
        & at_e["type"].isin(types).to_numpy()
        & (at_e["state_category"] != REMOVED).to_numpy()
        & ~at_c["state_category"].isin(done).to_numpy()  # finished before the sprint: not this sprint's work
    )

    items = _finish_items(c, None) if len(c) else pd.DataFrame(columns=ITEM_COLUMNS)
    added = _added(cache, pairs.loc[added_mask], at_e.loc[added_mask], assign, done, items, close_grace_hours)
    sprints = summarize_sprints(items, added["sprint_id"].value_counts().to_dict())
    report = {
        "unassigned_items": n_unassigned,
        "added_mid_sprint": int(added_mask.sum()),
        "dropped_empty_sprints": int(len(set(cal["sprint_id"]) - set(sprints["sprint_id"]))),
        "fallback_projects": sorted(set(cal.loc[cal["is_fallback"], "project"])),
        "n_sprints": len(sprints),
        "open_sprints": int(sprints["pct_done"].isna().sum()) if len(sprints) else 0,
        "n_committed_items": len(items),
        "unestimated_share": float(items["is_unestimated"].mean()) if len(items) else float("nan"),
        "end_state_mix": items.loc[items["done"].notna(), "state_category_at_end"]
        .fillna("(missing)").value_counts().to_dict(),
    }
    return SprintData(sprints, items, report, added, float(close_grace_hours))


def _added(
    cache: CacheData, pairs: pd.DataFrame, at_e: pd.DataFrame, assign: Assigner, done: set[str],
    history_items: pd.DataFrame, close_grace_hours: float = 0.0,
) -> pd.DataFrame:
    """Items added after the cutoff and still in the sprint at its end, described as of the end."""
    if pairs.empty:
        return pd.DataFrame(columns=ADDED_COLUMNS)
    a = pairs[["item_id", "target", "project", "start", "end", "cutoff"]].copy()
    a["type"] = at_e["type"].to_numpy()
    a["area"] = at_e["area"].to_numpy()
    a["assigned_to_sk"] = at_e["assigned_to_sk"].to_numpy()
    a["raw_points"] = raw_points(at_e).to_numpy()
    state = at_e["state_category"].to_numpy()
    a["team"] = assign(a["area"], a["target"], a["project"])
    keep = a["team"].notna().to_numpy()
    a, state = a[keep].rename(columns={"target": "iteration"}), state[keep]
    if a.empty:
        return pd.DataFrame(columns=ADDED_COLUMNS)
    a["team_key"] = a["project"] + "/" + a["team"]
    a["sprint_id"] = a["team_key"] + "|" + a["iteration"]
    a["carryover_count"] = carryover(cache.revisions, a.assign(cutoff=a["end"]), iteration_end_map(cache))
    is_open = (a["end"] > horizon(cache, a["project"])).to_numpy()
    graced, strict = done_at_end(cache, a["item_id"], a["iteration"], a["end"], a["project"], done, close_grace_hours)
    a["done"] = np.where(is_open, np.nan, graced.astype("float64"))
    a["done_strict"] = np.where(is_open, np.nan, strict.astype("float64"))
    a["state_category_at_end"] = np.where(is_open, None, state)
    a = impute_points(a, history_items if len(history_items) else a)
    return a[ADDED_COLUMNS].sort_values(["start", "sprint_id", "item_id"], kind="mergesort").reset_index(drop=True)


def scope_at(
    cache: CacheData,
    history: SprintData,
    *,
    iteration: str,
    cutoff,
    work_item_types: list[str],
    done_categories: list[str] = ("Completed",),
    commit_grace_days: float = 1.0,
    team: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Committed-style scope of one iteration as of `cutoff`, outcome unknown. Returns (sprints, items)."""
    cal = sprint_calendar(cache, commit_grace_days)
    group = iteration_group(cal, iteration, team)
    windows = group.iloc[:1][["project", "iteration", "start", "end"]].rename(columns={"iteration": "target"})
    windows = windows.assign(cutoff=to_utc(cutoff))
    pairs = pairs_for(cache.revisions, windows)
    at_c = as_of_many(cache.revisions, pairs[["item_id", "cutoff"]], "cutoff")
    c, _ = _committed(cache, pairs, at_c, Assigner(cache, cal), set(work_item_types), set(done_categories) | {REMOVED})
    if team is not None:
        c = c[c["team"] == team]
    if c.empty:
        return pd.DataFrame(columns=SPRINT_COLUMNS), pd.DataFrame(columns=ITEM_COLUMNS)
    c = c.assign(done=np.nan, state_category_at_end=None, assigned_at_end_sk=None, done_strict=np.nan)
    items = _finish_items(c, history.items)
    return summarize_sprints(items), items


def data_report(cache: CacheData, sd: SprintData) -> dict:
    its = cache.iterations
    dated = its["start_date"].notna() & its["end_date"].notna()
    running = set(zip(sd.sprints["project"], sd.sprints["team"]))
    projects = sorted(set(its["project"]) | set(cache.teams["project"]) | set(cache.revisions["project"]))
    per_project = []
    for p in projects:
        team_names = sorted(cache.teams.loc[cache.teams["project"] == p, "name"])
        per_project.append({
            "project": p,
            "teams": len(team_names),
            "teams_running_sprints": sum((p, t) in running for t in team_names),
            "dated_iterations": int((dated & (its["project"] == p)).sum()),
            "undated_iterations": int((~dated & (its["project"] == p)).sum()),
            "sprints": int((sd.sprints["project"] == p).sum()),
            "committed_items": int((sd.items["project"] == p).sum()),
            "fallback": p in sd.report.get("fallback_projects", []),
        })
    mix = sd.report.get("end_state_mix", {})
    ended = mix.get("Resolved", 0) + mix.get("Completed", 0)
    done_pts = sd.sprints["done_points"].sum()
    return {
        **sd.report,
        "projects": per_project,
        "resolved_share_of_resolved_or_completed": (mix.get("Resolved", 0) / ended) if ended else float("nan"),
        "close_grace_hours": sd.close_grace_hours,
        "graced_share": float(1 - sd.sprints["done_points_strict"].sum() / done_pts) if done_pts > 0 else float("nan"),
    }
