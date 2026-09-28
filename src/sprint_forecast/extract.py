"""Pull revisions, iterations and teams per project into cache.db. The only module that talks to ADO."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from sprint_forecast.analytics import (
    FetchJson,
    HttpError,
    list_projects,
    odata_string,
    odata_url,
    paginate,
    to_utc_iso,
)
from sprint_forecast.cache import (
    delete_project_revisions,
    get_meta,
    max_changed,
    replace_project_iterations,
    replace_project_teams,
    set_meta,
    upsert_revisions,
    watermark_key,
)

REVISION_SELECT = (
    "WorkItemId,Revision,ChangedDate,RevisedDate,WorkItemType,State,StateCategory,"
    "StoryPoints,Effort,ParentWorkItemId,CreatedDate"
)
REVISION_EXPAND = "Iteration($select=IterationPath),Area($select=AreaPath),AssignedTo($select=UserSK)"
ITERATION_SELECT = "IterationSK,IterationPath,IterationName,StartDate,EndDate,IsEnded"
TEAM_SELECT = "TeamSK,TeamName"
TEAM_EXPAND = "Areas($select=AreaPath),Iterations($select=IterationPath)"


@dataclass
class ExtractResult:
    project: str
    fetched: int = 0
    inserted: int = 0
    iterations: int = 0
    teams: int = 0
    skipped: str | None = None


def revisions_url(org: str, project: str, work_item_types: list[str], watermark: str | None) -> str:
    filters = [f"WorkItemType in ({','.join(odata_string(t) for t in work_item_types)})"]
    if watermark:
        filters.append(f"ChangedDate ge {watermark}")
    return odata_url(org, project, "WorkItemRevisions", {
        "$select": REVISION_SELECT,
        "$expand": REVISION_EXPAND,
        "$filter": " and ".join(filters),
        "$orderby": "WorkItemId,Revision",
    })


def iterations_url(org: str, project: str) -> str:
    return odata_url(org, project, "Iterations", {"$select": ITERATION_SELECT})


def teams_url(org: str, project: str) -> str:
    return odata_url(org, project, "Teams", {"$select": TEAM_SELECT, "$expand": TEAM_EXPAND})


def _nav(raw: dict, nav: str, prop: str):
    value = raw.get(nav)
    return value.get(prop) if isinstance(value, dict) else None


def revision_row(project: str, raw: dict) -> dict:
    parent = raw.get("ParentWorkItemId")
    return {
        "item_id": int(raw["WorkItemId"]),
        "rev": int(raw["Revision"]),
        "project": project,
        "changed": to_utc_iso(raw["ChangedDate"]),
        "revised": to_utc_iso(raw.get("RevisedDate")),
        "created": to_utc_iso(raw.get("CreatedDate")),
        "type": raw.get("WorkItemType"),
        "state": raw.get("State"),
        "state_category": raw.get("StateCategory"),
        "iteration": _nav(raw, "Iteration", "IterationPath"),
        "area": _nav(raw, "Area", "AreaPath"),
        "assigned_to_sk": _nav(raw, "AssignedTo", "UserSK"),
        "story_points": raw.get("StoryPoints"),
        "effort": raw.get("Effort"),
        "parent_id": int(parent) if parent is not None else None,
    }


def iteration_row(project: str, raw: dict) -> dict:
    return {
        "project": project,
        "iteration_sk": raw["IterationSK"],
        "path": raw["IterationPath"],
        "name": raw.get("IterationName"),
        "start_date": to_utc_iso(raw.get("StartDate")),
        "end_date": to_utc_iso(raw.get("EndDate")),
        "is_ended": int(bool(raw.get("IsEnded"))),
    }


def team_rows(raw: dict) -> tuple[dict, list[dict], list[dict]]:
    sk = raw["TeamSK"]
    team = {"team_sk": sk, "name": raw["TeamName"]}
    areas = [{"team_sk": sk, "area_path": a["AreaPath"]} for a in raw.get("Areas") or [] if a.get("AreaPath")]
    its = [
        {"team_sk": sk, "iteration_path": i["IterationPath"]}
        for i in raw.get("Iterations") or [] if i.get("IterationPath")
    ]
    return team, areas, its


def extract_project(
    fetch_json: FetchJson,
    conn: sqlite3.Connection,
    org: str,
    project: str,
    *,
    work_item_types: list[str],
    full: bool = False,
) -> ExtractResult:
    result = ExtractResult(project)
    try:
        iterations = [iteration_row(project, r) for r in paginate(fetch_json, iterations_url(org, project))]
        teams, areas, subs = [], [], []
        for raw in paginate(fetch_json, teams_url(org, project)):
            t, a, s = team_rows(raw)
            teams.append(t)
            areas.extend(a)
            subs.extend(s)
        watermark = None if full else get_meta(conn, watermark_key(project))
        url = revisions_url(org, project, work_item_types, watermark)
        rows = [revision_row(project, r) for r in paginate(fetch_json, url)]
    except HttpError as e:
        if e.status in (401, 403):
            result.skipped = f"HTTP {e.status}: no Analytics access"
            return result
        raise
    with conn:
        if full:
            delete_project_revisions(conn, project)
        result.inserted = upsert_revisions(conn, rows)
        replace_project_iterations(conn, project, iterations)
        replace_project_teams(conn, project, teams, areas, subs)
        newest = max_changed(conn, project)
        if newest:
            set_meta(conn, watermark_key(project), newest)
        set_meta(conn, "last_extract", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    result.fetched, result.iterations, result.teams = len(rows), len(iterations), len(teams)
    return result


def extract_all(
    fetch_json: FetchJson,
    conn: sqlite3.Connection,
    org: str,
    projects: list[str],
    *,
    work_item_types: list[str],
    full: bool = False,
    echo: Callable[[str], None] = print,
) -> list[ExtractResult]:
    names = list_projects(fetch_json, org) if projects == ["*"] else projects
    results = []
    for project in names:
        r = extract_project(fetch_json, conn, org, project, work_item_types=work_item_types, full=full)
        if r.skipped:
            echo(f"  {project}: skipped ({r.skipped})")
        else:
            echo(f"  {project}: {r.fetched} revisions fetched ({r.inserted} new), {r.iterations} iterations, {r.teams} teams")
        results.append(r)
    return results
