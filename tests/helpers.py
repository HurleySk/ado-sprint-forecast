"""Builders for hand-made cache fixtures (synthetic names only)."""
from __future__ import annotations

from pathlib import Path

from sprint_forecast.cache import (
    CacheData,
    connect,
    load_cache,
    replace_project_iterations,
    replace_project_teams,
    set_meta,
    upsert_revisions,
)


def rev(item_id: int, rev_no: int, changed: str, **overrides) -> dict:
    row = {
        "item_id": item_id,
        "rev": rev_no,
        "project": "Alpha",
        "changed": changed,
        "revised": None,
        "created": "2024-01-01T00:00:00.000Z",
        "type": "User Story",
        "state": "New",
        "state_category": "Proposed",
        "iteration": "Alpha",
        "area": "Alpha\\Red",
        "assigned_to_sk": None,
        "story_points": 3.0,
        "effort": None,
        "parent_id": None,
    }
    row.update(overrides)
    return row


def iteration(path: str, start: str | None, end: str | None, project: str = "Alpha") -> dict:
    return {
        "project": project,
        "iteration_sk": f"it-{path}",
        "path": path,
        "name": path.split("\\")[-1],
        "start_date": start,
        "end_date": end,
        "is_ended": 0,
    }


def team(name: str, areas: list[str], iterations: list[str], project: str = "Alpha") -> dict:
    return {"project": project, "name": name, "areas": areas, "iterations": iterations}


def build_cache(
    tmp_path: Path, revisions: list[dict], iterations: list[dict] = (), teams: list[dict] = (), meta: dict | None = None,
) -> CacheData:
    conn = connect(Path(tmp_path) / "cache.db")
    upsert_revisions(conn, revisions)
    for key, value in (meta or {}).items():
        set_meta(conn, key, value)
    projects = {i["project"] for i in iterations} | {t["project"] for t in teams}
    for project in projects:
        replace_project_iterations(conn, project, [i for i in iterations if i["project"] == project])
        ts = [t for t in teams if t["project"] == project]
        replace_project_teams(
            conn,
            project,
            [{"team_sk": f"sk-{t['name']}", "name": t["name"]} for t in ts],
            [{"team_sk": f"sk-{t['name']}", "area_path": a} for t in ts for a in t["areas"]],
            [{"team_sk": f"sk-{t['name']}", "iteration_path": p} for t in ts for p in t["iterations"]],
        )
    conn.commit()
    data = load_cache(conn)
    conn.close()
    return data
