"""SQLite schema and read helpers for `.sprint-forecast/cache.db`."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

SCHEMA_VERSION = "1"

REVISION_COLUMNS = [
    "item_id", "rev", "project", "changed", "revised", "created", "type", "state", "state_category",
    "iteration", "area", "assigned_to_sk", "story_points", "effort", "parent_id",
]
ITERATION_COLUMNS = ["project", "iteration_sk", "path", "name", "start_date", "end_date", "is_ended"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS revisions (
    item_id INTEGER NOT NULL, rev INTEGER NOT NULL, project TEXT NOT NULL,
    changed TEXT NOT NULL, revised TEXT, created TEXT,
    type TEXT, state TEXT, state_category TEXT, iteration TEXT, area TEXT,
    assigned_to_sk TEXT, story_points REAL, effort REAL, parent_id INTEGER,
    PRIMARY KEY (item_id, rev)
);
CREATE INDEX IF NOT EXISTS ix_revisions_project_changed ON revisions(project, changed);
CREATE TABLE IF NOT EXISTS iterations (
    project TEXT NOT NULL, iteration_sk TEXT PRIMARY KEY, path TEXT NOT NULL, name TEXT,
    start_date TEXT, end_date TEXT, is_ended INTEGER
);
CREATE TABLE IF NOT EXISTS teams (project TEXT NOT NULL, team_sk TEXT PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS team_areas (team_sk TEXT NOT NULL, area_path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS team_iterations (team_sk TEXT NOT NULL, iteration_path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass
class CacheData:
    revisions: pd.DataFrame
    iterations: pd.DataFrame
    teams: pd.DataFrame
    team_areas: pd.DataFrame
    team_iterations: pd.DataFrame


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    if get_meta(conn, "schema_version") is None:
        set_meta(conn, "schema_version", SCHEMA_VERSION)
    conn.commit()
    return conn


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return default if row is None else row[0]


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))


def watermark_key(project: str) -> str:
    return f"watermark:{project}"


def upsert_revisions(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    """Insert revisions keyed on (item_id, rev); existing rows are immutable and left untouched."""
    placeholders = ",".join("?" for _ in REVISION_COLUMNS)
    cur = conn.executemany(
        f"INSERT OR IGNORE INTO revisions({','.join(REVISION_COLUMNS)}) VALUES ({placeholders})",
        ([row.get(c) for c in REVISION_COLUMNS] for row in rows),
    )
    return cur.rowcount


def delete_project_revisions(conn: sqlite3.Connection, project: str) -> None:
    conn.execute("DELETE FROM revisions WHERE project = ?", (project,))


def max_changed(conn: sqlite3.Connection, project: str) -> str | None:
    return conn.execute("SELECT MAX(changed) FROM revisions WHERE project = ?", (project,)).fetchone()[0]


def replace_project_iterations(conn: sqlite3.Connection, project: str, rows: list[dict]) -> None:
    conn.execute("DELETE FROM iterations WHERE project = ?", (project,))
    conn.executemany(
        f"INSERT OR REPLACE INTO iterations({','.join(ITERATION_COLUMNS)}) VALUES (?,?,?,?,?,?,?)",
        ([r.get(c) for c in ITERATION_COLUMNS] for r in rows),
    )


def replace_project_teams(
    conn: sqlite3.Connection,
    project: str,
    teams: list[dict],
    team_areas: list[dict],
    team_iterations: list[dict],
) -> None:
    old = "SELECT team_sk FROM teams WHERE project = ?"
    conn.execute(f"DELETE FROM team_areas WHERE team_sk IN ({old})", (project,))
    conn.execute(f"DELETE FROM team_iterations WHERE team_sk IN ({old})", (project,))
    conn.execute("DELETE FROM teams WHERE project = ?", (project,))
    conn.executemany(
        "INSERT OR REPLACE INTO teams(project, team_sk, name) VALUES (?,?,?)",
        ((project, t["team_sk"], t["name"]) for t in teams),
    )
    conn.executemany(
        "INSERT INTO team_areas(team_sk, area_path) VALUES (?,?)",
        ((a["team_sk"], a["area_path"]) for a in team_areas),
    )
    conn.executemany(
        "INSERT INTO team_iterations(team_sk, iteration_path) VALUES (?,?)",
        ((i["team_sk"], i["iteration_path"]) for i in team_iterations),
    )


def _ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, format="ISO8601")


def load_cache(conn: sqlite3.Connection, projects: list[str] | None = None) -> CacheData:
    where, params = "", ()
    if projects:
        where = f" WHERE project IN ({','.join('?' for _ in projects)})"
        params = tuple(projects)
    revs = pd.read_sql_query(f"SELECT * FROM revisions{where}", conn, params=params)
    for col in ("changed", "revised", "created"):
        revs[col] = _ts(revs[col])
    for col in ("story_points", "effort", "parent_id"):
        revs[col] = revs[col].astype("float64")
    revs["item_id"] = revs["item_id"].astype("int64")
    revs["rev"] = revs["rev"].astype("int64")
    revs = revs.sort_values(["item_id", "changed", "rev"], kind="mergesort").reset_index(drop=True)

    its = pd.read_sql_query(f"SELECT * FROM iterations{where}", conn, params=params)
    its["start_date"] = _ts(its["start_date"])
    its["end_date"] = _ts(its["end_date"])
    its["is_ended"] = its["is_ended"].fillna(0).astype(bool)

    teams = pd.read_sql_query(f"SELECT * FROM teams{where}", conn, params=params)
    sks = tuple(teams["team_sk"])
    in_sk = f" WHERE team_sk IN ({','.join('?' for _ in sks)})" if sks else " WHERE 0"
    team_areas = pd.read_sql_query(f"SELECT * FROM team_areas{in_sk}", conn, params=sks)
    team_iterations = pd.read_sql_query(f"SELECT * FROM team_iterations{in_sk}", conn, params=sks)
    return CacheData(revs, its, teams, team_areas, team_iterations)
