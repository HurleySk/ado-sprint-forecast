"""Synthetic cache.db generator with planted effects. All names are synthetic."""
from __future__ import annotations

import itertools
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from sprint_forecast.cache import (
    connect,
    replace_project_iterations,
    replace_project_teams,
    set_meta,
    upsert_revisions,
    watermark_key,
)

B0 = 3.0
BETA_CARRY = 0.7
BETA_LOAD = 1.3
BETA_SIZE = 0.35
SHOCK_SD = 0.4
LOAD_MU = 0.0
LOAD_SD = 0.45
SIZES = [1, 2, 3, 5, 8, 13]
SIZE_P = [0.12, 0.20, 0.25, 0.22, 0.14, 0.07]
UNPOINTED_SHARE = 0.15
EPOCH = datetime(2024, 1, 8, 5, 0, tzinfo=timezone.utc)  # Monday 00:00 at UTC-05:00
PROJECT_OFFSET_DAYS = {"Alpha": 0, "Beta": 3}


@dataclass(frozen=True)
class SynthTeam:
    project: str
    name: str
    area: str
    capacity: float
    effect: float
    runs_sprints: bool = True
    uses_effort: bool = False


TEAMS = (
    SynthTeam("Alpha", "Team Red", "Alpha\\Red", 80.0, 0.3),
    SynthTeam("Alpha", "Team Blue", "Alpha\\Blue", 64.0, -0.3),
    SynthTeam("Beta", "Team Green", "Beta\\Green", 56.0, 0.0, uses_effort=True),
    SynthTeam("Beta", "Team Gray", "Beta\\Gray", 0.0, 0.0, runs_sprints=False),
)


def _iso(t: datetime | None) -> str | None:
    return None if t is None else t.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sprint_path(project: str, k: int) -> str:
    return f"{project}\\Sprint {k + 1}"


def sprint_window(project: str, k: int) -> tuple[datetime, datetime]:
    start = EPOCH + timedelta(days=PROJECT_OFFSET_DAYS[project] + 14 * k)
    return start, start + timedelta(days=14) - timedelta(milliseconds=1)


class _Item:
    def __init__(self, item_id: int, project: str, created: datetime, fields: dict):
        self.id = item_id
        self.project = project
        self.created = created
        self.fields = dict(fields)
        self.events: list[tuple[datetime, dict]] = [(created, dict(self.fields))]

    def set(self, t: datetime, **changes) -> None:
        t = max(t, self.events[-1][0] + timedelta(minutes=1))
        self.fields.update(changes)
        self.events.append((t, dict(self.fields)))


class _Sim:
    def __init__(self, seed: int, n_sprints: int):
        self.rng = np.random.default_rng(seed)
        self.n_sprints = n_sprints
        self.ids = itertools.count(1000)
        self.items: list[_Item] = []

    def u(self, lo: float, hi: float) -> float:
        return float(self.rng.uniform(lo, hi))

    def new_item(self, team: SynthTeam, members: list[str], created: datetime, iteration: str) -> tuple[_Item, float]:
        rng = self.rng
        pts = float(rng.choice(SIZES, p=SIZE_P))
        is_bug = rng.random() < 0.2
        typ = "Bug" if is_bug else ("Product Backlog Item" if team.uses_effort else "User Story")
        estimated = rng.random() >= UNPOINTED_SHARE
        r = rng.random()
        area = team.area + "\\Web" if r < 0.3 else (team.project if r < 0.33 else team.area)
        parent = int(rng.integers(100, 200)) if rng.random() < 0.5 else None
        assignee = members[int(rng.integers(len(members)))] if members and rng.random() < 0.9 else None
        item = _Item(next(self.ids), team.project, created, {
            "type": typ, "state": "New", "state_category": "Proposed", "iteration": iteration,
            "area": area, "assigned_to_sk": assignee,
            "story_points": pts if estimated and not team.uses_effort else None,
            "effort": pts if estimated and team.uses_effort else None,
            "parent_id": parent,
        })
        self.items.append(item)
        return item, pts

    def complete(self, item: _Item, cutoff: datetime, end: datetime) -> bool:
        """Finish the item; returns False when it only reaches Resolved by the end (Closed afterwards)."""
        if item.fields["state_category"] == "Proposed":
            item.set(cutoff + timedelta(days=self.u(0, 4)), state="Active", state_category="InProgress")
        done_at = cutoff + timedelta(days=self.u(5, 12.5))
        if self.rng.random() < 0.05:
            item.set(done_at, state="Resolved", state_category="Resolved")
            item.set(end + timedelta(days=self.u(1, 3)), state="Closed", state_category="Completed")
            return False
        item.set(done_at, state="Closed", state_category="Completed")
        return True

    def leave_unfinished(self, item: _Item, cutoff: datetime) -> None:
        if item.fields["state_category"] == "Proposed" and self.rng.random() < 0.6:
            item.set(cutoff + timedelta(days=self.u(0, 10)), state="Active", state_category="InProgress")

    def run_team(self, team: SynthTeam) -> None:
        rng = self.rng
        members = [str(uuid.UUID(bytes=rng.bytes(16), version=4)) for _ in range(4)]
        carry: list[tuple[_Item, int, float]] = []
        delivered_history: list[float] = []
        for k in range(self.n_sprints):
            start, end = sprint_window(team.project, k)
            cutoff = start + timedelta(days=1)
            path = sprint_path(team.project, k)
            last = k == self.n_sprints - 1
            velocity = float(np.mean(delivered_history[-3:])) if delivered_history else 0.8 * team.capacity
            basis = 0.85 * team.capacity
            target = basis * min(2.5, max(0.5, math.exp(rng.normal(LOAD_MU, LOAD_SD))))
            committed = []
            total = 0.0
            for item, count, pts in carry:
                if total >= target:
                    item.set(start + timedelta(hours=self.u(1, 20)), iteration=team.project)
                    continue
                item.set(start + timedelta(hours=self.u(1, 20)), iteration=path)
                committed.append((item, count, pts))
                total += pts
            while total < target:
                item, pts = self.new_item(team, members, start - timedelta(days=self.u(2, 40)), team.project)
                if rng.random() < 0.05:
                    item.set(item.created + timedelta(hours=self.u(12, 24)), iteration=f"{team.project}\\Someday")
                plan_at = start - timedelta(days=self.u(0, 3)) if rng.random() < 0.5 else start + timedelta(hours=self.u(0, 20))
                item.set(plan_at, iteration=path)
                committed.append((item, 0, pts))
                total += pts
            # load_ratio as the features define it: committed points / trailing-3 mean delivered points
            load_ratio = min(3.0, total / max(velocity, 1.0))
            shock = float(rng.normal(0.0, SHOCK_SD))
            carry = []
            delivered = 0.0
            for item, count, pts in committed:
                if rng.random() < 0.03:
                    item.set(cutoff + timedelta(days=self.u(1, 8)), iteration=team.project)
                    continue
                logit = (B0 - BETA_CARRY * count - BETA_LOAD * (load_ratio - 1) - BETA_SIZE * math.log(pts)
                         + team.effect + shock)
                if rng.random() < 1 / (1 + math.exp(-logit)):
                    delivered += pts if self.complete(item, cutoff, end) else 0.0
                    continue
                self.leave_unfinished(item, cutoff)
                if last:
                    continue
                fate = rng.random()
                after = end + timedelta(hours=self.u(1, 20))
                if fate < 0.05:
                    item.set(after, state="Removed", state_category="Removed")
                elif fate < (0.4 if count >= 2 else 0.15):
                    item.set(after, iteration=team.project)
                else:
                    carry.append((item, count + 1, pts))
            delivered_history.append(delivered)
            for _ in range(int(rng.poisson(1.0))):
                item, pts = self.new_item(team, members, cutoff + timedelta(days=self.u(1, 8)), path)
                if rng.random() < 0.6:
                    self.complete(item, cutoff, end)
                elif not last:
                    carry.append((item, 1, pts))

    def backlog_only(self, team: SynthTeam, n: int) -> None:
        for _ in range(n):
            self.new_item(team, [], EPOCH + timedelta(days=self.u(0, 60)), team.project)


def simulate(seed: int = 0, n_sprints: int = 40) -> dict[str, list[dict]]:
    sim = _Sim(seed, n_sprints)
    iterations: list[dict] = []
    for project in PROJECT_OFFSET_DAYS:
        iterations.append(_iteration_row(project, project, None, None))
        iterations.append(_iteration_row(project, f"{project}\\Someday", None, None))
        for k in range(n_sprints + 1):
            start, end = sprint_window(project, k)
            iterations.append(_iteration_row(project, sprint_path(project, k), start, end))
    teams, team_areas, team_iterations = [], [], []
    for team in TEAMS:
        sk = str(uuid.uuid5(uuid.NAMESPACE_URL, f"synthetic-team/{team.project}/{team.name}"))
        teams.append({"project": team.project, "team_sk": sk, "name": team.name})
        team_areas.append({"project": team.project, "team_sk": sk, "area_path": team.area})
        if team.runs_sprints:
            team_iterations.extend(
                {"project": team.project, "team_sk": sk, "iteration_path": sprint_path(team.project, k)}
                for k in range(n_sprints + 1)
            )
    for team in TEAMS:
        if team.runs_sprints:
            sim.run_team(team)
        else:
            sim.backlog_only(team, 10)
    revisions = []
    for item in sim.items:
        for i, (t, fields) in enumerate(item.events):
            nxt = item.events[i + 1][0] if i + 1 < len(item.events) else None
            revisions.append({
                "item_id": item.id, "rev": i + 1, "project": item.project, "changed": _iso(t),
                "revised": _iso(nxt), "created": _iso(item.created), **fields,
            })
    return {
        "revisions": revisions, "iterations": iterations, "teams": teams,
        "team_areas": team_areas, "team_iterations": team_iterations,
    }


def _iteration_row(project: str, path: str, start: datetime | None, end: datetime | None) -> dict:
    return {
        "project": project,
        "iteration_sk": str(uuid.uuid5(uuid.NAMESPACE_URL, f"synthetic-iteration/{path}")),
        "path": path,
        "name": path.split("\\")[-1],
        "start_date": _iso(start),
        "end_date": _iso(end),
        "is_ended": 0,
    }


def generate(path: Path | str, *, seed: int = 0, n_sprints: int = 40) -> Path:
    """Write a fresh synthetic cache.db at `path` and return the path."""
    path = Path(path)
    if path.exists():
        path.unlink()
    data = simulate(seed=seed, n_sprints=n_sprints)
    conn = connect(path)
    try:
        upsert_revisions(conn, data["revisions"])
        for project in PROJECT_OFFSET_DAYS:
            replace_project_iterations(conn, project, [r for r in data["iterations"] if r["project"] == project])
            replace_project_teams(
                conn,
                project,
                [t for t in data["teams"] if t["project"] == project],
                [a for a in data["team_areas"] if a["project"] == project],
                [i for i in data["team_iterations"] if i["project"] == project],
            )
            changed = [r["changed"] for r in data["revisions"] if r["project"] == project]
            set_meta(conn, watermark_key(project), max(changed))
        set_meta(conn, "synthetic", f"seed={seed};n_sprints={n_sprints}")
        conn.commit()
    finally:
        conn.close()
    return path
