import numpy as np
import pandas as pd
import pytest

from helpers import build_cache, iteration, rev, team
from sprint_forecast.golive import CURVE_COLUMNS, PARENT_COLUMNS, golive
from sprint_forecast.sprints import build_sprints

TYPES = ["User Story", "Bug"]
BACKLOG = "Alpha"
STARTS = pd.date_range("2024-01-01T05:00:00Z", periods=7, freq="14D")
PATHS = [f"Alpha\\Sprint {k}" for k in range(7)]
SPRINTS = [
    iteration(p, s.strftime("%Y-%m-%dT%H:%M:%S.000Z"), (s + pd.Timedelta(days=14, seconds=-1)).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"))
    for p, s in zip(PATHS, STARTS)
]
RED = team("Team Red", ["Alpha\\Red"], PATHS)
NOW = STARTS[4] + pd.Timedelta(days=3)  # Sprint 4 is running
ACTIVE = {"state": "Active", "state_category": "InProgress"}
CLOSED = {"state": "Closed", "state_category": "Completed"}
REMOVED = {"state": "Removed", "state_category": "Removed"}


def day(k: int, d: int) -> str:
    return (STARTS[k] + pd.Timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def finished(item_id: int, k: int, parent: int, points=5.0) -> list[dict]:
    """A child planned into Sprint k, started on its day 2 and closed on day 5."""
    it = PATHS[k]
    return [
        rev(item_id, 1, day(k, -3), parent_id=parent, story_points=points),
        rev(item_id, 2, day(k, 0), parent_id=parent, story_points=points, iteration=it),
        rev(item_id, 3, day(k, 2), parent_id=parent, story_points=points, iteration=it, **ACTIVE),
        rev(item_id, 4, day(k, 5), parent_id=parent, story_points=points, iteration=it, **CLOSED),
    ]


def run(tmp_path, revisions, **kw):
    cache = build_cache(tmp_path, revisions, SPRINTS, [RED])
    history = build_sprints(cache, work_item_types=TYPES)
    kw.setdefault("now", NOW)
    return golive(cache, history, work_item_types=TYPES, done_categories=["Completed"], **kw)


STEADY = finished(1, 1, 100) + finished(2, 2, 100) + finished(3, 3, 100)


def test_a_steady_feature_finishes_when_its_burn_covers_what_is_left(tmp_path):
    parents, curve = run(tmp_path, STEADY + [
        rev(4, 1, day(3, 1), parent_id=100, iteration=BACKLOG, story_points=5.0),
        rev(5, 1, day(3, 1), parent_id=100, iteration=BACKLOG, story_points=5.0),
    ])
    assert list(parents.columns) == PARENT_COLUMNS and list(curve.columns) == CURVE_COLUMNS
    p = parents.set_index("parent_id").loc[100]
    assert p["team_key"] == "Alpha/Team Red" and p["status"] == "forecast"
    assert (p["n_children"], p["n_open"], p["open_points"], p["done_points"]) == (5, 2, 10.0, 15.0)
    assert (p["burn_sprints"], p["mean_burn"]) == (3, 5.0)
    c = curve[curve["parent_id"] == 100].set_index("k")
    assert c.loc[0, "iteration"] == PATHS[4]
    assert c.loc[[0, 1, 2], "p_done_by"].tolist() == [0.0, 0.0, 1.0]
    assert p["p_this_sprint"] == 0.0
    assert p["p50_end"] == p["p85_end"] == c.loc[2, "end"] == pd.Timestamp(SPRINTS[6]["end_date"])


def test_the_curve_runs_past_the_calendar_on_the_teams_sprint_length(tmp_path):
    _, curve = run(tmp_path, STEADY + [rev(4, 1, day(3, 1), parent_id=100, iteration=BACKLOG, story_points=40.0)])
    c = curve.set_index("k")
    assert c.index.tolist() == list(range(13))
    assert c.loc[3, "iteration"] is None or pd.isna(c.loc[3, "iteration"])
    assert c.loc[3, "end"] - c.loc[2, "end"] == pd.Timedelta(days=14)
    assert c.loc[7, "p_done_by"] == 0.0 and c.loc[8, "p_done_by"] == 1.0
    assert c["p_done_by"].is_monotonic_increasing


def test_children_in_the_running_sprint_are_drawn_with_the_item_model(tmp_path):
    revs = STEADY + [
        rev(4, 1, day(3, 1), parent_id=100, iteration=BACKLOG, story_points=5.0),
        rev(4, 2, day(4, 0), parent_id=100, iteration=PATHS[4], story_points=5.0, **ACTIVE),
        rev(5, 1, day(3, 1), parent_id=100, iteration=BACKLOG, story_points=5.0),
    ]
    parents, curve = run(tmp_path, revs, p_open={4: 0.9999})
    c = curve.set_index("k")["p_done_by"]
    assert c[0] == 0.0 and c[1] > 0.99
    assert parents.set_index("parent_id").loc[100, "in_sprint_points"] == 5.0
    _, unlikely = run(tmp_path / "b", revs, p_open={4: 0.0001})
    assert unlikely.set_index("k")["p_done_by"][1] < 0.01


def test_removed_children_drop_out_and_unestimated_ones_take_the_team_median(tmp_path):
    parents, _ = run(tmp_path, STEADY + [
        rev(4, 1, day(3, 1), parent_id=100, iteration=BACKLOG, story_points=None),
        rev(5, 1, day(3, 1), parent_id=100, iteration=BACKLOG, story_points=5.0),
        rev(5, 2, day(3, 2), parent_id=100, iteration=BACKLOG, story_points=5.0, **REMOVED),
    ])
    p = parents.set_index("parent_id").loc[100]
    assert (p["n_children"], p["n_open"], p["n_unestimated_open"], p["open_points"]) == (4, 1, 1, 5.0)


def test_too_little_history_gets_no_forecast(tmp_path):
    parents, curve = run(tmp_path, finished(1, 3, 200) + [rev(2, 1, day(3, 1), parent_id=200, iteration=BACKLOG)])
    p = parents.set_index("parent_id").loc[200]
    assert p["status"] == "not enough history" and p["burn_sprints"] == 1
    assert np.isnan(p["p_this_sprint"]) and pd.isna(p["p50_end"])
    assert curve.empty


def test_no_burn_in_recent_sprints_gets_no_forecast(tmp_path):
    parents, curve = run(tmp_path, [
        rev(1, 1, day(1, 0), parent_id=300, iteration=PATHS[1]),
        rev(1, 2, day(1, 2), parent_id=300, iteration=PATHS[1], **ACTIVE),
        rev(1, 3, day(2, 0), parent_id=300, iteration=PATHS[4], **ACTIVE),
    ])
    assert parents.set_index("parent_id").loc[300, "status"] == "no recent progress"
    assert curve.empty


def test_finished_and_idle_parents_are_left_out(tmp_path):
    parents, _ = run(tmp_path, STEADY + finished(10, 1, 400) + [
        rev(11, 1, day(1, 1), parent_id=400, iteration=BACKLOG),
    ], active_days=30)
    assert parents["parent_id"].tolist() == []
    parents, _ = run(tmp_path / "b", STEADY + finished(10, 1, 400) + [
        rev(11, 1, day(1, 1), parent_id=400, iteration=BACKLOG),
    ])
    assert parents["parent_id"].tolist() == [400]


def test_items_of_other_types_do_not_count(tmp_path):
    parents, _ = run(tmp_path, STEADY + [
        rev(4, 1, day(3, 1), parent_id=100, iteration=BACKLOG, story_points=5.0),
        rev(5, 1, day(3, 1), parent_id=100, iteration=BACKLOG, story_points=5.0, type="Task"),
    ])
    assert parents.set_index("parent_id").loc[100, "n_open"] == 1


def test_the_forecast_is_reproducible(tmp_path):
    revs = STEADY + finished(6, 2, 100, points=13.0) + [rev(4, 1, day(3, 1), parent_id=100, iteration=BACKLOG,
                                                            story_points=30.0)]
    _, a = run(tmp_path, revs)
    _, b = run(tmp_path / "b", revs)
    pd.testing.assert_frame_equal(a, b)
    assert 0.0 < a["p_done_by"].iloc[3] < 1.0


@pytest.mark.parametrize("now", [STARTS[0] - pd.Timedelta(days=1)])
def test_nothing_to_forecast_still_has_columns(tmp_path, now):
    parents, curve = run(tmp_path, STEADY, now=now)
    assert list(parents.columns) == PARENT_COLUMNS and parents.empty
    assert list(curve.columns) == CURVE_COLUMNS and curve.empty
