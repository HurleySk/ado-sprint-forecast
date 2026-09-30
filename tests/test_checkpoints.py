import pandas as pd
import pytest

from conftest import SYNTH_TYPES
from helpers import build_cache, iteration, rev, team
from sprint_forecast.checkpoints import (
    CHECKPOINT_ROW_COLUMNS,
    OPEN_ROW_COLUMNS,
    PROGRESS_COLUMNS,
    build_checkpoint_rows,
    checkpoint_progress,
    checkpoint_time,
    open_rows,
    state_changes,
)
from sprint_forecast.sprints import build_sprints, sprint_calendar

TYPES = ["User Story", "Bug"]
DONE = ["Completed"]
IT0, IT1, IT2 = "Alpha\\Sprint 0", "Alpha\\Sprint 1", "Alpha\\Sprint 2"
S0 = iteration(IT0, "2024-02-19T05:00:00.000Z", "2024-03-04T04:59:59.999Z")
S1 = iteration(IT1, "2024-03-04T05:00:00.000Z", "2024-03-18T04:59:59.999Z")  # cutoff 2024-03-05T05:00Z
S2 = iteration(IT2, "2024-03-18T05:00:00.000Z", "2024-04-01T04:59:59.999Z")
PRE = "2024-03-01T00:00:00.000Z"    # before Sprint 1 starts
PLAN = "2024-03-04T12:00:00.000Z"   # after the start, before the cutoff
T = "2024-03-11T00:00:00.000Z"      # mid-sprint: the t of most tests
RED = team("Team Red", ["Alpha\\Red"], [IT0, IT1, IT2])
BLUE = team("Team Blue", ["Alpha\\Blue"], [IT1])
SPRINT1 = "Alpha/Team Red|Alpha\\Sprint 1"
BLUE1 = "Alpha/Team Blue|Alpha\\Sprint 1"


def at(tmp_path, revisions, t=T, teams=(RED,), sprint_ids=(SPRINT1,)):
    """open_rows of the given Sprint 1 team sprints at t (hand-built caches have no extraction time, so every
    2024 sprint counts as ended and y is known)."""
    cache = build_cache(tmp_path, revisions, [S0, S1, S2], list(teams))
    sd = build_sprints(cache, work_item_types=TYPES, done_categories=DONE)
    cal = sprint_calendar(cache)
    s = cal[cal["sprint_id"].isin(sprint_ids)]
    rows = open_rows(cache, sd, s, pd.Series(pd.Timestamp(t), index=s.index), work_item_types=TYPES, done_categories=DONE)
    return rows, sd


def test_items_done_before_t_are_not_rows_but_count_as_done(tmp_path):
    rows, _ = at(tmp_path, [
        rev(1, 1, PRE, iteration=IT1),
        rev(1, 2, "2024-03-08T00:00:00.000Z", iteration=IT1, state="Closed", state_category="Completed"),
        rev(2, 1, PRE, iteration=IT1, story_points=5.0),
    ])
    assert list(rows.columns) == OPEN_ROW_COLUMNS
    assert rows["item_id"].tolist() == [2]
    r = rows.iloc[0]
    assert r["done_points_at_t"] == 3.0 and r["committed_points"] == 8.0
    assert r["is_added"] == 0 and r["points_at_commit"] == 5.0 and r["state_category_at_commit"] == "Proposed"
    assert r["y"] == 0.0


def test_an_item_added_after_the_cutoff_is_a_row_with_its_outcome(tmp_path):
    rows, _ = at(tmp_path, [
        rev(1, 1, PRE, iteration=IT1),
        rev(6, 1, "2024-02-20T00:00:00.000Z", iteration=IT0, story_points=5.0),  # the team's prior median size: 5
        rev(2, 1, PRE, story_points=None),
        rev(2, 2, "2024-03-07T00:00:00.000Z", iteration=IT1, story_points=None),
        rev(2, 3, "2024-03-15T00:00:00.000Z", iteration=IT1, story_points=None, state="Closed",
            state_category="Completed"),
    ])
    added = rows.set_index("item_id").loc[2]
    assert added["is_added"] == 1 and added["y"] == 1.0
    assert added["is_unestimated"] and added["points"] == 5.0
    assert pd.isna(added["points_at_commit"]) and pd.isna(added["state_category_at_commit"])
    assert added["committed_points"] == 3.0


def test_committed_items_moved_out_before_t_drop_out_and_after_t_stay_with_y_0(tmp_path):
    rows, _ = at(tmp_path, [
        rev(1, 1, PRE, iteration=IT1),
        rev(1, 2, "2024-03-08T00:00:00.000Z", iteration=IT2),
        rev(2, 1, PRE, iteration=IT1),
        rev(2, 2, "2024-03-14T00:00:00.000Z", iteration=IT2),
        rev(3, 1, PRE, iteration=IT1),
        rev(3, 2, "2024-03-15T00:00:00.000Z", iteration=IT1, state="Closed", state_category="Completed"),
    ])
    r = rows.set_index("item_id")
    assert r.index.tolist() == [2, 3]
    assert r["y"].to_dict() == {2: 0.0, 3: 1.0}
    assert (r["committed_points"] == 9.0).all() and (r["done_points_at_t"] == 0.0).all()


def test_state_changes_and_reassignment_on_a_known_history(tmp_path):
    rows, _ = at(tmp_path, [
        rev(1, 1, PRE, assigned_to_sk="u1"),
        rev(1, 2, PLAN, iteration=IT1, assigned_to_sk="u1"),
        rev(1, 3, "2024-03-06T00:00:00.000Z", iteration=IT1, assigned_to_sk="u1", state="Active",
            state_category="InProgress"),
        rev(1, 4, "2024-03-08T00:00:00.000Z", iteration=IT1, assigned_to_sk="u2", state="Active",
            state_category="InProgress"),
        rev(1, 5, "2024-03-09T00:00:00.000Z", iteration=IT1, assigned_to_sk="u2", state="Review",
            state_category="InProgress"),
        rev(1, 6, "2024-03-12T00:00:00.000Z", iteration=IT1, assigned_to_sk="u2", state="Closed",
            state_category="Completed"),
        rev(2, 1, PRE, iteration=IT1, assigned_to_sk="u1"),
        rev(2, 2, "2024-03-07T00:00:00.000Z", iteration=IT1, assigned_to_sk="u1", story_points=5.0),
        rev(3, 1, PRE, iteration=IT1),
    ])
    r = rows.set_index("item_id")
    assert r.loc[1, "n_state_changes"] == 2 and r.loc[1, "reassigned"] == 1
    assert r.loc[1, "state_changed"] == pd.Timestamp("2024-03-09T00:00:00Z")
    assert r.loc[1, "state_category"] == "InProgress" and r.loc[1, "revisions_so_far"] == 5 and r.loc[1, "y"] == 1.0
    assert r.loc[2, "n_state_changes"] == 0 and r.loc[2, "reassigned"] == 0
    assert r.loc[2, "state_changed"] == pd.Timestamp(PRE)
    assert r.loc[2, "last_changed"] == pd.Timestamp("2024-03-07T00:00:00Z")
    assert r.loc[2, "points"] == 5.0 and r.loc[2, "points_at_commit"] == 3.0
    assert r.loc[3, "reassigned"] == 0


def test_an_area_move_to_another_team_makes_it_that_teams_added_item(tmp_path):
    rows, _ = at(tmp_path, [
        rev(1, 1, PRE, iteration=IT1),
        rev(1, 2, "2024-03-08T00:00:00.000Z", iteration=IT1, area="Alpha\\Blue"),
        rev(2, 1, PRE, iteration=IT1, area="Alpha\\Blue"),
        rev(3, 1, PRE, iteration=IT1),
    ], teams=(RED, BLUE), sprint_ids=(SPRINT1, BLUE1))
    red, blue = rows[rows["sprint_id"] == SPRINT1], rows[rows["sprint_id"] == BLUE1]
    assert red["item_id"].tolist() == [3] and (red["committed_points"] == 6.0).all()
    assert blue.set_index("item_id")["is_added"].to_dict() == {1: 1, 2: 0}
    assert (blue["committed_points"] == 3.0).all()


def test_before_the_cutoff_the_open_items_are_the_committed_scope(tmp_path):
    rows, sd = at(tmp_path, [
        rev(1, 1, PRE, iteration=IT1),
        rev(2, 1, PLAN, iteration=IT1, story_points=5.0),
        rev(3, 1, "2024-03-05T00:00:00.000Z", iteration=IT1),  # joins after t, before the cutoff
    ], t="2024-03-04T18:00:00.000Z")
    r = rows.set_index("item_id")
    assert r.index.tolist() == [1, 2]
    assert (r["is_added"] == 0).all() and (r["done_points_at_t"] == 0.0).all() and (r["reassigned"] == 0).all()
    assert (r["committed_points"] == 8.0).all()
    assert sd.sprints.set_index("sprint_id").loc[SPRINT1, "committed_points"] == 11.0
    assert r["points_at_commit"].tolist() == [3.0, 5.0] and (r["state_category_at_commit"] == "Proposed").all()


def test_rows_at_the_cutoff_are_the_committed_items(synth16):
    sd = synth16.sprints
    rows = build_checkpoint_rows(synth16.cache, sd, work_item_types=SYNTH_TYPES, done_categories=DONE,
                                 checkpoints=(0.0,))
    assert list(rows.columns) == CHECKPOINT_ROW_COLUMNS
    items = sd.items.reset_index(drop=True)
    assert list(zip(rows["sprint_id"], rows["item_id"])) == list(zip(items["sprint_id"], items["item_id"]))
    for col in ("points", "raw_points", "is_unestimated", "carryover_count", "revisions_so_far"):
        pd.testing.assert_series_equal(rows[col], items[col], check_dtype=False, check_names=False)
    assert rows["assigned_to_sk"].fillna("").tolist() == items["assigned_to_sk"].fillna("").tolist()
    assert rows["state_category"].tolist() == items["state_category_at_commit"].tolist()
    assert (rows["is_added"] == 0).all() and (rows["reassigned"] == 0).all() and (rows["done_points_at_t"] == 0).all()
    pd.testing.assert_series_equal(rows["y"], items["done"], check_dtype=False, check_names=False)


def test_checkpoint_progress_counts_committed_points_done_by_t(synth16):
    sd = synth16.sprints
    prog = checkpoint_progress(synth16.cache, sd, done_categories=DONE, checkpoints=(0.0, 1.0))
    assert list(prog.columns) == PROGRESS_COLUMNS and len(prog) == 2 * len(sd.sprints)
    first, last = prog[prog["checkpoint"] == 0.0], prog[prog["checkpoint"] == 1.0].set_index("sprint_id")
    assert (first["done_points"] == 0.0).all()
    closed = sd.sprints[sd.sprints["done_points"].notna()].set_index("sprint_id")
    assert last.loc[closed.index, "done_points"].to_numpy() == pytest.approx(closed["done_points"].to_numpy())
    assert (last["t"] == sd.sprints.set_index("sprint_id").loc[last.index, "end"]).all()


def test_checkpoint_time_is_a_share_of_the_way_from_cutoff_to_end():
    cutoff = pd.Series(pd.to_datetime(["2024-03-05T05:00:00Z"]))
    end = pd.Series(pd.to_datetime(["2024-03-15T05:00:00Z"]))
    assert checkpoint_time(cutoff, end, 0.0).iloc[0] == cutoff.iloc[0]
    assert checkpoint_time(cutoff, end, 0.25).iloc[0] == pd.Timestamp("2024-03-07T17:00:00Z")


def test_state_changes_keep_the_first_revision_and_each_change_of_state():
    revs = pd.DataFrame({
        "item_id": [1, 1, 1, 1, 2],
        "rev": [1, 2, 3, 4, 1],
        "changed": pd.to_datetime(["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04", "2024-03-01"], utc=True),
        "state": ["New", "New", "Active", "New", "Active"],
    })
    ev = state_changes(revs)
    assert list(zip(ev["item_id"], ev["changed"].dt.day, ev["first"])) == [
        (1, 1, True), (1, 3, False), (1, 4, False), (2, 1, True),
    ]


def test_y_counts_a_close_within_the_sprint_data_close_grace(tmp_path):
    revs = [
        rev(1, 1, PRE, iteration=IT1),
        rev(1, 2, "2024-03-18T11:00:00.000Z", iteration=IT1, state="Closed", state_category="Completed"),
    ]
    cache = build_cache(tmp_path, revs, [S0, S1, S2], [RED])
    cal = sprint_calendar(cache)
    s = cal[cal["sprint_id"] == SPRINT1]
    for grace, y in ((0.0, 0.0), (12.0, 1.0)):
        sd = build_sprints(cache, work_item_types=TYPES, done_categories=DONE, close_grace_hours=grace)
        rows = open_rows(cache, sd, s, pd.Series(pd.Timestamp(T), index=s.index), work_item_types=TYPES,
                         done_categories=DONE)
        assert rows["y"].tolist() == [y]
