import math

import pandas as pd
import pytest

from helpers import build_cache, iteration, rev, team
from sprint_forecast.cycle import CYCLE_COLUMNS, CYCLE_STATE_COLUMNS, build_cycle, business_days

TYPES = ["User Story", "Bug"]
S1 = iteration("Alpha\\Sprint 1", "2024-03-04T05:00:00.000Z", "2024-03-18T04:59:59.999Z")
RED = team("Team Red", ["Alpha\\Red"], ["Alpha\\Sprint 1"])
BLUE = team("Team Blue", ["Alpha\\Blue"], [])
MON_NOON = "2024-03-04T12:00:00.000Z"
WED = "2024-03-06T00:00:00.000Z"
THU = "2024-03-07T00:00:00.000Z"
FRI_NOON = "2024-03-08T12:00:00.000Z"
NEXT_MON_NOON = "2024-03-11T12:00:00.000Z"
ACTIVE = {"state": "Active", "state_category": "InProgress"}
TESTING = {"state": "Testing", "state_category": "InProgress"}
CLOSED = {"state": "Closed", "state_category": "Completed"}


def build(tmp_path, revisions, done=("Completed",)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cache = build_cache(tmp_path, revisions, [S1], [RED, BLUE])
    return build_cycle(cache, work_item_types=TYPES, done_categories=list(done))


def test_business_days_skip_weekends_and_keep_fractions():
    start = pd.Series(pd.to_datetime([MON_NOON, FRI_NOON, "2024-03-09T08:00:00.000Z"], utc=True))  # Sat 08:00
    end = pd.Series(pd.to_datetime(["2024-03-04T18:00:00.000Z", NEXT_MON_NOON, NEXT_MON_NOON], utc=True))
    assert business_days(start, end).tolist() == pytest.approx([0.25, 1.0, 0.5])


def test_cycle_runs_from_first_active_state_to_first_done_with_time_per_state(tmp_path):
    cy = build(tmp_path, [
        rev(1, 1, "2024-03-01T00:00:00.000Z"),
        rev(1, 2, MON_NOON, iteration="Alpha\\Sprint 1", **ACTIVE),
        rev(1, 3, WED, iteration="Alpha\\Sprint 1", **TESTING),
        rev(1, 4, THU, iteration="Alpha\\Sprint 1", story_points=5.0, **TESTING),
        rev(1, 5, NEXT_MON_NOON, iteration="Alpha\\Sprint 1", assigned_to_sk="u1", story_points=5.0, **CLOSED),
        rev(1, 6, "2024-03-12T00:00:00.000Z", iteration="Alpha\\Sprint 1", **ACTIVE),  # reopened: ignored
        rev(1, 7, "2024-03-13T00:00:00.000Z", iteration="Alpha\\Sprint 1", **CLOSED),
    ])
    assert list(cy.items.columns) == CYCLE_COLUMNS and list(cy.states.columns) == CYCLE_STATE_COLUMNS
    row = cy.items.set_index("item_id").loc[1]
    assert row["started"] == pd.Timestamp(MON_NOON) and row["closed"] == pd.Timestamp(NEXT_MON_NOON)
    assert row["days"] == pytest.approx(5.0)
    assert (row["points_at_start"], row["points"], row["re_estimated"]) == (3.0, 5.0, True)
    assert (row["team"], row["team_key"], row["assigned_to_sk"]) == ("Team Red", "Alpha/Team Red", "u1")
    states = cy.states.set_index("state")["days"].to_dict()
    assert states == pytest.approx({"Active": 1.5, "Testing": 3.5})


def test_items_closed_without_active_time_have_no_cycle_and_no_state_rows(tmp_path):
    cy = build(tmp_path, [
        rev(2, 1, "2024-03-01T00:00:00.000Z", story_points=None),
        rev(2, 2, WED, **CLOSED, story_points=None),
    ])
    row = cy.items.set_index("item_id").loc[2]
    assert pd.isna(row["started"]) and math.isnan(row["days"])
    assert math.isnan(row["points"]) and not row["re_estimated"]
    assert cy.states.empty


def test_only_finished_items_of_the_counted_types(tmp_path):
    cy = build(tmp_path, [
        rev(3, 1, MON_NOON, **ACTIVE),                    # never finished
        rev(4, 1, MON_NOON, type="Task", **ACTIVE),
        rev(4, 2, WED, type="Task", **CLOSED),            # not a counted type
        rev(5, 1, MON_NOON, type="Bug", **ACTIVE),
        rev(5, 2, WED, type="Bug", **CLOSED),
    ])
    assert cy.items["item_id"].tolist() == [5]


def test_team_comes_from_the_iteration_then_the_project_by_area(tmp_path):
    cy = build(tmp_path, [
        rev(6, 1, MON_NOON, area="Alpha\\Blue", **ACTIVE),
        rev(6, 2, WED, area="Alpha\\Blue", **CLOSED),     # backlog iteration: any team in the project by area
        rev(7, 1, MON_NOON, area="Alpha\\Elsewhere", **ACTIVE),
        rev(7, 2, WED, area="Alpha\\Elsewhere", **CLOSED),
    ])
    teams = cy.items.set_index("item_id")["team"]
    assert teams[6] == "Team Blue" and pd.isna(teams[7])


def test_resolved_ends_the_cycle_when_it_counts_as_done(tmp_path):
    revs = [
        rev(8, 1, MON_NOON, **ACTIVE),
        rev(8, 2, WED, state="Resolved", state_category="Resolved"),
        rev(8, 3, NEXT_MON_NOON, **CLOSED),
    ]
    assert build(tmp_path / "a", revs).items.set_index("item_id").loc[8, "days"] == pytest.approx(5.0)
    widened = build(tmp_path / "b", revs, done=("Resolved", "Completed"))
    assert widened.items.set_index("item_id").loc[8, "days"] == pytest.approx(1.5)
