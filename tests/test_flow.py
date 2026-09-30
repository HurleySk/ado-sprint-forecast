import pandas as pd

from helpers import build_cache, iteration, rev, team
from sprint_forecast.flow import FATES, idle_sprints, item_fates, sprint_state_changes
from sprint_forecast.sprints import build_sprints

TYPES = ["User Story", "Bug"]
IT0, IT1, IT2 = "Alpha\Sprint 0", "Alpha\Sprint 1", "Alpha\Sprint 2"
S0 = iteration(IT0, "2024-02-19T05:00:00.000Z", "2024-03-04T04:59:59.999Z")
S1 = iteration(IT1, "2024-03-04T05:00:00.000Z", "2024-03-18T04:59:59.999Z")
S2 = iteration(IT2, "2024-03-18T05:00:00.000Z", "2024-04-01T04:59:59.999Z")
BACKLOG = iteration("Alpha\Backlog", None, None)
RED = team("Team Red", ["Alpha\Red"], [IT0, IT1, IT2])
PLAN = "2024-03-04T12:00:00.000Z"
MID = "2024-03-10T00:00:00.000Z"
LATE_MID = "2024-03-15T00:00:00.000Z"
NEXT_MORNING = "2024-03-18T11:00:00.000Z"
LATER = "2024-03-25T00:00:00.000Z"
CLOSED = {"state": "Closed", "state_category": "Completed"}
REMOVED = {"state": "Removed", "state_category": "Removed"}
SPRINT1 = "Alpha/Team Red|Alpha\Sprint 1"


def sprint1(tmp_path, revisions, grace=12.0):
    cache = build_cache(tmp_path, revisions, [S0, S1, S2, BACKLOG], [RED])
    sd = build_sprints(cache, work_item_types=TYPES, close_grace_hours=grace)
    return cache, sd


def test_each_committed_item_gets_where_it_went(tmp_path):
    cache, sd = sprint1(tmp_path, [
        rev(1, 1, PLAN, iteration=IT1), rev(1, 2, MID, iteration=IT1, **CLOSED),                  # done
        rev(2, 1, PLAN, iteration=IT1), rev(2, 2, NEXT_MORNING, iteration=IT1, **CLOSED),         # done in grace
        rev(3, 1, PLAN, iteration=IT1), rev(3, 2, LATER, iteration=IT1, **CLOSED),                # closed later
        rev(4, 1, PLAN, iteration=IT1), rev(4, 2, LATER, iteration=IT2),                          # carried
        rev(5, 1, PLAN, iteration=IT1), rev(5, 2, LATE_MID, iteration=IT2),                       # carried mid-sprint
        rev(6, 1, PLAN, iteration=IT1), rev(6, 2, LATER, iteration="Alpha\Backlog"),             # back to backlog
        rev(7, 1, PLAN, iteration=IT1), rev(7, 2, LATE_MID, iteration=IT1, **REMOVED),            # removed
        rev(8, 1, PLAN, iteration=IT1),                                                           # still open
        rev(9, 1, PLAN, iteration=IT1), rev(9, 2, LATER, iteration=IT2, **CLOSED),                # moved and closed
    ])
    fates = item_fates(cache, sd).set_index("item_id")
    assert set(fates["fate"]) <= set(FATES)
    assert fates.loc[fates["sprint_id"] == SPRINT1, "fate"].to_dict() == {
        1: "done", 2: "done_in_grace", 3: "closed_later", 4: "carried", 5: "carried", 6: "backlog",
        7: "removed", 8: "open", 9: "carried",
    }


def test_added_items_get_a_fate_and_open_sprints_none(tmp_path):
    cache, sd = sprint1(tmp_path, [
        rev(1, 1, PLAN, iteration=IT1),
        rev(2, 1, "2024-03-01T00:00:00.000Z"), rev(2, 2, MID, iteration=IT1), rev(2, 3, LATER, iteration=IT2),
    ])
    fates = item_fates(cache, sd)
    row = fates[(fates["sprint_id"] == SPRINT1) & (fates["item_id"] == 2)]
    assert row["fate"].tolist() == ["carried"]


def test_state_changes_count_moves_within_the_sprint_window(tmp_path):
    cache, sd = sprint1(tmp_path, [
        rev(1, 1, "2024-02-01T00:00:00.000Z", iteration=IT1),
        rev(1, 2, PLAN, iteration=IT1, story_points=5.0),  # an edit, not a state change
        rev(1, 3, MID, iteration=IT1, state="Active", state_category="InProgress"),
        rev(1, 4, LATE_MID, iteration=IT1, state="Resolved", state_category="Resolved"),
        rev(1, 5, LATER, iteration=IT1, **CLOSED),  # after the end
        rev(2, 1, PLAN, iteration=IT1),
    ])
    changes = sprint_state_changes(cache, sd.items).set_axis(sd.items["item_id"].to_numpy())
    assert changes.to_dict() == {1: 2, 2: 0}


def test_idle_sprints_counts_sprints_in_a_row_without_a_state_change(tmp_path):
    cache = build_cache(tmp_path, [
        rev(1, 1, "2024-02-20T00:00:00.000Z", iteration=IT0, state="Active", state_category="InProgress"),
        rev(1, 2, "2024-03-05T00:00:00.000Z", iteration=IT1, state="Active", state_category="InProgress"),
        rev(1, 3, "2024-03-19T00:00:00.000Z", iteration=IT2, state="Active", state_category="InProgress"),
        rev(2, 1, "2024-02-20T00:00:00.000Z", iteration=IT0),
        rev(2, 2, "2024-03-20T00:00:00.000Z", iteration=IT2, state="Active", state_category="InProgress"),
        rev(3, 1, "2024-03-19T00:00:00.000Z", iteration=IT2),
    ], [S0, S1, S2, BACKLOG], [RED])
    rows = pd.DataFrame({"item_id": [1, 2, 3], "start": pd.Timestamp("2024-03-18T05:00:00Z"),
                         "t": pd.Timestamp("2024-03-25T00:00:00Z")})
    assert idle_sprints(cache, rows).tolist() == [3, 0, 1]
