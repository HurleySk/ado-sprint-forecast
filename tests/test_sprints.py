import math

import pandas as pd
import pytest

from helpers import build_cache, iteration, rev, team
from sprint_forecast.features import build_features
from sprint_forecast.sprints import (
    ADDED_COLUMNS,
    ITEM_COLUMNS,
    SPRINT_COLUMNS,
    build_sprints,
    data_report,
    iteration_group,
    match_team,
    scope_at,
    sprint_calendar,
)

TYPES = ["User Story", "Bug"]
S0 = iteration("Alpha\\Sprint 0", "2024-02-19T05:00:00.000Z", "2024-03-04T04:59:59.999Z")
S1 = iteration("Alpha\\Sprint 1", "2024-03-04T05:00:00.000Z", "2024-03-18T04:59:59.999Z")
S2 = iteration("Alpha\\Sprint 2", "2024-03-18T05:00:00.000Z", "2024-04-01T04:59:59.999Z")
UNDATED = iteration("Alpha\\Someday", None, None)
PRE = "2024-03-01T00:00:00.000Z"    # before Sprint 1 starts
PLAN = "2024-03-04T12:00:00.000Z"   # after start, before cutoff (start + 1 day)
MID = "2024-03-10T00:00:00.000Z"    # inside the sprint, after cutoff
POST = "2024-03-20T00:00:00.000Z"   # after Sprint 1 ends
RED = team("Team Red", ["Alpha\\Red"], ["Alpha\\Sprint 0", "Alpha\\Sprint 1", "Alpha\\Sprint 2"])
SPRINT1 = "Alpha/Team Red|Alpha\\Sprint 1"


def build(tmp_path, revisions, iterations=(S0, S1, S2, UNDATED), teams=(RED,), done=("Completed",)):
    cache = build_cache(tmp_path, revisions, list(iterations), list(teams))
    return build_sprints(cache, work_item_types=TYPES, done_categories=list(done)), cache


def items_of(sd, sprint_id=SPRINT1):
    return sd.items[sd.items["sprint_id"] == sprint_id].set_index("item_id")


def test_committed_vs_added_after_cutoff(tmp_path):
    sd, _ = build(tmp_path, [
        rev(1, 1, PRE), rev(1, 2, PLAN, iteration="Alpha\\Sprint 1"),
        rev(2, 1, PRE), rev(2, 2, MID, iteration="Alpha\\Sprint 1"),
    ])
    assert list(items_of(sd).index) == [1]
    assert sd.report["added_mid_sprint"] == 1
    assert sd.sprints.set_index("sprint_id").loc[SPRINT1, "n_added_mid"] == 1
    assert list(sd.sprints.columns) == SPRINT_COLUMNS
    assert list(sd.items.columns) == ITEM_COLUMNS


def test_items_added_after_cutoff_are_listed_with_outcome_and_end_holder(tmp_path):
    late = "2024-03-12T00:00:00.000Z"
    sd, _ = build(tmp_path, [
        rev(1, 1, PLAN, iteration="Alpha\\Sprint 1"),
        rev(6, 1, "2024-02-01T00:00:00.000Z", iteration="Alpha\\Sprint 0"),  # gives the team a prior median of 3
        rev(2, 1, PRE, story_points=None),
        rev(2, 2, MID, iteration="Alpha\\Sprint 1", assigned_to_sk="u1", story_points=None),
        rev(2, 3, late, iteration="Alpha\\Sprint 1", assigned_to_sk="u2", story_points=None,
            state="Closed", state_category="Completed"),
        rev(3, 1, PRE),
        rev(3, 2, MID, iteration="Alpha\\Sprint 1"),
        rev(3, 3, late, iteration="Alpha\\Sprint 1", state="Removed", state_category="Removed"),
        rev(4, 1, PRE, state="Closed", state_category="Completed"),
        rev(4, 2, MID, iteration="Alpha\\Sprint 1", state="Closed", state_category="Completed"),
        rev(5, 1, PRE, type="Bug"),
        rev(5, 2, MID, iteration="Alpha\\Sprint 1", type="Bug", story_points=5.0),
    ])
    added = sd.added.set_index("item_id")
    assert list(sd.added.columns) == ADDED_COLUMNS
    assert sorted(added.index) == [2, 5]  # removed (3) and already done before the sprint (4) are not added work
    assert added.loc[2, "sprint_id"] == SPRINT1 and added.loc[2, "assigned_to_sk"] == "u2"
    assert added["done"].to_dict() == {2: True, 5: False}
    assert added.loc[2, "is_unestimated"] and added.loc[2, "points"] == 3.0  # the team's prior median
    assert added.loc[5, "points"] == 5.0 and not added.loc[5, "is_unestimated"]
    assert sd.sprints.set_index("sprint_id").loc[SPRINT1, "n_added_mid"] == 2
    assert sd.report["added_mid_sprint"] == 2


def test_committed_items_record_who_held_them_at_sprint_end(tmp_path):
    sd, _ = build(tmp_path, [
        rev(1, 1, PLAN, iteration="Alpha\\Sprint 1", assigned_to_sk="u1"),
        rev(1, 2, MID, iteration="Alpha\\Sprint 1", assigned_to_sk="u2"),
        rev(2, 1, PLAN, iteration="Alpha\\Sprint 1", assigned_to_sk="u1"),
    ])
    items = items_of(sd)
    assert items["assigned_to_sk"].to_dict() == {1: "u1", 2: "u1"}
    assert items["assigned_at_end_sk"].to_dict() == {1: "u2", 2: "u1"}


def test_moved_out_mid_sprint_is_not_done(tmp_path):
    sd, _ = build(tmp_path, [
        rev(1, 1, PLAN, iteration="Alpha\\Sprint 1"),
        rev(1, 2, MID, iteration="Alpha\\Sprint 2"),
        rev(1, 3, "2024-03-12T00:00:00.000Z", iteration="Alpha\\Sprint 2", state="Closed", state_category="Completed"),
    ])
    row = items_of(sd).loc[1]
    assert not row["done"]
    assert row["state_category_at_end"] == "Completed"


def test_removed_during_sprint_not_done_and_removed_before_cutoff_not_committed(tmp_path):
    sd, _ = build(tmp_path, [
        rev(1, 1, PLAN, iteration="Alpha\\Sprint 1"),
        rev(1, 2, MID, iteration="Alpha\\Sprint 1", state="Removed", state_category="Removed"),
        rev(2, 1, PRE, iteration="Alpha\\Sprint 1", state="Removed", state_category="Removed"),
        rev(3, 1, PRE, iteration="Alpha\\Sprint 1"),
    ])
    items = items_of(sd)
    assert sorted(items.index) == [1, 3]
    assert not items.loc[1, "done"]


def _resolved_completed_revs():
    return [
        rev(1, 1, PLAN, iteration="Alpha\\Sprint 1"),
        rev(1, 2, MID, iteration="Alpha\\Sprint 1", state="Resolved", state_category="Resolved"),
        rev(1, 3, POST, iteration="Alpha\\Sprint 1", state="Closed", state_category="Completed"),
        rev(2, 1, PLAN, iteration="Alpha\\Sprint 1"),
        rev(2, 2, MID, iteration="Alpha\\Sprint 1", state="Closed", state_category="Completed"),
        rev(3, 1, PRE, iteration="Alpha\\Sprint 1", state="Resolved", state_category="Resolved"),
    ]


def test_resolved_is_not_done_under_default_done_set(tmp_path):
    sd, _ = build(tmp_path, _resolved_completed_revs())
    items = items_of(sd)
    assert sorted(items.index) == [1, 2, 3]
    assert items["done"].to_dict() == {1: False, 2: True, 3: False}
    assert sd.report["end_state_mix"] == {"Resolved": 2, "Completed": 1}


def test_resolved_counts_as_done_when_widened(tmp_path):
    sd, cache = build(tmp_path, _resolved_completed_revs(), done=("Resolved", "Completed"))
    items = items_of(sd)
    assert sorted(items.index) == [1, 2]
    assert items["done"].all()
    report = data_report(cache, sd)
    assert report["resolved_share_of_resolved_or_completed"] == pytest.approx(0.5)


def test_carryover_counts_distinct_earlier_ended_iterations(tmp_path):
    sd, _ = build(tmp_path, [
        rev(1, 1, "2024-02-01T00:00:00.000Z", iteration="Alpha\\Sprint 0"),
        rev(1, 2, "2024-02-20T00:00:00.000Z", iteration="Alpha\\Someday"),
        rev(1, 3, "2024-02-21T00:00:00.000Z", iteration="Alpha\\Sprint 0"),
        rev(1, 4, PLAN, iteration="Alpha\\Sprint 1"),
        rev(2, 1, PLAN, iteration="Alpha\\Sprint 1"),
        rev(3, 1, PLAN, iteration="Alpha\\Sprint 1"),
        rev(3, 2, MID, iteration="Alpha\\Sprint 0"),
    ])
    items = items_of(sd)
    assert items["carryover_count"].to_dict() == {1: 1, 2: 0, 3: 0}


def test_bounce_before_cutoff_counts_once(tmp_path):
    sd, _ = build(tmp_path, [
        rev(1, 1, PRE, iteration="Alpha\\Sprint 1"),
        rev(1, 2, "2024-03-02T00:00:00.000Z", iteration="Alpha\\Someday"),
        rev(1, 3, PLAN, iteration="Alpha\\Sprint 1"),
        rev(1, 4, MID, iteration="Alpha\\Sprint 1", state="Closed", state_category="Completed"),
    ])
    assert list(items_of(sd).index) == [1]
    assert items_of(sd).loc[1, "done"]
    assert sd.report["added_mid_sprint"] == 0


def test_match_team_rules():
    areas = {"Team Red": ["Alpha\\Red"], "Team Web": ["Alpha\\Red\\Web"], "Team Blue": ["Alpha\\Blue"]}
    three = ["Team Red", "Team Web", "Team Blue"]
    assert match_team("Alpha\\Red", three, areas) == "Team Red"
    assert match_team("alpha\\RED", three, areas) == "Team Red"
    assert match_team("Alpha\\Red\\Web\\UI", three, areas) == "Team Web"
    assert match_team("Alpha\\Red\\Api", three, areas) == "Team Red"
    assert match_team("Alpha\\Redder", three, areas) is None
    assert match_team("Alpha", three, areas) is None
    assert match_team(None, three, areas) is None
    assert match_team("Alpha\\Other", ["Team Red"], areas) == "Team Red"
    tie = {"Team Red": ["Alpha\\Red"], "Team Blue": ["Alpha\\Red"]}
    assert match_team("Alpha\\Red", ["Team Red", "Team Blue"], tie) is None


def test_team_assignment_exact_prefix_single_and_unassigned(tmp_path):
    teams = (
        team("Team Red", ["Alpha\\Red"], ["Alpha\\Sprint 1", "Alpha\\Sprint 2"]),
        team("Team Web", ["Alpha\\Red\\Web"], ["Alpha\\Sprint 1"]),
    )
    sd, _ = build(tmp_path, [
        rev(1, 1, PLAN, iteration="Alpha\\Sprint 1", area="Alpha\\Red"),
        rev(2, 1, PLAN, iteration="Alpha\\Sprint 1", area="Alpha\\Red\\Web\\UI"),
        rev(3, 1, PLAN, iteration="Alpha\\Sprint 1", area="Alpha"),
        rev(4, 1, "2024-03-18T12:00:00.000Z", iteration="Alpha\\Sprint 2", area="Alpha"),
    ], teams=teams)
    by_item = sd.items.set_index("item_id")["team"].to_dict()
    assert by_item == {1: "Team Red", 2: "Team Web", 4: "Team Red"}
    assert sd.report["unassigned_items"] == 1


def test_project_level_fallback_when_no_team_subscribes(tmp_path):
    beta_it = iteration("Beta\\Sprint 1", "2024-03-04T05:00:00.000Z", "2024-03-18T04:59:59.999Z", project="Beta")
    sd, cache = build(
        tmp_path,
        [
            rev(1, 1, PLAN, iteration="Alpha\\Sprint 1"),
            rev(9, 1, PLAN, project="Beta", iteration="Beta\\Sprint 1", area="Beta\\Anything"),
        ],
        iterations=(S1, beta_it),
        teams=(RED, team("Team Green", ["Beta\\Green"], [], project="Beta")),
    )
    beta = sd.sprints[sd.sprints["project"] == "Beta"]
    assert list(beta["team"]) == ["Beta"]
    assert list(beta["team_key"]) == ["Beta/Beta"]
    assert sd.report["fallback_projects"] == ["Beta"]
    assert set(sd.sprints["team"]) == {"Team Red", "Beta"}
    report = data_report(cache, sd)
    by_project = {p["project"]: p for p in report["projects"]}
    assert by_project["Beta"]["fallback"] is True
    assert by_project["Beta"]["teams_running_sprints"] == 0
    assert by_project["Alpha"]["teams_running_sprints"] == 1


def test_undated_iterations_never_become_sprints(tmp_path):
    cache = build_cache(tmp_path, [], [S1, UNDATED], [team("Team Red", ["Alpha\\Red"], ["Alpha\\Sprint 1", "Alpha\\Someday"])])
    cal = sprint_calendar(cache)
    assert list(cal["iteration"]) == ["Alpha\\Sprint 1"]
    assert cal.loc[0, "cutoff"] == pd.Timestamp("2024-03-05T05:00:00Z")


def test_half_dated_and_inverted_iterations_are_not_sprints(tmp_path):
    odd = [
        iteration("Alpha\\Start Only", "2024-03-04T05:00:00.000Z", None),
        iteration("Alpha\\End Only", None, "2024-03-18T04:59:59.999Z"),
        iteration("Alpha\\Inverted", "2024-03-18T05:00:00.000Z", "2024-03-04T05:00:00.000Z"),
        iteration("Alpha\\Zero", "2024-03-04T05:00:00.000Z", "2024-03-04T05:00:00.000Z"),
    ]
    subs = [i["path"] for i in odd] + ["Alpha\\Sprint 1"]
    sd, _ = build(
        tmp_path,
        [rev(1, 1, PLAN, iteration="Alpha\\Sprint 1"), rev(2, 1, PLAN, iteration="Alpha\\Inverted")],
        iterations=[S1, *odd],
        teams=(team("Team Red", ["Alpha\\Red"], subs),),
    )
    assert list(sd.sprints["iteration"]) == ["Alpha\\Sprint 1"]
    assert list(sd.items["item_id"]) == [1]


def test_points_fallback_imputation_and_pct_done(tmp_path):
    sd, _ = build(tmp_path, [
        # Sprint 0: two estimated items (3 and 5 points), one done
        rev(10, 1, "2024-02-19T12:00:00.000Z", iteration="Alpha\\Sprint 0", story_points=3.0),
        rev(10, 2, "2024-02-25T00:00:00.000Z", iteration="Alpha\\Sprint 0", story_points=3.0,
            state="Closed", state_category="Completed"),
        rev(11, 1, "2024-02-19T12:00:00.000Z", iteration="Alpha\\Sprint 0", story_points=5.0),
        rev(12, 1, "2024-02-19T12:00:00.000Z", iteration="Alpha\\Sprint 0", story_points=None),
        # Sprint 1: effort fallback, unestimated (imputed from Sprint 0 median = 4), zero points = missing
        rev(1, 1, PLAN, iteration="Alpha\\Sprint 1", story_points=None, effort=8.0),
        rev(1, 2, MID, iteration="Alpha\\Sprint 1", story_points=None, effort=8.0, state="Closed", state_category="Completed"),
        rev(2, 1, PLAN, iteration="Alpha\\Sprint 1", story_points=None, effort=None),
        rev(3, 1, PLAN, iteration="Alpha\\Sprint 1", story_points=0.0),
    ])
    s0 = items_of(sd, "Alpha/Team Red|Alpha\\Sprint 0")
    assert s0.loc[12, "points"] == 1.0  # no prior history anywhere -> 1
    s1 = items_of(sd)
    assert s1.loc[1, "points"] == 8.0 and not s1.loc[1, "is_unestimated"]
    assert s1.loc[2, "points"] == 4.0 and s1.loc[2, "is_unestimated"]
    assert s1.loc[3, "points"] == 4.0 and s1.loc[3, "is_unestimated"]
    row = sd.sprints.set_index("sprint_id").loc[SPRINT1]
    assert row["committed_points"] == 16.0
    assert row["done_points"] == 8.0
    assert row["pct_done"] == pytest.approx(0.5)
    assert row["pct_done_count"] == pytest.approx(1 / 3)
    assert row["n_unestimated"] == 2


def test_empty_sprints_dropped_and_counted(tmp_path):
    sd, _ = build(tmp_path, [rev(1, 1, PLAN, iteration="Alpha\\Sprint 1")])
    assert list(sd.sprints["sprint_id"]) == [SPRINT1]
    assert sd.report["dropped_empty_sprints"] == 2


def test_build_on_empty_cache(tmp_path):
    sd, _ = build(tmp_path, [])
    assert sd.sprints.empty and sd.items.empty
    assert math.isnan(sd.report["unestimated_share"])


def test_scope_at_uses_given_cutoff_and_history(tmp_path):
    revs = [
        rev(10, 1, "2024-02-19T12:00:00.000Z", iteration="Alpha\\Sprint 0", story_points=6.0),
        rev(1, 1, PLAN, iteration="Alpha\\Sprint 1", story_points=None),
        rev(2, 1, MID, iteration="Alpha\\Sprint 1"),
    ]
    sd, cache = build(tmp_path, revs)
    sprints, items = scope_at(cache, sd, iteration="Alpha\\Sprint 1", cutoff=pd.Timestamp(MID) + pd.Timedelta(hours=1),
                              work_item_types=TYPES)
    assert sorted(items["item_id"]) == [1, 2]
    assert items.set_index("item_id").loc[1, "points"] == 6.0
    assert items["done"].isna().all()
    assert sprints.loc[0, "cutoff"] == pd.Timestamp(MID) + pd.Timedelta(hours=1)
    assert math.isnan(sprints.loc[0, "pct_done"])
    with pytest.raises(ValueError, match="no team runs"):
        scope_at(cache, sd, iteration="Alpha\\Nope", cutoff=MID, work_item_types=TYPES)
    with pytest.raises(ValueError, match="Team Red"):
        scope_at(cache, sd, iteration="Alpha\\Sprint 1", cutoff=MID, work_item_types=TYPES, team="Team Pink")


def test_synthetic_cache_reconstructs(tmp_path):
    from sprint_forecast.cache import connect, load_cache
    from sprint_forecast.synth import generate

    conn = connect(generate(tmp_path / "cache.db", seed=1, n_sprints=8))
    cache = load_cache(conn)
    conn.close()
    sd = build_sprints(cache, work_item_types=["User Story", "Product Backlog Item", "Bug"])
    assert set(sd.sprints["team"]) == {"Team Red", "Team Blue", "Team Green"}
    assert sd.sprints.groupby("team").size().min() >= 7
    assert sd.report["added_mid_sprint"] > 0
    assert sd.report["unassigned_items"] > 0
    assert 0.3 < sd.sprints["pct_done"].mean() < 0.95
    assert sd.items["carryover_count"].max() >= 1
    assert sd.items["is_unestimated"].any()


def test_sprints_not_ended_at_extraction_have_unknown_outcome(tmp_path):
    s3 = iteration("Alpha\Sprint 3", "2024-04-01T05:00:00.000Z", "2024-04-15T04:59:59.999Z")
    red = team("Team Red", ["Alpha\Red"], ["Alpha\Sprint 1", "Alpha\Sprint 2", "Alpha\Sprint 3"])
    revs = [
        rev(1, 1, PLAN, iteration="Alpha\Sprint 1"),
        rev(1, 2, MID, iteration="Alpha\Sprint 1", state="Closed", state_category="Completed"),
        rev(2, 1, "2024-03-18T12:00:00.000Z", iteration="Alpha\Sprint 2", state="Active", state_category="InProgress"),
        rev(3, 1, "2024-03-20T00:00:00.000Z", iteration="Alpha\Sprint 3"),
    ]
    cache = build_cache(tmp_path, revs, [S1, S2, s3], [red], meta={"extracted_at:Alpha": "2024-03-25T00:00:00.000Z"})
    sd = build_sprints(cache, work_item_types=TYPES)
    pct = sd.sprints.set_index("iteration")["pct_done"]
    assert pct["Alpha\Sprint 1"] == 1.0
    assert math.isnan(pct["Alpha\Sprint 2"]) and math.isnan(pct["Alpha\Sprint 3"])
    assert sd.items.set_index("item_id")["done"].isna().to_dict() == {1: False, 2: True, 3: True}
    assert sd.report["open_sprints"] == 2
    assert sd.report["end_state_mix"] == {"Completed": 1}
    frame = build_features(sd).set_index("item_id")
    assert frame.loc[3, "team_sprint_index"] == 1
    assert math.isnan(frame.loc[3, "y"]) and math.isnan(frame.loc[2, "y"])


def test_no_dated_iterations_builds_no_sprints(tmp_path):
    sd, cache = build(tmp_path, [rev(1, 1, PLAN, iteration="Alpha\Someday")], iterations=(UNDATED,))
    assert sd.report["n_sprints"] == 0 and sd.items.empty
    assert data_report(cache, sd)["projects"][0]["dated_iterations"] == 0


def test_iteration_group_filters_to_a_team_and_explains_bad_input(tmp_path):
    blue = team("Team Blue", ["Alpha\\Blue"], ["Alpha\\Sprint 1"])
    cache = build_cache(tmp_path, [rev(1, 1, PRE)], [S1], [RED, blue])
    cal = sprint_calendar(cache)
    assert set(iteration_group(cal, "Alpha\\Sprint 1")["team"]) == {"Team Red", "Team Blue"}
    assert iteration_group(cal, "Alpha\\Sprint 1", "Team Blue")["team"].tolist() == ["Team Blue"]
    with pytest.raises(ValueError, match="no team runs"):
        iteration_group(cal, "Alpha\\Nope")
    with pytest.raises(ValueError, match="does not run"):
        iteration_group(cal, "Alpha\\Sprint 1", "Team Pink")
