import pandas as pd
import pytest

from conftest import SYNTH_TYPES
from helpers import build_cache, iteration, rev, team
from sprint_forecast.features import FEATURE_VERSION
from sprint_forecast.forecast import check_bundle, score_iteration
from sprint_forecast.rollup import fit_forecaster
from sprint_forecast.sprints import build_sprints, scope_at

DONE = ["Completed"]
RED12 = "Alpha/Team Red|Alpha\\Sprint 12"  # synth16: 2024-06-10 05:00 to 2024-06-24 04:59:59.999 UTC
TYPES = ["User Story", "Bug"]
IT0, IT1 = "Alpha\\Sprint 0", "Alpha\\Sprint 1"
S0 = iteration(IT0, "2024-02-19T05:00:00.000Z", "2024-03-04T04:59:59.999Z")
S1 = iteration(IT1, "2024-03-04T05:00:00.000Z", "2024-03-18T04:59:59.999Z")  # cutoff 2024-03-05T05:00Z
RED = team("Team Red", ["Alpha\\Red"], [IT0, IT1])
PRE = "2024-03-01T00:00:00.000Z"
MID = pd.Timestamp("2024-03-11T00:00:00Z")


@pytest.fixture(scope="module")
def fc(synth16):
    return fit_forecaster(synth16.ckpt, seed=0)


def score(fc, cache, history, t, iteration="Alpha\\Sprint 12", types=SYNTH_TYPES):
    return score_iteration(fc, cache, history, iteration=iteration, t=t, work_item_types=types,
                           done_categories=DONE, commit_grace_days=1.0, team="Team Red")


def test_a_running_sprint_is_scored_at_t_on_top_of_the_points_done(fc, synth16):
    t = pd.Timestamp("2024-06-19T12:00:00Z")
    (sc,) = score(fc, synth16.cache, synth16.sprints, t)
    assert sc.t == t and sc.sprint["sprint_id"] == RED12
    committed = float(sc.sprint["committed_points"])
    assert sc.done_points > 0
    assert sc.summary["p10"] >= sc.done_points / committed
    assert (sc.items["sprint_id"] == RED12).all() and sc.items["p"].between(0, 1).all()
    still_committed = sc.items.loc[sc.items["is_added"] == 0, "item_id"]
    assert set(still_committed) <= set(sc.committed["item_id"])


def test_before_the_cutoff_the_scope_is_the_open_items_and_nothing_is_done(fc, synth16):
    t = pd.Timestamp("2024-06-10T12:00:00Z")
    (sc,) = score(fc, synth16.cache, synth16.sprints, t)
    _, items = scope_at(synth16.cache, synth16.sprints, iteration="Alpha\\Sprint 12", cutoff=t,
                        work_item_types=SYNTH_TYPES, done_categories=DONE, team="Team Red")
    assert sorted(sc.items["item_id"]) == sorted(items["item_id"])
    assert (sc.items["is_added"] == 0).all() and sc.done_points == 0.0 and sc.t == t
    assert sc.sprint["cutoff"] == pd.Timestamp("2024-06-11T05:00:00Z")


def test_an_ended_sprint_is_scored_at_its_end_with_the_observed_outcome(fc, synth16):
    (sc,) = score(fc, synth16.cache, synth16.sprints, pd.Timestamp("2024-06-29T00:00:00Z"))
    actual = synth16.sprints.sprints.set_index("sprint_id").loc[RED12]
    assert sc.t == actual["end"]
    assert sc.done_points == pytest.approx(actual["done_points"])
    assert sc.summary["expected"] == pytest.approx(actual["pct_done"])
    assert sc.summary["p10"] == sc.summary["p90"]
    assert (sc.items["p"] == 0.0).all()


def test_a_sprint_that_ended_after_the_last_extract_is_scored_at_the_extract(fc, tmp_path):
    cache = build_cache(tmp_path, [
        rev(1, 1, PRE, iteration=IT1),
        rev(1, 2, "2024-03-08T00:00:00.000Z", iteration=IT1, state="Closed", state_category="Completed"),
        rev(2, 1, PRE, iteration=IT1, story_points=5.0),
    ], [S0, S1], [RED], meta={"extracted_at:Alpha": "2024-03-11T00:00:00.000Z"})
    history = build_sprints(cache, work_item_types=TYPES, done_categories=DONE)
    (sc,) = score(fc, cache, history, pd.Timestamp("2024-03-25T00:00:00Z"), iteration=IT1, types=TYPES)
    assert sc.t == MID and sc.done_points == 3.0
    assert 0.0 < sc.items.set_index("item_id").loc[2, "p"] < 1.0
    assert 3 / 8 < sc.summary["expected"] < 1.0


def test_a_sprint_with_nothing_left_open_is_a_point_mass_at_its_done_share(fc, tmp_path):
    cache = build_cache(tmp_path, [
        rev(1, 1, PRE, iteration=IT1),
        rev(1, 2, "2024-03-08T00:00:00.000Z", iteration=IT1, state="Closed", state_category="Completed"),
        rev(2, 1, PRE, iteration=IT1, story_points=5.0),
        rev(2, 2, "2024-03-09T00:00:00.000Z", iteration="Alpha"),
    ], [S0, S1], [RED])
    history = build_sprints(cache, work_item_types=TYPES, done_categories=DONE)
    (sc,) = score(fc, cache, history, MID, iteration=IT1, types=TYPES)
    assert sc.items.empty and sc.done_points == 3.0
    assert sc.summary["p10"] == sc.summary["p90"] == pytest.approx(3 / 8)
    assert sorted(sc.committed["item_id"]) == [1, 2]


def test_an_unestimated_added_item_takes_the_teams_usual_size(fc, tmp_path):
    cache = build_cache(tmp_path, [
        rev(6, 1, "2024-02-20T00:00:00.000Z", iteration=IT0, story_points=5.0),  # the team's prior median size: 5
        rev(1, 1, PRE, iteration=IT1),
        rev(2, 1, PRE, story_points=None),
        rev(2, 2, "2024-03-07T00:00:00.000Z", iteration=IT1, story_points=None),
    ], [S0, S1], [RED])
    history = build_sprints(cache, work_item_types=TYPES, done_categories=DONE)
    (sc,) = score(fc, cache, history, MID, iteration=IT1, types=TYPES)
    added = sc.items.set_index("item_id").loc[2]
    assert added["is_added"] == 1.0 and added["is_unestimated"] == 1.0 and added["points"] == 5.0
    assert 0.0 < added["p"] < 1.0


def test_check_bundle_needs_the_current_feature_version():
    with pytest.raises(ValueError, match="older version; run `sprint-forecast train`"):
        check_bundle({"forecaster": None})
    with pytest.raises(ValueError, match="older version"):
        check_bundle({"feature_version": FEATURE_VERSION - 1})
    check_bundle({"feature_version": FEATURE_VERSION})
