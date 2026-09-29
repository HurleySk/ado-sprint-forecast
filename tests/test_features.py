import dataclasses
import math

import pandas as pd
import pytest

from helpers import build_cache, iteration, rev, team
from sprint_forecast.cache import connect, load_cache
from sprint_forecast.checkpoints import open_rows
from sprint_forecast.features import (
    CHECKPOINT_FRAME_COLUMNS,
    DAY1_FEATURES,
    FEATURE_FRAME_COLUMNS,
    FEATURES,
    ID_COLUMNS,
    assignee_load,
    build_checkpoint_features,
    build_checkpoint_frame,
    build_features,
    team_history,
)
from sprint_forecast.sprints import build_sprints
from sprint_forecast.synth import generate

TYPES = ["User Story", "Product Backlog Item", "Bug"]
T0 = pd.Timestamp("2024-01-08T05:00:00Z")
DAY = pd.Timedelta(days=1)
MS = pd.Timedelta(milliseconds=1)


def _history(done_points, pct, team_key="Alpha/Team Red"):
    n = len(done_points)
    starts = [T0 + 14 * k * DAY for k in range(n)]
    return pd.DataFrame({
        "sprint_id": [f"{team_key}|S{k}" for k in range(n)],
        "team_key": team_key,
        "start": starts,
        "end": [s + 14 * DAY - MS for s in starts],
        "done_points": [float(x) for x in done_points],
        "pct_done": [float(x) for x in pct],
    })


def _target(k, team_key="Alpha/Team Red"):
    return pd.DataFrame({"sprint_id": ["target"], "team_key": [team_key], "start": [T0 + 14 * k * DAY]})


def test_team_history_uses_only_sprints_ended_before_start():
    hist = _history([10, 20, 30, 40, 50], [0.1, 0.2, 0.3, 0.4, 0.5])
    row = team_history(_target(4), hist).iloc[0]
    assert row["trailing_velocity"] == pytest.approx(30.0)      # sprints 1..3
    assert row["team_trailing_completion"] == pytest.approx(0.25)  # sprints 0..3
    assert row["team_sprint_index"] == 4


def test_team_history_ignores_overlapping_future_and_other_team_sprints():
    hist = _history([10, 20, 30, 40], [0.1, 0.2, 0.3, 0.4])
    base = team_history(_target(4), hist).iloc[0]
    noise = pd.DataFrame({
        "sprint_id": ["overlap", "future", "other-team"],
        "team_key": ["Alpha/Team Red", "Alpha/Team Red", "Alpha/Team Blue"],
        "start": [T0 + 50 * DAY, T0 + 70 * DAY, T0],
        "end": [T0 + 56 * DAY, T0 + 84 * DAY, T0 + 14 * DAY],   # 56 days == target start: not strictly before
        "done_points": [999.0, 999.0, 999.0],
        "pct_done": [1.0, 1.0, 1.0],
    })
    row = team_history(_target(4), pd.concat([hist, noise], ignore_index=True)).iloc[0]
    assert row.equals(base)


def test_team_history_without_history_is_nan():
    row = team_history(_target(0), _history([], [])).iloc[0]
    assert math.isnan(row["trailing_velocity"]) and math.isnan(row["team_trailing_completion"])
    assert row["team_sprint_index"] == 0


def test_assignee_load_uses_last_three_earlier_sprints():
    ends = [T0 + 14 * k * DAY for k in range(1, 6)]
    history = pd.DataFrame({
        "sprint_id": ["h0", "h1", "h2", "h3", "h3", "future"],
        "assigned_to_sk": ["p1"] * 6,
        "end": [ends[0], ends[1], ends[2], ends[3], ends[3], ends[4] + 30 * DAY],
        "points": [100.0, 4.0, 6.0, 5.0, 3.0, 100.0],
        "done": [True, True, True, True, False, True],
    })
    target = pd.DataFrame({
        "sprint_id": ["t", "t", "t", "t"],
        "start": [ends[4]] * 4,
        "assigned_to_sk": ["p1", "p1", None, "p2"],
        "points": [3.0, 7.0, 5.0, 2.0],
    })
    ratio = assignee_load(target, history)
    assert ratio.iloc[0] == pytest.approx(10.0 / 5.0)  # delivered 4, 6, 5 in h1..h3
    assert ratio.iloc[1] == pytest.approx(2.0)
    assert math.isnan(ratio.iloc[2]) and math.isnan(ratio.iloc[3])


@pytest.fixture(scope="module")
def synth_cache(tmp_path_factory):
    conn = connect(generate(tmp_path_factory.mktemp("synth") / "cache.db", seed=2, n_sprints=10))
    cache = load_cache(conn)
    conn.close()
    return cache


def _features(cache):
    sd = build_sprints(cache, work_item_types=TYPES)
    return sd, build_features(sd)


def test_feature_frame_shape_and_first_sprint_nans(synth_cache):
    sd, frame = _features(synth_cache)
    assert list(frame.columns) == FEATURE_FRAME_COLUMNS
    assert len(frame) == len(sd.items)
    assert set(frame["y"].unique()) <= {0.0, 1.0}
    first = frame[frame["team_sprint_index"] == 0]
    assert len(first) > 0
    assert first["load_ratio"].isna().all() and first["points_rel"].isna().all()
    later = frame[frame["team_sprint_index"] >= 3]
    assert later["load_ratio"].notna().all()
    unassigned = frame["is_unassigned"] == 1.0
    assert unassigned.any() and frame.loc[unassigned, "assignee_load_ratio"].isna().all()
    assert (frame["age_days"] >= 0).all() and (frame["days_since_change"] >= 0).all()


def test_revisions_after_cutoff_do_not_change_features(synth_cache):
    sd, before = _features(synth_cache)
    target = sd.sprints[sd.sprints["team"] == "Team Red"].iloc[5]
    ids = sd.items.loc[sd.items["sprint_id"] == target["sprint_id"], "item_id"]
    revs = synth_cache.revisions
    last = revs[revs["item_id"].isin(ids)].sort_values(["item_id", "rev"]).drop_duplicates("item_id", keep="last")
    late = last.assign(
        rev=last["rev"] + 1, changed=target["cutoff"] + pd.Timedelta(hours=1),
        state="Closed", state_category="Completed", story_points=99.0, assigned_to_sk="late-person",
    )
    cache = dataclasses.replace(synth_cache, revisions=pd.concat([revs, late], ignore_index=True))
    _, after = _features(cache)
    cols = ID_COLUMNS + DAY1_FEATURES
    b = before[before["sprint_id"] == target["sprint_id"]][cols].reset_index(drop=True)
    a = after[after["sprint_id"] == target["sprint_id"]][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(b, a)
    # control: the same edits one minute before the cutoff do change the target sprint
    early = late.assign(changed=target["cutoff"] - pd.Timedelta(minutes=1))
    _, moved = _features(dataclasses.replace(synth_cache, revisions=pd.concat([revs, early], ignore_index=True)))
    assert (moved["sprint_id"] == target["sprint_id"]).sum() < len(b)


def _checkpoint_frame(cache, checkpoints=(0.5,)):
    sd = build_sprints(cache, work_item_types=TYPES)
    frame = build_checkpoint_frame(cache, sd, work_item_types=TYPES, done_categories=["Completed"],
                                   checkpoints=checkpoints)
    return sd, frame


def test_at_the_cutoff_checkpoint_features_equal_the_day1_features(synth16):
    ck = synth16.ckpt[synth16.ckpt["checkpoint"] == 0.0].reset_index(drop=True)
    day1 = synth16.frame.reset_index(drop=True)
    assert list(synth16.ckpt.columns) == CHECKPOINT_FRAME_COLUMNS
    shared = [f for f in FEATURES if f in DAY1_FEATURES]
    pd.testing.assert_frame_equal(ck[ID_COLUMNS + shared], day1[ID_COLUMNS + shared], check_dtype=False)
    assert (ck["state_category"] == day1["state_category_at_commit"]).all()
    assert (ck[["is_added", "reassigned", "done_share", "elapsed"]] == 0.0).all().all()
    pd.testing.assert_series_equal(ck["y"], day1["y"], check_names=False)


def test_later_checkpoints_have_added_items_and_progress(synth16):
    late = synth16.ckpt[synth16.ckpt["checkpoint"] == 0.75]
    assert (late["is_added"] == 1.0).any() and (late["done_share"] > 0).any()
    added = late[late["is_added"] == 1.0]
    assert added["assignee_load_ratio"].isna().all() and added["points_at_commit"].isna().all()
    assert late["elapsed"].to_numpy() == pytest.approx(0.75)


def test_revisions_after_t_do_not_change_checkpoint_features(synth_cache):
    sd, before = _checkpoint_frame(synth_cache)
    target = sd.sprints[sd.sprints["team"] == "Team Red"].iloc[5]
    rows = before[before["sprint_id"] == target["sprint_id"]]
    assert len(rows) > 0
    t = rows["t"].iloc[0]
    revs = synth_cache.revisions
    last = revs[revs["item_id"].isin(rows["item_id"])].sort_values(["item_id", "rev"]).drop_duplicates(
        "item_id", keep="last")
    late = last.assign(
        rev=last["rev"] + 1, changed=t + pd.Timedelta(hours=1),
        state="Closed", state_category="Completed", story_points=99.0, assigned_to_sk="late-person",
    )
    _, after = _checkpoint_frame(dataclasses.replace(synth_cache, revisions=pd.concat([revs, late], ignore_index=True)))
    cols = ID_COLUMNS + ["checkpoint", "t"] + FEATURES
    b = rows[cols].reset_index(drop=True)
    a = after[after["sprint_id"] == target["sprint_id"]][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(b, a)
    # control: the same edits one minute before t close the items, so they are no longer open rows
    early = late.assign(changed=t - pd.Timedelta(minutes=1))
    _, moved = _checkpoint_frame(dataclasses.replace(synth_cache, revisions=pd.concat([revs, early], ignore_index=True)))
    assert (moved["sprint_id"] == target["sprint_id"]).sum() < len(b)


def test_progress_features_before_the_cutoff_and_after_the_end(tmp_path):
    s1 = iteration("Alpha\\Sprint 1", "2024-03-04T05:00:00.000Z", "2024-03-18T04:59:59.999Z")
    red = team("Team Red", ["Alpha\\Red"], ["Alpha\\Sprint 1"])
    cache = build_cache(tmp_path, [rev(1, 1, "2024-03-01T00:00:00.000Z", iteration="Alpha\\Sprint 1")], [s1], [red])
    sd = build_sprints(cache, work_item_types=["User Story", "Bug"])
    day1, s = build_features(sd), sd.sprints

    def features_at(t):
        rows = open_rows(cache, sd, s, pd.Series(pd.Timestamp(t), index=s.index),
                         work_item_types=["User Story", "Bug"], done_categories=["Completed"])
        return build_checkpoint_features(rows, day1, sd.sprints).iloc[0]

    after_end = features_at("2024-03-19T05:00:00Z")
    assert after_end["elapsed"] == 1.0 and after_end["days_left"] == 0.0
    early = features_at("2024-03-04T18:00:00Z")
    span_days = (s["end"].iloc[0] - s["cutoff"].iloc[0]) / DAY
    assert early["elapsed"] == 0.0 and early["days_left"] == pytest.approx(span_days)
    assert early["done_share"] == 0.0 and early["is_added"] == 0.0
    assert early["days_in_state"] == pytest.approx(3.75) and early["age_days"] == pytest.approx(63.75)
