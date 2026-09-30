import pandas as pd
import pytest

from conftest import SYNTH_TYPES
from helpers import build_cache, iteration, rev, team
from sprint_forecast.cycle import CYCLE_STATE_COLUMNS
from sprint_forecast.export import (
    CYCLE_FILE_COLUMNS,
    ITEM_FORECAST_COLUMNS,
    ITEM_HISTORY_COLUMNS,
    SPRINT_FORECAST_COLUMNS,
    assignee_names,
    export_forecasts,
    progress_at,
    select_sprints,
    write_csv,
)
from sprint_forecast.features import FEATURE_VERSION
from sprint_forecast.rollup import fit_forecaster
from sprint_forecast.sprints import SPRINT_COLUMNS

NOW = pd.Timestamp("2024-06-12T12:00:00Z")  # synth16: Alpha Sprint 12 and Beta Sprint 11 are running


def _cal(rows):
    frame = pd.DataFrame(rows, columns=["sprint_id", "team_key", "start", "end"])
    for col in ("start", "end"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


def test_select_sprints_takes_running_sprints_and_each_teams_next_one():
    cal = _cal([
        ("A|S1", "A", "2024-06-01", "2024-06-10"),
        ("A|S2", "A", "2024-06-10", "2024-06-20"),
        ("A|S3", "A", "2024-06-20", "2024-06-30"),
        ("A|S4", "A", "2024-06-30", "2024-07-10"),
        ("B|S2", "B", "2024-06-10", "2024-06-20"),
        ("C|F2", "C", "2024-07-01", "2024-07-10"),
        ("C|F1", "C", "2024-06-25", "2024-07-01"),
    ])
    chosen = select_sprints(cal, pd.Timestamp("2024-06-15T00:00:00Z"))
    assert dict(zip(chosen["sprint_id"], chosen["status"])) == {
        "A|S2": "running", "B|S2": "running", "A|S3": "upcoming", "C|F1": "upcoming",
    }


def test_progress_counts_only_items_done_and_still_in_the_sprint(tmp_path):
    it = "Alpha\\Sprint 1"
    cache = build_cache(
        tmp_path,
        [
            rev(1, 1, "2024-03-04T10:00:00Z", iteration=it),
            rev(1, 2, "2024-03-08T10:00:00Z", iteration=it, state="Closed", state_category="Completed"),
            rev(2, 1, "2024-03-04T10:00:00Z", iteration=it),
            rev(2, 2, "2024-03-07T10:00:00Z", iteration="Alpha\\Sprint 2", state="Closed", state_category="Completed"),
            rev(3, 1, "2024-03-04T10:00:00Z", iteration=it),
            rev(3, 2, "2024-03-06T10:00:00Z", iteration=it, state="Active", state_category="InProgress"),
            rev(4, 1, "2024-03-04T10:00:00Z", iteration=it),
            rev(4, 2, "2024-03-12T10:00:00Z", iteration=it, state="Closed", state_category="Completed"),
        ],
    )
    items = pd.DataFrame({"item_id": [1, 2, 3, 4], "iteration": [it] * 4}, index=[10, 11, 12, 13])
    got = progress_at(cache, items, pd.Timestamp("2024-03-10T00:00:00Z"), ["Completed"])
    assert list(got.index) == [10, 11, 12, 13]
    assert list(got["done_now"]) == [True, False, False, False]
    assert list(got["in_sprint_now"]) == [True, False, True, True]
    assert list(got["state_category_now"]) == ["Completed", "Completed", "InProgress", "Proposed"]


def test_write_csv_uses_utc_iso_timestamps_and_leaves_no_temp_file(tmp_path):
    frame = pd.DataFrame({
        "at": pd.to_datetime(["2024-06-10T05:00:00Z", None], utc=True),
        "flag": [True, False],
        "x": [0.5, float("nan")],
    })
    path = tmp_path / "sub" / "t.csv"
    write_csv(frame, path)
    assert path.read_text(encoding="utf-8").splitlines() == ["at,flag,x", "2024-06-10T05:00:00Z,True,0.5", ",False,"]
    assert [p.name for p in path.parent.iterdir()] == ["t.csv"]


@pytest.fixture(scope="module")
def bundle(synth16):
    return {
        "forecaster": fit_forecaster(synth16.ckpt, seed=0),
        "feature_version": FEATURE_VERSION,
        "work_item_types": SYNTH_TYPES,
        "done_categories": ["Completed"],
        "commit_grace_days": 1.0,
        "trained_at": "2024-08-20T00:00:00+00:00",
    }


@pytest.fixture(scope="module")
def exported(synth16, bundle, tmp_path_factory):
    out = tmp_path_factory.mktemp("export")
    backtest = out.parent / "backtest-source.csv"
    backtest.write_text("sprint_id,model\nx,a\n", encoding="utf-8")
    first = export_forecasts(synth16.cache, bundle, out, now=NOW, backtest_csv=backtest)
    second = export_forecasts(synth16.cache, bundle, out, now=NOW + pd.Timedelta(days=1), backtest_csv=None)
    return out, first, second


def test_export_forecasts_running_and_upcoming_sprints(exported):
    out, first, _ = exported
    sprints = pd.read_csv(out / "sprint_forecasts" / f"{first.run_id}.csv")
    assert first.run_id == "20240612T120000Z"
    assert list(sprints.columns) == SPRINT_FORECAST_COLUMNS
    running = sprints[sprints["status"] == "running"]
    assert set(running["sprint_id"]) == {
        "Alpha/Team Red|Alpha\\Sprint 12", "Alpha/Team Blue|Alpha\\Sprint 12", "Beta/Team Green|Beta\\Sprint 11",
    }
    assert set(sprints["status"]) <= {"running", "upcoming"}
    assert (sprints["basis"] == "now").all()
    day1 = ["day1_expected", "day1_p10", "day1_p90"]
    assert sprints.loc[sprints["status"] == "running", day1].notna().all().all()
    assert sprints.loc[sprints["status"] == "upcoming", day1].isna().all().all()
    assert (sprints["day1_p10"].dropna() <= sprints["day1_p90"].dropna()).all()
    assert (sprints["p_full"] <= sprints["p_80"]).all() and (sprints["p10"] <= sprints["p90"]).all()
    assert (sprints["forecast_key"] == sprints["run_id"] + "|" + sprints["sprint_id"]).all()
    red = sprints.set_index("sprint_id").loc["Alpha/Team Red|Alpha\\Sprint 12"]
    assert red["scored_as_of"] == "2024-06-12T12:00:00Z"
    assert red["start"] == "2024-06-10T05:00:00Z" and red["run_at"] == "2024-06-12T12:00:00Z"
    assert red["elapsed_share"] == pytest.approx(55 / (14 * 24), abs=1e-3)
    assert red["data_as_of"] == "2024-08-19T04:59:59Z" and red["model_trained_at"] == "2024-08-20T00:00:00Z"


def test_export_item_rows_match_their_sprint(exported):
    out, first, _ = exported
    sprints = pd.read_csv(out / "sprint_forecasts" / f"{first.run_id}.csv")
    items = pd.read_csv(out / "item_forecasts" / f"{first.run_id}.csv")
    assert list(items.columns) == ITEM_FORECAST_COLUMNS
    committed = items[~items["is_added"]]
    per_sprint = committed.groupby("forecast_key").agg(n=("item_id", "size"), pts=("points", "sum"))
    joined = sprints.set_index("forecast_key").join(per_sprint)
    assert (joined["n"] == joined["n_items"]).all()
    assert joined["pts"].to_numpy() == pytest.approx(joined["committed_points"].to_numpy())
    assert items["p_done"].between(0, 1).all()
    assert items["risk_factor_1"].notna().any()
    done = committed.assign(w=committed["points"] * committed["done_now"]).groupby("forecast_key")["w"].sum()
    assert joined["points_done_so_far"].to_numpy() == pytest.approx(done.reindex(joined.index).to_numpy())


def test_the_rank_model_orders_items_but_leaves_sprint_forecasts_alone(synth16, bundle, tmp_path):
    ranked = {**bundle, "forecaster": fit_forecaster(synth16.ckpt, seed=0, with_rank=True)}
    runs = {}
    for name, b in (("plain", bundle), ("ranked", ranked)):
        result = export_forecasts(synth16.cache, b, tmp_path / name, now=NOW)
        runs[name] = (
            pd.read_csv(tmp_path / name / "sprint_forecasts" / f"{result.run_id}.csv"),
            pd.read_csv(tmp_path / name / "item_forecasts" / f"{result.run_id}.csv"),
        )
    cols = ["sprint_id", "expected", "p10", "p50", "p90", "p_full"]
    pd.testing.assert_frame_equal(runs["plain"][0][cols], runs["ranked"][0][cols])
    plain, rank = (runs[k][1].set_index(["forecast_key", "item_id"])["p_done"] for k in ("plain", "ranked"))
    assert not (plain - rank.reindex(plain.index)).abs().lt(1e-9).all()


@pytest.fixture(scope="module")
def late(synth16, bundle, tmp_path_factory):
    """A week later (2024-06-19 12:00 UTC): committed items are finishing and Team Blue has two open added items."""
    out = tmp_path_factory.mktemp("late")
    result = export_forecasts(synth16.cache, bundle, out, now=NOW + pd.Timedelta(days=7))
    return result.sprints, pd.read_csv(out / "item_forecasts" / f"{result.run_id}.csv")


def test_export_lists_committed_items_that_are_done_or_gone(late):
    sprints, items = late
    running = sprints[sprints["status"] == "running"].set_index("forecast_key")
    committed = items[~items["is_added"]]
    per_sprint = committed.groupby("forecast_key").agg(n=("item_id", "size"), pts=("points", "sum"))
    assert (per_sprint["n"].reindex(running.index) == running["n_items"]).all()
    assert per_sprint["pts"].reindex(running.index).to_numpy() == pytest.approx(running["committed_points"].to_numpy())
    done = committed[committed["done_now"]]
    assert len(done) > 0 and (done["p_done"] == 1.0).all() and done["risk_factor_1"].isna().all()
    assert (committed.loc[~committed["in_sprint_now"], "p_done"] == 0.0).all()
    done_points = done.groupby("forecast_key")["points"].sum().reindex(running.index).fillna(0.0)
    assert running["points_done_so_far"].to_numpy() == pytest.approx(done_points.to_numpy())
    assert (running["p10"] >= running["pct_done_so_far"] - 1e-12).all()


def test_export_marks_items_added_after_the_cutoff(late):
    sprints, items = late
    added = items[items["is_added"]]
    assert len(added) > 0 and added["state_category_at_commit"].isna().all()
    assert added["p_done"].between(0, 1).all() and not added["done_now"].any()
    assert set(added["forecast_key"]) <= set(sprints.loc[sprints["status"] == "running", "forecast_key"])


def test_a_running_sprint_with_nothing_left_open_lists_each_committed_item(bundle, tmp_path):
    it0, it1 = "Alpha\\Sprint 0", "Alpha\\Sprint 1"
    cache = build_cache(tmp_path, [
        rev(1, 1, "2024-03-01T00:00:00.000Z", iteration=it1),
        rev(1, 2, "2024-03-08T00:00:00.000Z", iteration=it1, state="Closed", state_category="Completed"),
        rev(2, 1, "2024-03-01T00:00:00.000Z", iteration=it1),
        rev(2, 2, "2024-03-09T00:00:00.000Z", iteration="Alpha"),
    ], [
        iteration(it0, "2024-02-19T05:00:00.000Z", "2024-03-04T04:59:59.999Z"),
        iteration(it1, "2024-03-04T05:00:00.000Z", "2024-03-18T04:59:59.999Z"),
    ], [team("Team Red", ["Alpha\\Red"], [it0, it1])])
    result = export_forecasts(cache, bundle, tmp_path / "out", now=pd.Timestamp("2024-03-11T00:00:00Z"))
    (row,) = result.sprints.to_dict("records")
    assert row["status"] == "running" and row["points_done_so_far"] == 3.0
    assert row["expected"] == pytest.approx(0.5) and row["p10"] == row["p90"] == pytest.approx(0.5)
    items = pd.read_csv(tmp_path / "out" / "item_forecasts" / f"{result.run_id}.csv")
    assert dict(zip(items["item_id"], items["p_done"])) == {1: 1.0, 2: 0.0}
    assert not items["is_added"].any() and items["risk_factor_1"].isna().all()


def test_export_refuses_a_model_from_an_older_version(synth16, bundle, tmp_path):
    old = {k: v for k, v in bundle.items() if k != "feature_version"}
    with pytest.raises(ValueError, match="older version; run `sprint-forecast train`"):
        export_forecasts(synth16.cache, old, tmp_path, now=NOW)
    assert not any(tmp_path.iterdir())


def test_export_appends_forecasts_per_run_and_replaces_snapshots(exported, synth16):
    out, first, second = exported
    assert sorted(p.name for p in (out / "sprint_forecasts").iterdir()) == [
        f"{first.run_id}.csv", f"{second.run_id}.csv",
    ]
    assert len(list((out / "item_forecasts").iterdir())) == 2
    outcomes = pd.read_csv(out / "sprints.csv")
    assert list(outcomes.columns) == SPRINT_COLUMNS and len(outcomes) == len(synth16.sprints.sprints)
    assert (out / "backtest.csv").read_text(encoding="utf-8") == "sprint_id,model\nx,a\n"


def test_assignee_names_are_blank_when_unassigned_and_flag_unknown_keys():
    users = pd.DataFrame({"user_sk": ["u1", "u2"], "name": ["Pat Example", "Sam Sample"]})
    names = assignee_names(pd.Series(["u1", None, "u9", "u2"]), users)
    assert names.tolist() == ["Pat Example", None, "Unknown user", "Sam Sample"]


def test_export_writes_every_committed_and_added_item_with_whoever_held_it_at_the_end(exported, synth16):
    out, _, _ = exported
    items = pd.read_csv(out / "items.csv")
    assert list(items.columns) == ITEM_HISTORY_COLUMNS
    names = dict(zip(synth16.cache.users["user_sk"], synth16.cache.users["name"]))
    committed, added = items[~items["added_mid"]], items[items["added_mid"]]
    history, history_added = synth16.sprints.items, synth16.sprints.added
    assert committed["item_id"].tolist() == history["item_id"].tolist()
    assert committed["assignee"].fillna("").tolist() == history["assigned_at_end_sk"].map(names).fillna("").tolist()
    assert len(added) == len(history_added) > 0
    assert sorted(zip(added["sprint_id"], added["item_id"])) == sorted(
        zip(history_added["sprint_id"], history_added["item_id"]))
    assert items["assignee"].isna().any() and items["assignee"].nunique() >= 8
    outcomes = pd.read_csv(out / "sprints.csv").set_index("sprint_id")
    closed = outcomes.index[outcomes["done_points"].notna()]
    done = committed.assign(w=committed["points"] * committed["done"]).groupby("sprint_id")["w"].sum()
    assert done[closed].to_numpy() == pytest.approx(outcomes.loc[closed, "done_points"].to_numpy())
    assert added.groupby("sprint_id").size().to_dict() == outcomes.loc[outcomes["n_added_mid"] > 0, "n_added_mid"].to_dict()
    assert items.loc[items["sprint_id"].isin(closed), "done"].isin([0, 1]).all()
    assert items.loc[~items["sprint_id"].isin(closed), "done"].isna().all()


def test_export_writes_cycle_time_per_finished_item_and_state(exported, synth16):
    out, _, _ = exported
    cycle = pd.read_csv(out / "cycle.csv")
    states = pd.read_csv(out / "cycle_states.csv")
    assert list(cycle.columns) == CYCLE_FILE_COLUMNS and list(states.columns) == CYCLE_STATE_COLUMNS
    assert len(cycle) > 100 and cycle["item_id"].is_unique
    timed = cycle[cycle["days"].notna()].set_index("item_id")
    per_item = states.groupby("item_id")["days"].sum()
    assert per_item.reindex(timed.index).to_numpy() == pytest.approx(timed["days"].to_numpy())
    assert set(states["state"]) <= {"Active", "Resolved"}
    names = set(synth16.cache.users["name"])
    assert cycle["assignee"].dropna().isin(names).all()
    assert cycle["team"].notna().mean() > 0.8  # synth also files some items at the project root, which no team owns


def test_export_with_nothing_running_still_writes_headers(synth16, bundle, tmp_path):
    result = export_forecasts(synth16.cache, bundle, tmp_path, now=pd.Timestamp("2030-01-01T00:00:00Z"))
    assert result.sprints.empty
    assert list(pd.read_csv(tmp_path / "sprint_forecasts" / f"{result.run_id}.csv").columns) == SPRINT_FORECAST_COLUMNS
    assert list(pd.read_csv(tmp_path / "item_forecasts" / f"{result.run_id}.csv").columns) == ITEM_FORECAST_COLUMNS
    assert not (tmp_path / "backtest.csv").exists()
    assert list(pd.read_csv(tmp_path / "items.csv").columns) == ITEM_HISTORY_COLUMNS
    assert list(pd.read_csv(tmp_path / "cycle.csv").columns) == CYCLE_FILE_COLUMNS


def test_export_items_say_where_missed_work_went_and_how_it_moved(exported):
    from sprint_forecast.flow import FATES
    out, first, _ = exported
    items = pd.read_csv(out / "items.csv")
    ended = items[items["done"].notna()]
    assert ended["fate"].isin(FATES).all() and ended["fate"].nunique() >= 3
    assert ((ended["fate"] == "done") == (ended["done_strict"] == 1)).all()
    assert items.loc[items["done"].isna(), "fate"].isna().all()
    assert (items["state_changes"] >= 0).all() and (items["state_changes"] == 0).any()
    forecasts = pd.read_csv(out / "item_forecasts" / f"{first.run_id}.csv")
    open_rows = forecasts[forecasts["risk_factor_1"].notna()]
    assert (open_rows["idle_sprints"] >= 0).all() and open_rows["idle_sprints"].max() >= 1
    assert (pd.read_csv(out / "cycle.csv")["returns"] >= 0).all()


def test_export_writes_go_live_forecasts_for_parents_with_open_children(exported):
    from sprint_forecast.golive import CURVE_COLUMNS, PARENT_COLUMNS
    out, _, _ = exported
    parents = pd.read_csv(out / "golive.csv")
    curve = pd.read_csv(out / "golive_curve.csv")
    assert list(parents.columns) == PARENT_COLUMNS and list(curve.columns) == CURVE_COLUMNS
    assert parents["parent_id"].is_unique and (parents["n_open"] > 0).all()
    forecast = parents[parents["status"] == "forecast"]
    assert len(forecast) > 0 and set(curve["parent_id"]) == set(forecast["parent_id"])
    assert curve["p_done_by"].between(0, 1).all()
    assert curve.groupby("parent_id")["p_done_by"].apply(lambda s: s.is_monotonic_increasing).all()
    assert (parents["p50_end"].isna() | (parents["p50_end"] <= parents["p85_end"]) | parents["p85_end"].isna()).all()
    assert curve.groupby("parent_id")["p_done_by_no_growth"].apply(lambda s: s.is_monotonic_increasing).all()
    assert (curve["p_done_by"] <= curve["p_done_by_no_growth"]).all()
    floor, dated = parents["p50_end_no_growth"], parents["p50_end"].notna()
    assert (floor[dated].notna() & (floor[dated] <= parents.loc[dated, "p50_end"])).all()
