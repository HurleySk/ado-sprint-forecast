import pandas as pd
import pytest

from conftest import SYNTH_TYPES
from helpers import build_cache, rev
from sprint_forecast.export import (
    ITEM_FORECAST_COLUMNS,
    ITEM_HISTORY_COLUMNS,
    SPRINT_FORECAST_COLUMNS,
    assignee_names,
    export_forecasts,
    progress_at,
    select_sprints,
    write_csv,
)
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
        "forecaster": fit_forecaster(synth16.frame, seed=0),
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
    assert (sprints.loc[sprints["status"] == "running", "basis"] == "commit cutoff").all()
    assert (sprints.loc[sprints["status"] == "upcoming", "basis"] == "now").all()
    assert (sprints["p_full"] <= sprints["p_80"]).all() and (sprints["p10"] <= sprints["p90"]).all()
    assert (sprints["forecast_key"] == sprints["run_id"] + "|" + sprints["sprint_id"]).all()
    red = sprints.set_index("sprint_id").loc["Alpha/Team Red|Alpha\\Sprint 12"]
    assert red["start"] == "2024-06-10T05:00:00Z" and red["run_at"] == "2024-06-12T12:00:00Z"
    assert red["scored_as_of"] == "2024-06-11T05:00:00Z"
    assert red["elapsed_share"] == pytest.approx(55 / (14 * 24), abs=1e-3)
    assert red["data_as_of"] == "2024-08-19T04:59:59Z" and red["model_trained_at"] == "2024-08-20T00:00:00Z"


def test_export_item_rows_match_their_sprint(exported):
    out, first, _ = exported
    sprints = pd.read_csv(out / "sprint_forecasts" / f"{first.run_id}.csv")
    items = pd.read_csv(out / "item_forecasts" / f"{first.run_id}.csv")
    assert list(items.columns) == ITEM_FORECAST_COLUMNS
    per_sprint = items.groupby("forecast_key").agg(n=("item_id", "size"), pts=("points", "sum"))
    joined = sprints.set_index("forecast_key").join(per_sprint)
    assert (joined["n"] == joined["n_items"]).all()
    assert joined["pts"].to_numpy() == pytest.approx(joined["committed_points"].to_numpy())
    assert items["p_done"].between(0, 1).all()
    assert items["risk_factor_1"].notna().any()
    done = items.assign(w=items["points"] * items["done_now"]).groupby("forecast_key")["w"].sum()
    assert joined["points_done_so_far"].to_numpy() == pytest.approx(done.reindex(joined.index).to_numpy())


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


def test_export_writes_every_committed_item_with_its_assignee(exported, synth16):
    out, _, _ = exported
    items = pd.read_csv(out / "items.csv")
    assert list(items.columns) == ITEM_HISTORY_COLUMNS
    history = synth16.sprints.items
    assert items["item_id"].tolist() == history["item_id"].tolist()
    names = dict(zip(synth16.cache.users["user_sk"], synth16.cache.users["name"]))
    assert items["assignee"].fillna("").tolist() == history["assigned_to_sk"].map(names).fillna("").tolist()
    assert items["assignee"].isna().any() and items["assignee"].nunique() >= 8
    outcomes = pd.read_csv(out / "sprints.csv").set_index("sprint_id")
    closed = outcomes.index[outcomes["done_points"].notna()]
    done = items.assign(w=items["points"] * items["done"]).groupby("sprint_id")["w"].sum()
    assert done[closed].to_numpy() == pytest.approx(outcomes.loc[closed, "done_points"].to_numpy())
    assert items.loc[items["sprint_id"].isin(closed), "done"].isin([0, 1]).all()
    assert items.loc[~items["sprint_id"].isin(closed), "done"].isna().all()


def test_export_with_nothing_running_still_writes_headers(synth16, bundle, tmp_path):
    result = export_forecasts(synth16.cache, bundle, tmp_path, now=pd.Timestamp("2030-01-01T00:00:00Z"))
    assert result.sprints.empty
    assert list(pd.read_csv(tmp_path / "sprint_forecasts" / f"{result.run_id}.csv").columns) == SPRINT_FORECAST_COLUMNS
    assert list(pd.read_csv(tmp_path / "item_forecasts" / f"{result.run_id}.csv").columns) == ITEM_FORECAST_COLUMNS
    assert not (tmp_path / "backtest.csv").exists()
    assert list(pd.read_csv(tmp_path / "items.csv").columns) == ITEM_HISTORY_COLUMNS
