import pandas as pd

from helpers import iteration, rev
from sprint_forecast.cache import (
    SCHEMA_VERSION,
    connect,
    get_meta,
    load_cache,
    max_changed,
    replace_project_iterations,
    replace_project_teams,
    set_meta,
    upsert_revisions,
)


def test_connect_creates_schema_and_version(tmp_path):
    conn = connect(tmp_path / "sub" / "cache.db")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"revisions", "iterations", "teams", "team_areas", "team_iterations", "meta"} <= tables
    assert get_meta(conn, "schema_version") == SCHEMA_VERSION
    conn.close()


def test_upsert_is_insert_if_absent(tmp_path):
    conn = connect(tmp_path / "cache.db")
    assert upsert_revisions(conn, [rev(1, 1, "2024-01-02T00:00:00.000Z"), rev(1, 2, "2024-01-03T00:00:00.000Z")]) == 2
    assert upsert_revisions(conn, [rev(1, 2, "2024-01-03T00:00:00.000Z", state="Active")]) == 0
    assert conn.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 2
    assert conn.execute("SELECT state FROM revisions WHERE rev = 2").fetchone()[0] == "New"
    assert max_changed(conn, "Alpha") == "2024-01-03T00:00:00.000Z"
    assert max_changed(conn, "Beta") is None
    conn.close()


def test_meta_round_trip(tmp_path):
    conn = connect(tmp_path / "cache.db")
    set_meta(conn, "watermark:Alpha", "2024-01-01T00:00:00.000Z")
    assert get_meta(conn, "watermark:Alpha") == "2024-01-01T00:00:00.000Z"
    assert get_meta(conn, "missing", "d") == "d"
    conn.close()


def test_replace_is_per_project(tmp_path):
    conn = connect(tmp_path / "cache.db")
    replace_project_iterations(conn, "Alpha", [iteration("Alpha\\S1", None, None)])
    replace_project_iterations(conn, "Beta", [iteration("Beta\\S1", None, None, project="Beta")])
    replace_project_teams(conn, "Alpha", [{"team_sk": "a1", "name": "Team Red"}],
                          [{"team_sk": "a1", "area_path": "Alpha\\Red"}], [{"team_sk": "a1", "iteration_path": "Alpha\\S1"}])
    replace_project_teams(conn, "Beta", [{"team_sk": "b1", "name": "Team Green"}],
                          [{"team_sk": "b1", "area_path": "Beta\\Green"}], [])
    replace_project_iterations(conn, "Alpha", [iteration("Alpha\\S2", None, None)])
    replace_project_teams(conn, "Alpha", [{"team_sk": "a2", "name": "Team Blue"}], [], [])
    conn.commit()
    data = load_cache(conn)
    assert sorted(data.iterations["path"]) == ["Alpha\\S2", "Beta\\S1"]
    assert sorted(data.teams["name"]) == ["Team Blue", "Team Green"]
    assert list(data.team_areas["team_sk"]) == ["b1"]
    assert data.team_iterations.empty
    conn.close()


def test_load_cache_types_and_project_filter(tmp_path):
    conn = connect(tmp_path / "cache.db")
    upsert_revisions(conn, [
        rev(2, 1, "2024-01-05T00:00:00.000Z", story_points=None, parent_id=7),
        rev(1, 1, "2024-01-02T00:00:00.000Z"),
        rev(3, 1, "2024-01-02T00:00:00.000Z", project="Beta"),
    ])
    replace_project_iterations(conn, "Alpha", [iteration("Alpha\\S1", "2024-01-01T05:00:00.000Z", None)])
    conn.commit()
    data = load_cache(conn, projects=["Alpha"])
    revs = data.revisions
    assert list(revs["item_id"]) == [1, 2]
    assert str(revs["changed"].dt.tz) == "UTC"
    assert revs["revised"].isna().all()
    assert pd.isna(revs.loc[revs.item_id == 2, "story_points"]).all()
    assert revs.loc[revs.item_id == 2, "parent_id"].iloc[0] == 7.0
    assert data.iterations["start_date"].iloc[0] == pd.Timestamp("2024-01-01T05:00:00Z")
    assert pd.isna(data.iterations["end_date"].iloc[0])
    assert data.iterations["is_ended"].dtype == bool
    conn.close()


def test_load_cache_empty_db(tmp_path):
    conn = connect(tmp_path / "cache.db")
    data = load_cache(conn)
    assert data.revisions.empty and data.iterations.empty and data.teams.empty
    conn.close()


def test_load_cache_reads_extraction_times(tmp_path):
    conn = connect(tmp_path / "cache.db")
    set_meta(conn, "extracted_at:Alpha", "2024-03-25T00:00:00.000Z")
    set_meta(conn, "extracted_at:Beta Two", "2024-04-01T12:30:00.000Z")
    conn.commit()
    assert load_cache(conn).extracted_at == {
        "Alpha": pd.Timestamp("2024-03-25T00:00:00Z"), "Beta Two": pd.Timestamp("2024-04-01T12:30:00Z"),
    }
    assert set(load_cache(conn, projects=["Alpha"]).extracted_at) == {"Alpha"}
    conn.close()


def test_missing_state_category_is_inferred_from_the_state(tmp_path):
    conn = connect(tmp_path / "cache.db")
    t1, t2 = "2024-01-02T00:00:00.000Z", "2024-01-03T00:00:00.000Z"
    upsert_revisions(conn, [
        rev(1, 1, t1, state="Done", state_category="Resolved"),
        rev(1, 2, t2, state="Done", state_category=None),       # same project uses Done as Resolved
        rev(2, 1, t1, state="Removed", state_category=None),    # never categorized here: standard meaning
        rev(3, 1, t1, state="In Progress", state_category=None),
        rev(4, 1, t1, state="Parked", state_category=None),     # unknown name: stays missing
        rev(5, 1, t1, state="Done", state_category=None, project="Beta"),
    ])
    conn.commit()
    got = load_cache(conn).revisions.set_index(["item_id", "rev"])["state_category"]
    assert got[(1, 2)] == "Resolved"
    assert got[(2, 1)] == "Removed"
    assert got[(3, 1)] == "InProgress"
    assert pd.isna(got[(4, 1)])
    assert got[(5, 1)] == "Completed"
    conn.close()
