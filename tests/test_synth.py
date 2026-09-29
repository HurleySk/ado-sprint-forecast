import pandas as pd

from sprint_forecast.cache import ITERATION_COLUMNS, REVISION_COLUMNS, connect, get_meta, load_cache, watermark_key
from sprint_forecast.synth import generate, simulate


def test_deterministic_per_seed():
    assert simulate(seed=3, n_sprints=6) == simulate(seed=3, n_sprints=6)
    assert simulate(seed=3, n_sprints=6)["revisions"] != simulate(seed=4, n_sprints=6)["revisions"]


def test_rows_match_cache_schema():
    data = simulate(seed=0, n_sprints=6)
    assert all(set(r) == set(REVISION_COLUMNS) for r in data["revisions"])
    assert all(set(r) == set(ITERATION_COLUMNS) for r in data["iterations"])
    by_item = {}
    for r in data["revisions"]:
        by_item.setdefault(r["item_id"], []).append(r)
    for revs in by_item.values():
        changed = [r["changed"] for r in revs]
        assert changed == sorted(changed)
        assert [r["rev"] for r in revs] == list(range(1, len(revs) + 1))
        assert revs[-1]["revised"] is None


def test_names_are_synthetic():
    data = simulate(seed=0, n_sprints=6)
    assert {t["name"] for t in data["teams"]} == {"Team Red", "Team Blue", "Team Green", "Team Gray"}
    assert {r["project"] for r in data["revisions"]} == {"Alpha", "Beta"}


def test_every_synthetic_assignee_has_a_synthetic_name(tmp_path):
    data = simulate(seed=0, n_sprints=6)
    sks = {r["assigned_to_sk"] for r in data["revisions"] if r["assigned_to_sk"]}
    names = {u["user_sk"]: u["name"] for u in data["users"]}
    assert sks and sks <= set(names)
    assert len(set(names.values())) == len(names)
    cache = load_cache(connect(generate(tmp_path / "cache.db", seed=0, n_sprints=6)))
    assert dict(zip(cache.users["user_sk"], cache.users["name"])) == names


def test_planted_mess_is_present():
    data = simulate(seed=0, n_sprints=12)
    revs = pd.DataFrame(data["revisions"])
    first = revs.sort_values(["item_id", "rev"]).drop_duplicates("item_id")
    unpointed = (first["story_points"].isna() & first["effort"].isna()).mean()
    assert 0.08 < unpointed < 0.25
    assert first["effort"].notna().any() and first["story_points"].notna().any()
    its = pd.DataFrame(data["iterations"])
    assert its["start_date"].isna().sum() >= 4                      # roots and Someday are undated
    assert revs["iteration"].str.endswith("\\Someday").any()           # iteration bounces
    gray = next(t["team_sk"] for t in data["teams"] if t["name"] == "Team Gray")
    assert not any(i["team_sk"] == gray for i in data["team_iterations"])  # a team that runs no sprints
    assert (revs["state_category"] == "Removed").any()
    assert (revs["state_category"] == "Resolved").any()


def test_generate_writes_a_loadable_cache(tmp_path):
    path = generate(tmp_path / "demo" / "cache.db", seed=1, n_sprints=8)
    conn = connect(path)
    cache = load_cache(conn)
    assert get_meta(conn, "synthetic") == "seed=1;n_sprints=8"
    assert get_meta(conn, watermark_key("Alpha")) is not None
    conn.close()
    assert len(cache.revisions) == len(simulate(seed=1, n_sprints=8)["revisions"])
    assert sorted(cache.teams["name"]) == ["Team Blue", "Team Gray", "Team Green", "Team Red"]
    assert cache.revisions["changed"].dt.tz is not None


def test_generate_overwrites_existing_file(tmp_path):
    path = tmp_path / "cache.db"
    generate(path, seed=0, n_sprints=4)
    generate(path, seed=0, n_sprints=4)
    conn = connect(path)
    n = conn.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]
    conn.close()
    assert n == len(simulate(seed=0, n_sprints=4)["revisions"])
