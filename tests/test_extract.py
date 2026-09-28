import re
from urllib.parse import unquote

import pytest

from sprint_forecast.analytics import HttpError, to_utc_iso
from sprint_forecast.cache import connect, get_meta, load_cache, watermark_key
from sprint_forecast.extract import extract_all, extract_project, revision_row

ORG = "https://dev.azure.com/contoso"
TYPES = ["User Story", "Bug"]


def raw_rev(item_id, rev, changed, **kw):
    row = {
        "WorkItemId": item_id, "Revision": rev, "ChangedDate": changed,
        "RevisedDate": "9999-01-01T00:00:00Z", "CreatedDate": "2024-05-01T09:00:00-04:00",
        "WorkItemType": "User Story", "State": "New", "StateCategory": "Proposed",
        "StoryPoints": 3.0, "Effort": None, "ParentWorkItemId": None,
        "Iteration": {"IterationPath": "Alpha\\Sprint 1"}, "Area": {"AreaPath": "Alpha\\Red"},
        "AssignedTo": {"UserSK": "00000000-0000-0000-0000-000000000001"},
    }
    row.update(kw)
    return row


class FakeAnalytics:
    """Routes Analytics URLs to in-memory rows; supports the ChangedDate ge filter and per-project 401s."""

    def __init__(self):
        self.data = {}
        self.denied = set()
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        text = unquote(url)
        m = re.match(r"https://analytics\.dev\.azure\.com/contoso/([^/]+)/_odata/v4\.0-preview/(\w+)", text)
        project, entity = m.group(1), m.group(2)
        if project in self.denied:
            raise HttpError(401, url)
        rows = self.data.get((project, entity), [])
        wm = re.search(r"ChangedDate ge ([^&\s]+)", text)
        if wm:
            rows = [r for r in rows if to_utc_iso(r["ChangedDate"]) >= wm.group(1)]
        return {"value": rows}


@pytest.fixture
def fake():
    f = FakeAnalytics()
    f.data[("Alpha", "Iterations")] = [
        {"IterationSK": "i1", "IterationPath": "Alpha\\Sprint 1", "IterationName": "Sprint 1",
         "StartDate": "2025-12-24T00:00:00-05:00", "EndDate": "2026-01-06T23:59:59.999-05:00", "IsEnded": True},
        {"IterationSK": "i0", "IterationPath": "Alpha", "IterationName": "Alpha",
         "StartDate": None, "EndDate": None, "IsEnded": False},
    ]
    f.data[("Alpha", "Teams")] = [
        {"TeamSK": "t1", "TeamName": "Team Red", "Areas": [{"AreaPath": "Alpha\\Red"}],
         "Iterations": [{"IterationPath": "Alpha\\Sprint 1"}]},
        {"TeamSK": "t2", "TeamName": "Team Gray", "Areas": [{"AreaPath": "Alpha\\Gray"}], "Iterations": []},
    ]
    f.data[("Alpha", "WorkItemRevisions")] = [
        raw_rev(1, 1, "2024-05-24T15:41:38.597-04:00", RevisedDate="2024-05-25T10:00:00-04:00"),
        raw_rev(1, 2, "2024-05-25T10:00:00-04:00", Iteration=None, AssignedTo=None, ParentWorkItemId=77),
    ]
    return f


def test_revision_row_normalizes_timestamps_and_navigations():
    row = revision_row("Alpha", raw_rev(5, 1, "2024-05-24T15:41:38.597-04:00", Iteration=None, ParentWorkItemId=9))
    assert row["changed"] == "2024-05-24T19:41:38.597Z"
    assert row["revised"] is None
    assert row["created"] == "2024-05-01T13:00:00.000Z"
    assert row["iteration"] is None
    assert row["area"] == "Alpha\\Red"
    assert row["assigned_to_sk"] == "00000000-0000-0000-0000-000000000001"
    assert row["parent_id"] == 9


def test_first_extract_stores_everything_and_sets_watermark(tmp_path, fake):
    conn = connect(tmp_path / "cache.db")
    r = extract_project(fake, conn, ORG, "Alpha", work_item_types=TYPES)
    assert (r.fetched, r.inserted, r.iterations, r.teams, r.skipped) == (2, 2, 2, 2, None)
    assert get_meta(conn, watermark_key("Alpha")) == "2024-05-25T14:00:00.000Z"
    data = load_cache(conn)
    assert len(data.revisions) == 2
    assert set(data.team_iterations["iteration_path"]) == {"Alpha\\Sprint 1"}
    it = data.iterations.set_index("path").loc["Alpha\\Sprint 1"]
    assert str(it["end_date"]) == "2026-01-07 04:59:59.999000+00:00"
    rev_url = [u for u in fake.calls if "WorkItemRevisions" in u][0]
    assert "ChangedDate ge" not in unquote(rev_url)
    assert "$filter=WorkItemType%20in%20('User%20Story','Bug')" in rev_url
    assert "$orderby=WorkItemId,Revision" in rev_url
    assert " " not in rev_url
    conn.close()


def test_incremental_uses_ge_watermark_and_dedupes_boundary(tmp_path, fake):
    conn = connect(tmp_path / "cache.db")
    extract_project(fake, conn, ORG, "Alpha", work_item_types=TYPES)
    fake.data[("Alpha", "WorkItemRevisions")].append(raw_rev(2, 1, "2024-06-01T00:00:00Z"))
    fake.calls.clear()
    r = extract_project(fake, conn, ORG, "Alpha", work_item_types=TYPES)
    rev_url = unquote([u for u in fake.calls if "WorkItemRevisions" in u][0])
    assert "ChangedDate ge 2024-05-25T14:00:00.000Z" in rev_url
    assert r.fetched == 2 and r.inserted == 1
    assert conn.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 3
    assert get_meta(conn, watermark_key("Alpha")) == "2024-06-01T00:00:00.000Z"
    conn.close()


def test_iterations_and_teams_replaced_per_project(tmp_path, fake):
    conn = connect(tmp_path / "cache.db")
    fake.data[("Beta", "Iterations")] = [{"IterationSK": "b1", "IterationPath": "Beta\\S1", "IsEnded": False}]
    extract_all(fake, conn, ORG, ["Alpha", "Beta"], work_item_types=TYPES, echo=lambda s: None)
    fake.data[("Alpha", "Iterations")] = fake.data[("Alpha", "Iterations")][:1]
    fake.data[("Alpha", "Teams")] = fake.data[("Alpha", "Teams")][:1]
    extract_project(fake, conn, ORG, "Alpha", work_item_types=TYPES)
    data = load_cache(conn)
    assert sorted(data.iterations["path"]) == ["Alpha\\Sprint 1", "Beta\\S1"]
    assert list(data.teams["name"]) == ["Team Red"]
    assert list(data.team_areas["area_path"]) == ["Alpha\\Red"]
    conn.close()


def test_full_reextract_replaces_project_revisions(tmp_path, fake):
    conn = connect(tmp_path / "cache.db")
    extract_project(fake, conn, ORG, "Alpha", work_item_types=TYPES)
    fake.data[("Alpha", "WorkItemRevisions")] = fake.data[("Alpha", "WorkItemRevisions")][:1]
    r = extract_project(fake, conn, ORG, "Alpha", work_item_types=TYPES, full=True)
    assert r.inserted == 1
    assert conn.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 1
    conn.close()


def test_full_reextract_leaves_other_projects_alone(tmp_path, fake):
    conn = connect(tmp_path / "cache.db")
    fake.data[("Beta", "WorkItemRevisions")] = [raw_rev(50, 1, "2024-05-01T00:00:00Z")]
    extract_all(fake, conn, ORG, ["Alpha", "Beta"], work_item_types=TYPES, echo=lambda s: None)
    fake.data[("Alpha", "WorkItemRevisions")] = []
    extract_project(fake, conn, ORG, "Alpha", work_item_types=TYPES, full=True)
    rows = conn.execute("SELECT project, COUNT(*) FROM revisions GROUP BY project").fetchall()
    assert rows == [("Beta", 1)]
    assert get_meta(conn, watermark_key("Beta")) == "2024-05-01T00:00:00.000Z"
    conn.close()


def test_failure_mid_pagination_leaves_cache_and_watermark_unchanged(tmp_path, fake):
    conn = connect(tmp_path / "cache.db")
    extract_project(fake, conn, ORG, "Alpha", work_item_types=TYPES)

    def flaky(url):
        if "WorkItemRevisions" not in url:
            return fake(url)
        if "skiptoken" in url:
            raise HttpError(503, url)
        return {"value": [raw_rev(3, 1, "2024-07-01T00:00:00Z")],
                "@odata.nextLink": "https://analytics.dev.azure.com/contoso/Alpha/_odata/v4.0-preview/WorkItemRevisions?$skiptoken=1"}

    with pytest.raises(HttpError):
        extract_project(flaky, conn, ORG, "Alpha", work_item_types=TYPES)
    assert conn.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 2
    assert get_meta(conn, watermark_key("Alpha")) == "2024-05-25T14:00:00.000Z"
    conn.close()


def test_401_project_is_skipped_not_fatal(tmp_path, fake):
    conn = connect(tmp_path / "cache.db")
    fake.denied.add("Beta")
    lines = []
    results = extract_all(fake, conn, ORG, ["Beta", "Alpha"], work_item_types=TYPES, echo=lines.append)
    assert results[0].skipped.startswith("HTTP 401")
    assert results[1].fetched == 2
    assert any("Beta: skipped" in line for line in lines)
    assert get_meta(conn, watermark_key("Beta")) is None
    conn.close()


def test_other_http_errors_propagate(tmp_path):
    def boom(url):
        raise HttpError(500, url)

    conn = connect(tmp_path / "cache.db")
    with pytest.raises(HttpError):
        extract_project(boom, conn, ORG, "Alpha", work_item_types=TYPES)
    conn.close()


def test_star_lists_projects_via_rest(tmp_path, fake):
    def fetch(url):
        if "_apis/projects" in url:
            return {"count": 1, "value": [{"name": "Alpha"}]}
        return fake(url)

    conn = connect(tmp_path / "cache.db")
    results = extract_all(fetch, conn, ORG, ["*"], work_item_types=TYPES, echo=lambda s: None)
    assert [r.project for r in results] == ["Alpha"]
    conn.close()
