import pandas as pd

from helpers import build_cache, rev
from sprint_forecast.timeline import as_of, as_of_many


def _revs(tmp_path):
    return build_cache(tmp_path, [
        rev(1, 1, "2024-02-01T10:00:00.000Z", state_category="Proposed"),
        rev(1, 2, "2024-02-02T10:00:00.000Z", state_category="InProgress"),
        rev(1, 3, "2024-02-03T10:00:00.000Z", state_category="Completed"),
        rev(2, 1, "2024-02-05T00:00:00.000Z"),
    ]).revisions


def test_exact_boundary_includes_revision(tmp_path):
    revs = _revs(tmp_path)
    at = as_of(revs, pd.Timestamp("2024-02-02T10:00:00Z"))
    assert at.set_index("item_id").loc[1, "state_category"] == "InProgress"


def test_one_millisecond_before_boundary_uses_previous(tmp_path):
    revs = _revs(tmp_path)
    at = as_of(revs, pd.Timestamp("2024-02-02T09:59:59.999Z"))
    assert at.set_index("item_id").loc[1, "state_category"] == "Proposed"


def test_before_creation_item_absent(tmp_path):
    revs = _revs(tmp_path)
    at = as_of(revs, pd.Timestamp("2024-02-04T00:00:00Z"))
    assert list(at["item_id"]) == [1]
    assert as_of(revs, pd.Timestamp("2024-01-01T00:00:00Z")).empty


def test_naive_timestamp_treated_as_utc(tmp_path):
    revs = _revs(tmp_path)
    at = as_of(revs, "2024-02-03T10:00:00")
    assert at.set_index("item_id").loc[1, "state_category"] == "Completed"


def test_offset_timestamp_converted(tmp_path):
    revs = _revs(tmp_path)
    at = as_of(revs, "2024-02-03T05:00:00-05:00")
    assert at.set_index("item_id").loc[1, "rev"] == 3


def test_as_of_many_matches_as_of_and_keeps_order(tmp_path):
    revs = _revs(tmp_path)
    keys = pd.DataFrame({
        "item_id": [2, 1, 1, 1],
        "t": pd.to_datetime([
            "2024-02-06T00:00:00Z", "2024-02-02T10:00:00Z", "2024-01-01T00:00:00Z", "2024-02-02T09:59:59.999Z",
        ], utc=True, format="ISO8601").as_unit("ns"),
        "tag": ["a", "b", "c", "d"],
    })
    out = as_of_many(revs, keys, "t")
    assert list(out["tag"]) == ["a", "b", "c", "d"]
    assert out.loc[0, "rev"] == 1 and out.loc[0, "revisions_so_far"] == 1
    assert out.loc[1, "state_category"] == "InProgress" and out.loc[1, "revisions_so_far"] == 2
    assert pd.isna(out.loc[2, "rev"])
    assert out.loc[3, "state_category"] == "Proposed"


def test_as_of_many_same_instant_takes_highest_rev(tmp_path):
    revs = build_cache(tmp_path, [
        rev(1, 1, "2024-02-01T10:00:00.000Z", iteration="Alpha\\Sprint 1"),
        rev(1, 2, "2024-02-01T10:00:00.000Z", iteration="Alpha\\Sprint 2"),
    ]).revisions
    keys = pd.DataFrame({"item_id": [1], "t": [pd.Timestamp("2024-02-01T10:00:00Z")]})
    out = as_of_many(revs, keys, "t")
    assert out.loc[0, "iteration"] == "Alpha\\Sprint 2"
    assert out.loc[0, "revisions_so_far"] == 2
