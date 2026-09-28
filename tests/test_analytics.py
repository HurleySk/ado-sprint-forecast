import io
import json
from email.message import Message
from urllib.error import HTTPError

import pytest

from sprint_forecast import analytics
from sprint_forecast.analytics import (
    HttpError,
    list_projects,
    make_fetch_json,
    odata_string,
    odata_url,
    org_name,
    paginate,
    to_utc_iso,
)


class FakeResponse:
    def __init__(self, payload, headers=None):
        self._body = json.dumps(payload).encode()
        self.headers = Message()
        for k, v in (headers or {}).items():
            self.headers[k] = v

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(url, code, headers=None):
    hdrs = Message()
    for k, v in (headers or {}).items():
        hdrs[k] = v
    return HTTPError(url, code, "err", hdrs, io.BytesIO(b'{"message":"nope"}'))


def test_org_name_forms():
    assert org_name("https://dev.azure.com/contoso") == "contoso"
    assert org_name("https://dev.azure.com/contoso/") == "contoso"
    assert org_name("https://contoso.visualstudio.com") == "contoso"
    assert org_name("contoso") == "contoso"


def test_odata_url_encodes_spaces_but_keeps_odata_punctuation():
    url = odata_url(
        "https://dev.azure.com/contoso",
        "Alpha Project",
        "WorkItemRevisions",
        {"$select": "WorkItemId,Revision", "$filter": "WorkItemType in ('User Story','Bug')"},
    )
    assert url.startswith("https://analytics.dev.azure.com/contoso/Alpha%20Project/_odata/v4.0-preview/WorkItemRevisions?")
    assert "$select=WorkItemId,Revision" in url
    assert "$filter=WorkItemType%20in%20('User%20Story','Bug')" in url
    assert " " not in url


def test_odata_url_encodes_special_project_names_and_literals():
    literal = odata_string("Dev's Story")
    url = odata_url("contoso", "R&D Team's #1", "Iterations", {"$filter": "WorkItemType eq " + literal})
    assert "/contoso/R%26D%20Team%27s%20%231/_odata/v4.0-preview/Iterations?" in url
    assert "$filter=WorkItemType%20eq%20'Dev''s%20Story'" in url
    assert odata_string("Dev's Story") == "'Dev''s Story'"


def test_to_utc_iso_normalizes_offsets_and_open_dates():
    assert to_utc_iso("2024-05-24T15:41:38.597-04:00") == "2024-05-24T19:41:38.597Z"
    assert to_utc_iso("2026-01-06T23:59:59.999-05:00") == "2026-01-07T04:59:59.999Z"
    assert to_utc_iso("2025-12-24T00:00:00-05:00") == "2025-12-24T05:00:00.000Z"
    assert to_utc_iso("2024-01-01T00:00:00Z") == "2024-01-01T00:00:00.000Z"
    assert to_utc_iso("2024-01-01T00:00:00.1234567Z") == "2024-01-01T00:00:00.123Z"
    assert to_utc_iso("9999-01-01T00:00:00Z") is None
    assert to_utc_iso("9999-12-31T23:59:59.997-05:00") is None
    assert to_utc_iso(None) is None
    with pytest.raises(ValueError):
        to_utc_iso("yesterday")


def test_paginate_client_driven_skip_until_short_page():
    calls = []
    pages = {0: [{"i": 1}, {"i": 2}], 2: [{"i": 3}, {"i": 4}], 4: [{"i": 5}]}

    def fetch(url):
        calls.append(url)
        skip = int(url.rsplit("$skip=", 1)[1])
        return {"value": pages[skip]}

    rows = list(paginate(fetch, "https://x/_odata/v4.0-preview/Iterations?$select=IterationSK", top=2))
    assert [r["i"] for r in rows] == [1, 2, 3, 4, 5]
    assert calls[0].endswith("?$select=IterationSK&$top=2&$skip=0")
    assert calls[-1].endswith("&$top=2&$skip=4")
    assert len(calls) == 3


def test_paginate_exactly_full_last_page_makes_one_empty_request():
    def fetch(url):
        skip = int(url.rsplit("$skip=", 1)[1])
        return {"value": [{"i": 1}, {"i": 2}] if skip == 0 else []}

    assert len(list(paginate(fetch, "https://x/E", top=2))) == 2


def test_paginate_follows_next_link():
    calls = []

    def fetch(url):
        calls.append(url)
        if "skiptoken" not in url:
            return {"value": [{"i": 1}], "@odata.nextLink": "https://x/E?$skiptoken=abc"}
        return {"value": [{"i": 2}]}

    rows = list(paginate(fetch, "https://x/E", top=5000))
    assert [r["i"] for r in rows] == [1, 2]
    assert calls[1] == "https://x/E?$skiptoken=abc"
    assert len(calls) == 2


def test_fetch_json_pat_header_and_retry_on_429():
    seen = []
    sleeps = []

    def opener(req, timeout):
        seen.append(req)
        if len(seen) == 1:
            raise http_error(req.full_url, 429, {"Retry-After": "3"})
        if len(seen) == 2:
            raise http_error(req.full_url, 503)
        return FakeResponse({"value": [1]})

    fetch = make_fetch_json("pat", "secret", opener=opener, sleep=sleeps.append)
    assert fetch("https://x/E") == {"value": [1]}
    assert seen[0].get_header("Authorization") == "Basic OnNlY3JldA=="
    assert sleeps == [3.0, 2]


def test_fetch_json_date_retry_after_falls_back_to_backoff():
    sleeps = []
    calls = []

    def opener(req, timeout):
        calls.append(req)
        if len(calls) == 1:
            raise http_error(req.full_url, 429, {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})
        return FakeResponse({"value": []})

    fetch = make_fetch_json("pat", "secret", opener=opener, sleep=sleeps.append)
    assert fetch("https://x/E") == {"value": []}
    assert sleeps == [1]


def test_fetch_json_raises_http_error_on_401_without_retry():
    def opener(req, timeout):
        raise http_error(req.full_url, 401)

    fetch = make_fetch_json("pat", "secret", opener=opener, sleep=lambda s: None)
    with pytest.raises(HttpError) as ei:
        fetch("https://x/E")
    assert ei.value.status == 401


def test_fetch_json_gives_up_after_max_retries():
    calls = []

    def opener(req, timeout):
        calls.append(1)
        raise http_error(req.full_url, 500)

    fetch = make_fetch_json("pat", "secret", opener=opener, sleep=lambda s: None, max_retries=2)
    with pytest.raises(HttpError):
        fetch("https://x/E")
    assert len(calls) == 3


def test_fetch_json_az_cli_bearer_token():
    seen = []

    def opener(req, timeout):
        seen.append(req)
        return FakeResponse({})

    fetch = make_fetch_json("az-cli", opener=opener, token_provider=lambda: "tok123")
    fetch("https://x/E")
    assert seen[0].get_header("Authorization") == "Bearer tok123"


def test_pat_required():
    with pytest.raises(ValueError, match="ADO_PAT"):
        make_fetch_json("pat", "")


def test_continuation_header_is_surfaced_in_body():
    def opener(req, timeout):
        return FakeResponse({"value": []}, headers={"x-ms-continuationtoken": "tok"})

    fetch = make_fetch_json("pat", "p", opener=opener)
    assert fetch("https://x")["continuationToken"] == "tok"


def test_list_projects_follows_continuation_token():
    calls = []

    def fetch(url):
        calls.append(url)
        if "continuationToken" not in url:
            return {"count": 2, "value": [{"name": "Beta"}, {"name": "Alpha"}], "continuationToken": "next 1"}
        return {"count": 1, "value": [{"name": "Gamma"}]}

    assert list_projects(fetch, "https://dev.azure.com/contoso") == ["Alpha", "Beta", "Gamma"]
    assert calls[0] == "https://dev.azure.com/contoso/_apis/projects?api-version=7.1&$top=100"
    assert calls[1].endswith("&continuationToken=next%201")


def test_fetch_titles_chunks_and_tolerates_failure():
    def fetch(url):
        if "ids=1,2" in url:
            return {"value": [{"id": 1, "fields": {"System.Title": "First"}}, None]}
        raise HttpError(404, url)

    assert analytics.fetch_titles(fetch, "https://dev.azure.com/contoso", [1, 2]) == {1: "First"}
    assert analytics.fetch_titles(fetch, "https://dev.azure.com/contoso", [9]) == {}
