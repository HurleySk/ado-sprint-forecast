"""Minimal Azure DevOps Analytics OData client (stdlib urllib only)."""
from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from http.client import HTTPException
from typing import Callable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

ADO_RESOURCE_ID = "499b84ac-1321-427f-aa17-267ca6975798"
ANALYTICS_BASE = "https://analytics.dev.azure.com"
ODATA_VERSION = "v4.0-preview"
PAGE_SIZE = 5000
RETRY_STATUSES = {429, 500, 502, 503, 504}
_SAFE = "$(),'/:"

FetchJson = Callable[[str], dict]

_TS = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})?$"
)


class HttpError(Exception):
    def __init__(self, status: int, url: str, body: str = ""):
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")
        self.status = status
        self.url = url
        self.body = body


def org_name(org_url: str) -> str:
    """'https://dev.azure.com/contoso' -> 'contoso'; 'https://contoso.visualstudio.com' -> 'contoso'."""
    text = org_url.strip().rstrip("/")
    if "://" not in text:
        return text
    parsed = urlparse(text)
    host = parsed.netloc.lower()
    if host.endswith(".visualstudio.com"):
        return host.split(".")[0]
    return parsed.path.strip("/").split("/")[0]


def odata_url(org: str, project: str | None, entity: str, params: dict[str, str]) -> str:
    base = f"{ANALYTICS_BASE}/{org_name(org)}"
    if project:
        base += "/" + quote(project, safe="")
    query = "&".join(f"{k}={quote(str(v), safe=_SAFE)}" for k, v in params.items())
    return f"{base}/_odata/{ODATA_VERSION}/{entity}" + (f"?{query}" if query else "")


def odata_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def to_utc_iso(value: str | None) -> str | None:
    """Normalize an OData timestamp to UTC 'YYYY-MM-DDTHH:MM:SS.fffZ'. Year >= 9999 means open -> None."""
    if value is None or value == "":
        return None
    m = _TS.match(value)
    if not m:
        raise ValueError(f"unrecognized timestamp: {value!r}")
    year = int(m.group(1))
    if year >= 9999:
        return None
    frac = (m.group(7) or "0")[:6].ljust(6, "0")
    tz_text = m.group(8) or "Z"
    if tz_text == "Z":
        tz = timezone.utc
    else:
        sign = 1 if tz_text[0] == "+" else -1
        tz = timezone(sign * timedelta(hours=int(tz_text[1:3]), minutes=int(tz_text[4:6])))
    dt = datetime(
        year, int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6)),
        int(frac), tzinfo=tz,
    )
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def with_paging(url: str, top: int, skip: int) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}$top={top}&$skip={skip}"


def paginate(fetch_json: FetchJson, url: str, top: int = PAGE_SIZE) -> Iterator[dict]:
    """Yield rows. Follow @odata.nextLink when present, else advance $skip while pages are full."""
    skip = 0
    next_url = with_paging(url, top, skip)
    server_paging = False
    while True:
        data = fetch_json(next_url)
        rows = data.get("value", [])
        yield from rows
        link = data.get("@odata.nextLink")
        if link:
            server_paging = True
            next_url = link
            continue
        if server_paging or len(rows) < top:
            return
        skip += top
        next_url = with_paging(url, top, skip)


def pat_headers(pat: str) -> dict[str, str]:
    token = base64.b64encode(f":{pat}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def az_cli_token(run: Callable = subprocess.run) -> str:
    exe = shutil.which("az")
    if exe is None:
        raise RuntimeError("Azure CLI `az` not found on PATH; install it or use --auth pat")
    proc = run(
        [exe, "account", "get-access-token", "--resource", ADO_RESOURCE_ID, "--query", "accessToken", "-o", "tsv"],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"az account get-access-token failed: {proc.stderr.strip()[:300]}")
    return proc.stdout.strip()


def make_fetch_json(
    auth_method: str,
    pat: str = "",
    *,
    opener: Callable = urlopen,
    sleep: Callable[[float], None] = time.sleep,
    token_provider: Callable[[], str] = az_cli_token,
    max_retries: int = 5,
) -> FetchJson:
    if auth_method == "pat":
        if not pat:
            raise ValueError("no PAT: set the ADO_PAT environment variable or use --auth az-cli")
        headers = pat_headers(pat)
    elif auth_method == "az-cli":
        headers = {"Authorization": f"Bearer {token_provider()}"}
    else:
        raise ValueError(f"unknown auth method {auth_method!r}")
    headers["Accept"] = "application/json"

    def fetch_json(url: str) -> dict:
        attempt = 0
        while True:
            try:
                with opener(Request(url, headers=headers, method="GET"), timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    token = resp.headers.get("x-ms-continuationtoken")
                    if token and isinstance(body, dict) and "continuationToken" not in body:
                        body["continuationToken"] = token
                    return body
            except HTTPError as e:
                text = e.read().decode("utf-8", errors="replace") if e.fp else ""
                if e.code in RETRY_STATUSES and attempt < max_retries:
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    sleep(float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 30))
                    attempt += 1
                    continue
                raise HttpError(e.code, url, text) from None
            except (OSError, HTTPException):  # URLError, timeouts, resets, truncated reads
                if attempt < max_retries:
                    sleep(min(2 ** attempt, 30))
                    attempt += 1
                    continue
                raise

    return fetch_json


def list_projects(fetch_json: FetchJson, org_url: str) -> list[str]:
    base = f"https://dev.azure.com/{org_name(org_url)}/_apis/projects?api-version=7.1&$top=100"
    names: list[str] = []
    url = base
    while True:
        data = fetch_json(url)
        names.extend(p["name"] for p in data.get("value", []))
        token = data.get("continuationToken")
        if not token:
            return sorted(names)
        url = f"{base}&continuationToken={quote(str(token), safe='')}"


def fetch_titles(fetch_json: FetchJson, org_url: str, ids: list[int]) -> dict[int, str]:
    """Titles for display only; never cached. Returns {} on any HTTP failure."""
    titles: dict[int, str] = {}
    for i in range(0, len(ids), 200):
        chunk = ",".join(str(x) for x in ids[i : i + 200])
        url = (
            f"https://dev.azure.com/{org_name(org_url)}/_apis/wit/workitems"
            f"?ids={chunk}&fields=System.Title&errorPolicy=omit&api-version=7.1"
        )
        try:
            data = fetch_json(url)
        except (HttpError, URLError, RuntimeError):
            return titles
        for wi in data.get("value", []) or []:
            if wi and "id" in wi:
                titles[int(wi["id"])] = wi.get("fields", {}).get("System.Title", "")
    return titles
