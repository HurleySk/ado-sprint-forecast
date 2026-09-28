# sprint-forecast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `sprint-forecast`, a Python CLI that rebuilds Azure DevOps sprints from Analytics revision history and forecasts the distribution of "% of day-1 committed points delivered". Every model is backtested against a velocity baseline.

**Architecture:** The package uses a `src/` layout. Its small, single-purpose modules pass pandas DataFrames and dataclasses to each other. The pipeline runs in this order:
- `extract`: Analytics OData into a SQLite cache
- `timeline`: as-of reconstruction
- `sprints`
- `features`
- the models: baseline C and model A (LightGBM plus calibration)
- `rollup`: sprint-shock Monte Carlo
- `backtest`

A click CLI sits on top. A synthetic cache generator backs the tests and `demo`. All HTTP goes through an injected `fetch_json(url) -> dict`, so every test runs offline.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy, scikit-learn, LightGBM, click and joblib, plus stdlib `urllib`, `sqlite3` and `tomllib`. Tests use pytest; CI uses GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-28-sprint-forecast-design.md`

## Global Constraints

**Platform and dependencies**
- `requires-python = ">=3.11"`. CI runs a matrix of `ubuntu-latest` and `windows-latest` × Python `3.11`, `3.12`, `3.13`. All code must be cross-platform: use `pathlib`, keep every timestamp in UTC, and add no shell-specific code.
- Runtime dependencies are exactly: pandas, numpy, scipy, scikit-learn, lightgbm, click, joblib.
- Do not use `requests`; use stdlib `urllib`. Do not use `shap`; use LightGBM `pred_contrib=True`. Do not use `tomli` or `tomli-w`; read with `tomllib` and write with the small serializer in `config.py`. **Do not depend on ado-search.**
- Packaging follows the sibling ado-search repo:
  - setuptools `pyproject.toml` with a `src/` layout
  - entry point `sprint-forecast = "sprint_forecast.cli:main"`
  - `[project.optional-dependencies] dev = ["pytest>=7.0"]`
  - tests in `tests/`

**Public-repo hygiene**
- The repo is public. **No real org, project, team, person, or work-item names anywhere**: not in code, tests, docs, examples, or commit messages. Use only the synthetic names `contoso`, `Alpha`, `Beta`, `Team Red`, `Team Blue`, `Team Green` and `Team Gray`.
- `.gitignore` must contain `.sprint-forecast/`, `*.db`, `*.sqlite`, `*.jsonl`, `*.parquet`, `*.joblib` and `.env`. The cache, models and backtest output live only under `.sprint-forecast/`.
- Assignees are only opaque Analytics `UserSK` GUIDs, used only as numeric load features. Titles are fetched on demand for display and never cached.

**Data handling**
- Timestamps are stored as UTC `YYYY-MM-DDTHH:MM:SS.fffZ` strings. A `RevisedDate` with year >= 9999 is stored as NULL.
- Analytics base URL: `https://analytics.dev.azure.com/{org}/{project}/_odata/v4.0-preview/{EntitySet}`.
- Paging:
  - Page size is 5000.
  - Follow `@odata.nextLink` when the response has one. Otherwise keep increasing `$skip` while `len(value) == $top`.
  - Encode spaces as `%20` and keep `$ ( ) ' , /` readable.
- Errors: retry 429/500/502/503/504 with backoff. A 401/403 skips that project.

**Model defaults**
- Work item types default to `User Story, Product Backlog Item, Requirement, Bug`. Done categories default to `{Completed}`.
- Commit grace is 1 day, `min_history` is 8, and `retrain_every` is 4.
- The Monte Carlo uses 10,000 draws. σ is bounded to [0, 3] and fitted with 32-node Gauss–Hermite quadrature.
- LightGBM uses:
  - the spec's values: `num_leaves` 15, `min_child_samples` 20, `learning_rate` 0.03, `n_estimators` 300, `subsample` 0.8, `colsample_bytree` 0.8
  - for reproducibility: `subsample_freq=1`, `n_jobs=1`, `deterministic=True`, `force_row_wise=True`, `verbose=-1`

**Tests and output**
- Tests are offline and deterministic, with the fixed seed written in each test. The whole suite must run well under a minute; reference runs on a laptop take 25-40 s.
- CLI output is ASCII-only, because Windows consoles may use legacy code pages.

**Commands** (Windows paths; on Linux/macOS use `.venv/bin/python`)
- Create the venv: `python -m venv .venv`
- Install: `.venv/Scripts/python -m pip install -e ".[dev]"`
- Run tests: `.venv/Scripts/python -m pytest`

## Review Focus

1. **Special characters in project names.** Names with spaces, apostrophes, `&` or `#`, and type names with apostrophes, must still produce valid OData URLs. Test: Task 3 `test_odata_url_encodes_special_project_names_and_literals`.
2. **Extract failing mid-pagination.** If an extract fails partway through paging (a network drop, or a 503 after retries), the cache and watermark must stay untouched so a rerun is safe. Test: Task 6 `test_failure_mid_pagination_leaves_cache_and_watermark_unchanged`.
3. **Full re-extract of one project.** `extract --full --project Alpha` must not delete other projects' revisions or watermarks. Test: Task 6 `test_full_reextract_leaves_other_projects_alone`.
4. **Partly dated or inverted iterations.** An iteration with only one date, or with end <= start, must be ignored rather than becoming a sprint or crashing. Test: Task 8 `test_half_dated_and_inverted_iterations_are_not_sprints`.
5. **Shared iterations.** Running `predict` on an iteration shared by several teams without `--team` must print one forecast per team, not a merged one. Test: Task 14 `test_predict_shared_iteration_without_team_prints_each_team`.

## Decisions on spec ambiguities

The spec leaves these points open. They are resolved as follows, and the code below implements these resolutions.

**Extraction**

1. **Paging order.** `$skip` paging has no guaranteed order, so the revisions query adds `$orderby=WorkItemId,Revision`. Iterations and Teams are small, single-page queries and need no ordering.

**Sprint reconstruction**

2. **Where `carryover_count` lives.** It needs revision history, so `sprints.py` computes it and stores it on each item row. `features.py` copies it. The spec's "carryover counting" sprint fixture therefore lives in Task 8.
3. **Points.** StoryPoints or Effort values <= 0 count as missing. Missing points are imputed and set `is_unestimated`.
4. **"Team's trailing median item size."** This is the median raw points of the team's items in *all* sprints that ended before this sprint started. The fallbacks are the global median over the same prior sprints, then 1.
5. **"Added mid-sprint."** An item counts if, at E, it is in the iteration with a configured type and not Removed, but was not in the iteration at C.
6. **Team assignment.**
   - Area matching is case-insensitive.
   - A tie between two teams on the same longest matching area leaves the item unassigned.
   - In project-level fallback sprints, every item in the iteration belongs to the project.
7. **Iteration edge cases.** Iterations with end <= start are ignored. The cutoff is clipped to the end date for very short iterations.
8. **Sprint and team keys.**
   - `team_key` is `"project/team"`, so teams with the same name in different projects stay separate.
   - `sprint_id` is `"team_key|iteration"`.
   - `team_sprint_index` is the number of the team's sprints that ended before this sprint started.

**Features**

9. **Velocity.** Velocity is delivered committed points. A trailing velocity <= 0 becomes NaN, so `load_ratio` and `points_rel` become NaN rather than inf.

**Models**

10. **LightGBM parameters.** "colsample" means `colsample_bytree`. The plan adds `subsample_freq=1`, because without it LightGBM silently ignores `subsample`. It also adds the determinism flags listed in Global Constraints.
11. **Calibration.**
    - The model fitted on the first 80% of sprints is kept; it is not refit on 100%.
    - If the calibration set lacks either class, no calibration is applied.
    - `train_item_model` raises `ValueError` on single-class data. The backtest catches this, skips the affected targets and counts them.
12. **σ fit.** σ is fitted on the *calibrated* predictions for the calibration sprints.
13. **Negative SHAP for `carryover_count`.** SHAP values are centered over the training data, so a mean over all items proves nothing. The test instead checks that the mean over items with `carryover_count > 0` is negative, and lower than the mean over items without carryover.
14. **Model C with little history.** With fewer than 3 prior team sprints, model C draws %done from the project's last 20 finished sprints (any team, `end < start`). With no history at all, the target is skipped for C.

**Synthetic data**

15. **Synthetic `load_ratio`.** The latent formula uses the same `load_ratio` definition as the feature (committed points ÷ trailing-3 delivered), so the planted effect is learnable. `b0 = 3.0` was chosen by experiment; seed 0 gives an out-of-time item AUC of 0.715 against the 0.65 threshold.

**Backtest**

16. **Reference models and LOTO.** The team-mean reference is always reported. Leave-one-team-out is run for model A only.
17. **Retraining schedule.** The model is retrained at every `retrain_every`-th target, on sprints that ended before that target started. The leakage check runs for every target, raising `LeakageError`. It is a real exception rather than an `assert`, so it still runs under `python -O`.

**CLI**

18. **`predict --as-of`.** The option accepts `auto|now|commit`, and `auto` (the default) implements the spec's rule. With `--as-of now`, items already in a done category drop out of scope, because committed scope excludes done categories.
19. **Small CLI additions.**
    - `init` refuses to overwrite an existing config unless `--force` is given.
    - `predict` has `--top N` and `--titles/--no-titles`.
    - `demo` has `--seed` and `--n-sprints` (minimum 12), so tests can use a small, fast dataset.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `pyproject.toml`, `.gitignore`, `LICENSE`, `README.md`, `.github/workflows/ci.yml` | Packaging, hygiene, CI | 1 (README finished in 15) |
| `src/sprint_forecast/__init__.py` | Package marker + `__version__` | 1 |
| `src/sprint_forecast/config.py` | Read/write `.sprint-forecast/config.toml` | 2 |
| `src/sprint_forecast/analytics.py` | OData URLs, timestamp normalization, paging, auth, retries, REST project list and titles | 3 |
| `src/sprint_forecast/cache.py` | SQLite schema, writes, typed DataFrame loads | 4 |
| `src/sprint_forecast/synth.py` | Deterministic synthetic `cache.db` with planted effects | 5 |
| `src/sprint_forecast/extract.py` | Per-project extraction with watermarks; the only module that talks to ADO | 6 |
| `src/sprint_forecast/timeline.py` | As-of lookups (single instant and vectorized) | 7 |
| `src/sprint_forecast/sprints.py` | Sprint calendar, team assignment, committed scope, outcomes, data report | 8 |
| `src/sprint_forecast/features.py` | Leak-free feature matrix at the commit cutoff | 9 |
| `src/sprint_forecast/baseline.py` | Model C (velocity bootstrap) and team-mean reference | 10 |
| `src/sprint_forecast/model.py` | Model A (LightGBM + calibration), LR sanity model, SHAP drivers | 11 |
| `src/sprint_forecast/rollup.py` | σ fit (Gauss–Hermite), Monte Carlo %done, `Forecaster` | 12 |
| `src/sprint_forecast/backtest.py` | Expanding-window backtest, metrics, LOTO, report text | 13 |
| `src/sprint_forecast/cli.py` | Click commands `init/extract/data/backtest/train/predict/demo` | 14 |
| `tests/helpers.py` | Hand-built cache fixtures (synthetic names) | 4 |
| `tests/conftest.py` | Session-scoped synthetic datasets `synth40`, `synth16` | 11 |
| `tests/test_*.py` | One test module per source module (+ `test_smoke.py`) | each task |

## Conventions for every task

- Run all commands from the repo root: the `sprint-forecast` checkout that contains `docs/superpowers/`.
- Tests import `from helpers import ...`. This works because pytest's default `prepend` import mode puts `tests/` on `sys.path`. Do **not** add `tests/__init__.py`.
- Iteration and area paths use one backslash as the separator (`Alpha\Sprint 1`). In Python source that is written `"Alpha\\Sprint 1"`.
- Every code block below is the complete file content. Create the file exactly as shown.
- Commit at the end of each task, using only the synthetic names from Global Constraints in commit messages.

---

### Task 1: Project scaffold, hygiene files and CI

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `README.md` (stub)
- Create: `.github/workflows/ci.yml`
- Create: `src/sprint_forecast/__init__.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `sprint_forecast.__version__ == "0.1.0"`.
  - The console script `sprint-forecast`, which points at `sprint_forecast.cli:main`. That module is created in Task 14; until then the script exists but fails on import, which is expected.

- [ ] **Step 1: Write the failing smoke test**

`tests/test_smoke.py`:
```python
import sprint_forecast


def test_package_imports_with_version():
    assert sprint_forecast.__version__ == "0.1.0"
```

- [ ] **Step 2: Create the packaging and hygiene files**

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=77.0"]
build-backend = "setuptools.build_meta"

[project]
name = "sprint-forecast"
version = "0.1.0"
description = "Forecast how much of an Azure DevOps sprint's day-1 committed scope will be delivered"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
authors = [{name = "Samuel Hurley"}]
keywords = ["azure-devops", "forecasting", "sprint", "lightgbm", "monte-carlo"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
dependencies = [
    "click>=8.1",
    "joblib>=1.3",
    "lightgbm>=4.0",
    "numpy>=1.26",
    "pandas>=2.2",
    "scikit-learn>=1.4",
    "scipy>=1.11",
]

[project.urls]
Homepage = "https://github.com/HurleySk/sprint-forecast"
Repository = "https://github.com/HurleySk/sprint-forecast"
Issues = "https://github.com/HurleySk/sprint-forecast/issues"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
]

[project.scripts]
sprint-forecast = "sprint_forecast.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:
```gitignore
# Local cache, models and backtest output can contain real org data: never commit them.
.sprint-forecast/
*.db
*.sqlite
*.jsonl
*.parquet
*.joblib
.env

# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
build/
dist/
.pytest_cache/
```

`LICENSE`:
```text
MIT License

Copyright (c) 2026 Samuel Hurley

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

`README.md` (a stub; Task 15 replaces it):
```markdown
# sprint-forecast

Forecast how much of an Azure DevOps sprint's day-1 committed scope will be delivered.

Work in progress. Design: `docs/superpowers/specs/2026-09-28-sprint-forecast-design.md`.
```

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.11", "3.12", "3.13"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: python -m pip install -e ".[dev]"
      - name: Test
        run: python -m pytest -q
```

- [ ] **Step 3: Create the venv, install only the test runner, and confirm the test fails**

The editable install cannot run yet: setuptools fails with "Failed to build ... when getting requirements to build editable" while `src/` does not exist. So install pytest alone for now.

Run:
```bash
python -m venv .venv
.venv/Scripts/python -m pip install "pytest>=7.0"
.venv/Scripts/python -m pytest tests/test_smoke.py -v
```
Expected: collection error `ModuleNotFoundError: No module named 'sprint_forecast'`.

- [ ] **Step 4: Create the package and install it**

`src/sprint_forecast/__init__.py`:
```python
"""Forecast how much of an Azure DevOps sprint's day-1 committed scope gets delivered."""

__version__ = "0.1.0"
```

Now do the editable install. It also installs every runtime dependency used by later tasks:
```bash
.venv/Scripts/python -m pip install -e ".[dev]"
```

- [ ] **Step 5: Run the test and confirm it passes**

Run: `.venv/Scripts/python -m pytest tests/test_smoke.py -v`
Expected: `1 passed`.

- [ ] **Step 6: Verify the ignore rules**

Run: `git check-ignore -v .sprint-forecast/cache.db x.db x.sqlite x.jsonl x.parquet x.joblib .env`
Expected: seven lines, each naming the matching `.gitignore` rule. Then run `git status --porcelain`. It must list only the new source and hygiene files, with no `.venv/` and no `*.egg-info`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore LICENSE README.md .github/workflows/ci.yml src/sprint_forecast/__init__.py tests/test_smoke.py
git commit -m "chore: scaffold sprint-forecast package, hygiene files and CI"
```

---

### Task 2: Config (`config.py`)

**Files:**
- Create: `src/sprint_forecast/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - Constants:
    - `DATA_DIR_NAME = ".sprint-forecast"`
    - `CONFIG_FILE = "config.toml"`
    - `AUTH_METHODS = ("pat", "az-cli")`
    - `DEFAULT_TYPES = ["User Story", "Product Backlog Item", "Requirement", "Bug"]`
    - `DEFAULT_DONE = ["Completed"]`
    - `VALID_CATEGORIES = {"Proposed", "InProgress", "Resolved", "Completed", "Removed"}`
  - `@dataclass Config` with these fields:
    - `org_url: str`
    - `projects: list[str]`
    - `auth_method: str = "pat"`
    - `pat: str = ""`
    - `work_item_types: list[str]`, defaulting to a copy of `DEFAULT_TYPES`
    - `done_categories: list[str]`, defaulting to `["Completed"]`
    - `commit_grace_days: float = 1.0`
  - Paths: `data_dir(root: Path) -> Path` returns `root/.sprint-forecast`. `config_path(root: Path) -> Path` returns `root/.sprint-forecast/config.toml`.
  - `parse_done_categories(text: str) -> list[str]` splits on commas. It raises `ValueError` for unknown values or `Removed`.
  - `load_config(path: Path, env: Mapping[str, str] | None = None) -> Config`:
    - `env` defaults to `os.environ`, and `ADO_PAT` in it overrides the file's PAT.
    - A missing file raises `FileNotFoundError` with a message containing `sprint-forecast init`.
    - A bad auth value raises `ValueError` with a message containing `ado.auth`.
  - `save_config(cfg: Config, path: Path) -> None` and `to_toml(doc: dict) -> str`. `to_toml` writes a two-level dict of scalars and lists and quotes non-bare keys.
  - TOML layout:
    - `[ado]`: `org_url`, `projects`, `auth`, and an optional `pat`
    - `[model]`: `work_item_types`, `done_categories`, `commit_grace_days`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
import tomllib

import pytest

from sprint_forecast.config import (
    Config,
    config_path,
    load_config,
    parse_done_categories,
    save_config,
    to_toml,
)


def test_round_trip(tmp_path):
    cfg = Config(org_url="https://dev.azure.com/contoso", projects=["Alpha", "Beta Two"], auth_method="az-cli")
    path = config_path(tmp_path)
    save_config(cfg, path)
    loaded = load_config(path, env={})
    assert loaded == cfg
    assert path.parent.name == ".sprint-forecast"


def test_env_pat_overrides_file(tmp_path):
    cfg = Config(org_url="https://dev.azure.com/contoso", projects=["*"], pat="from-file")
    path = config_path(tmp_path)
    save_config(cfg, path)
    assert load_config(path, env={}).pat == "from-file"
    assert load_config(path, env={"ADO_PAT": "from-env"}).pat == "from-env"


def test_to_toml_quotes_non_bare_keys_and_escapes_strings():
    text = to_toml({"t": {"plain_key": 1, "needs quotes": 'a "b" \\ c', "names": ["x", "Ünïcode"]}, "flag": True})
    parsed = tomllib.loads(text)
    assert parsed == {"flag": True, "t": {"plain_key": 1, "needs quotes": 'a "b" \\ c', "names": ["x", "Ünïcode"]}}
    assert '"needs quotes" =' in text


def test_missing_config_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="sprint-forecast init"):
        load_config(config_path(tmp_path), env={})


def test_bad_auth_method_rejected(tmp_path):
    path = config_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('[ado]\norg_url = "x"\nprojects = ["A"]\nauth = "basic"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="ado.auth"):
        load_config(path, env={})


def test_parse_done_categories():
    assert parse_done_categories("Resolved, Completed") == ["Resolved", "Completed"]
    with pytest.raises(ValueError):
        parse_done_categories("Done")
    with pytest.raises(ValueError):
        parse_done_categories("Removed")
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'sprint_forecast.config'`.

- [ ] **Step 3: Implement**

`src/sprint_forecast/config.py`:
```python
"""Load and save `.sprint-forecast/config.toml`."""
from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

DATA_DIR_NAME = ".sprint-forecast"
CONFIG_FILE = "config.toml"
AUTH_METHODS = ("pat", "az-cli")
DEFAULT_TYPES = ["User Story", "Product Backlog Item", "Requirement", "Bug"]
DEFAULT_DONE = ["Completed"]
VALID_CATEGORIES = {"Proposed", "InProgress", "Resolved", "Completed", "Removed"}

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class Config:
    org_url: str
    projects: list[str]
    auth_method: str = "pat"
    pat: str = ""
    work_item_types: list[str] = field(default_factory=lambda: list(DEFAULT_TYPES))
    done_categories: list[str] = field(default_factory=lambda: list(DEFAULT_DONE))
    commit_grace_days: float = 1.0


def data_dir(root: Path) -> Path:
    return Path(root) / DATA_DIR_NAME


def config_path(root: Path) -> Path:
    return data_dir(root) / CONFIG_FILE


def parse_done_categories(text: str) -> list[str]:
    cats = [c.strip() for c in text.split(",") if c.strip()]
    bad = [c for c in cats if c not in VALID_CATEGORIES - {"Removed"}]
    if not cats or bad:
        raise ValueError(
            f"invalid done categories {bad or text!r}; choose from Proposed, InProgress, Resolved, Completed"
        )
    return cats


def load_config(path: Path, env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path} (run `sprint-forecast init` first)")
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    ado = raw.get("ado", {})
    model = raw.get("model", {})
    auth = ado.get("auth", "pat")
    if auth not in AUTH_METHODS:
        raise ValueError(f"ado.auth must be one of {AUTH_METHODS}, got {auth!r}")
    return Config(
        org_url=ado.get("org_url", ""),
        projects=list(ado.get("projects", [])),
        auth_method=auth,
        pat=env.get("ADO_PAT") or ado.get("pat", ""),
        work_item_types=list(model.get("work_item_types", DEFAULT_TYPES)),
        done_categories=list(model.get("done_categories", DEFAULT_DONE)),
        commit_grace_days=float(model.get("commit_grace_days", 1.0)),
    )


def save_config(cfg: Config, path: Path) -> None:
    ado: dict = {"org_url": cfg.org_url, "projects": cfg.projects, "auth": cfg.auth_method}
    if cfg.pat:
        ado["pat"] = cfg.pat
    doc = {
        "ado": ado,
        "model": {
            "work_item_types": cfg.work_item_types,
            "done_categories": cfg.done_categories,
            "commit_grace_days": float(cfg.commit_grace_days),
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_toml(doc), encoding="utf-8")


def to_toml(doc: dict) -> str:
    """Tiny TOML writer for a two-level dict of scalars and lists of scalars."""
    lines: list[str] = []
    for key, value in doc.items():
        if not isinstance(value, dict):
            lines.append(f"{_key(key)} = {_value(value)}")
    for key, value in doc.items():
        if isinstance(value, dict):
            lines.append("")
            lines.append(f"[{_key(key)}]")
            for k, v in value.items():
                lines.append(f"{_key(k)} = {_value(v)}")
    return "\n".join(lines).strip() + "\n"


def _key(k: str) -> str:
    return k if _BARE_KEY.match(k) else json.dumps(k, ensure_ascii=False)


def _value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/sprint_forecast/config.py tests/test_config.py
git commit -m "feat: config.toml loading and writing with ADO_PAT override"
```

---

### Task 3: Analytics client (`analytics.py`)

**Files:**
- Create: `src/sprint_forecast/analytics.py`
- Test: `tests/test_analytics.py`

**Interfaces:**
- Consumes: nothing. This module uses only the stdlib.
- Produces:
  - Constants:
    - `ADO_RESOURCE_ID = "499b84ac-1321-427f-aa17-267ca6975798"` (the public Azure DevOps resource ID)
    - `ANALYTICS_BASE = "https://analytics.dev.azure.com"`
    - `ODATA_VERSION = "v4.0-preview"`
    - `PAGE_SIZE = 5000`
    - `RETRY_STATUSES = {429, 500, 502, 503, 504}`
  - `FetchJson = Callable[[str], dict]`: the injected HTTP seam that every later module uses.
  - `class HttpError(Exception)`, with `.status: int`, `.url: str` and `.body: str`.
  - URL helpers:
    - `org_name(org_url: str) -> str` accepts `https://dev.azure.com/contoso`, `https://contoso.visualstudio.com` or a bare `contoso`, and returns `contoso`.
    - `odata_url(org: str, project: str | None, entity: str, params: dict[str, str]) -> str` quotes the project with `safe=""` and parameter values with `safe="$(),'/:"`.
    - `odata_string(value: str) -> str` wraps a value in quotes for OData, doubling `'` as `''`.
    - `with_paging(url: str, top: int, skip: int) -> str`.
  - `to_utc_iso(value: str | None) -> str | None` returns `YYYY-MM-DDTHH:MM:SS.fffZ`. It returns `None` for `None`, `""` or a year >= 9999, and raises `ValueError` for anything unrecognized.
  - `paginate(fetch_json: FetchJson, url: str, top: int = PAGE_SIZE) -> Iterator[dict]` yields rows.
  - Auth helpers: `pat_headers(pat: str) -> dict[str, str]` and `az_cli_token(run=subprocess.run) -> str`.
  - `make_fetch_json(auth_method: str, pat: str = "", *, opener=urlopen, sleep=time.sleep, token_provider=az_cli_token, max_retries: int = 5) -> FetchJson`:
    - With PAT auth and no PAT, it raises `ValueError` with a message containing `ADO_PAT`.
    - The returned function copies the `x-ms-continuationtoken` header into the body as `continuationToken`.
  - `list_projects(fetch_json: FetchJson, org_url: str) -> list[str]` returns project names sorted.
  - `fetch_titles(fetch_json: FetchJson, org_url: str, ids: list[int]) -> dict[int, str]` fetches titles in chunks of 200 for display only. It returns partial results on failure.

- [ ] **Step 1: Write the failing test**

`tests/test_analytics.py`:
```python
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
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_analytics.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'sprint_forecast.analytics'`.

- [ ] **Step 3: Implement**

`src/sprint_forecast/analytics.py`:
```python
"""Minimal Azure DevOps Analytics OData client (stdlib urllib only)."""
from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
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
            except URLError:
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
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_analytics.py -v`
Expected: `16 passed`. This includes the Review Focus test `test_odata_url_encodes_special_project_names_and_literals`.

- [ ] **Step 5: Commit**

```bash
git add src/sprint_forecast/analytics.py tests/test_analytics.py
git commit -m "feat: minimal Analytics OData client with paging, retries and auth"
```

---

### Task 4: SQLite cache (`cache.py`) and test helpers

**Files:**
- Create: `src/sprint_forecast/cache.py`
- Create: `tests/helpers.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - Constants:
    - `SCHEMA_VERSION = "1"`
    - `REVISION_COLUMNS = ["item_id","rev","project","changed","revised","created","type","state","state_category","iteration","area","assigned_to_sk","story_points","effort","parent_id"]`
    - `ITERATION_COLUMNS = ["project","iteration_sk","path","name","start_date","end_date","is_ended"]`
  - `@dataclass CacheData` holds five DataFrames:
    - `revisions`:
      - columns are `REVISION_COLUMNS`
      - `changed`, `revised` and `created` are `datetime64[..., UTC]`
      - `story_points`, `effort` and `parent_id` are float64
      - `item_id` and `rev` are int64
      - rows are sorted by `(item_id, changed, rev)`
    - `iterations`: columns are `ITERATION_COLUMNS`. `start_date` and `end_date` are UTC datetimes, NaT when undated. `is_ended` is bool.
    - `teams`: `project, team_sk, name`
    - `team_areas`: `team_sk, area_path`
    - `team_iterations`: `team_sk, iteration_path`
  - Connection and metadata:
    - `connect(path: Path | str) -> sqlite3.Connection` creates parent directories, creates the schema and sets `schema_version`.
    - `get_meta(conn, key, default=None) -> str | None` and `set_meta(conn, key, value) -> None`. The caller commits.
    - `watermark_key(project: str) -> str` returns `"watermark:{project}"`.
  - Revision writes:
    - `upsert_revisions(conn, rows: Iterable[dict]) -> int` runs `INSERT OR IGNORE` on `(item_id, rev)` and returns the number of rows inserted.
    - `delete_project_revisions(conn, project) -> None`.
    - `max_changed(conn, project) -> str | None`.
  - Per-project replacement:
    - `replace_project_iterations(conn, project, rows: list[dict]) -> None`.
    - `replace_project_teams(conn, project, teams: list[dict], team_areas: list[dict], team_iterations: list[dict]) -> None`. Team rows need `team_sk` and `name`; area rows need `team_sk` and `area_path`; iteration rows need `team_sk` and `iteration_path`.
  - `load_cache(conn, projects: list[str] | None = None) -> CacheData`.
  - `tests/helpers.py`:
    - `rev(item_id, rev_no, changed, **overrides) -> dict`. The defaults are project `Alpha`, type `User Story`, state `New`/`Proposed`, iteration `Alpha`, area `Alpha\Red` and 3 story points.
    - `iteration(path, start, end, project="Alpha") -> dict`.
    - `team(name, areas, iterations, project="Alpha") -> dict`.
    - `build_cache(tmp_path, revisions, iterations=(), teams=()) -> CacheData`, which uses `team_sk = "sk-{name}"`.

- [ ] **Step 1: Write the test helpers and the failing test**

`tests/helpers.py`:
```python
"""Builders for hand-made cache fixtures (synthetic names only)."""
from __future__ import annotations

from pathlib import Path

from sprint_forecast.cache import (
    CacheData,
    connect,
    load_cache,
    replace_project_iterations,
    replace_project_teams,
    upsert_revisions,
)


def rev(item_id: int, rev_no: int, changed: str, **overrides) -> dict:
    row = {
        "item_id": item_id,
        "rev": rev_no,
        "project": "Alpha",
        "changed": changed,
        "revised": None,
        "created": "2024-01-01T00:00:00.000Z",
        "type": "User Story",
        "state": "New",
        "state_category": "Proposed",
        "iteration": "Alpha",
        "area": "Alpha\\Red",
        "assigned_to_sk": None,
        "story_points": 3.0,
        "effort": None,
        "parent_id": None,
    }
    row.update(overrides)
    return row


def iteration(path: str, start: str | None, end: str | None, project: str = "Alpha") -> dict:
    return {
        "project": project,
        "iteration_sk": f"it-{path}",
        "path": path,
        "name": path.split("\\")[-1],
        "start_date": start,
        "end_date": end,
        "is_ended": 0,
    }


def team(name: str, areas: list[str], iterations: list[str], project: str = "Alpha") -> dict:
    return {"project": project, "name": name, "areas": areas, "iterations": iterations}


def build_cache(tmp_path: Path, revisions: list[dict], iterations: list[dict] = (), teams: list[dict] = ()) -> CacheData:
    conn = connect(Path(tmp_path) / "cache.db")
    upsert_revisions(conn, revisions)
    projects = {i["project"] for i in iterations} | {t["project"] for t in teams}
    for project in projects:
        replace_project_iterations(conn, project, [i for i in iterations if i["project"] == project])
        ts = [t for t in teams if t["project"] == project]
        replace_project_teams(
            conn,
            project,
            [{"team_sk": f"sk-{t['name']}", "name": t["name"]} for t in ts],
            [{"team_sk": f"sk-{t['name']}", "area_path": a} for t in ts for a in t["areas"]],
            [{"team_sk": f"sk-{t['name']}", "iteration_path": p} for t in ts for p in t["iterations"]],
        )
    conn.commit()
    data = load_cache(conn)
    conn.close()
    return data
```

`tests/test_cache.py`:
```python
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
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_cache.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'sprint_forecast.cache'`.

- [ ] **Step 3: Implement**

`src/sprint_forecast/cache.py`:
```python
"""SQLite schema and read helpers for `.sprint-forecast/cache.db`."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

SCHEMA_VERSION = "1"

REVISION_COLUMNS = [
    "item_id", "rev", "project", "changed", "revised", "created", "type", "state", "state_category",
    "iteration", "area", "assigned_to_sk", "story_points", "effort", "parent_id",
]
ITERATION_COLUMNS = ["project", "iteration_sk", "path", "name", "start_date", "end_date", "is_ended"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS revisions (
    item_id INTEGER NOT NULL, rev INTEGER NOT NULL, project TEXT NOT NULL,
    changed TEXT NOT NULL, revised TEXT, created TEXT,
    type TEXT, state TEXT, state_category TEXT, iteration TEXT, area TEXT,
    assigned_to_sk TEXT, story_points REAL, effort REAL, parent_id INTEGER,
    PRIMARY KEY (item_id, rev)
);
CREATE INDEX IF NOT EXISTS ix_revisions_project_changed ON revisions(project, changed);
CREATE TABLE IF NOT EXISTS iterations (
    project TEXT NOT NULL, iteration_sk TEXT PRIMARY KEY, path TEXT NOT NULL, name TEXT,
    start_date TEXT, end_date TEXT, is_ended INTEGER
);
CREATE TABLE IF NOT EXISTS teams (project TEXT NOT NULL, team_sk TEXT PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS team_areas (team_sk TEXT NOT NULL, area_path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS team_iterations (team_sk TEXT NOT NULL, iteration_path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass
class CacheData:
    revisions: pd.DataFrame
    iterations: pd.DataFrame
    teams: pd.DataFrame
    team_areas: pd.DataFrame
    team_iterations: pd.DataFrame


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    if get_meta(conn, "schema_version") is None:
        set_meta(conn, "schema_version", SCHEMA_VERSION)
    conn.commit()
    return conn


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return default if row is None else row[0]


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))


def watermark_key(project: str) -> str:
    return f"watermark:{project}"


def upsert_revisions(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    """Insert revisions keyed on (item_id, rev); existing rows are immutable and left untouched."""
    placeholders = ",".join("?" for _ in REVISION_COLUMNS)
    cur = conn.executemany(
        f"INSERT OR IGNORE INTO revisions({','.join(REVISION_COLUMNS)}) VALUES ({placeholders})",
        ([row.get(c) for c in REVISION_COLUMNS] for row in rows),
    )
    return cur.rowcount


def delete_project_revisions(conn: sqlite3.Connection, project: str) -> None:
    conn.execute("DELETE FROM revisions WHERE project = ?", (project,))


def max_changed(conn: sqlite3.Connection, project: str) -> str | None:
    return conn.execute("SELECT MAX(changed) FROM revisions WHERE project = ?", (project,)).fetchone()[0]


def replace_project_iterations(conn: sqlite3.Connection, project: str, rows: list[dict]) -> None:
    conn.execute("DELETE FROM iterations WHERE project = ?", (project,))
    conn.executemany(
        f"INSERT OR REPLACE INTO iterations({','.join(ITERATION_COLUMNS)}) VALUES (?,?,?,?,?,?,?)",
        ([r.get(c) for c in ITERATION_COLUMNS] for r in rows),
    )


def replace_project_teams(
    conn: sqlite3.Connection,
    project: str,
    teams: list[dict],
    team_areas: list[dict],
    team_iterations: list[dict],
) -> None:
    old = "SELECT team_sk FROM teams WHERE project = ?"
    conn.execute(f"DELETE FROM team_areas WHERE team_sk IN ({old})", (project,))
    conn.execute(f"DELETE FROM team_iterations WHERE team_sk IN ({old})", (project,))
    conn.execute("DELETE FROM teams WHERE project = ?", (project,))
    conn.executemany(
        "INSERT OR REPLACE INTO teams(project, team_sk, name) VALUES (?,?,?)",
        ((project, t["team_sk"], t["name"]) for t in teams),
    )
    conn.executemany(
        "INSERT INTO team_areas(team_sk, area_path) VALUES (?,?)",
        ((a["team_sk"], a["area_path"]) for a in team_areas),
    )
    conn.executemany(
        "INSERT INTO team_iterations(team_sk, iteration_path) VALUES (?,?)",
        ((i["team_sk"], i["iteration_path"]) for i in team_iterations),
    )


def _ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, format="ISO8601")


def load_cache(conn: sqlite3.Connection, projects: list[str] | None = None) -> CacheData:
    where, params = "", ()
    if projects:
        where = f" WHERE project IN ({','.join('?' for _ in projects)})"
        params = tuple(projects)
    revs = pd.read_sql_query(f"SELECT * FROM revisions{where}", conn, params=params)
    for col in ("changed", "revised", "created"):
        revs[col] = _ts(revs[col])
    for col in ("story_points", "effort", "parent_id"):
        revs[col] = revs[col].astype("float64")
    revs["item_id"] = revs["item_id"].astype("int64")
    revs["rev"] = revs["rev"].astype("int64")
    revs = revs.sort_values(["item_id", "changed", "rev"], kind="mergesort").reset_index(drop=True)

    its = pd.read_sql_query(f"SELECT * FROM iterations{where}", conn, params=params)
    its["start_date"] = _ts(its["start_date"])
    its["end_date"] = _ts(its["end_date"])
    its["is_ended"] = its["is_ended"].fillna(0).astype(bool)

    teams = pd.read_sql_query(f"SELECT * FROM teams{where}", conn, params=params)
    sks = tuple(teams["team_sk"])
    in_sk = f" WHERE team_sk IN ({','.join('?' for _ in sks)})" if sks else " WHERE 0"
    team_areas = pd.read_sql_query(f"SELECT * FROM team_areas{in_sk}", conn, params=sks)
    team_iterations = pd.read_sql_query(f"SELECT * FROM team_iterations{in_sk}", conn, params=sks)
    return CacheData(revs, its, teams, team_areas, team_iterations)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_cache.py -v`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/sprint_forecast/cache.py tests/helpers.py tests/test_cache.py
git commit -m "feat: SQLite cache schema, per-project replace and typed loads"
```

---

### Task 5: Synthetic data generator (`synth.py`)

**Files:**
- Create: `src/sprint_forecast/synth.py`
- Test: `tests/test_synth.py`

**Interfaces:**
- Consumes (Task 4): `connect`, `upsert_revisions`, `replace_project_iterations`, `replace_project_teams`, `set_meta`, `watermark_key`.
- Produces:
  - Constants for the planted logit: `B0=3.0`, `BETA_CARRY=0.7`, `BETA_LOAD=1.3`, `BETA_SIZE=0.35`, `SHOCK_SD=0.4`.
  - `EPOCH = 2024-01-08T05:00Z` and `PROJECT_OFFSET_DAYS = {"Alpha": 0, "Beta": 3}`.
  - `SynthTeam(project, name, area, capacity, effect, runs_sprints=True, uses_effort=False)` and `TEAMS`:

    | Team | Project | Area | Capacity | Effect | Notes |
    |---|---|---|---|---|---|
    | Team Red | Alpha | `Alpha\Red` | 80 | +0.3 | |
    | Team Blue | Alpha | `Alpha\Blue` | 64 | -0.3 | |
    | Team Green | Beta | `Beta\Green` | 56 | 0 | uses Product Backlog Item and Effort |
    | Team Gray | Beta | `Beta\Gray` | 0 | 0 | subscribed to no iterations, so it runs no sprints |

  - `sprint_path(project, k) -> str` returns `"{project}\\Sprint {k+1}"`.
  - `sprint_window(project, k) -> tuple[datetime, datetime]` returns a 14-day window ending 1 ms before the next sprint.
  - `simulate(seed: int = 0, n_sprints: int = 40) -> dict[str, list[dict]]`:
    - Its keys are `revisions`, `iterations`, `teams`, `team_areas` and `team_iterations`.
    - Rows are shaped like the cache tables, and team rows also carry `project`.
    - Output is deterministic per seed.
  - `generate(path: Path | str, *, seed: int = 0, n_sprints: int = 40) -> Path` writes a fresh `cache.db`, replacing any existing file. It sets watermarks and meta `synthetic = "seed={seed};n_sprints={n}"`.
  - The planted mess:
    - about 15% unpointed items
    - undated project-root and `\Someday` iterations
    - 5% of items bounce to `\Someday`
    - Poisson(1) mid-sprint additions per sprint
    - 3% of items at the project-root area, which are unassigned in Alpha
    - 5% of items only Resolved at the end
    - 3% of items moved out mid-sprint
    - unfinished items carried over, sent back to the backlog, or Removed

- [ ] **Step 1: Write the failing test**

`tests/test_synth.py`:
```python
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
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_synth.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'sprint_forecast.synth'`.

- [ ] **Step 3: Implement**

`src/sprint_forecast/synth.py`:
```python
"""Synthetic cache.db generator with planted effects. All names are synthetic."""
from __future__ import annotations

import itertools
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from sprint_forecast.cache import (
    connect,
    replace_project_iterations,
    replace_project_teams,
    set_meta,
    upsert_revisions,
    watermark_key,
)

B0 = 3.0
BETA_CARRY = 0.7
BETA_LOAD = 1.3
BETA_SIZE = 0.35
SHOCK_SD = 0.4
LOAD_MU = 0.0
LOAD_SD = 0.45
SIZES = [1, 2, 3, 5, 8, 13]
SIZE_P = [0.12, 0.20, 0.25, 0.22, 0.14, 0.07]
UNPOINTED_SHARE = 0.15
EPOCH = datetime(2024, 1, 8, 5, 0, tzinfo=timezone.utc)  # Monday 00:00 at UTC-05:00
PROJECT_OFFSET_DAYS = {"Alpha": 0, "Beta": 3}


@dataclass(frozen=True)
class SynthTeam:
    project: str
    name: str
    area: str
    capacity: float
    effect: float
    runs_sprints: bool = True
    uses_effort: bool = False


TEAMS = (
    SynthTeam("Alpha", "Team Red", "Alpha\\Red", 80.0, 0.3),
    SynthTeam("Alpha", "Team Blue", "Alpha\\Blue", 64.0, -0.3),
    SynthTeam("Beta", "Team Green", "Beta\\Green", 56.0, 0.0, uses_effort=True),
    SynthTeam("Beta", "Team Gray", "Beta\\Gray", 0.0, 0.0, runs_sprints=False),
)


def _iso(t: datetime | None) -> str | None:
    return None if t is None else t.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sprint_path(project: str, k: int) -> str:
    return f"{project}\\Sprint {k + 1}"


def sprint_window(project: str, k: int) -> tuple[datetime, datetime]:
    start = EPOCH + timedelta(days=PROJECT_OFFSET_DAYS[project] + 14 * k)
    return start, start + timedelta(days=14) - timedelta(milliseconds=1)


class _Item:
    def __init__(self, item_id: int, project: str, created: datetime, fields: dict):
        self.id = item_id
        self.project = project
        self.created = created
        self.fields = dict(fields)
        self.events: list[tuple[datetime, dict]] = [(created, dict(self.fields))]

    def set(self, t: datetime, **changes) -> None:
        t = max(t, self.events[-1][0] + timedelta(minutes=1))
        self.fields.update(changes)
        self.events.append((t, dict(self.fields)))


class _Sim:
    def __init__(self, seed: int, n_sprints: int):
        self.rng = np.random.default_rng(seed)
        self.n_sprints = n_sprints
        self.ids = itertools.count(1000)
        self.items: list[_Item] = []

    def u(self, lo: float, hi: float) -> float:
        return float(self.rng.uniform(lo, hi))

    def new_item(self, team: SynthTeam, members: list[str], created: datetime, iteration: str) -> tuple[_Item, float]:
        rng = self.rng
        pts = float(rng.choice(SIZES, p=SIZE_P))
        is_bug = rng.random() < 0.2
        typ = "Bug" if is_bug else ("Product Backlog Item" if team.uses_effort else "User Story")
        estimated = rng.random() >= UNPOINTED_SHARE
        r = rng.random()
        area = team.area + "\\Web" if r < 0.3 else (team.project if r < 0.33 else team.area)
        parent = int(rng.integers(100, 200)) if rng.random() < 0.5 else None
        assignee = members[int(rng.integers(len(members)))] if members and rng.random() < 0.9 else None
        item = _Item(next(self.ids), team.project, created, {
            "type": typ, "state": "New", "state_category": "Proposed", "iteration": iteration,
            "area": area, "assigned_to_sk": assignee,
            "story_points": pts if estimated and not team.uses_effort else None,
            "effort": pts if estimated and team.uses_effort else None,
            "parent_id": parent,
        })
        self.items.append(item)
        return item, pts

    def complete(self, item: _Item, cutoff: datetime, end: datetime) -> bool:
        """Finish the item; returns False when it only reaches Resolved by the end (Closed afterwards)."""
        if item.fields["state_category"] == "Proposed":
            item.set(cutoff + timedelta(days=self.u(0, 4)), state="Active", state_category="InProgress")
        done_at = cutoff + timedelta(days=self.u(5, 12.5))
        if self.rng.random() < 0.05:
            item.set(done_at, state="Resolved", state_category="Resolved")
            item.set(end + timedelta(days=self.u(1, 3)), state="Closed", state_category="Completed")
            return False
        item.set(done_at, state="Closed", state_category="Completed")
        return True

    def leave_unfinished(self, item: _Item, cutoff: datetime) -> None:
        if item.fields["state_category"] == "Proposed" and self.rng.random() < 0.6:
            item.set(cutoff + timedelta(days=self.u(0, 10)), state="Active", state_category="InProgress")

    def run_team(self, team: SynthTeam) -> None:
        rng = self.rng
        members = [str(uuid.UUID(bytes=rng.bytes(16), version=4)) for _ in range(4)]
        carry: list[tuple[_Item, int, float]] = []
        delivered_history: list[float] = []
        for k in range(self.n_sprints):
            start, end = sprint_window(team.project, k)
            cutoff = start + timedelta(days=1)
            path = sprint_path(team.project, k)
            last = k == self.n_sprints - 1
            velocity = float(np.mean(delivered_history[-3:])) if delivered_history else 0.8 * team.capacity
            basis = 0.85 * team.capacity
            target = basis * min(2.5, max(0.5, math.exp(rng.normal(LOAD_MU, LOAD_SD))))
            committed = []
            total = 0.0
            for item, count, pts in carry:
                if total >= target:
                    item.set(start + timedelta(hours=self.u(1, 20)), iteration=team.project)
                    continue
                item.set(start + timedelta(hours=self.u(1, 20)), iteration=path)
                committed.append((item, count, pts))
                total += pts
            while total < target:
                item, pts = self.new_item(team, members, start - timedelta(days=self.u(2, 40)), team.project)
                if rng.random() < 0.05:
                    item.set(item.created + timedelta(hours=self.u(12, 24)), iteration=f"{team.project}\\Someday")
                plan_at = start - timedelta(days=self.u(0, 3)) if rng.random() < 0.5 else start + timedelta(hours=self.u(0, 20))
                item.set(plan_at, iteration=path)
                committed.append((item, 0, pts))
                total += pts
            # load_ratio as the features define it: committed points / trailing-3 mean delivered points
            load_ratio = min(3.0, total / max(velocity, 1.0))
            shock = float(rng.normal(0.0, SHOCK_SD))
            carry = []
            delivered = 0.0
            for item, count, pts in committed:
                if rng.random() < 0.03:
                    item.set(cutoff + timedelta(days=self.u(1, 8)), iteration=team.project)
                    continue
                logit = (B0 - BETA_CARRY * count - BETA_LOAD * (load_ratio - 1) - BETA_SIZE * math.log(pts)
                         + team.effect + shock)
                if rng.random() < 1 / (1 + math.exp(-logit)):
                    delivered += pts if self.complete(item, cutoff, end) else 0.0
                    continue
                self.leave_unfinished(item, cutoff)
                if last:
                    continue
                fate = rng.random()
                after = end + timedelta(hours=self.u(1, 20))
                if fate < 0.05:
                    item.set(after, state="Removed", state_category="Removed")
                elif fate < (0.4 if count >= 2 else 0.15):
                    item.set(after, iteration=team.project)
                else:
                    carry.append((item, count + 1, pts))
            delivered_history.append(delivered)
            for _ in range(int(rng.poisson(1.0))):
                item, pts = self.new_item(team, members, cutoff + timedelta(days=self.u(1, 8)), path)
                if rng.random() < 0.6:
                    self.complete(item, cutoff, end)
                elif not last:
                    carry.append((item, 1, pts))

    def backlog_only(self, team: SynthTeam, n: int) -> None:
        for _ in range(n):
            self.new_item(team, [], EPOCH + timedelta(days=self.u(0, 60)), team.project)


def simulate(seed: int = 0, n_sprints: int = 40) -> dict[str, list[dict]]:
    sim = _Sim(seed, n_sprints)
    iterations: list[dict] = []
    for project in PROJECT_OFFSET_DAYS:
        iterations.append(_iteration_row(project, project, None, None))
        iterations.append(_iteration_row(project, f"{project}\\Someday", None, None))
        for k in range(n_sprints + 1):
            start, end = sprint_window(project, k)
            iterations.append(_iteration_row(project, sprint_path(project, k), start, end))
    teams, team_areas, team_iterations = [], [], []
    for team in TEAMS:
        sk = str(uuid.uuid5(uuid.NAMESPACE_URL, f"synthetic-team/{team.project}/{team.name}"))
        teams.append({"project": team.project, "team_sk": sk, "name": team.name})
        team_areas.append({"project": team.project, "team_sk": sk, "area_path": team.area})
        if team.runs_sprints:
            team_iterations.extend(
                {"project": team.project, "team_sk": sk, "iteration_path": sprint_path(team.project, k)}
                for k in range(n_sprints + 1)
            )
    for team in TEAMS:
        if team.runs_sprints:
            sim.run_team(team)
        else:
            sim.backlog_only(team, 10)
    revisions = []
    for item in sim.items:
        for i, (t, fields) in enumerate(item.events):
            nxt = item.events[i + 1][0] if i + 1 < len(item.events) else None
            revisions.append({
                "item_id": item.id, "rev": i + 1, "project": item.project, "changed": _iso(t),
                "revised": _iso(nxt), "created": _iso(item.created), **fields,
            })
    return {
        "revisions": revisions, "iterations": iterations, "teams": teams,
        "team_areas": team_areas, "team_iterations": team_iterations,
    }


def _iteration_row(project: str, path: str, start: datetime | None, end: datetime | None) -> dict:
    return {
        "project": project,
        "iteration_sk": str(uuid.uuid5(uuid.NAMESPACE_URL, f"synthetic-iteration/{path}")),
        "path": path,
        "name": path.split("\\")[-1],
        "start_date": _iso(start),
        "end_date": _iso(end),
        "is_ended": 0,
    }


def generate(path: Path | str, *, seed: int = 0, n_sprints: int = 40) -> Path:
    """Write a fresh synthetic cache.db at `path` and return the path."""
    path = Path(path)
    if path.exists():
        path.unlink()
    data = simulate(seed=seed, n_sprints=n_sprints)
    conn = connect(path)
    try:
        upsert_revisions(conn, data["revisions"])
        for project in PROJECT_OFFSET_DAYS:
            replace_project_iterations(conn, project, [r for r in data["iterations"] if r["project"] == project])
            replace_project_teams(
                conn,
                project,
                [t for t in data["teams"] if t["project"] == project],
                [a for a in data["team_areas"] if a["project"] == project],
                [i for i in data["team_iterations"] if i["project"] == project],
            )
            changed = [r["changed"] for r in data["revisions"] if r["project"] == project]
            set_meta(conn, watermark_key(project), max(changed))
        set_meta(conn, "synthetic", f"seed={seed};n_sprints={n_sprints}")
        conn.commit()
    finally:
        conn.close()
    return path
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_synth.py -v`
Expected: `6 passed` in about 2 s.

- [ ] **Step 5: Commit**

```bash
git add src/sprint_forecast/synth.py tests/test_synth.py
git commit -m "feat: deterministic synthetic cache generator with planted effects"
```

---

### Task 6: Extraction (`extract.py`)

**Files:**
- Create: `src/sprint_forecast/extract.py`
- Test: `tests/test_extract.py`

**Interfaces:**
- Consumes:
  - Task 3: `FetchJson`, `HttpError`, `list_projects`, `odata_string`, `odata_url`, `paginate`, `to_utc_iso`.
  - Task 4: `delete_project_revisions`, `get_meta`, `max_changed`, `replace_project_iterations`, `replace_project_teams`, `set_meta`, `upsert_revisions`, `watermark_key`.
- Produces:
  - Query constants:
    - `REVISION_SELECT = "WorkItemId,Revision,ChangedDate,RevisedDate,WorkItemType,State,StateCategory,StoryPoints,Effort,ParentWorkItemId,CreatedDate"`
    - `REVISION_EXPAND = "Iteration($select=IterationPath),Area($select=AreaPath),AssignedTo($select=UserSK)"`
    - `ITERATION_SELECT`, `TEAM_SELECT`, `TEAM_EXPAND`
  - `@dataclass ExtractResult(project: str, fetched: int = 0, inserted: int = 0, iterations: int = 0, teams: int = 0, skipped: str | None = None)`.
  - URL builders:
    - `revisions_url(org, project, work_item_types, watermark: str | None) -> str` filters `WorkItemType in (...)`, adds `and ChangedDate ge <watermark>` when a watermark is given, and orders by `WorkItemId,Revision`.
    - `iterations_url(org, project) -> str` and `teams_url(org, project) -> str`.
  - Row normalizers:
    - `revision_row(project, raw) -> dict` returns the `REVISION_COLUMNS` keys. Null navigation properties become `None`.
    - `iteration_row(project, raw) -> dict`.
    - `team_rows(raw) -> tuple[dict, list[dict], list[dict]]`.
  - `extract_project(fetch_json, conn, org, project, *, work_item_types: list[str], full: bool = False) -> ExtractResult`:
    - It fetches everything first, then writes everything in one transaction.
    - A 401/403 returns `skipped="HTTP 401: ..."` and writes nothing. Other `HttpError`s propagate.
  - `extract_all(fetch_json, conn, org, projects: list[str], *, work_item_types, full=False, echo=print) -> list[ExtractResult]`. `["*"]` expands through `list_projects`.

- [ ] **Step 1: Write the failing test**

`tests/test_extract.py`:
```python
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
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_extract.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'sprint_forecast.extract'`.

- [ ] **Step 3: Implement**

`src/sprint_forecast/extract.py`:
```python
"""Pull revisions, iterations and teams per project into cache.db. The only module that talks to ADO."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from sprint_forecast.analytics import (
    FetchJson,
    HttpError,
    list_projects,
    odata_string,
    odata_url,
    paginate,
    to_utc_iso,
)
from sprint_forecast.cache import (
    delete_project_revisions,
    get_meta,
    max_changed,
    replace_project_iterations,
    replace_project_teams,
    set_meta,
    upsert_revisions,
    watermark_key,
)

REVISION_SELECT = (
    "WorkItemId,Revision,ChangedDate,RevisedDate,WorkItemType,State,StateCategory,"
    "StoryPoints,Effort,ParentWorkItemId,CreatedDate"
)
REVISION_EXPAND = "Iteration($select=IterationPath),Area($select=AreaPath),AssignedTo($select=UserSK)"
ITERATION_SELECT = "IterationSK,IterationPath,IterationName,StartDate,EndDate,IsEnded"
TEAM_SELECT = "TeamSK,TeamName"
TEAM_EXPAND = "Areas($select=AreaPath),Iterations($select=IterationPath)"


@dataclass
class ExtractResult:
    project: str
    fetched: int = 0
    inserted: int = 0
    iterations: int = 0
    teams: int = 0
    skipped: str | None = None


def revisions_url(org: str, project: str, work_item_types: list[str], watermark: str | None) -> str:
    filters = [f"WorkItemType in ({','.join(odata_string(t) for t in work_item_types)})"]
    if watermark:
        filters.append(f"ChangedDate ge {watermark}")
    return odata_url(org, project, "WorkItemRevisions", {
        "$select": REVISION_SELECT,
        "$expand": REVISION_EXPAND,
        "$filter": " and ".join(filters),
        "$orderby": "WorkItemId,Revision",
    })


def iterations_url(org: str, project: str) -> str:
    return odata_url(org, project, "Iterations", {"$select": ITERATION_SELECT})


def teams_url(org: str, project: str) -> str:
    return odata_url(org, project, "Teams", {"$select": TEAM_SELECT, "$expand": TEAM_EXPAND})


def _nav(raw: dict, nav: str, prop: str):
    value = raw.get(nav)
    return value.get(prop) if isinstance(value, dict) else None


def revision_row(project: str, raw: dict) -> dict:
    parent = raw.get("ParentWorkItemId")
    return {
        "item_id": int(raw["WorkItemId"]),
        "rev": int(raw["Revision"]),
        "project": project,
        "changed": to_utc_iso(raw["ChangedDate"]),
        "revised": to_utc_iso(raw.get("RevisedDate")),
        "created": to_utc_iso(raw.get("CreatedDate")),
        "type": raw.get("WorkItemType"),
        "state": raw.get("State"),
        "state_category": raw.get("StateCategory"),
        "iteration": _nav(raw, "Iteration", "IterationPath"),
        "area": _nav(raw, "Area", "AreaPath"),
        "assigned_to_sk": _nav(raw, "AssignedTo", "UserSK"),
        "story_points": raw.get("StoryPoints"),
        "effort": raw.get("Effort"),
        "parent_id": int(parent) if parent is not None else None,
    }


def iteration_row(project: str, raw: dict) -> dict:
    return {
        "project": project,
        "iteration_sk": raw["IterationSK"],
        "path": raw["IterationPath"],
        "name": raw.get("IterationName"),
        "start_date": to_utc_iso(raw.get("StartDate")),
        "end_date": to_utc_iso(raw.get("EndDate")),
        "is_ended": int(bool(raw.get("IsEnded"))),
    }


def team_rows(raw: dict) -> tuple[dict, list[dict], list[dict]]:
    sk = raw["TeamSK"]
    team = {"team_sk": sk, "name": raw["TeamName"]}
    areas = [{"team_sk": sk, "area_path": a["AreaPath"]} for a in raw.get("Areas") or [] if a.get("AreaPath")]
    its = [
        {"team_sk": sk, "iteration_path": i["IterationPath"]}
        for i in raw.get("Iterations") or [] if i.get("IterationPath")
    ]
    return team, areas, its


def extract_project(
    fetch_json: FetchJson,
    conn: sqlite3.Connection,
    org: str,
    project: str,
    *,
    work_item_types: list[str],
    full: bool = False,
) -> ExtractResult:
    result = ExtractResult(project)
    try:
        iterations = [iteration_row(project, r) for r in paginate(fetch_json, iterations_url(org, project))]
        teams, areas, subs = [], [], []
        for raw in paginate(fetch_json, teams_url(org, project)):
            t, a, s = team_rows(raw)
            teams.append(t)
            areas.extend(a)
            subs.extend(s)
        watermark = None if full else get_meta(conn, watermark_key(project))
        url = revisions_url(org, project, work_item_types, watermark)
        rows = [revision_row(project, r) for r in paginate(fetch_json, url)]
    except HttpError as e:
        if e.status in (401, 403):
            result.skipped = f"HTTP {e.status}: no Analytics access"
            return result
        raise
    with conn:
        if full:
            delete_project_revisions(conn, project)
        result.inserted = upsert_revisions(conn, rows)
        replace_project_iterations(conn, project, iterations)
        replace_project_teams(conn, project, teams, areas, subs)
        newest = max_changed(conn, project)
        if newest:
            set_meta(conn, watermark_key(project), newest)
        set_meta(conn, "last_extract", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    result.fetched, result.iterations, result.teams = len(rows), len(iterations), len(teams)
    return result


def extract_all(
    fetch_json: FetchJson,
    conn: sqlite3.Connection,
    org: str,
    projects: list[str],
    *,
    work_item_types: list[str],
    full: bool = False,
    echo: Callable[[str], None] = print,
) -> list[ExtractResult]:
    names = list_projects(fetch_json, org) if projects == ["*"] else projects
    results = []
    for project in names:
        r = extract_project(fetch_json, conn, org, project, work_item_types=work_item_types, full=full)
        if r.skipped:
            echo(f"  {project}: skipped ({r.skipped})")
        else:
            echo(f"  {project}: {r.fetched} revisions fetched ({r.inserted} new), {r.iterations} iterations, {r.teams} teams")
        results.append(r)
    return results
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_extract.py -v`
Expected: `10 passed`. This includes the Review Focus tests `test_full_reextract_leaves_other_projects_alone` and `test_failure_mid_pagination_leaves_cache_and_watermark_unchanged`.

- [ ] **Step 5: Commit**

```bash
git add src/sprint_forecast/extract.py tests/test_extract.py
git commit -m "feat: incremental per-project extraction with watermarks and 401/403 skip"
```

---

### Task 7: As-of timeline (`timeline.py`)

**Files:**
- Create: `src/sprint_forecast/timeline.py`
- Test: `tests/test_timeline.py`

**Interfaces:**
- Consumes (Task 4): `CacheData.revisions`, plus `tests/helpers.py` (`build_cache`, `rev`).
- Produces:
  - `STATE_COLUMNS = ["rev","changed","created","type","state","state_category","iteration","area","assigned_to_sk","story_points","effort","parent_id"]`.
  - `to_utc(t) -> pd.Timestamp`: a naive value is treated as UTC; an offset value is converted.
  - `as_of(revisions, t) -> pd.DataFrame` returns one row per item: the last revision with `changed <= t`, where ties on `changed` go to the highest `rev`. Items with no revision by `t` are absent.
  - `as_of_many(revisions, keys: pd.DataFrame, time_col: str) -> pd.DataFrame`:
    - `keys` has `item_id` and `time_col`.
    - It returns the `keys` columns plus `revisions_so_far` (the number of revisions up to that instant) and `STATE_COLUMNS`.
    - Output rows keep the order of `keys`. Rows with no revision yet get NaN/NaT.
    - Internally it uses `pd.merge_asof(direction="backward", by="item_id")`, after casting `keys[time_col]` to the dtype of `revisions.changed`. pandas refuses to merge ns against us timestamps.

- [ ] **Step 1: Write the failing test**

`tests/test_timeline.py`:
```python
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
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_timeline.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'sprint_forecast.timeline'`.

- [ ] **Step 3: Implement**

`src/sprint_forecast/timeline.py`:
```python
"""As-of reconstruction: an item's state at instant t is its last revision with changed <= t."""
from __future__ import annotations

import numpy as np
import pandas as pd

STATE_COLUMNS = [
    "rev", "changed", "created", "type", "state", "state_category", "iteration", "area",
    "assigned_to_sk", "story_points", "effort", "parent_id",
]


def to_utc(t) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def as_of(revisions: pd.DataFrame, t) -> pd.DataFrame:
    """One row per item: its last revision with changed <= t. Items with no revision by t are absent."""
    t = to_utc(t)
    sub = revisions[revisions["changed"] <= t]
    sub = sub.sort_values(["item_id", "changed", "rev"], kind="mergesort")
    return sub.drop_duplicates("item_id", keep="last").reset_index(drop=True)


def as_of_many(revisions: pd.DataFrame, keys: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """For each row of `keys` (columns item_id, time_col, plus any non-revision columns), attach the item's
    last revision with changed <= keys[time_col] (STATE_COLUMNS) and `revisions_so_far`, the number of
    revisions up to that instant. Rows with no revision yet get NaN/NaT. Output keeps the order of `keys`."""
    right = revisions.sort_values(["changed", "rev"], kind="mergesort")
    right = right.assign(revisions_so_far=right.groupby("item_id").cumcount() + 1)
    right = right[["item_id", "revisions_so_far"] + STATE_COLUMNS]
    left = keys.reset_index(drop=True).assign(_row=np.arange(len(keys)))
    left[time_col] = left[time_col].astype(right["changed"].dtype)
    left = left.sort_values(time_col, kind="mergesort")
    out = pd.merge_asof(
        left, right, left_on=time_col, right_on="changed", by="item_id",
        direction="backward", allow_exact_matches=True,
    )
    return out.sort_values("_row").drop(columns="_row").reset_index(drop=True)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_timeline.py -v`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/sprint_forecast/timeline.py tests/test_timeline.py
git commit -m "feat: as-of reconstruction (single instant and vectorized)"
```

---

### Task 8: Sprint reconstruction (`sprints.py`)

**Files:**
- Create: `src/sprint_forecast/sprints.py`
- Test: `tests/test_sprints.py`

**Interfaces:**
- Consumes:
  - Task 4: `CacheData`.
  - Task 7: `as_of_many`, `to_utc`.
  - Tests use `tests/helpers.py` and, from Task 5, `synth.generate`.
- Produces:
  - Column lists:
    - `REMOVED = "Removed"`.
    - `CALENDAR_COLUMNS = ["sprint_id","project","team","team_key","iteration","start","end","cutoff"]`.
    - `SPRINT_COLUMNS` = `CALENDAR_COLUMNS` + `["n_items","committed_points","done_points","pct_done","pct_done_count","n_unestimated","n_added_mid"]`.
    - `ITEM_COLUMNS` = `CALENDAR_COLUMNS` + `["item_id","type","state_category_at_commit","area","assigned_to_sk","parent_id","created","last_changed","revisions_so_far","carryover_count","raw_points","points","is_unestimated","done","state_category_at_end"]`.
  - Key formats:
    - `team_key = "{project}/{team}"`.
    - `sprint_id = "{team_key}|{iteration}"`.
    - For project-level fallback sprints, `team == project`, so `team_key` is `"Beta/Beta"`.
  - `@dataclass SprintData(sprints: pd.DataFrame, items: pd.DataFrame, report: dict)`. The `report` keys are:
    - `unassigned_items`
    - `added_mid_sprint`
    - `dropped_empty_sprints`
    - `fallback_projects` (a list)
    - `n_sprints`
    - `n_committed_items`
    - `unestimated_share`
    - `end_state_mix` (a dict)
  - `raw_points(frame) -> pd.Series`: story points, falling back to effort, with values <= 0 treated as NaN.
  - `match_team(area, candidates: list[str], team_areas: dict[str, list[str]]) -> str | None`.
  - `sprint_calendar(cache, commit_grace_days=1.0) -> pd.DataFrame`: `CALENDAR_COLUMNS` + `is_fallback`, sorted by `(start, sprint_id)`.
  - Point helpers:
    - `impute_points(items, history_items) -> pd.DataFrame` adds `points` and `is_unestimated`.
    - `summarize_sprints(items, n_added: dict[str, int] | None = None) -> pd.DataFrame` returns `SPRINT_COLUMNS`.
  - `build_sprints(cache, *, work_item_types: list[str], done_categories=("Completed",), commit_grace_days=1.0) -> SprintData`. In `items`, `done` is bool.
  - `scope_at(cache, history: SprintData, *, iteration: str, cutoff, work_item_types, done_categories=("Completed",), commit_grace_days=1.0, team: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]`:
    - Returns the committed-style scope as of `cutoff`, with `done` set to NaN.
    - Raises `ValueError("no team runs ...")` for an unknown iteration, or `"team 'X' does not run ...; teams: ..."` for a team that does not run it.
  - `data_report(cache, sd) -> dict` returns `sd.report` plus:
    - `projects`: a list of dicts with keys `project, teams, teams_running_sprints, dated_iterations, undated_iterations, sprints, committed_items, fallback`
    - `resolved_share_of_resolved_or_completed`

- [ ] **Step 1: Write the failing test** (one hand-built fixture per spec case)

`tests/test_sprints.py`:
```python
import math

import pandas as pd
import pytest

from helpers import build_cache, iteration, rev, team
from sprint_forecast.sprints import (
    ITEM_COLUMNS,
    SPRINT_COLUMNS,
    build_sprints,
    data_report,
    match_team,
    scope_at,
    sprint_calendar,
)

TYPES = ["User Story", "Bug"]
S0 = iteration("Alpha\\Sprint 0", "2024-02-19T05:00:00.000Z", "2024-03-04T04:59:59.999Z")
S1 = iteration("Alpha\\Sprint 1", "2024-03-04T05:00:00.000Z", "2024-03-18T04:59:59.999Z")
S2 = iteration("Alpha\\Sprint 2", "2024-03-18T05:00:00.000Z", "2024-04-01T04:59:59.999Z")
UNDATED = iteration("Alpha\\Someday", None, None)
PRE = "2024-03-01T00:00:00.000Z"    # before Sprint 1 starts
PLAN = "2024-03-04T12:00:00.000Z"   # after start, before cutoff (start + 1 day)
MID = "2024-03-10T00:00:00.000Z"    # inside the sprint, after cutoff
POST = "2024-03-20T00:00:00.000Z"   # after Sprint 1 ends
RED = team("Team Red", ["Alpha\\Red"], ["Alpha\\Sprint 0", "Alpha\\Sprint 1", "Alpha\\Sprint 2"])
SPRINT1 = "Alpha/Team Red|Alpha\\Sprint 1"


def build(tmp_path, revisions, iterations=(S0, S1, S2, UNDATED), teams=(RED,), done=("Completed",)):
    cache = build_cache(tmp_path, revisions, list(iterations), list(teams))
    return build_sprints(cache, work_item_types=TYPES, done_categories=list(done)), cache


def items_of(sd, sprint_id=SPRINT1):
    return sd.items[sd.items["sprint_id"] == sprint_id].set_index("item_id")


def test_committed_vs_added_after_cutoff(tmp_path):
    sd, _ = build(tmp_path, [
        rev(1, 1, PRE), rev(1, 2, PLAN, iteration="Alpha\\Sprint 1"),
        rev(2, 1, PRE), rev(2, 2, MID, iteration="Alpha\\Sprint 1"),
    ])
    assert list(items_of(sd).index) == [1]
    assert sd.report["added_mid_sprint"] == 1
    assert sd.sprints.set_index("sprint_id").loc[SPRINT1, "n_added_mid"] == 1
    assert list(sd.sprints.columns) == SPRINT_COLUMNS
    assert list(sd.items.columns) == ITEM_COLUMNS


def test_moved_out_mid_sprint_is_not_done(tmp_path):
    sd, _ = build(tmp_path, [
        rev(1, 1, PLAN, iteration="Alpha\\Sprint 1"),
        rev(1, 2, MID, iteration="Alpha\\Sprint 2"),
        rev(1, 3, "2024-03-12T00:00:00.000Z", iteration="Alpha\\Sprint 2", state="Closed", state_category="Completed"),
    ])
    row = items_of(sd).loc[1]
    assert not row["done"]
    assert row["state_category_at_end"] == "Completed"


def test_removed_during_sprint_not_done_and_removed_before_cutoff_not_committed(tmp_path):
    sd, _ = build(tmp_path, [
        rev(1, 1, PLAN, iteration="Alpha\\Sprint 1"),
        rev(1, 2, MID, iteration="Alpha\\Sprint 1", state="Removed", state_category="Removed"),
        rev(2, 1, PRE, iteration="Alpha\\Sprint 1", state="Removed", state_category="Removed"),
        rev(3, 1, PRE, iteration="Alpha\\Sprint 1"),
    ])
    items = items_of(sd)
    assert sorted(items.index) == [1, 3]
    assert not items.loc[1, "done"]


def _resolved_completed_revs():
    return [
        rev(1, 1, PLAN, iteration="Alpha\\Sprint 1"),
        rev(1, 2, MID, iteration="Alpha\\Sprint 1", state="Resolved", state_category="Resolved"),
        rev(1, 3, POST, iteration="Alpha\\Sprint 1", state="Closed", state_category="Completed"),
        rev(2, 1, PLAN, iteration="Alpha\\Sprint 1"),
        rev(2, 2, MID, iteration="Alpha\\Sprint 1", state="Closed", state_category="Completed"),
        rev(3, 1, PRE, iteration="Alpha\\Sprint 1", state="Resolved", state_category="Resolved"),
    ]


def test_resolved_is_not_done_under_default_done_set(tmp_path):
    sd, _ = build(tmp_path, _resolved_completed_revs())
    items = items_of(sd)
    assert sorted(items.index) == [1, 2, 3]
    assert items["done"].to_dict() == {1: False, 2: True, 3: False}
    assert sd.report["end_state_mix"] == {"Resolved": 2, "Completed": 1}


def test_resolved_counts_as_done_when_widened(tmp_path):
    sd, cache = build(tmp_path, _resolved_completed_revs(), done=("Resolved", "Completed"))
    items = items_of(sd)
    assert sorted(items.index) == [1, 2]
    assert items["done"].all()
    report = data_report(cache, sd)
    assert report["resolved_share_of_resolved_or_completed"] == pytest.approx(0.5)


def test_carryover_counts_distinct_earlier_ended_iterations(tmp_path):
    sd, _ = build(tmp_path, [
        rev(1, 1, "2024-02-01T00:00:00.000Z", iteration="Alpha\\Sprint 0"),
        rev(1, 2, "2024-02-20T00:00:00.000Z", iteration="Alpha\\Someday"),
        rev(1, 3, "2024-02-21T00:00:00.000Z", iteration="Alpha\\Sprint 0"),
        rev(1, 4, PLAN, iteration="Alpha\\Sprint 1"),
        rev(2, 1, PLAN, iteration="Alpha\\Sprint 1"),
        rev(3, 1, PLAN, iteration="Alpha\\Sprint 1"),
        rev(3, 2, MID, iteration="Alpha\\Sprint 0"),
    ])
    items = items_of(sd)
    assert items["carryover_count"].to_dict() == {1: 1, 2: 0, 3: 0}


def test_bounce_before_cutoff_counts_once(tmp_path):
    sd, _ = build(tmp_path, [
        rev(1, 1, PRE, iteration="Alpha\\Sprint 1"),
        rev(1, 2, "2024-03-02T00:00:00.000Z", iteration="Alpha\\Someday"),
        rev(1, 3, PLAN, iteration="Alpha\\Sprint 1"),
        rev(1, 4, MID, iteration="Alpha\\Sprint 1", state="Closed", state_category="Completed"),
    ])
    assert list(items_of(sd).index) == [1]
    assert items_of(sd).loc[1, "done"]
    assert sd.report["added_mid_sprint"] == 0


def test_match_team_rules():
    areas = {"Team Red": ["Alpha\\Red"], "Team Web": ["Alpha\\Red\\Web"], "Team Blue": ["Alpha\\Blue"]}
    three = ["Team Red", "Team Web", "Team Blue"]
    assert match_team("Alpha\\Red", three, areas) == "Team Red"
    assert match_team("alpha\\RED", three, areas) == "Team Red"
    assert match_team("Alpha\\Red\\Web\\UI", three, areas) == "Team Web"
    assert match_team("Alpha\\Red\\Api", three, areas) == "Team Red"
    assert match_team("Alpha\\Redder", three, areas) is None
    assert match_team("Alpha", three, areas) is None
    assert match_team(None, three, areas) is None
    assert match_team("Alpha\\Other", ["Team Red"], areas) == "Team Red"
    tie = {"Team Red": ["Alpha\\Red"], "Team Blue": ["Alpha\\Red"]}
    assert match_team("Alpha\\Red", ["Team Red", "Team Blue"], tie) is None


def test_team_assignment_exact_prefix_single_and_unassigned(tmp_path):
    teams = (
        team("Team Red", ["Alpha\\Red"], ["Alpha\\Sprint 1", "Alpha\\Sprint 2"]),
        team("Team Web", ["Alpha\\Red\\Web"], ["Alpha\\Sprint 1"]),
    )
    sd, _ = build(tmp_path, [
        rev(1, 1, PLAN, iteration="Alpha\\Sprint 1", area="Alpha\\Red"),
        rev(2, 1, PLAN, iteration="Alpha\\Sprint 1", area="Alpha\\Red\\Web\\UI"),
        rev(3, 1, PLAN, iteration="Alpha\\Sprint 1", area="Alpha"),
        rev(4, 1, "2024-03-18T12:00:00.000Z", iteration="Alpha\\Sprint 2", area="Alpha"),
    ], teams=teams)
    by_item = sd.items.set_index("item_id")["team"].to_dict()
    assert by_item == {1: "Team Red", 2: "Team Web", 4: "Team Red"}
    assert sd.report["unassigned_items"] == 1


def test_project_level_fallback_when_no_team_subscribes(tmp_path):
    beta_it = iteration("Beta\\Sprint 1", "2024-03-04T05:00:00.000Z", "2024-03-18T04:59:59.999Z", project="Beta")
    sd, cache = build(
        tmp_path,
        [
            rev(1, 1, PLAN, iteration="Alpha\\Sprint 1"),
            rev(9, 1, PLAN, project="Beta", iteration="Beta\\Sprint 1", area="Beta\\Anything"),
        ],
        iterations=(S1, beta_it),
        teams=(RED, team("Team Green", ["Beta\\Green"], [], project="Beta")),
    )
    beta = sd.sprints[sd.sprints["project"] == "Beta"]
    assert list(beta["team"]) == ["Beta"]
    assert list(beta["team_key"]) == ["Beta/Beta"]
    assert sd.report["fallback_projects"] == ["Beta"]
    assert set(sd.sprints["team"]) == {"Team Red", "Beta"}
    report = data_report(cache, sd)
    by_project = {p["project"]: p for p in report["projects"]}
    assert by_project["Beta"]["fallback"] is True
    assert by_project["Beta"]["teams_running_sprints"] == 0
    assert by_project["Alpha"]["teams_running_sprints"] == 1


def test_undated_iterations_never_become_sprints(tmp_path):
    cache = build_cache(tmp_path, [], [S1, UNDATED], [team("Team Red", ["Alpha\\Red"], ["Alpha\\Sprint 1", "Alpha\\Someday"])])
    cal = sprint_calendar(cache)
    assert list(cal["iteration"]) == ["Alpha\\Sprint 1"]
    assert cal.loc[0, "cutoff"] == pd.Timestamp("2024-03-05T05:00:00Z")


def test_half_dated_and_inverted_iterations_are_not_sprints(tmp_path):
    odd = [
        iteration("Alpha\\Start Only", "2024-03-04T05:00:00.000Z", None),
        iteration("Alpha\\End Only", None, "2024-03-18T04:59:59.999Z"),
        iteration("Alpha\\Inverted", "2024-03-18T05:00:00.000Z", "2024-03-04T05:00:00.000Z"),
        iteration("Alpha\\Zero", "2024-03-04T05:00:00.000Z", "2024-03-04T05:00:00.000Z"),
    ]
    subs = [i["path"] for i in odd] + ["Alpha\\Sprint 1"]
    sd, _ = build(
        tmp_path,
        [rev(1, 1, PLAN, iteration="Alpha\\Sprint 1"), rev(2, 1, PLAN, iteration="Alpha\\Inverted")],
        iterations=[S1, *odd],
        teams=(team("Team Red", ["Alpha\\Red"], subs),),
    )
    assert list(sd.sprints["iteration"]) == ["Alpha\\Sprint 1"]
    assert list(sd.items["item_id"]) == [1]


def test_points_fallback_imputation_and_pct_done(tmp_path):
    sd, _ = build(tmp_path, [
        # Sprint 0: two estimated items (3 and 5 points), one done
        rev(10, 1, "2024-02-19T12:00:00.000Z", iteration="Alpha\\Sprint 0", story_points=3.0),
        rev(10, 2, "2024-02-25T00:00:00.000Z", iteration="Alpha\\Sprint 0", story_points=3.0,
            state="Closed", state_category="Completed"),
        rev(11, 1, "2024-02-19T12:00:00.000Z", iteration="Alpha\\Sprint 0", story_points=5.0),
        rev(12, 1, "2024-02-19T12:00:00.000Z", iteration="Alpha\\Sprint 0", story_points=None),
        # Sprint 1: effort fallback, unestimated (imputed from Sprint 0 median = 4), zero points = missing
        rev(1, 1, PLAN, iteration="Alpha\\Sprint 1", story_points=None, effort=8.0),
        rev(1, 2, MID, iteration="Alpha\\Sprint 1", story_points=None, effort=8.0, state="Closed", state_category="Completed"),
        rev(2, 1, PLAN, iteration="Alpha\\Sprint 1", story_points=None, effort=None),
        rev(3, 1, PLAN, iteration="Alpha\\Sprint 1", story_points=0.0),
    ])
    s0 = items_of(sd, "Alpha/Team Red|Alpha\\Sprint 0")
    assert s0.loc[12, "points"] == 1.0  # no prior history anywhere -> 1
    s1 = items_of(sd)
    assert s1.loc[1, "points"] == 8.0 and not s1.loc[1, "is_unestimated"]
    assert s1.loc[2, "points"] == 4.0 and s1.loc[2, "is_unestimated"]
    assert s1.loc[3, "points"] == 4.0 and s1.loc[3, "is_unestimated"]
    row = sd.sprints.set_index("sprint_id").loc[SPRINT1]
    assert row["committed_points"] == 16.0
    assert row["done_points"] == 8.0
    assert row["pct_done"] == pytest.approx(0.5)
    assert row["pct_done_count"] == pytest.approx(1 / 3)
    assert row["n_unestimated"] == 2


def test_empty_sprints_dropped_and_counted(tmp_path):
    sd, _ = build(tmp_path, [rev(1, 1, PLAN, iteration="Alpha\\Sprint 1")])
    assert list(sd.sprints["sprint_id"]) == [SPRINT1]
    assert sd.report["dropped_empty_sprints"] == 2


def test_build_on_empty_cache(tmp_path):
    sd, _ = build(tmp_path, [])
    assert sd.sprints.empty and sd.items.empty
    assert math.isnan(sd.report["unestimated_share"])


def test_scope_at_uses_given_cutoff_and_history(tmp_path):
    revs = [
        rev(10, 1, "2024-02-19T12:00:00.000Z", iteration="Alpha\\Sprint 0", story_points=6.0),
        rev(1, 1, PLAN, iteration="Alpha\\Sprint 1", story_points=None),
        rev(2, 1, MID, iteration="Alpha\\Sprint 1"),
    ]
    sd, cache = build(tmp_path, revs)
    sprints, items = scope_at(cache, sd, iteration="Alpha\\Sprint 1", cutoff=pd.Timestamp(MID) + pd.Timedelta(hours=1),
                              work_item_types=TYPES)
    assert sorted(items["item_id"]) == [1, 2]
    assert items.set_index("item_id").loc[1, "points"] == 6.0
    assert items["done"].isna().all()
    assert sprints.loc[0, "cutoff"] == pd.Timestamp(MID) + pd.Timedelta(hours=1)
    assert math.isnan(sprints.loc[0, "pct_done"])
    with pytest.raises(ValueError, match="no team runs"):
        scope_at(cache, sd, iteration="Alpha\\Nope", cutoff=MID, work_item_types=TYPES)
    with pytest.raises(ValueError, match="Team Red"):
        scope_at(cache, sd, iteration="Alpha\\Sprint 1", cutoff=MID, work_item_types=TYPES, team="Team Pink")


def test_synthetic_cache_reconstructs(tmp_path):
    from sprint_forecast.cache import connect, load_cache
    from sprint_forecast.synth import generate

    conn = connect(generate(tmp_path / "cache.db", seed=1, n_sprints=8))
    cache = load_cache(conn)
    conn.close()
    sd = build_sprints(cache, work_item_types=["User Story", "Product Backlog Item", "Bug"])
    assert set(sd.sprints["team"]) == {"Team Red", "Team Blue", "Team Green"}
    assert sd.sprints.groupby("team").size().min() >= 7
    assert sd.report["added_mid_sprint"] > 0
    assert sd.report["unassigned_items"] > 0
    assert 0.3 < sd.sprints["pct_done"].mean() < 0.95
    assert sd.items["carryover_count"].max() >= 1
    assert sd.items["is_unestimated"].any()
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_sprints.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'sprint_forecast.sprints'`.

- [ ] **Step 3: Implement**

`src/sprint_forecast/sprints.py`:
```python
"""Sprint reconstruction: (project, team, iteration) -> committed scope at the cutoff and outcome at the end."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sprint_forecast.cache import CacheData
from sprint_forecast.timeline import as_of_many, to_utc

REMOVED = "Removed"

CALENDAR_COLUMNS = ["sprint_id", "project", "team", "team_key", "iteration", "start", "end", "cutoff"]
SPRINT_COLUMNS = CALENDAR_COLUMNS + [
    "n_items", "committed_points", "done_points", "pct_done", "pct_done_count", "n_unestimated", "n_added_mid",
]
ITEM_COLUMNS = CALENDAR_COLUMNS + [
    "item_id", "type", "state_category_at_commit", "area", "assigned_to_sk", "parent_id", "created",
    "last_changed", "revisions_so_far", "carryover_count", "raw_points", "points", "is_unestimated",
    "done", "state_category_at_end",
]


@dataclass
class SprintData:
    sprints: pd.DataFrame
    items: pd.DataFrame
    report: dict = field(default_factory=dict)


def raw_points(frame: pd.DataFrame) -> pd.Series:
    """Story points, falling back to effort; non-positive values count as missing."""
    sp = frame["story_points"].where(frame["story_points"] > 0)
    ef = frame["effort"].where(frame["effort"] > 0)
    return sp.fillna(ef).astype("float64")


def match_team(area: str | None, candidates: list[str], team_areas: dict[str, list[str]]) -> str | None:
    """Exact area match, else longest area-path prefix, else the only candidate; ties -> None."""
    target = (area if isinstance(area, str) else "").casefold()
    best_len, best = -1, []
    for team in candidates:
        for team_area in team_areas.get(team, []):
            ta = team_area.casefold()
            if target == ta or target.startswith(ta + "\\"):
                if len(ta) > best_len:
                    best_len, best = len(ta), [team]
                elif len(ta) == best_len and team not in best:
                    best.append(team)
    if len(best) == 1:
        return best[0]
    if not best and len(candidates) == 1:
        return candidates[0]
    return None


def sprint_calendar(cache: CacheData, commit_grace_days: float = 1.0) -> pd.DataFrame:
    """One row per (project, team, dated iteration); project-level rows when no team in a project subscribes."""
    its = cache.iterations
    ok = its["start_date"].notna() & its["end_date"].notna() & (its["end_date"] > its["start_date"])
    dated = its.loc[ok, ["project", "path", "start_date", "end_date"]]
    subs = cache.teams.rename(columns={"name": "team"}).merge(cache.team_iterations, on="team_sk")
    team_rows = subs.merge(dated, left_on=["project", "iteration_path"], right_on=["project", "path"])
    fallback = sorted(set(dated["project"]) - set(team_rows["project"]))
    team_rows = team_rows.assign(is_fallback=False)
    proj_rows = dated[dated["project"].isin(fallback)].assign(team=lambda d: d["project"], is_fallback=True)
    cols = ["project", "team", "path", "start_date", "end_date", "is_fallback"]
    cal = pd.concat([team_rows[cols], proj_rows[cols]], ignore_index=True)
    cal = cal.rename(columns={"path": "iteration", "start_date": "start", "end_date": "end"})
    cal["is_fallback"] = cal["is_fallback"].astype(bool)
    cal["team_key"] = cal["project"] + "/" + cal["team"]
    cal["sprint_id"] = cal["team_key"] + "|" + cal["iteration"]
    cutoff = cal["start"] + pd.Timedelta(days=commit_grace_days)
    cal["cutoff"] = cutoff.where(cutoff < cal["end"], cal["end"])
    cal = cal.drop_duplicates("sprint_id").sort_values(["start", "sprint_id"], kind="mergesort")
    return cal[CALENDAR_COLUMNS + ["is_fallback"]].reset_index(drop=True)


class _Assigner:
    """Assign (area, iteration, project) rows to one of the teams running that iteration."""

    def __init__(self, cache: CacheData, cal: pd.DataFrame):
        self.candidates = cal.groupby("iteration")["team"].apply(list).to_dict()
        self.fallback = set(cal.loc[cal["is_fallback"], "project"])
        merged = cache.teams.merge(cache.team_areas, on="team_sk")
        self.areas: dict[str, dict[str, list[str]]] = {}
        for row in merged.itertuples(index=False):
            self.areas.setdefault(row.project, {}).setdefault(row.name, []).append(row.area_path)

    def __call__(self, areas: pd.Series, iterations: pd.Series, projects: pd.Series) -> pd.Series:
        teams = [
            match_team(a, self.candidates.get(it, []), {} if p in self.fallback else self.areas.get(p, {}))
            for a, it, p in zip(areas, iterations, projects)
        ]
        return pd.Series(teams, index=areas.index, dtype=object)


def _iteration_end_map(cache: CacheData) -> pd.Series:
    its = cache.iterations.dropna(subset=["end_date"]).drop_duplicates("path")
    return its.set_index("path")["end_date"]


def _pairs(revisions: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """(item_id, target) for every item ever in a window's iteration, joined to the window's dates."""
    seen = revisions.loc[revisions["iteration"].isin(windows["target"]), ["item_id", "iteration"]]
    seen = seen.drop_duplicates().rename(columns={"iteration": "target"})
    return seen.merge(windows, on="target").reset_index(drop=True)


def _carryover(revisions: pd.DataFrame, rows: pd.DataFrame, iteration_end: pd.Series) -> np.ndarray:
    """Distinct earlier iterations each row's item was in (revisions <= cutoff) whose iteration ended before start."""
    left = rows[["item_id", "cutoff", "start"]].reset_index(drop=True)
    left["_row"] = np.arange(len(left))
    hist = left.merge(revisions[["item_id", "changed", "iteration"]], on="item_id")
    hist = hist[hist["changed"] <= hist["cutoff"]]
    hist = hist[hist["iteration"].map(iteration_end) < hist["start"]]
    counts = hist.groupby("_row")["iteration"].nunique()
    return left["_row"].map(counts).fillna(0).astype("int64").to_numpy()


def _committed(
    cache: CacheData,
    pairs: pd.DataFrame,
    at_c: pd.DataFrame,
    assign: _Assigner,
    types: set[str],
    excluded: set[str],
) -> tuple[pd.DataFrame, int]:
    """Committed rows (index aligned with `pairs`) with team assigned; also returns the unassigned count."""
    mask = (
        (at_c["iteration"].to_numpy() == pairs["target"].to_numpy())
        & at_c["type"].isin(types).to_numpy()
        & ~at_c["state_category"].isin(excluded).to_numpy()
    )
    st = at_c.loc[mask]
    c = pairs.loc[mask, ["item_id", "target", "project", "start", "end", "cutoff"]].copy()
    c["type"] = st["type"].to_numpy()
    c["state_category_at_commit"] = st["state_category"].to_numpy()
    c["area"] = st["area"].to_numpy()
    c["assigned_to_sk"] = st["assigned_to_sk"].to_numpy()
    c["parent_id"] = st["parent_id"].to_numpy()
    c["created"] = st["created"].to_numpy()
    c["last_changed"] = st["changed"].to_numpy()
    c["revisions_so_far"] = st["revisions_so_far"].astype("int64").to_numpy()
    c["raw_points"] = raw_points(st).to_numpy()
    c["team"] = assign(c["area"], c["target"], c["project"])
    n_unassigned = int(c["team"].isna().sum())
    c = c[c["team"].notna()].rename(columns={"target": "iteration"})
    c["team_key"] = c["project"] + "/" + c["team"]
    c["sprint_id"] = c["team_key"] + "|" + c["iteration"]
    c["carryover_count"] = _carryover(cache.revisions, c, _iteration_end_map(cache))
    return c, n_unassigned


def impute_points(items: pd.DataFrame, history_items: pd.DataFrame) -> pd.DataFrame:
    """Fill missing raw_points with the team's prior median item size, then the global prior median, then 1."""
    items = items.copy()
    items["is_unestimated"] = items["raw_points"].isna()
    items["points"] = items["raw_points"].astype("float64")
    known = history_items[history_items["raw_points"].notna()]
    need = items[items["is_unestimated"]]
    for (team_key, start), idx in need.groupby(["team_key", "start"]).groups.items():
        prior = known[known["end"] < start]
        team_prior = prior[prior["team_key"] == team_key]
        if not team_prior.empty:
            fill = float(team_prior["raw_points"].median())
        elif not prior.empty:
            fill = float(prior["raw_points"].median())
        else:
            fill = 1.0
        items.loc[idx, "points"] = fill
    return items


def summarize_sprints(items: pd.DataFrame, n_added: dict[str, int] | None = None) -> pd.DataFrame:
    if items.empty:
        return pd.DataFrame(columns=SPRINT_COLUMNS)
    done = items["done"].astype("float64")
    work = items.assign(_done=done, _done_pts=items["points"] * done, _unest=items["is_unestimated"].astype("int64"))
    g = work.groupby("sprint_id", sort=False)
    out = g[CALENDAR_COLUMNS[1:]].first()
    out["n_items"] = g.size()
    out["committed_points"] = g["points"].sum()
    out["done_points"] = g["_done_pts"].sum(min_count=1)
    out["pct_done"] = out["done_points"] / out["committed_points"].where(out["committed_points"] > 0)
    out["pct_done_count"] = g["_done"].mean()
    out["n_unestimated"] = g["_unest"].sum()
    out = out.reset_index()
    out["n_added_mid"] = out["sprint_id"].map(n_added or {}).fillna(0).astype("int64")
    return out[SPRINT_COLUMNS].sort_values(["start", "sprint_id"], kind="mergesort").reset_index(drop=True)


def _finish_items(c: pd.DataFrame, history_items: pd.DataFrame | None) -> pd.DataFrame:
    items = impute_points(c, c if history_items is None else history_items)
    items = items[ITEM_COLUMNS].sort_values(["start", "sprint_id", "item_id"], kind="mergesort")
    return items.reset_index(drop=True)


def build_sprints(
    cache: CacheData,
    *,
    work_item_types: list[str],
    done_categories: list[str] = ("Completed",),
    commit_grace_days: float = 1.0,
) -> SprintData:
    revs = cache.revisions
    cal = sprint_calendar(cache, commit_grace_days)
    types, done = set(work_item_types), set(done_categories)
    assign = _Assigner(cache, cal)
    windows = cal.drop_duplicates("iteration")[["project", "iteration", "start", "end", "cutoff"]]
    pairs = _pairs(revs, windows.rename(columns={"iteration": "target"}))
    at_c = as_of_many(revs, pairs[["item_id", "cutoff"]], "cutoff")
    at_e = as_of_many(revs, pairs[["item_id", "end"]], "end")
    c, n_unassigned = _committed(cache, pairs, at_c, assign, types, done | {REMOVED})
    e = at_e.loc[c.index]
    c["done"] = (e["iteration"].to_numpy() == c["iteration"].to_numpy()) & e["state_category"].isin(done).to_numpy()
    c["state_category_at_end"] = e["state_category"].to_numpy()

    target = pairs["target"].to_numpy()
    added_mask = (
        (at_c["iteration"].to_numpy() != target)
        & (at_e["iteration"].to_numpy() == target)
        & at_e["type"].isin(types).to_numpy()
        & (at_e["state_category"] != REMOVED).to_numpy()
    )
    added = pairs.loc[added_mask]
    added_team = assign(at_e.loc[added_mask, "area"], added["target"], added["project"])
    added_ids = (added["project"] + "/" + added_team + "|" + added["target"])[added_team.notna()]

    items = _finish_items(c, None) if len(c) else pd.DataFrame(columns=ITEM_COLUMNS)
    sprints = summarize_sprints(items, added_ids.value_counts().to_dict())
    report = {
        "unassigned_items": n_unassigned,
        "added_mid_sprint": int(added_mask.sum()),
        "dropped_empty_sprints": int(len(set(cal["sprint_id"]) - set(sprints["sprint_id"]))),
        "fallback_projects": sorted(set(cal.loc[cal["is_fallback"], "project"])),
        "n_sprints": len(sprints),
        "n_committed_items": len(items),
        "unestimated_share": float(items["is_unestimated"].mean()) if len(items) else float("nan"),
        "end_state_mix": items["state_category_at_end"].fillna("(missing)").value_counts().to_dict(),
    }
    return SprintData(sprints, items, report)


def scope_at(
    cache: CacheData,
    history: SprintData,
    *,
    iteration: str,
    cutoff,
    work_item_types: list[str],
    done_categories: list[str] = ("Completed",),
    commit_grace_days: float = 1.0,
    team: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Committed-style scope of one iteration as of `cutoff`, outcome unknown. Returns (sprints, items)."""
    cal = sprint_calendar(cache, commit_grace_days)
    group = cal[cal["iteration"] == iteration]
    if group.empty:
        raise ValueError(f"no team runs a dated iteration with path {iteration!r}")
    if team is not None and team not in set(group["team"]):
        raise ValueError(f"team {team!r} does not run {iteration!r}; teams: {', '.join(group['team'])}")
    windows = group.iloc[:1][["project", "iteration", "start", "end"]].rename(columns={"iteration": "target"})
    windows = windows.assign(cutoff=to_utc(cutoff))
    pairs = _pairs(cache.revisions, windows)
    at_c = as_of_many(cache.revisions, pairs[["item_id", "cutoff"]], "cutoff")
    c, _ = _committed(cache, pairs, at_c, _Assigner(cache, cal), set(work_item_types), set(done_categories) | {REMOVED})
    if team is not None:
        c = c[c["team"] == team]
    if c.empty:
        return pd.DataFrame(columns=SPRINT_COLUMNS), pd.DataFrame(columns=ITEM_COLUMNS)
    c = c.assign(done=np.nan, state_category_at_end=None)
    items = _finish_items(c, history.items)
    return summarize_sprints(items), items


def data_report(cache: CacheData, sd: SprintData) -> dict:
    its = cache.iterations
    dated = its["start_date"].notna() & its["end_date"].notna()
    running = set(zip(sd.sprints["project"], sd.sprints["team"]))
    projects = sorted(set(its["project"]) | set(cache.teams["project"]) | set(cache.revisions["project"]))
    per_project = []
    for p in projects:
        team_names = sorted(cache.teams.loc[cache.teams["project"] == p, "name"])
        per_project.append({
            "project": p,
            "teams": len(team_names),
            "teams_running_sprints": sum((p, t) in running for t in team_names),
            "dated_iterations": int((dated & (its["project"] == p)).sum()),
            "undated_iterations": int((~dated & (its["project"] == p)).sum()),
            "sprints": int((sd.sprints["project"] == p).sum()),
            "committed_items": int((sd.items["project"] == p).sum()),
            "fallback": p in sd.report.get("fallback_projects", []),
        })
    mix = sd.report.get("end_state_mix", {})
    ended = mix.get("Resolved", 0) + mix.get("Completed", 0)
    return {
        **sd.report,
        "projects": per_project,
        "resolved_share_of_resolved_or_completed": (mix.get("Resolved", 0) / ended) if ended else float("nan"),
    }
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_sprints.py -v`
Expected: `17 passed` in under 10 s. This includes the Review Focus test `test_half_dated_and_inverted_iterations_are_not_sprints`.

- [ ] **Step 5: Commit**

```bash
git add src/sprint_forecast/sprints.py tests/test_sprints.py
git commit -m "feat: sprint reconstruction, team assignment, committed scope and outcomes"
```

---

### Task 9: Features (`features.py`)

**Files:**
- Create: `src/sprint_forecast/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes:
  - Task 8: `SprintData`, `build_sprints`.
  - Tests also use Task 4's `connect` and `load_cache`, and Task 5's `generate`.
- Produces:
  - Feature lists:
    - `CATEGORICAL = ["type", "state_category_at_commit"]`
    - `ITEM_FEATURES = ["type","state_category_at_commit","points","points_rel","is_unestimated","carryover_count","age_days","days_since_change","revisions_so_far","has_parent"]`
    - `SPRINT_FEATURES = ["load_ratio","n_items","bug_share","carryover_share","unestimated_share","team_trailing_completion","sprint_length_days","team_sprint_index"]`
    - `ASSIGNEE_FEATURES = ["assignee_load_ratio","is_unassigned"]`
    - `FEATURES = ITEM_FEATURES + SPRINT_FEATURES + ASSIGNEE_FEATURES`
    - `NUMERIC` = `FEATURES` without `CATEGORICAL`
  - Frame columns:
    - `ID_COLUMNS = ["sprint_id","project","team","team_key","iteration","start","end","cutoff","item_id"]`
    - `FEATURE_FRAME_COLUMNS = ID_COLUMNS + FEATURES + ["y"]`. `y` is float: 1.0, 0.0, or NaN when the outcome is unknown. Booleans are encoded as 0.0/1.0.
  - Window sizes: `VELOCITY_WINDOW = 3`, `COMPLETION_WINDOW = 5`, `ASSIGNEE_WINDOW = 3`.
  - `team_history(target_sprints, history_sprints) -> pd.DataFrame`:
    - Returns columns `sprint_id, trailing_velocity, team_trailing_completion, team_sprint_index`.
    - Uses only the team's sprints with `end < target start`, and only those whose `done_points` is known.
    - `target_sprints` needs `sprint_id, team_key, start`. `history_sprints` needs `team_key, end, done_points, pct_done`.
  - `assignee_load(target_items, history_items) -> pd.Series`, aligned to `target_items.index`.
  - `build_features_for(target_sprints, target_items, history_sprints, history_items) -> pd.DataFrame` and `build_features(sd: SprintData) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test**

`tests/test_features.py`:
```python
import dataclasses
import math

import pandas as pd
import pytest

from sprint_forecast.cache import connect, load_cache
from sprint_forecast.features import (
    FEATURE_FRAME_COLUMNS,
    FEATURES,
    ID_COLUMNS,
    assignee_load,
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
    cols = ID_COLUMNS + FEATURES
    b = before[before["sprint_id"] == target["sprint_id"]][cols].reset_index(drop=True)
    a = after[after["sprint_id"] == target["sprint_id"]][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(b, a)
    # control: the same edits one minute before the cutoff do change the target sprint
    early = late.assign(changed=target["cutoff"] - pd.Timedelta(minutes=1))
    _, moved = _features(dataclasses.replace(synth_cache, revisions=pd.concat([revs, early], ignore_index=True)))
    assert (moved["sprint_id"] == target["sprint_id"]).sum() < len(b)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_features.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'sprint_forecast.features'`.

- [ ] **Step 3: Implement**

`src/sprint_forecast/features.py`:
```python
"""Feature matrix at the commit cutoff. Team history uses only sprints with end < this sprint's start."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sprint_forecast.sprints import SprintData

CATEGORICAL = ["type", "state_category_at_commit"]
ITEM_FEATURES = [
    "type", "state_category_at_commit", "points", "points_rel", "is_unestimated", "carryover_count",
    "age_days", "days_since_change", "revisions_so_far", "has_parent",
]
SPRINT_FEATURES = [
    "load_ratio", "n_items", "bug_share", "carryover_share", "unestimated_share",
    "team_trailing_completion", "sprint_length_days", "team_sprint_index",
]
ASSIGNEE_FEATURES = ["assignee_load_ratio", "is_unassigned"]
FEATURES = ITEM_FEATURES + SPRINT_FEATURES + ASSIGNEE_FEATURES
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
ID_COLUMNS = ["sprint_id", "project", "team", "team_key", "iteration", "start", "end", "cutoff", "item_id"]
FEATURE_FRAME_COLUMNS = ID_COLUMNS + FEATURES + ["y"]

VELOCITY_WINDOW = 3
COMPLETION_WINDOW = 5
ASSIGNEE_WINDOW = 3
DAY = pd.Timedelta(days=1)


def _ns(series: pd.Series) -> np.ndarray:
    return series.dt.tz_convert("UTC").dt.as_unit("ns").astype("int64").to_numpy()


def _ts_ns(ts) -> int:
    return pd.Timestamp(ts).tz_convert("UTC").as_unit("ns").value


def _trailing_mean(values: np.ndarray, k: int, window: int) -> float:
    return float(values[max(0, k - window):k].mean()) if k > 0 else float("nan")


def team_history(target_sprints: pd.DataFrame, history_sprints: pd.DataFrame) -> pd.DataFrame:
    """Per target sprint: trailing velocity (mean delivered points of last 3), trailing completion (last 5),
    and the number of earlier team sprints, all from team sprints with end < target start."""
    hist = history_sprints.dropna(subset=["done_points"]).sort_values("end", kind="mergesort")
    by_team = {
        tk: (_ns(g["end"]), g["done_points"].to_numpy(float), g["pct_done"].to_numpy(float))
        for tk, g in hist.groupby("team_key")
    }
    rows = []
    for sprint_id, team_key, start in zip(target_sprints["sprint_id"], target_sprints["team_key"], target_sprints["start"]):
        ends, dp, pct = by_team.get(team_key, (np.array([], dtype="int64"), np.array([]), np.array([])))
        k = int(np.searchsorted(ends, _ts_ns(start), side="left"))
        rows.append({
            "sprint_id": sprint_id,
            "trailing_velocity": _trailing_mean(dp, k, VELOCITY_WINDOW),
            "team_trailing_completion": _trailing_mean(pct, k, COMPLETION_WINDOW),
            "team_sprint_index": k,
        })
    return pd.DataFrame(rows, columns=["sprint_id", "trailing_velocity", "team_trailing_completion", "team_sprint_index"])


def assignee_load(target_items: pd.DataFrame, history_items: pd.DataFrame) -> pd.Series:
    """Per target item: its assignee's committed points this sprint / their mean delivered points over the
    last 3 earlier sprints (end < start) they appear in. NaN when unassigned or no history."""
    h = history_items[history_items["assigned_to_sk"].notna()]
    h = h.assign(_delivered=h["points"] * h["done"].astype("float64"))
    per_sprint = h.groupby(["assigned_to_sk", "sprint_id"], as_index=False).agg(
        end=("end", "first"), delivered=("_delivered", "sum")
    ).sort_values("end", kind="mergesort")
    by_person = {a: (_ns(g["end"]), g["delivered"].to_numpy(float)) for a, g in per_sprint.groupby("assigned_to_sk")}
    t = target_items[target_items["assigned_to_sk"].notna()]
    committed = t.groupby(["sprint_id", "assigned_to_sk"])["points"].sum()
    starts = t.groupby("sprint_id")["start"].first()
    ratio = {}
    for (sprint_id, person), pts in committed.items():
        ends, delivered = by_person.get(person, (np.array([], dtype="int64"), np.array([])))
        k = int(np.searchsorted(ends, _ts_ns(starts[sprint_id]), side="left"))
        mean = _trailing_mean(delivered, k, ASSIGNEE_WINDOW)
        ratio[(sprint_id, person)] = pts / mean if mean > 0 else float("nan")
    keys = list(zip(target_items["sprint_id"], target_items["assigned_to_sk"]))
    return pd.Series([ratio.get(k, float("nan")) for k in keys], index=target_items.index, dtype="float64")


def build_features_for(
    target_sprints: pd.DataFrame,
    target_items: pd.DataFrame,
    history_sprints: pd.DataFrame,
    history_items: pd.DataFrame,
) -> pd.DataFrame:
    items = target_items.reset_index(drop=True)
    if items.empty:
        return pd.DataFrame(columns=FEATURE_FRAME_COLUMNS)
    hist = team_history(target_sprints, history_sprints).set_index("sprint_id")
    agg = items.assign(
        _bug=(items["type"] == "Bug").astype(float),
        _carry=(items["carryover_count"] > 0).astype(float),
        _unest=items["is_unestimated"].astype(float),
    ).groupby("sprint_id").agg(
        n_items=("item_id", "size"), committed_points=("points", "sum"),
        bug_share=("_bug", "mean"), carryover_share=("_carry", "mean"), unestimated_share=("_unest", "mean"),
    )
    sid = items["sprint_id"]
    velocity = sid.map(hist["trailing_velocity"])
    velocity = velocity.where(velocity > 0)
    out = items[ID_COLUMNS].copy()
    out["type"] = items["type"].astype(str)
    out["state_category_at_commit"] = items["state_category_at_commit"].astype(str)
    out["points"] = items["points"].astype(float)
    out["points_rel"] = out["points"] / velocity
    out["is_unestimated"] = items["is_unestimated"].astype(float)
    out["carryover_count"] = items["carryover_count"].astype(float)
    out["age_days"] = (items["cutoff"] - items["created"]) / DAY
    out["days_since_change"] = (items["cutoff"] - items["last_changed"]) / DAY
    out["revisions_so_far"] = items["revisions_so_far"].astype(float)
    out["has_parent"] = items["parent_id"].notna().astype(float)
    out["load_ratio"] = sid.map(agg["committed_points"]) / velocity
    for col in ("n_items", "bug_share", "carryover_share", "unestimated_share"):
        out[col] = sid.map(agg[col]).astype(float)
    out["team_trailing_completion"] = sid.map(hist["team_trailing_completion"])
    out["sprint_length_days"] = (items["end"] - items["start"]) / DAY
    out["team_sprint_index"] = sid.map(hist["team_sprint_index"]).astype(float)
    out["assignee_load_ratio"] = assignee_load(items, history_items)
    out["is_unassigned"] = items["assigned_to_sk"].isna().astype(float)
    out["y"] = items["done"].astype("float64")
    return out[FEATURE_FRAME_COLUMNS]


def build_features(sd: SprintData) -> pd.DataFrame:
    return build_features_for(sd.sprints, sd.items, sd.sprints, sd.items)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_features.py -v`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/sprint_forecast/features.py tests/test_features.py
git commit -m "feat: leak-free item, sprint and assignee features at the commit cutoff"
```

---

### Task 10: Baseline C and team-mean reference (`baseline.py`)

**Files:**
- Create: `src/sprint_forecast/baseline.py`
- Test: `tests/test_baseline.py`

**Interfaces:**
- Consumes: the sprint frame shape from Task 8. It needs the columns `sprint_id, team_key, project, start, end, done_points, pct_done`.
- Produces:
  - Constants: `VELOCITY_DRAWS_FROM = 8`, `MIN_TEAM_SPRINTS = 3`, `PROJECT_WINDOW = 20`.
  - `velocity_bootstrap(history_sprints: pd.DataFrame, target: pd.Series, n_draws: int = 10_000, seed: int = 0) -> np.ndarray | None`:
    - `target` needs `team_key, project, start, committed_points`.
    - It returns `n_draws` samples of %done in [0, 1], or `None` when there is no history.
  - `team_mean_samples(team_trailing_completion: float) -> np.ndarray | None` returns a 1-element array, or `None` for NaN.

- [ ] **Step 1: Write the failing test**

`tests/test_baseline.py`:
```python
import numpy as np
import pandas as pd

from sprint_forecast.baseline import team_mean_samples, velocity_bootstrap

T0 = pd.Timestamp("2024-01-08T05:00:00Z")
DAY = pd.Timedelta(days=1)


def _sprints(team_key, project, done_points, pct, first=0):
    rows = []
    for k, (dp, pc) in enumerate(zip(done_points, pct), start=first):
        start = T0 + 14 * k * DAY
        rows.append({
            "sprint_id": f"{team_key}|S{k}", "team_key": team_key, "project": project,
            "start": start, "end": start + 14 * DAY - pd.Timedelta(milliseconds=1),
            "done_points": float(dp), "pct_done": float(pc),
        })
    return pd.DataFrame(rows, columns=["sprint_id", "team_key", "project", "start", "end", "done_points", "pct_done"])


def _target(k, team_key="Alpha/Team Red", project="Alpha", committed=40.0):
    return pd.Series({"team_key": team_key, "project": project, "start": T0 + 14 * k * DAY, "committed_points": committed})


def test_draws_from_last_eight_team_velocities_capped_at_one():
    hist = _sprints("Alpha/Team Red", "Alpha", [1, 2, 10, 20, 30, 40, 50, 60, 70, 80], [0.5] * 10)
    samples = velocity_bootstrap(hist, _target(10), n_draws=5000, seed=0)
    assert samples.shape == (5000,)
    assert set(np.round(samples, 6)) == {0.25, 0.5, 0.75, 1.0}
    assert np.isclose(samples, 0.25).any() and np.isclose(samples, 1.0).mean() > 0.4


def test_ignores_sprints_not_ended_before_start():
    hist = _sprints("Alpha/Team Red", "Alpha", [10, 10, 10, 999, 999], [0.5] * 5)
    samples = velocity_bootstrap(hist, _target(3), n_draws=1000, seed=0)
    assert np.allclose(samples, 0.25)


def test_falls_back_to_project_completion_with_few_team_sprints():
    red = _sprints("Alpha/Team Red", "Alpha", [10, 10], [0.9, 0.9], first=23)
    blue = _sprints("Alpha/Team Blue", "Alpha", [5] * 25, [0.1] * 5 + [0.6] * 20)
    other = _sprints("Beta/Team Green", "Beta", [5] * 25, [0.0] * 25)
    hist = pd.concat([red, blue, other], ignore_index=True)
    samples = velocity_bootstrap(hist, _target(25), n_draws=2000, seed=0)
    assert set(np.round(samples, 6)) <= {0.6, 0.9}  # last 20 Alpha sprints only; Blue's early 0.1 are too old
    assert (samples == 0.9).any()


def test_no_history_returns_none_and_is_deterministic():
    empty = _sprints("Alpha/Team Red", "Alpha", [], [])
    assert velocity_bootstrap(empty, _target(0), seed=0) is None
    hist = _sprints("Alpha/Team Red", "Alpha", [10, 20, 30, 40], [0.5] * 4)
    assert np.array_equal(velocity_bootstrap(hist, _target(4), seed=7), velocity_bootstrap(hist, _target(4), seed=7))


def test_team_mean_is_a_point_mass():
    assert team_mean_samples(0.72).tolist() == [0.72]
    assert team_mean_samples(float("nan")) is None
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_baseline.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'sprint_forecast.baseline'`.

- [ ] **Step 3: Implement**

`src/sprint_forecast/baseline.py`:
```python
"""Model C (velocity bootstrap) and the team-mean reference. Both return samples of %done, or None."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

VELOCITY_DRAWS_FROM = 8
MIN_TEAM_SPRINTS = 3
PROJECT_WINDOW = 20


def velocity_bootstrap(
    history_sprints: pd.DataFrame,
    target: pd.Series,
    n_draws: int = 10_000,
    seed: int = 0,
) -> np.ndarray | None:
    """%done = min(1, V / committed) with V drawn from the team's last 8 delivered velocities (end < start).
    With fewer than 3 prior team sprints, draw %done from the project's last 20 sprints instead.
    `target` needs team_key, project, start, committed_points. Returns None when there is no history."""
    prior = history_sprints[(history_sprints["end"] < target["start"]) & history_sprints["done_points"].notna()]
    prior = prior.sort_values(["end", "sprint_id"], kind="mergesort")
    rng = np.random.default_rng(seed)
    team = prior[prior["team_key"] == target["team_key"]].tail(VELOCITY_DRAWS_FROM)
    committed = float(target["committed_points"])
    if len(team) >= MIN_TEAM_SPRINTS and committed > 0:
        v = rng.choice(team["done_points"].to_numpy(float), size=n_draws, replace=True)
        return np.minimum(1.0, v / committed)
    ratios = prior[prior["project"] == target["project"]].tail(PROJECT_WINDOW)["pct_done"].dropna()
    if ratios.empty:
        return None
    return rng.choice(ratios.to_numpy(float), size=n_draws, replace=True)


def team_mean_samples(team_trailing_completion: float) -> np.ndarray | None:
    """Reference forecast: a point mass at the team's trailing mean %done (None without history)."""
    if team_trailing_completion is None or math.isnan(team_trailing_completion):
        return None
    return np.array([float(team_trailing_completion)])
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_baseline.py -v`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/sprint_forecast/baseline.py tests/test_baseline.py
git commit -m "feat: velocity-bootstrap baseline and team-mean reference"
```

---

### Task 11: Model A, the LR sanity model and SHAP drivers (`model.py`)

**Files:**
- Create: `src/sprint_forecast/model.py`
- Create: `tests/conftest.py` (session-scoped synthetic datasets, also used by Tasks 12 and 13)
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes:
  - Task 9: `CATEGORICAL`, `FEATURES`, `NUMERIC`, `build_features`.
  - `conftest.py` also uses Task 4 (`connect`, `load_cache`), Task 5 (`generate`) and Task 8 (`build_sprints`).
- Produces:
  - `LGBM_PARAMS`, as listed in Global Constraints.
  - Constants:
    - `CALIBRATION_SHARE = 0.2`
    - `ISOTONIC_MIN_ITEMS = 1000`
    - `P_CLIP = 1e-4`
    - `FEATURE_LABELS`: each feature mapped to plain words
    - `BOOLEAN_FEATURES`
  - `@dataclass ItemModel` has these fields:
    - `lgbm: LGBMClassifier`
    - `lr: Pipeline`
    - `calibrator: IsotonicRegression | LogisticRegression | None`
    - `calibration: str`, one of `"isotonic"`, `"platt"` or `"none"`
    - `categories: dict[str, list[str]]`
    - `calib_frame: pd.DataFrame`, with columns `sprint_id, p, y`: calibrated predictions for the calibration sprints
  - Data prep:
    - `split_by_time(frame, share=0.2) -> tuple[list[str], list[str]]` returns `(fit_ids, cal_ids)`, ordered by sprint start.
    - `to_matrix(frame, categories) -> pd.DataFrame`. Unknown category values become NaN.
  - Training and prediction:
    - `train_item_model(frame, seed: int = 0) -> ItemModel` raises `ValueError("...both...")` when `y` has one class.
    - `predict_proba(model, frame) -> np.ndarray` returns calibrated probabilities clipped to `[P_CLIP, 1-P_CLIP]`.
    - `predict_proba_lr(model, frame) -> np.ndarray`.
  - Explanations:
    - `contributions(model, frame) -> pd.DataFrame`, with columns `FEATURES + ["bias"]` on the log-odds scale.
    - `describe_drivers(contrib: pd.Series, values: pd.Series, k: int = 2) -> list[str]` returns strings like `"sprints already carried over = 2"`.
  - `tests/conftest.py`:
    - `Synth(cache, sprints: SprintData, frame)`.
    - Fixture `synth40`: seed 0, 40 sprints, about 1,700 items.
    - Fixture `synth16`: seed 1, 16 sprints.
    - `SYNTH_TYPES`.

- [ ] **Step 1: Write the shared fixtures and the failing test**

`tests/conftest.py`:
```python
"""Shared, session-scoped synthetic data so the expensive pipeline runs once per test session."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from sprint_forecast.cache import CacheData, connect, load_cache
from sprint_forecast.features import build_features
from sprint_forecast.sprints import SprintData, build_sprints
from sprint_forecast.synth import generate

SYNTH_TYPES = ["User Story", "Product Backlog Item", "Bug"]


@dataclass
class Synth:
    cache: CacheData
    sprints: SprintData
    frame: pd.DataFrame


def _synth(tmp_path_factory, seed: int, n_sprints: int) -> Synth:
    path = generate(tmp_path_factory.mktemp(f"synth{seed}_{n_sprints}") / "cache.db", seed=seed, n_sprints=n_sprints)
    conn = connect(path)
    cache = load_cache(conn)
    conn.close()
    sd = build_sprints(cache, work_item_types=SYNTH_TYPES)
    return Synth(cache, sd, build_features(sd))


@pytest.fixture(scope="session")
def synth40(tmp_path_factory) -> Synth:
    """Seed 0, 40 sprints per team: ~1700 committed items; used for model-quality tests."""
    return _synth(tmp_path_factory, seed=0, n_sprints=40)


@pytest.fixture(scope="session")
def synth16(tmp_path_factory) -> Synth:
    """Seed 1, 16 sprints per team: small and fast; used for backtest and plumbing tests."""
    return _synth(tmp_path_factory, seed=1, n_sprints=16)
```

`tests/test_model.py`:
```python
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from sprint_forecast import model as model_module
from sprint_forecast.features import FEATURES
from sprint_forecast.model import (
    P_CLIP,
    contributions,
    describe_drivers,
    predict_proba,
    predict_proba_lr,
    split_by_time,
    train_item_model,
)


def _time_split(frame):
    """Train on sprints that ended before the median sprint start; test on sprints starting at or after it."""
    split = frame.drop_duplicates("sprint_id")["start"].median()
    return frame[frame["end"] < split], frame[frame["start"] >= split]


@pytest.fixture(scope="module")
def trained(synth40):
    train, test = _time_split(synth40.frame)
    return train_item_model(train, seed=0), train, test


def test_item_auc_beats_065_on_synth(trained):
    model, _, test = trained
    assert roc_auc_score(test["y"], predict_proba(model, test)) > 0.65
    assert roc_auc_score(test["y"], predict_proba_lr(model, test)) > 0.6


def test_carryover_shap_is_negative_for_carried_items(trained):
    model, _, test = trained
    contrib = contributions(model, test)
    assert list(contrib.columns) == FEATURES + ["bias"]
    carried = test["carryover_count"] > 0
    assert contrib.loc[carried, "carryover_count"].mean() < 0
    assert contrib.loc[carried, "carryover_count"].mean() < contrib.loc[~carried, "carryover_count"].mean()


def test_calibration_uses_last_sprints_and_platt_when_small(trained):
    model, train, _ = trained
    assert model.calibration == "platt"
    fit_ids, cal_ids = split_by_time(train)
    assert len(cal_ids) == round(0.2 * (len(fit_ids) + len(cal_ids)))
    starts = train.drop_duplicates("sprint_id").set_index("sprint_id")["start"]
    assert starts[cal_ids].min() >= starts[fit_ids].max()
    assert set(model.calib_frame["sprint_id"]) == set(cal_ids)
    assert model.calib_frame["p"].between(P_CLIP, 1 - P_CLIP).all()


def test_isotonic_when_calibration_set_is_large(synth40, monkeypatch):
    monkeypatch.setattr(model_module, "ISOTONIC_MIN_ITEMS", 50)
    train, test = _time_split(synth40.frame)
    model = train_item_model(train, seed=0)
    assert model.calibration == "isotonic"
    p = predict_proba(model, test)
    assert np.all((p >= P_CLIP) & (p <= 1 - P_CLIP))


def test_training_is_deterministic(synth40):
    train, test = _time_split(synth40.frame)
    a = predict_proba(train_item_model(train, seed=0), test)
    b = predict_proba(train_item_model(train, seed=0), test)
    assert np.array_equal(a, b)


def test_single_class_training_raises(synth40):
    frame = synth40.frame.assign(y=1.0)
    with pytest.raises(ValueError, match="both"):
        train_item_model(frame)


def test_unknown_category_and_empty_frame_predict(trained):
    model, _, test = trained
    odd = test.head(3).assign(type="Requirement", state_category_at_commit="Resolved")
    p = predict_proba(model, odd)
    assert p.shape == (3,) and np.isfinite(p).all()
    assert predict_proba_lr(model, odd).shape == (3,)
    assert predict_proba(model, test.iloc[0:0]).shape == (0,)


def test_describe_drivers_in_plain_words():
    contrib = pd.Series({f: 0.0 for f in FEATURES} | {"carryover_count": -0.9, "load_ratio": -0.4, "points": 0.3})
    values = pd.Series({f: np.nan for f in FEATURES} | {"carryover_count": 2.0, "load_ratio": 1.456})
    assert describe_drivers(contrib, values) == [
        "sprints already carried over = 2",
        "sprint load vs team velocity = 1.46",
    ]
    assert describe_drivers(contrib.clip(lower=0), values) == []
    flags = pd.Series({f: 0.0 for f in FEATURES} | {"is_unestimated": -0.5, "team_trailing_completion": -0.2})
    flag_values = pd.Series({f: np.nan for f in FEATURES} | {"is_unestimated": 1.0})
    assert describe_drivers(flags, flag_values) == [
        "item is unestimated = yes",
        "team's recent completion rate = missing",
    ]
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_model.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'sprint_forecast.model'`.

- [ ] **Step 3: Implement**

`src/sprint_forecast/model.py`:
```python
"""Model A: LightGBM item classifier with time-split calibration, plus a logistic-regression sanity model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sprint_forecast.features import CATEGORICAL, FEATURES, NUMERIC

LGBM_PARAMS = {
    "num_leaves": 15,
    "min_child_samples": 20,
    "learning_rate": 0.03,
    "n_estimators": 300,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_jobs": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
CALIBRATION_SHARE = 0.2
ISOTONIC_MIN_ITEMS = 1000
P_CLIP = 1e-4

FEATURE_LABELS = {
    "type": "work item type",
    "state_category_at_commit": "state at commit",
    "points": "item size in points",
    "points_rel": "item size vs team velocity",
    "is_unestimated": "item is unestimated",
    "carryover_count": "sprints already carried over",
    "age_days": "item age in days",
    "days_since_change": "days since last change",
    "revisions_so_far": "number of edits so far",
    "has_parent": "has a parent item",
    "load_ratio": "sprint load vs team velocity",
    "n_items": "items committed to the sprint",
    "bug_share": "share of bugs in the sprint",
    "carryover_share": "share of carried-over items in the sprint",
    "unestimated_share": "share of unestimated items in the sprint",
    "team_trailing_completion": "team's recent completion rate",
    "sprint_length_days": "sprint length in days",
    "team_sprint_index": "team's number of earlier sprints",
    "assignee_load_ratio": "assignee load vs their recent delivery",
    "is_unassigned": "item is unassigned",
}


@dataclass
class ItemModel:
    lgbm: LGBMClassifier
    lr: Pipeline
    calibrator: IsotonicRegression | LogisticRegression | None
    calibration: str
    categories: dict[str, list[str]]
    calib_frame: pd.DataFrame


def split_by_time(frame: pd.DataFrame, share: float = CALIBRATION_SHARE) -> tuple[list[str], list[str]]:
    """Split sprint ids by start: the first ~80% fit the model, the last ~20% calibrate it."""
    sprints = frame.drop_duplicates("sprint_id").sort_values(["start", "sprint_id"], kind="mergesort")
    ids = list(sprints["sprint_id"])
    n_cal = max(1, int(round(len(ids) * share))) if len(ids) >= 2 else 0
    return ids[: len(ids) - n_cal], ids[len(ids) - n_cal:]


def to_matrix(frame: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    X = frame[FEATURES].copy()
    for col in CATEGORICAL:
        values = X[col].astype(str)
        X[col] = pd.Categorical(values.where(values.isin(categories[col])), categories=categories[col])
    for col in NUMERIC:
        X[col] = X[col].astype("float64")
    return X


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, P_CLIP, 1 - P_CLIP)
    return np.log(p / (1 - p))


def _make_lr() -> Pipeline:
    pre = ColumnTransformer([
        ("num", make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler()
        ), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=2000))


def _lr_input(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = X[col].astype(str).astype(object)
    for col in NUMERIC:
        X[col] = X[col].astype("float64")
    return X


def _calibrate(model: ItemModel, raw: np.ndarray) -> np.ndarray:
    if model.calibration == "isotonic":
        p = model.calibrator.predict(raw)
    elif model.calibration == "platt":
        p = model.calibrator.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
    else:
        p = raw
    return np.clip(p, P_CLIP, 1 - P_CLIP)


def train_item_model(frame: pd.DataFrame, seed: int = 0) -> ItemModel:
    frame = frame[frame["y"].notna()]
    if frame["y"].nunique() < 2:
        raise ValueError("training data needs both delivered and undelivered items")
    categories = {c: sorted(frame[c].astype(str).unique()) for c in CATEGORICAL}
    fit_ids, cal_ids = split_by_time(frame)
    fit = frame[frame["sprint_id"].isin(fit_ids)]
    cal = frame[frame["sprint_id"].isin(cal_ids)]
    if fit["y"].nunique() < 2:
        fit, cal = frame, frame.iloc[0:0]
    lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=seed)
    lgbm.fit(to_matrix(fit, categories), fit["y"].astype(int))
    lr = _make_lr().fit(_lr_input(fit), fit["y"].astype(int))
    model = ItemModel(lgbm, lr, None, "none", categories, pd.DataFrame(columns=["sprint_id", "p", "y"]))
    if len(cal) and cal["y"].nunique() == 2:
        raw = lgbm.predict_proba(to_matrix(cal, categories))[:, 1]
        y = cal["y"].astype(int).to_numpy()
        if len(cal) >= ISOTONIC_MIN_ITEMS:
            model.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y)
            model.calibration = "isotonic"
        else:
            model.calibrator = LogisticRegression().fit(_logit(raw).reshape(-1, 1), y)
            model.calibration = "platt"
    if len(cal):
        model.calib_frame = pd.DataFrame({
            "sprint_id": cal["sprint_id"].to_numpy(),
            "p": predict_proba(model, cal),
            "y": cal["y"].astype(int).to_numpy(),
        })
    return model


def predict_proba(model: ItemModel, frame: pd.DataFrame) -> np.ndarray:
    if len(frame) == 0:
        return np.array([], dtype=float)
    raw = model.lgbm.predict_proba(to_matrix(frame, model.categories))[:, 1]
    return _calibrate(model, raw)


def predict_proba_lr(model: ItemModel, frame: pd.DataFrame) -> np.ndarray:
    if len(frame) == 0:
        return np.array([], dtype=float)
    return model.lr.predict_proba(_lr_input(frame))[:, 1]


def contributions(model: ItemModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Per-item SHAP contributions (log-odds scale, uncalibrated) from LightGBM pred_contrib; last column 'bias'."""
    values = model.lgbm.predict(to_matrix(frame, model.categories), pred_contrib=True)
    return pd.DataFrame(np.asarray(values), columns=FEATURES + ["bias"], index=frame.index)


BOOLEAN_FEATURES = {"is_unestimated", "has_parent", "is_unassigned"}


def _fmt(feature: str, value) -> str:
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "missing"
    if feature in BOOLEAN_FEATURES:
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3g}"
    return str(value)


def describe_drivers(contrib: pd.Series, values: pd.Series, k: int = 2) -> list[str]:
    """The k most negative contributions, in plain words, e.g. 'sprints already carried over = 2'."""
    neg = contrib[FEATURES]
    neg = neg[neg < 0].sort_values().head(k)
    return [f"{FEATURE_LABELS[f]} = {_fmt(f, values[f])}" for f in neg.index]
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_model.py -v`
Expected: `8 passed`.

Reference values with seed 0 and a train/test split at the median sprint start:
- LightGBM AUC is 0.715 and LR AUC is 0.739. Seeds 0–3 give LightGBM AUCs from 0.656 to 0.722.
- The mean carryover SHAP over carried items is -0.105, against +0.041 over the other items.

- [ ] **Step 5: Commit**

```bash
git add src/sprint_forecast/model.py tests/conftest.py tests/test_model.py
git commit -m "feat: LightGBM item model with time-split calibration, LR sanity model and SHAP drivers"
```

---

### Task 12: Sprint-shock rollup (`rollup.py`)

**Files:**
- Create: `src/sprint_forecast/rollup.py`
- Test: `tests/test_rollup.py`

**Interfaces:**
- Consumes:
  - Task 11: `P_CLIP`, `ItemModel`, `predict_proba`, `train_item_model`.
  - Tests use the `synth40` fixture.
- Produces:
  - Constants: `GH_NODES = 32`, `SIGMA_MAX = 3.0`, `N_DRAWS = 10_000`, `FULL_TOL = 1e-9`.
  - σ fitting:
    - `sprint_log_likelihood(sigma, p, y, groups) -> float`.
    - `fit_sigma(p: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float` returns a value in [0, 3]. It returns 0.0 when there are fewer than 2 sprints.
  - Simulation:
    - `simulate_pct_done(p, points, sigma, n_draws=N_DRAWS, seed=0) -> np.ndarray` returns `n_draws` samples in [0, 1], or all NaN when there are no items.
    - `summarize(samples) -> dict` returns the keys `p_full, p_80, expected, p10, p50, p90`.
  - Forecasting:
    - `@dataclass Forecaster(item_model: ItemModel, sigma: float)`.
    - `fit_forecaster(frame, seed=0) -> Forecaster` fits σ on `item_model.calib_frame`.
    - `forecast_sprint(fc, items, n_draws=N_DRAWS, seed=0) -> tuple[np.ndarray, np.ndarray]` returns item probabilities and %done samples.

- [ ] **Step 1: Write the failing test**

`tests/test_rollup.py`:
```python
import numpy as np
import pytest
from scipy.special import expit

from sprint_forecast.rollup import (
    fit_forecaster,
    fit_sigma,
    forecast_sprint,
    simulate_pct_done,
    sprint_log_likelihood,
    summarize,
)

P = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.6])
W = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 3.0])


def test_sigma_zero_matches_independent_poisson_binomial():
    samples = simulate_pct_done(P, W, sigma=0.0, n_draws=10_000, seed=0)
    s = summarize(samples)
    assert s["expected"] == pytest.approx(float((P * W).sum() / W.sum()), abs=0.01)
    assert s["p_full"] == pytest.approx(float(np.prod(P)), abs=0.01)
    assert samples.min() >= 0.0 and samples.max() <= 1.0


def test_shock_widens_the_band():
    narrow = summarize(simulate_pct_done(P, W, sigma=0.0, seed=0))
    wide = summarize(simulate_pct_done(P, W, sigma=1.5, seed=0))
    assert wide["p90"] - wide["p10"] > narrow["p90"] - narrow["p10"]


def test_simulation_is_deterministic_and_handles_empty():
    a = simulate_pct_done(P, W, sigma=0.7, n_draws=500, seed=3)
    assert np.array_equal(a, simulate_pct_done(P, W, sigma=0.7, n_draws=500, seed=3))
    assert np.isnan(simulate_pct_done(np.array([]), np.array([]), 0.5, n_draws=10)).all()


def _planted(sigma, n_sprints=400, n_items=15, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.8, 1.0, size=(n_sprints, n_items))
    z = rng.normal(0.0, sigma, size=(n_sprints, 1))
    y = (rng.random((n_sprints, n_items)) < expit(base + z)).astype(float)
    groups = np.repeat(np.arange(n_sprints), n_items)
    return expit(base).ravel(), y.ravel(), groups


def test_sigma_fit_recovers_planted_sigma():
    p, y, groups = _planted(0.8, seed=0)
    assert fit_sigma(p, y, groups) == pytest.approx(0.8, abs=0.15)  # seed 0 gives 0.83; seeds 0-9 give 0.76-0.87


def test_sigma_fit_near_zero_without_shock():
    p, y, groups = _planted(0.0, seed=0)
    assert fit_sigma(p, y, groups) < 0.35  # seed 0 gives 0.26; seeds 0-9 give 0.0-0.26


def test_log_likelihood_at_zero_is_independent_bernoulli():
    p, y, groups = _planted(0.0, n_sprints=5, n_items=4, seed=1)
    direct = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    assert sprint_log_likelihood(0.0, p, y, groups) == pytest.approx(direct, rel=1e-9)


def test_fit_sigma_degenerate_inputs():
    assert fit_sigma(np.array([]), np.array([]), np.array([])) == 0.0
    assert fit_sigma(np.array([0.5, 0.5]), np.array([1.0, 0.0]), np.array(["a", "a"])) == 0.0


def test_forecaster_on_synth(synth40):
    frame = synth40.frame
    last = frame["start"].max()
    fc = fit_forecaster(frame[frame["end"] < last], seed=0)
    assert 0.0 <= fc.sigma <= 3.0
    target = frame[frame["start"] == last]
    sid = target["sprint_id"].iloc[0]
    p, samples = forecast_sprint(fc, target[target["sprint_id"] == sid], n_draws=2000, seed=0)
    s = summarize(samples)
    assert 0.0 <= s["p10"] <= s["p50"] <= s["p90"] <= 1.0
    assert 0.0 <= s["p_full"] <= s["p_80"] <= 1.0
    assert len(p) == (target["sprint_id"] == sid).sum()
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_rollup.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'sprint_forecast.rollup'`.

- [ ] **Step 3: Implement**

`src/sprint_forecast/rollup.py`:
```python
"""Sprint-shock rollup: fit sigma by Gauss-Hermite marginal likelihood, then Monte Carlo %done."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp

from sprint_forecast.model import P_CLIP, ItemModel, predict_proba, train_item_model

GH_NODES = 32
SIGMA_MAX = 3.0
N_DRAWS = 10_000
FULL_TOL = 1e-9


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), P_CLIP, 1 - P_CLIP)
    return np.log(p / (1 - p))


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def sprint_log_likelihood(sigma: float, p: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """sum over sprints of log  integral prod_i Bern(y_i | sigmoid(logit p_i + z)) N(z; 0, sigma^2) dz."""
    nodes, weights = np.polynomial.hermite.hermgauss(GH_NODES)
    z = np.sqrt(2.0) * sigma * nodes
    eta = _logit(p)[:, None] + z[None, :]
    y = np.asarray(y, dtype=float)[:, None]
    item_ll = y * _log_sigmoid(eta) + (1.0 - y) * _log_sigmoid(-eta)
    codes, uniques = pd.factorize(pd.Series(groups))
    per_sprint = np.zeros((len(uniques), GH_NODES))
    np.add.at(per_sprint, codes, item_ll)
    log_w = np.log(weights / np.sqrt(np.pi))
    return float(logsumexp(per_sprint + log_w[None, :], axis=1).sum())


def fit_sigma(p: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Maximum-likelihood sprint shock sigma in [0, 3]; 0 when there are fewer than two sprints."""
    if len(p) == 0 or pd.Series(groups).nunique() < 2:
        return 0.0
    res = minimize_scalar(
        lambda s: -sprint_log_likelihood(s, p, y, groups), bounds=(0.0, SIGMA_MAX), method="bounded",
    )
    best = float(res.x)
    if sprint_log_likelihood(0.0, p, y, groups) >= sprint_log_likelihood(best, p, y, groups):
        return 0.0
    return best


def simulate_pct_done(
    p: np.ndarray, points: np.ndarray, sigma: float, n_draws: int = N_DRAWS, seed: int = 0,
) -> np.ndarray:
    """Samples of sum(points * done) / sum(points), with one shared shock z ~ N(0, sigma^2) per draw."""
    p = np.asarray(p, dtype=float)
    w = np.asarray(points, dtype=float)
    if len(p) == 0 or w.sum() <= 0:
        return np.full(n_draws, np.nan)
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, sigma, size=(n_draws, 1)) if sigma > 0 else np.zeros((n_draws, 1))
    prob = expit(_logit(p)[None, :] + z)
    done = rng.random((n_draws, len(p))) < prob
    return done @ w / w.sum()


def summarize(samples: np.ndarray) -> dict[str, float]:
    s = np.asarray(samples, dtype=float)
    q10, q50, q90 = np.quantile(s, [0.1, 0.5, 0.9])
    return {
        "p_full": float(np.mean(s >= 1.0 - FULL_TOL)),
        "p_80": float(np.mean(s >= 0.8 - FULL_TOL)),
        "expected": float(np.mean(s)),
        "p10": float(q10),
        "p50": float(q50),
        "p90": float(q90),
    }


@dataclass
class Forecaster:
    item_model: ItemModel
    sigma: float


def fit_forecaster(frame: pd.DataFrame, seed: int = 0) -> Forecaster:
    """Train model A and fit sigma on its calibrated predictions for the calibration sprints."""
    model = train_item_model(frame, seed=seed)
    cf = model.calib_frame
    sigma = fit_sigma(cf["p"].to_numpy(), cf["y"].to_numpy(), cf["sprint_id"].to_numpy()) if len(cf) else 0.0
    return Forecaster(model, sigma)


def forecast_sprint(
    fc: Forecaster, items: pd.DataFrame, n_draws: int = N_DRAWS, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Item probabilities and %done samples for one sprint's feature rows."""
    p = predict_proba(fc.item_model, items)
    return p, simulate_pct_done(p, items["points"].to_numpy(float), fc.sigma, n_draws, seed)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_rollup.py -v`
Expected: `8 passed`.

Reference values with seed 0:
- With 400 sprints × 15 items, a planted σ of 0.8 is recovered as 0.83. Seeds 0–9 give 0.76–0.87.
- With a planted σ of 0, the fit gives 0.26. Seeds 0–9 give 0.00–0.26.

- [ ] **Step 5: Commit**

```bash
git add src/sprint_forecast/rollup.py tests/test_rollup.py
git commit -m "feat: sprint-shock sigma fit and Monte Carlo %done rollup"
```

---

### Task 13: Backtest (`backtest.py`)

**Files:**
- Create: `src/sprint_forecast/backtest.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes:
  - Task 10: `team_mean_samples`, `velocity_bootstrap`.
  - Task 9: `team_history`.
  - Task 11: `P_CLIP`, `predict_proba_lr`.
  - Task 12: `FULL_TOL`, `N_DRAWS`, `fit_forecaster`, `forecast_sprint`, `summarize`.
  - Task 8: `SprintData`.
  - Tests use the `synth16` fixture.
- Produces:
  - Constants:
    - `SPRINT_MODELS = ("a", "c", "team_mean")`
    - `METRICS = ["crps","pinball_10","pinball_50","pinball_90","brier_full","brier_80","coverage","mae"]`
    - `TARGET_COLUMNS = ["sprint_id","project","team","team_key","iteration","start","n_items","committed_points"]`
  - `class LeakageError(RuntimeError)`.
  - `@dataclass BacktestResult` has these fields:
    - `sprint_rows`: `TARGET_COLUMNS` + `model` + the `summarize` keys + `actual` + `METRICS`. Model `a` rows also have `sigma`, `train_sprints` and `train_end_max`.
    - `item_rows`: `sprint_id, team_key, item_id, y, p_a, p_lr`.
    - `summary`: `model, team_key, n_sprints` + `METRICS`. `team_key == "(all)"` marks the overall rows.
    - `item_summary`: `model` (`"a"` or `"lr"`), `n_items, brier, log_loss, auc`.
    - `calibration`: `range, n, mean_p, frac_done`.
    - `loto`: `team_key, n_sprints, crps, mae, coverage, item_auc`.
    - `skipped`: `{"a": int, "c": int, "team_mean": int}`.
  - Metric helpers:
    - `crps_samples(samples, y) -> float` and `pinball(q, y, tau) -> float`.
    - `sprint_metrics(samples, actual) -> dict`.
    - `item_metrics(y, p) -> dict`.
    - `calibration_table(y, p, bins=10) -> pd.DataFrame`.
  - Target selection and checks:
    - `check_no_leakage(train: pd.DataFrame, target_start) -> None` raises `LeakageError`.
    - `select_targets(sprints, min_history, project=None, team=None) -> pd.DataFrame`.
  - `leave_one_team_out(sprints, frame, held_out: list[str], n_draws, seed) -> pd.DataFrame`.
  - `run_backtest(sd, frame, *, models=SPRINT_MODELS, min_history=8, retrain_every=4, n_draws=N_DRAWS, seed=0, project=None, team=None, loto=True) -> BacktestResult`.
  - `format_report(result) -> str`. When both models A and C ran, it includes a line of the form `Model A beats|does NOT beat baseline C on CRPS (...)`.

- [ ] **Step 1: Write the failing test**

`tests/test_backtest.py`:
```python
import numpy as np
import pandas as pd
import pytest

from sprint_forecast.backtest import (
    METRICS,
    LeakageError,
    calibration_table,
    check_no_leakage,
    crps_samples,
    format_report,
    pinball,
    run_backtest,
    select_targets,
)


def test_crps_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    x = rng.random(200)
    y = 0.4
    pairwise = np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))
    assert crps_samples(x, y) == pytest.approx(pairwise, rel=1e-12)
    assert crps_samples(np.array([0.7]), 0.4) == pytest.approx(0.3)


def test_pinball_loss():
    assert pinball(0.5, 0.8, 0.9) == pytest.approx(0.27)
    assert pinball(0.5, 0.2, 0.9) == pytest.approx(0.03)


def test_leakage_check_raises_on_overlap():
    start = pd.Timestamp("2024-03-04T05:00:00Z")
    ok = pd.DataFrame({"sprint_id": ["s0"], "end": [start - pd.Timedelta(milliseconds=1)]})
    check_no_leakage(ok, start)
    bad = pd.DataFrame({"sprint_id": ["s0", "s1"], "end": [start - pd.Timedelta(days=1), start]})
    with pytest.raises(LeakageError, match="s1"):
        check_no_leakage(bad, start)


def test_select_targets_needs_min_history(synth16):
    targets = select_targets(synth16.sprints.sprints, min_history=8)
    assert set(targets["team"]) == {"Team Red", "Team Blue", "Team Green"}
    assert targets.groupby("team").size().max() <= 8
    assert select_targets(synth16.sprints.sprints, min_history=8, team="Team Red")["team"].eq("Team Red").all()


def test_calibration_table_bins():
    table = calibration_table(np.array([0, 1, 1, 0]), np.array([0.05, 0.95, 1.0, 0.12]))
    assert list(table["range"]) == ["0.0-0.1", "0.1-0.2", "0.9-1.0"]
    assert list(table["n"]) == [1, 1, 2]


@pytest.fixture(scope="module")
def result(synth16):
    return run_backtest(synth16.sprints, synth16.frame, min_history=8, retrain_every=4, n_draws=2000, seed=0)


def test_backtest_has_no_leakage(result):
    a = result.sprint_rows[result.sprint_rows["model"] == "a"]
    assert len(a) > 0
    assert (a["train_end_max"] < a["start"]).all()


def test_backtest_metrics_are_finite(result):
    overall = result.summary[result.summary["team_key"] == "(all)"].set_index("model")
    assert set(overall.index) == {"a", "c", "team_mean"}
    assert np.isfinite(overall[METRICS].to_numpy(dtype=float)).all()
    assert set(result.summary["team_key"]) == {"(all)", "Alpha/Team Red", "Alpha/Team Blue", "Beta/Team Green"}
    assert np.isfinite(result.item_summary[["brier", "log_loss", "auc"]].to_numpy(dtype=float)).all()
    assert result.calibration["n"].sum() == len(result.item_rows)
    assert len(result.loto) == 3 and np.isfinite(result.loto[["crps", "mae"]].to_numpy(dtype=float)).all()
    text = format_report(result)
    assert "baseline C on CRPS" in text and "Leave-one-team-out" in text


def test_backtest_model_filter_and_team_filter(synth16):
    res = run_backtest(synth16.sprints, synth16.frame, models=("c", "team_mean"), team="Team Green", n_draws=500)
    assert set(res.sprint_rows["model"]) == {"c", "team_mean"}
    assert set(res.sprint_rows["team"]) == {"Team Green"}
    assert res.item_rows.empty and res.loto.empty


def test_backtest_with_no_targets(synth16):
    res = run_backtest(synth16.sprints, synth16.frame, min_history=99, n_draws=100)
    assert res.sprint_rows.empty
    assert "No sprints qualified" in format_report(res)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_backtest.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'sprint_forecast.backtest'`.

- [ ] **Step 3: Implement**

`src/sprint_forecast/backtest.py`:
```python
"""Expanding-window backtest of model A (plus LR item metrics), model C and the team-mean reference."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from sprint_forecast.baseline import team_mean_samples, velocity_bootstrap
from sprint_forecast.features import team_history
from sprint_forecast.model import P_CLIP, predict_proba_lr
from sprint_forecast.rollup import FULL_TOL, N_DRAWS, fit_forecaster, forecast_sprint, summarize
from sprint_forecast.sprints import SprintData

SPRINT_MODELS = ("a", "c", "team_mean")
METRICS = ["crps", "pinball_10", "pinball_50", "pinball_90", "brier_full", "brier_80", "coverage", "mae"]
TARGET_COLUMNS = ["sprint_id", "project", "team", "team_key", "iteration", "start", "n_items", "committed_points"]


class LeakageError(RuntimeError):
    """A training sprint did not end before the target sprint started."""


@dataclass
class BacktestResult:
    sprint_rows: pd.DataFrame
    item_rows: pd.DataFrame
    summary: pd.DataFrame
    item_summary: pd.DataFrame
    calibration: pd.DataFrame
    loto: pd.DataFrame
    skipped: dict = field(default_factory=dict)


def crps_samples(samples: np.ndarray, y: float) -> float:
    """CRPS = E|X - y| - 0.5 E|X - X'|, using the sorted-sample identity for the second term."""
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    i = np.arange(1, n + 1)
    return float(np.mean(np.abs(x - y)) - np.sum((2 * i - n - 1) * x) / n**2)


def pinball(q: float, y: float, tau: float) -> float:
    d = y - q
    return float(max(tau * d, (tau - 1) * d))


def sprint_metrics(samples: np.ndarray, actual: float) -> dict:
    s = summarize(samples)
    full = float(actual >= 1.0 - FULL_TOL)
    at_80 = float(actual >= 0.8 - FULL_TOL)
    return {
        **s,
        "actual": float(actual),
        "crps": crps_samples(samples, actual),
        "pinball_10": pinball(s["p10"], actual, 0.1),
        "pinball_50": pinball(s["p50"], actual, 0.5),
        "pinball_90": pinball(s["p90"], actual, 0.9),
        "brier_full": (s["p_full"] - full) ** 2,
        "brier_80": (s["p_80"] - at_80) ** 2,
        "coverage": float(s["p10"] <= actual <= s["p90"]),
        "mae": abs(s["expected"] - actual),
    }


def check_no_leakage(train: pd.DataFrame, target_start: pd.Timestamp) -> None:
    if len(train) and (train["end"] >= target_start).any():
        late = train.loc[train["end"] >= target_start, "sprint_id"].iloc[0]
        raise LeakageError(f"training sprint {late!r} did not end before the target started at {target_start}")


def select_targets(
    sprints: pd.DataFrame, min_history: int, project: str | None = None, team: str | None = None,
) -> pd.DataFrame:
    """Sprints with a known outcome and at least `min_history` earlier sprints (end < start) of their team."""
    s = sprints[sprints["pct_done"].notna()].sort_values(["start", "sprint_id"], kind="mergesort")
    prior = team_history(s, s).set_index("sprint_id")["team_sprint_index"]
    s = s[s["sprint_id"].map(prior) >= min_history]
    if project is not None:
        s = s[s["project"] == project]
    if team is not None:
        s = s[s["team"] == team]
    return s.reset_index(drop=True)


def _item_block(fc, items: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "sprint_id": items["sprint_id"].to_numpy(),
        "team_key": items["team_key"].to_numpy(),
        "item_id": items["item_id"].to_numpy(),
        "y": items["y"].to_numpy(),
        "p_a": p,
        "p_lr": predict_proba_lr(fc.item_model, items),
    })


def _summaries(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["model", "team_key", "n_sprints"] + METRICS)
    overall = rows.groupby("model")[METRICS].mean().assign(team_key="(all)")
    overall["n_sprints"] = rows.groupby("model").size()
    per_team = rows.groupby(["model", "team_key"])[METRICS].mean()
    per_team["n_sprints"] = rows.groupby(["model", "team_key"]).size()
    out = pd.concat([overall.reset_index(), per_team.reset_index()], ignore_index=True)
    return out[["model", "team_key", "n_sprints"] + METRICS]


def item_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), P_CLIP, 1 - P_CLIP)
    both = len(np.unique(y)) == 2
    return {
        "n_items": len(y),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else float("nan"),
        "auc": float(roc_auc_score(y, p)) if both else float("nan"),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    frame["bin"] = np.clip((frame["p"] * bins).astype(int), 0, bins - 1)
    table = frame.groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"), frac_done=("y", "mean")).reset_index()
    table["range"] = [f"{b / bins:.1f}-{(b + 1) / bins:.1f}" for b in table["bin"]]
    return table[["range", "n", "mean_p", "frac_done"]]


def leave_one_team_out(
    sprints: pd.DataFrame, frame: pd.DataFrame, held_out: list[str], n_draws: int, seed: int,
) -> pd.DataFrame:
    """Cross-team generalization check (not time-ordered): train on the other teams, predict the held-out team."""
    rows = []
    known = sprints[sprints["pct_done"].notna()]
    for team_key in held_out:
        try:
            fc = fit_forecaster(frame[frame["team_key"] != team_key], seed=seed)
        except ValueError:
            continue
        ys, ps, metrics = [], [], []
        for t in known[known["team_key"] == team_key].itertuples(index=False):
            items = frame[frame["sprint_id"] == t.sprint_id]
            p, samples = forecast_sprint(fc, items, n_draws, seed)
            metrics.append(sprint_metrics(samples, t.pct_done))
            ys.append(items["y"].to_numpy())
            ps.append(p)
        if not metrics:
            continue
        m = pd.DataFrame(metrics)
        rows.append({
            "team_key": team_key, "n_sprints": len(m), "crps": m["crps"].mean(), "mae": m["mae"].mean(),
            "coverage": m["coverage"].mean(), "item_auc": item_metrics(np.concatenate(ys), np.concatenate(ps))["auc"],
        })
    return pd.DataFrame(rows, columns=["team_key", "n_sprints", "crps", "mae", "coverage", "item_auc"])


def run_backtest(
    sd: SprintData,
    frame: pd.DataFrame,
    *,
    models: tuple[str, ...] = SPRINT_MODELS,
    min_history: int = 8,
    retrain_every: int = 4,
    n_draws: int = N_DRAWS,
    seed: int = 0,
    project: str | None = None,
    team: str | None = None,
    loto: bool = True,
) -> BacktestResult:
    frame = frame[frame["y"].notna()]
    targets = select_targets(sd.sprints, min_history, project, team)
    sprint_rows, item_blocks = [], []
    skipped = {"a": 0, "c": 0, "team_mean": 0}
    fc, train = None, None
    for i, t in enumerate(targets.itertuples(index=False)):
        items = frame[frame["sprint_id"] == t.sprint_id]
        base = {c: getattr(t, c) for c in TARGET_COLUMNS}
        if "a" in models:
            if i % max(1, retrain_every) == 0:
                train = frame[frame["end"] < t.start]
                try:
                    fc = fit_forecaster(train, seed=seed)
                except ValueError:
                    fc = None
            if fc is None:
                skipped["a"] += 1
            else:
                check_no_leakage(train, t.start)
                p, samples = forecast_sprint(fc, items, n_draws, seed)
                sprint_rows.append({
                    **base, "model": "a", **sprint_metrics(samples, t.pct_done), "sigma": fc.sigma,
                    "train_sprints": train["sprint_id"].nunique(), "train_end_max": train["end"].max(),
                })
                item_blocks.append(_item_block(fc, items, p))
        if "c" in models:
            samples = velocity_bootstrap(sd.sprints, pd.Series(t._asdict()), n_draws, seed)
            if samples is None:
                skipped["c"] += 1
            else:
                sprint_rows.append({**base, "model": "c", **sprint_metrics(samples, t.pct_done)})
        if "team_mean" in models:
            samples = team_mean_samples(float(items["team_trailing_completion"].iloc[0]))
            if samples is None:
                skipped["team_mean"] += 1
            else:
                sprint_rows.append({**base, "model": "team_mean", **sprint_metrics(samples, t.pct_done)})
    rows = pd.DataFrame(sprint_rows)
    item_rows = pd.concat(item_blocks, ignore_index=True) if item_blocks else pd.DataFrame(
        columns=["sprint_id", "team_key", "item_id", "y", "p_a", "p_lr"])
    item_summary = pd.DataFrame([
        {"model": name, **item_metrics(item_rows["y"].to_numpy(), item_rows[col].to_numpy())}
        for name, col in (("a", "p_a"), ("lr", "p_lr"))
    ]) if len(item_rows) else pd.DataFrame(columns=["model", "n_items", "brier", "log_loss", "auc"])
    calibration = calibration_table(item_rows["y"], item_rows["p_a"]) if len(item_rows) else pd.DataFrame(
        columns=["range", "n", "mean_p", "frac_done"])
    held_out = sorted(targets["team_key"].unique())
    loto_rows = pd.DataFrame(columns=["team_key", "n_sprints", "crps", "mae", "coverage", "item_auc"])
    if loto and "a" in models and sd.sprints["team_key"].nunique() >= 2 and held_out:
        loto_rows = leave_one_team_out(sd.sprints, frame, held_out, n_draws, seed)
    return BacktestResult(rows, item_rows, _summaries(rows), item_summary, calibration, loto_rows, skipped)


def format_report(result: BacktestResult) -> str:
    """Plain-text report: sprint metrics per model (overall, then per team), item metrics, calibration, LOTO."""
    if result.sprint_rows.empty:
        return "No sprints qualified for the backtest (try a smaller --min-history)."
    lines = ["Sprint-level metrics (lower is better, except coverage: target 0.80)"]
    lines.append(result.summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    overall = result.summary[result.summary["team_key"] == "(all)"].set_index("model")
    if {"a", "c"} <= set(overall.index):
        a, c = overall.loc["a", "crps"], overall.loc["c", "crps"]
        verdict = "beats" if a < c else "does NOT beat"
        lines.append(f"\nModel A {verdict} baseline C on CRPS ({a:.3f} vs {c:.3f}).")
    if len(result.item_summary):
        lines.append("\nItem-level metrics")
        lines.append(result.item_summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        lines.append("\nCalibration of model A (10 bins)")
        lines.append(result.calibration.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(result.loto):
        lines.append("\nLeave-one-team-out (cross-team generalization check, not time-ordered)")
        lines.append(result.loto.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if any(result.skipped.values()):
        lines.append(f"\nSkipped targets per model: {result.skipped}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_backtest.py -v`
Expected: `9 passed`, in about 7 s including building `synth16`.

- [ ] **Step 5: Commit**

```bash
git add src/sprint_forecast/backtest.py tests/test_backtest.py
git commit -m "feat: expanding-window backtest with leakage check, sprint and item metrics, LOTO"
```

---

### Task 14: CLI (`cli.py`), including `demo`

**Files:**
- Create: `src/sprint_forecast/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes every earlier module:
  - Task 2 config: `AUTH_METHODS`, `DEFAULT_DONE`, `DEFAULT_TYPES`, `Config`, `config_path`, `data_dir`, `load_config`, `parse_done_categories`, `save_config`.
  - Task 3 analytics: `fetch_titles`, `make_fetch_json`.
  - Task 4 cache: `CacheData`, `connect`, `load_cache`.
  - Task 5 synth: `generate`.
  - Task 6 extract: `extract_all`.
  - Task 8 sprints: `SprintData`, `build_sprints`, `data_report`, `scope_at`, `sprint_calendar`.
  - Task 9 features: `build_features`, `build_features_for`, `team_history`.
  - Task 11 model: `contributions`, `describe_drivers`.
  - Task 12 rollup: `fit_forecaster`, `forecast_sprint`, `summarize`.
  - Task 13 backtest: `SPRINT_MODELS`, `format_report`, `run_backtest`.
- Produces:
  - `main`: a click group with the option `--root DIR` (default `.`). It stores the root path in `ctx.obj`.
  - Commands:
    - `init --org URL --projects "A,B"|"*" [--auth pat|az-cli] [--force]`
    - `extract [--project P ...] [--full]`
    - `data [--done-categories X,Y]`
    - `backtest [--project P] [--team T] [--model a|c|all] [--min-history N] [--retrain-every K] [--done-categories X,Y]`
    - `train [--done-categories X,Y]`
    - `predict --iteration PATH [--team T] [--as-of auto|now|commit] [--top N] [--titles/--no-titles]`
    - `demo [--seed S] [--n-sprints N>=12]`
  - Files, all under the working directory `root/.sprint-forecast/`, or `root/.sprint-forecast/demo/` for `demo`:
    - `cache.db`
    - `model.joblib`: a dict with keys `forecaster, work_item_types, done_categories, commit_grace_days, trained_at, n_sprints, n_items, version`
    - `backtest.csv`: the per-sprint rows
  - Helpers used by tests:
    - `_predict(workdir: Path, *, iteration, team, as_of, top, title_lookup, now: pd.Timestamp | None = None) -> list[dict]` returns one summary dict per sprint, each with `sprint_id` plus the `summarize` keys.
    - `format_data_report(report: dict) -> str`.
  - Module-level `fetch_titles` and `make_fetch_json` names, so tests can monkeypatch them.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
import tomllib

import pandas as pd
import pytest
from click.testing import CliRunner

from sprint_forecast import cli
from sprint_forecast.cache import connect, load_cache
from sprint_forecast.synth import generate


def run(root, *args):
    result = CliRunner().invoke(cli.main, ["--root", str(root), *args], catch_exceptions=False)
    return result


def test_demo_runs_end_to_end(tmp_path):
    result = run(tmp_path, "demo", "--n-sprints", "12")
    assert result.exit_code == 0, result.output
    demo = tmp_path / ".sprint-forecast" / "demo"
    assert (demo / "cache.db").exists() and (demo / "model.joblib").exists()
    rows = pd.read_csv(demo / "backtest.csv")
    assert set(rows["model"]) == {"a", "c", "team_mean"}
    for text in ("== data ==", "baseline C on CRPS", "Trained on", "P(full)", "riskiest items", "why:"):
        assert text in result.output


def test_init_writes_config_and_refuses_overwrite(tmp_path):
    result = run(tmp_path, "init", "--org", "https://dev.azure.com/contoso/", "--projects", "Alpha, Beta")
    assert result.exit_code == 0
    path = tmp_path / ".sprint-forecast" / "config.toml"
    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    assert doc["ado"] == {"org_url": "https://dev.azure.com/contoso", "projects": ["Alpha", "Beta"], "auth": "pat"}
    again = CliRunner().invoke(cli.main, ["--root", str(tmp_path), "init", "--org", "x", "--projects", "*"])
    assert again.exit_code != 0 and "--force" in again.output
    forced = run(tmp_path, "init", "--org", "https://dev.azure.com/contoso", "--projects", "*", "--auth", "az-cli", "--force")
    assert forced.exit_code == 0
    assert tomllib.loads(path.read_text(encoding="utf-8"))["ado"]["projects"] == ["*"]


def test_extract_without_pat_explains(tmp_path, monkeypatch):
    monkeypatch.delenv("ADO_PAT", raising=False)
    run(tmp_path, "init", "--org", "https://dev.azure.com/contoso", "--projects", "Alpha")
    result = CliRunner().invoke(cli.main, ["--root", str(tmp_path), "extract"])
    assert result.exit_code != 0 and "ADO_PAT" in result.output


def test_extract_uses_fetch_json(tmp_path, monkeypatch):
    monkeypatch.setenv("ADO_PAT", "not-a-real-token")
    calls = []

    def fake_fetch(url):
        calls.append(url)
        if "/Iterations" in url:
            return {"value": [{"IterationSK": "i1", "IterationPath": "Alpha\\Sprint 1", "IsEnded": False,
                               "StartDate": "2024-03-04T00:00:00Z", "EndDate": "2024-03-17T23:59:59.999Z"}]}
        return {"value": []}

    monkeypatch.setattr(cli, "make_fetch_json", lambda method, pat: fake_fetch)
    run(tmp_path, "init", "--org", "https://dev.azure.com/contoso", "--projects", "Alpha")
    result = run(tmp_path, "extract", "--project", "Alpha")
    assert result.exit_code == 0, result.output
    assert "Alpha: 0 revisions fetched" in result.output and "1 projects extracted" in result.output
    conn = connect(tmp_path / ".sprint-forecast" / "cache.db")
    assert list(load_cache(conn).iterations["path"]) == ["Alpha\\Sprint 1"]
    conn.close()


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    generate(root / ".sprint-forecast" / "cache.db", seed=5, n_sprints=12)
    assert run(root, "train").exit_code == 0
    return root


def test_data_report(workspace):
    result = run(workspace, "data", "--done-categories", "Resolved,Completed")
    assert result.exit_code == 0
    assert "Alpha: 2 teams (2 running sprints)" in result.output
    assert "Beta: 2 teams (1 running sprints)" in result.output
    assert "Items added mid-sprint" in result.output


def test_bad_done_categories_rejected(workspace):
    result = CliRunner().invoke(cli.main, ["--root", str(workspace), "data", "--done-categories", "Removed"])
    assert result.exit_code != 0 and "--done-categories" in result.output


def test_backtest_writes_csv(workspace):
    result = run(workspace, "backtest", "--model", "c", "--min-history", "6", "--team", "Team Red")
    assert result.exit_code == 0, result.output
    rows = pd.read_csv(workspace / ".sprint-forecast" / "backtest.csv")
    assert set(rows["model"]) == {"c", "team_mean"} and set(rows["team"]) == {"Team Red"}


def test_predict_past_sprint_as_of_commit_with_titles(workspace, monkeypatch):
    run(workspace, "init", "--org", "https://dev.azure.com/contoso", "--projects", "Alpha,Beta", "--force")
    try:
        monkeypatch.setenv("ADO_PAT", "not-a-real-token")
        monkeypatch.setattr(cli, "fetch_titles", lambda fetch, org, ids: {i: f"Synthetic item {i}" for i in ids})
        result = run(workspace, "predict", "--iteration", "Alpha\\Sprint 12", "--team", "Team Blue", "--top", "2")
    finally:
        (workspace / ".sprint-forecast" / "config.toml").unlink()
    assert result.exit_code == 0, result.output
    assert "Alpha/Team Blue | Alpha\\Sprint 12" in result.output
    assert "scored as of commit cutoff" in result.output
    assert "Synthetic item" in result.output
    assert result.output.count("p=") == 2


def test_predict_auto_uses_now_before_cutoff(workspace, capsys):
    now = pd.Timestamp("2024-06-10T12:00:00Z")  # Alpha Sprint 12 starts 2024-06-10 05:00 UTC; cutoff is a day later
    kwargs = dict(iteration="Alpha\\Sprint 12", team="Team Red", top=0, title_lookup=None)
    early = cli._predict(workspace / ".sprint-forecast", as_of="auto", now=now, **kwargs)
    assert "scored as of now (2024-06-10 12:00 UTC)" in capsys.readouterr().out
    later = cli._predict(workspace / ".sprint-forecast", as_of="auto", now=now + pd.Timedelta(days=2), **kwargs)
    assert "scored as of commit cutoff (2024-06-11 05:00 UTC)" in capsys.readouterr().out
    assert len(early) == len(later) == 1
    assert 0.0 <= early[0]["p_full"] <= early[0]["p_80"] <= 1.0


def test_predict_shared_iteration_without_team_prints_each_team(workspace):
    result = run(workspace, "predict", "--iteration", "Alpha\\Sprint 12", "--no-titles", "--top", "1")
    assert result.exit_code == 0, result.output
    assert "Alpha/Team Red | Alpha\\Sprint 12" in result.output
    assert "Alpha/Team Blue | Alpha\\Sprint 12" in result.output
    assert result.output.count("P(full)") == 2


def test_predict_unknown_iteration_and_missing_model(workspace, tmp_path):
    result = CliRunner().invoke(cli.main, ["--root", str(workspace), "predict", "--iteration", "Alpha\\Nope"])
    assert result.exit_code != 0 and "no team runs" in result.output
    empty = CliRunner().invoke(cli.main, ["--root", str(tmp_path), "predict", "--iteration", "Alpha\\Sprint 1"])
    assert empty.exit_code != 0 and "sprint-forecast train" in empty.output
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: collection error `ImportError: cannot import name 'cli' from 'sprint_forecast'`.

- [ ] **Step 3: Implement**

`src/sprint_forecast/cli.py`:
```python
"""Click CLI: init, extract, data, backtest, train, predict, demo."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import click
import joblib
import pandas as pd

from sprint_forecast import __version__
from sprint_forecast.analytics import fetch_titles, make_fetch_json
from sprint_forecast.backtest import SPRINT_MODELS, format_report, run_backtest
from sprint_forecast.cache import CacheData, connect, load_cache
from sprint_forecast.config import (
    AUTH_METHODS,
    DEFAULT_DONE,
    DEFAULT_TYPES,
    Config,
    config_path,
    data_dir,
    load_config,
    parse_done_categories,
    save_config,
)
from sprint_forecast.extract import extract_all
from sprint_forecast.features import build_features, build_features_for, team_history
from sprint_forecast.model import contributions, describe_drivers
from sprint_forecast.rollup import fit_forecaster, forecast_sprint, summarize
from sprint_forecast.sprints import SprintData, build_sprints, data_report, scope_at, sprint_calendar
from sprint_forecast.synth import generate

CACHE_FILE = "cache.db"
MODEL_FILE = "model.joblib"
BACKTEST_FILE = "backtest.csv"
DEMO_DIR = "demo"
TITLE_WIDTH = 50


@dataclass
class Settings:
    work_item_types: list[str]
    done_categories: list[str]
    commit_grace_days: float = 1.0


def _settings(root: Path, done_categories: str | None) -> Settings:
    path = config_path(root)
    cfg = load_config(path) if path.exists() else None
    try:
        done = parse_done_categories(done_categories) if done_categories else None
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="--done-categories") from None
    return Settings(
        work_item_types=cfg.work_item_types if cfg else list(DEFAULT_TYPES),
        done_categories=done or (cfg.done_categories if cfg else list(DEFAULT_DONE)),
        commit_grace_days=cfg.commit_grace_days if cfg else 1.0,
    )


def _require_config(root: Path) -> Config:
    try:
        return load_config(config_path(root))
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e)) from None


def _load_cache(workdir: Path) -> CacheData:
    path = workdir / CACHE_FILE
    if not path.exists():
        raise click.ClickException(f"no cache at {path}; run `sprint-forecast extract` (or try `sprint-forecast demo`)")
    conn = connect(path)
    try:
        return load_cache(conn)
    finally:
        conn.close()


def _build(cache: CacheData, s: Settings) -> SprintData:
    return build_sprints(
        cache, work_item_types=s.work_item_types, done_categories=s.done_categories,
        commit_grace_days=s.commit_grace_days,
    )


def _pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f}%"


def format_data_report(report: dict) -> str:
    lines = ["Projects"]
    for p in report["projects"]:
        fallback = "  (project-level sprints: no team subscribes to dated iterations)" if p["fallback"] else ""
        lines.append(
            f"  {p['project']}: {p['teams']} teams ({p['teams_running_sprints']} running sprints), "
            f"{p['dated_iterations']} dated / {p['undated_iterations']} undated iterations, "
            f"{p['sprints']} sprints, {p['committed_items']} committed items{fallback}"
        )
    lines += [
        f"Sprints reconstructed: {report['n_sprints']} ({report['dropped_empty_sprints']} empty sprints dropped)",
        f"Committed items: {report['n_committed_items']}; unestimated share {_pct(report['unestimated_share'])}",
        f"Unassigned items (no unique team match, excluded): {report['unassigned_items']}",
        f"Items added mid-sprint (after the commit cutoff, excluded): {report['added_mid_sprint']}",
        "Done-state mix at sprint end: "
        + (", ".join(f"{k} {v}" for k, v in sorted(report["end_state_mix"].items())) or "n/a"),
        f"Resolved share of Resolved+Completed at end: {_pct(report['resolved_share_of_resolved_or_completed'])}"
        " (high values suggest --done-categories Resolved,Completed)",
    ]
    return "\n".join(lines)


def _data(workdir: Path, s: Settings) -> None:
    cache = _load_cache(workdir)
    click.echo(format_data_report(data_report(cache, _build(cache, s))))


def _backtest(workdir: Path, s: Settings, **kwargs) -> None:
    sd = _build(_load_cache(workdir), s)
    result = run_backtest(sd, build_features(sd), **kwargs)
    click.echo(format_report(result))
    out = workdir / BACKTEST_FILE
    result.sprint_rows.to_csv(out, index=False)
    click.echo(f"\nPer-sprint rows written to {out}")


def _train(workdir: Path, s: Settings) -> dict:
    sd = _build(_load_cache(workdir), s)
    frame = build_features(sd)
    try:
        fc = fit_forecaster(frame, seed=0)
    except ValueError as e:
        raise click.ClickException(f"cannot train: {e}") from None
    bundle = {
        "forecaster": fc,
        "work_item_types": s.work_item_types,
        "done_categories": s.done_categories,
        "commit_grace_days": s.commit_grace_days,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_sprints": int(len(sd.sprints)),
        "n_items": int(frame["y"].notna().sum()),
        "version": __version__,
    }
    path = workdir / MODEL_FILE
    joblib.dump(bundle, path)
    click.echo(
        f"Trained on {bundle['n_sprints']} sprints / {bundle['n_items']} items "
        f"(calibration: {fc.item_model.calibration}, sprint shock sigma = {fc.sigma:.2f}) -> {path}"
    )
    return bundle


def _load_model(workdir: Path) -> dict:
    path = workdir / MODEL_FILE
    if not path.exists():
        raise click.ClickException(f"no model at {path}; run `sprint-forecast train` first")
    return joblib.load(path)


def _title_lookup(root: Path) -> Callable[[list[int]], dict[int, str]] | None:
    """Title fetcher for display (never cached); None when there is no config to authenticate with."""
    path = config_path(root)
    if not path.exists():
        return None
    cfg = load_config(path)

    def lookup(ids: list[int]) -> dict[int, str]:
        return fetch_titles(make_fetch_json(cfg.auth_method, cfg.pat), cfg.org_url, ids)

    return lookup


def _truncate(text: str, width: int = TITLE_WIDTH) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def _predict(
    workdir: Path,
    *,
    iteration: str,
    team: str | None,
    as_of: str,
    top: int,
    title_lookup: Callable[[list[int]], dict[int, str]] | None,
    now: pd.Timestamp | None = None,
) -> list[dict]:
    bundle = _load_model(workdir)
    fc = bundle["forecaster"]
    s = Settings(bundle["work_item_types"], bundle["done_categories"], bundle["commit_grace_days"])
    cache = _load_cache(workdir)
    history = _build(cache, s)
    cal = sprint_calendar(cache, s.commit_grace_days)
    group = cal[cal["iteration"] == iteration]
    if group.empty:
        raise click.ClickException(f"no team runs a dated iteration with path {iteration!r}")
    cutoff = group["cutoff"].iloc[0]
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    if as_of == "now" or (as_of == "auto" and now < cutoff):
        t, label = now, "now"
    else:
        t, label = cutoff, "commit cutoff"
    try:
        sprints, items = scope_at(
            cache, history, iteration=iteration, cutoff=t, work_item_types=s.work_item_types,
            done_categories=s.done_categories, commit_grace_days=s.commit_grace_days, team=team,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    if items.empty:
        click.echo(f"No committed items in {iteration} as of {label} ({t:%Y-%m-%d %H:%M} UTC).")
        return []
    feats = build_features_for(sprints, items, history.sprints, history.items)
    velocity = team_history(sprints, history.sprints).set_index("sprint_id")["trailing_velocity"]
    titles: dict[int, str] = {}
    if title_lookup is not None and top > 0:
        try:
            titles = title_lookup(sorted(int(i) for i in feats["item_id"]))
        except Exception:  # titles are cosmetic; never fail a forecast over them
            titles = {}
    results = []
    for sid, rows in feats.groupby("sprint_id", sort=False):
        srow = sprints.set_index("sprint_id").loc[sid]
        p, samples = forecast_sprint(fc, rows, seed=0)
        summary = summarize(samples)
        v = velocity.get(sid, float("nan"))
        click.echo(f"\n{srow['team_key']} | {iteration}")
        click.echo(
            f"  window {srow['start']:%Y-%m-%d} .. {srow['end']:%Y-%m-%d} UTC; "
            f"scored as of {label} ({t:%Y-%m-%d %H:%M} UTC)"
        )
        click.echo(
            f"  committed: {int(srow['n_items'])} items, {srow['committed_points']:.1f} points "
            f"({int(srow['n_unestimated'])} unestimated)"
        )
        if v > 0:
            click.echo(f"  load: {srow['committed_points'] / v:.2f}x trailing velocity ({v:.1f} points)")
        else:
            click.echo("  load: n/a (no velocity history for this team)")
        click.echo(
            f"  P(full) {_pct(summary['p_full'])}   P(>=80%) {_pct(summary['p_80'])}   "
            f"expected {_pct(summary['expected'])}   p10/p50/p90 "
            f"{_pct(summary['p10'])} / {_pct(summary['p50'])} / {_pct(summary['p90'])}"
        )
        risky = rows.assign(p=p).sort_values("p", kind="mergesort").head(top)
        if len(risky):
            click.echo("  riskiest items:")
            contrib = contributions(fc.item_model, risky)
            for idx, r in risky.iterrows():
                drivers = describe_drivers(contrib.loc[idx], r)
                title = _truncate(titles.get(int(r["item_id"]), "")) if titles else ""
                click.echo(f"    #{int(r['item_id'])}  p={r['p']:.2f}  {r['points']:.1f} pts  {title}".rstrip())
                if drivers:
                    click.echo(f"        why: {'; '.join(drivers)}")
        results.append({"sprint_id": sid, **summary})
    return results


done_option = click.option(
    "--done-categories", "done_categories", default=None,
    help="Comma-separated StateCategory values that count as done, e.g. Resolved,Completed "
         "(default: from config, else Completed).",
)


@click.group()
@click.version_option(__version__, prog_name="sprint-forecast")
@click.option(
    "--root", type=click.Path(file_okay=False, path_type=Path), default=".", show_default=True,
    help="Directory that holds .sprint-forecast/.",
)
@click.pass_context
def main(ctx: click.Context, root: Path) -> None:
    """Forecast how much of a sprint's day-1 committed scope an Azure DevOps team will deliver."""
    ctx.obj = root


@main.command()
@click.option("--org", required=True, help="Organization URL, e.g. https://dev.azure.com/contoso")
@click.option("--projects", required=True, help='Comma-separated project names, or "*" for every project.')
@click.option("--auth", type=click.Choice(AUTH_METHODS), default="pat", show_default=True)
@click.option("--force", is_flag=True, help="Overwrite an existing config.")
@click.pass_obj
def init(root: Path, org: str, projects: str, auth: str, force: bool) -> None:
    """Write .sprint-forecast/config.toml."""
    path = config_path(root)
    if path.exists() and not force:
        raise click.ClickException(f"{path} already exists; pass --force to overwrite")
    names = ["*"] if projects.strip() == "*" else [p.strip() for p in projects.split(",") if p.strip()]
    if not names:
        raise click.BadParameter("no project names given", param_hint="--projects")
    save_config(Config(org_url=org.strip().rstrip("/"), projects=names, auth_method=auth), path)
    click.echo(f"Wrote {path}")
    if auth == "pat":
        click.echo("Set ADO_PAT (a PAT with Analytics read scope) before running `sprint-forecast extract`.")


@main.command()
@click.option("--project", "projects", multiple=True, help="Only extract this project (repeatable).")
@click.option("--full", is_flag=True, help="Drop and re-download the selected projects' revisions.")
@click.pass_obj
def extract(root: Path, projects: tuple[str, ...], full: bool) -> None:
    """Pull revisions, iterations and teams into .sprint-forecast/cache.db."""
    cfg = _require_config(root)
    try:
        fetch_json = make_fetch_json(cfg.auth_method, cfg.pat)
    except (ValueError, RuntimeError) as e:
        raise click.ClickException(str(e)) from None
    conn = connect(data_dir(root) / CACHE_FILE)
    try:
        results = extract_all(
            fetch_json, conn, cfg.org_url, list(projects) or cfg.projects,
            work_item_types=cfg.work_item_types, full=full, echo=click.echo,
        )
    finally:
        conn.close()
    skipped = sum(1 for r in results if r.skipped)
    click.echo(f"Done: {len(results) - skipped} projects extracted, {skipped} skipped.")


@main.command()
@done_option
@click.pass_obj
def data(root: Path, done_categories: str | None) -> None:
    """Data-quality report for the cached data."""
    _data(data_dir(root), _settings(root, done_categories))


@main.command()
@click.option("--project", default=None, help="Only score this project's sprints.")
@click.option("--team", default=None, help="Only score this team's sprints.")
@click.option("--model", "model_choice", type=click.Choice(["a", "c", "all"]), default="all", show_default=True)
@click.option("--min-history", type=click.IntRange(min=1), default=8, show_default=True)
@click.option("--retrain-every", type=click.IntRange(min=1), default=4, show_default=True)
@done_option
@click.pass_obj
def backtest(root, project, team, model_choice, min_history, retrain_every, done_categories) -> None:
    """Expanding-window backtest; writes .sprint-forecast/backtest.csv."""
    models = {"a": ("a", "team_mean"), "c": ("c", "team_mean"), "all": SPRINT_MODELS}[model_choice]
    _backtest(
        data_dir(root), _settings(root, done_categories), models=models, min_history=min_history,
        retrain_every=retrain_every, project=project, team=team,
    )


@main.command()
@done_option
@click.pass_obj
def train(root: Path, done_categories: str | None) -> None:
    """Train model A on every completed sprint; writes .sprint-forecast/model.joblib."""
    _train(data_dir(root), _settings(root, done_categories))


@main.command()
@click.option("--iteration", required=True, help="Iteration path, e.g. 'Alpha\\Sprint 12'.")
@click.option("--team", default=None, help="Only this team (default: every team running the iteration).")
@click.option(
    "--as-of", "as_of", type=click.Choice(["auto", "now", "commit"]), default="auto", show_default=True,
    help="auto: the commit cutoff once it has passed, else now. now: current contents (done items drop out).",
)
@click.option("--top", type=click.IntRange(min=0), default=5, show_default=True, help="Riskiest items to list.")
@click.option("--titles/--no-titles", default=True, help="Fetch titles from ADO for display (never cached).")
@click.pass_obj
def predict(root: Path, iteration: str, team: str | None, as_of: str, top: int, titles: bool) -> None:
    """Forecast one iteration's committed scope."""
    lookup = _title_lookup(root) if titles else None
    _predict(data_dir(root), iteration=iteration, team=team, as_of=as_of, top=top, title_lookup=lookup)


@main.command()
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--n-sprints", type=click.IntRange(min=12), default=40, show_default=True)
@click.pass_obj
def demo(root: Path, seed: int, n_sprints: int) -> None:
    """Run the whole pipeline on synthetic data (no ADO needed) under .sprint-forecast/demo/."""
    workdir = data_dir(root) / DEMO_DIR
    generate(workdir / CACHE_FILE, seed=seed, n_sprints=n_sprints)
    s = Settings(list(DEFAULT_TYPES), list(DEFAULT_DONE))
    click.echo(f"Synthetic cache written to {workdir / CACHE_FILE}\n\n== data ==")
    _data(workdir, s)
    click.echo("\n== backtest ==")
    _backtest(workdir, s)
    click.echo("\n== train ==")
    _train(workdir, s)
    history = _build(_load_cache(workdir), s)
    red = history.sprints[history.sprints["team"] == "Team Red"]
    latest = red.sort_values("start").iloc[-1]["iteration"]
    click.echo("\n== predict ==")
    _predict(workdir, iteration=latest, team="Team Red", as_of="commit", top=3, title_lookup=None)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/Scripts/python -m pytest tests/test_cli.py -v`
Expected: `11 passed`, in about 10 s. This includes the Review Focus test `test_predict_shared_iteration_without_team_prints_each_team`.

- [ ] **Step 5: Commit**

```bash
git add src/sprint_forecast/cli.py tests/test_cli.py
git commit -m "feat: click CLI (init, extract, data, backtest, train, predict, demo)"
```

---

### Task 15: README, full verification and pre-push leak check

**Files:**
- Modify: `README.md` (replace the stub completely)

**Interfaces:**
- Consumes: the whole package.
- Produces: user-facing docs. This task adds no new code.

- [ ] **Step 1: Replace `README.md`**

`README.md`:
````markdown
# sprint-forecast

Forecast how much of an Azure DevOps sprint's **day-1 committed scope** a team will deliver.

The output is a distribution over "% of committed points reaching Done": P(full scope), P(>=80%),
the expected %, and a p10/p50/p90 band. It is meant as a planning-time sanity check ("is this sprint
overloaded?") and as an honest ML exercise: every model is backtested against a dumb velocity baseline,
and losing to the baseline is reported as a result.

## Quick start (no Azure DevOps needed)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Linux/macOS: .venv/bin/python
.venv/Scripts/sprint-forecast demo
```

`demo` generates a synthetic cache (projects `Alpha` and `Beta`, teams `Team Red`, `Team Blue`,
`Team Green`, plus `Team Gray`, which runs no sprints), then runs the data report, the backtest,
training and a forecast. Everything is written under `.sprint-forecast/demo/`.

## Using it on your organization

```bash
sprint-forecast init --org https://dev.azure.com/contoso --projects "Alpha,Beta"   # or --projects "*"
export ADO_PAT=...            # PAT with Analytics (read) scope; or: init ... --auth az-cli
sprint-forecast extract       # incremental; --full re-downloads, --project P limits the run
sprint-forecast data          # data-quality report: teams, iterations, unassigned/unestimated items, done-state mix
sprint-forecast backtest      # expanding-window backtest; writes .sprint-forecast/backtest.csv
sprint-forecast train         # writes .sprint-forecast/model.joblib
sprint-forecast predict --iteration "Alpha\Sprint 12" --team "Team Red"
```

`--done-categories Resolved,Completed` (on `data`, `backtest` and `train`) widens what counts as done
when your process closes work in a Resolved state. `predict --as-of now` scores the iteration's current
contents; by default it uses the contents at the commit cutoff once that has passed.

## How it works

- **Extract**: Analytics OData `WorkItemRevisions`, `Iterations` and `Teams` per project into a local
  SQLite cache, with per-project watermarks. A project that returns 401/403 is skipped.
- **Sprint** = (project, team, iteration with dates). Items are assigned to the subscribed team whose area
  path matches exactly, else by longest prefix, else the only subscribed team; otherwise they are
  reported as unassigned. Committed scope is what sits in the iteration at start + 1 day (the commit
  cutoff); an item is done if at the end date it is still in the iteration and in a done category.
- **Features** are computed as of the cutoff and use only sprints that ended before the sprint started.
  Team identity is never a feature.
- **Models**: C, a velocity bootstrap baseline; a team-mean reference; A, a LightGBM item classifier with
  time-split calibration (plus a logistic-regression sanity check). Item probabilities roll up to a sprint
  distribution through a Monte Carlo with a shared per-sprint shock whose size is fitted by maximum
  likelihood.
- **Backtest**: expanding window, retrained every few sprints, with a leakage check, CRPS, pinball loss,
  Brier scores, coverage and MAE per model and team, plus a leave-one-team-out generalization check.

## Privacy

This repository contains code only. The cache, trained models and backtest output live in
`.sprint-forecast/`, which is git-ignored along with `*.db`, `*.sqlite`, `*.jsonl`, `*.parquet`,
`*.joblib` and `.env`. Assignees are kept only as opaque Analytics user keys and used only as numeric
load features. Work item titles are fetched on demand for display and never stored.

## Development

```bash
.venv/Scripts/python -m pytest
```

Tests are offline and deterministic: HTTP is mocked through an injected `fetch_json(url)` function, and
model tests run on synthetic data from `sprint_forecast.synth`. CI runs on Ubuntu and Windows with
Python 3.11-3.13.

## License

MIT
````

- [ ] **Step 2: Run the full suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: `116 passed` in well under 60 s; reference runs take 25-40 s. The run must report no warnings. Any warning about `ConvergenceWarning` or `Pandas4Warning` means something drifted; investigate it, don't suppress it.

- [ ] **Step 3: Run the demo end to end**

Run: `.venv/Scripts/sprint-forecast demo`

Expected, in under a minute (reference runs take 25-40 s):
- The sections `== data ==`, `== backtest ==`, `== train ==` and `== predict ==`.
- A line `Model A beats baseline C on CRPS (...)` or `Model A does NOT beat baseline C on CRPS (...)`. Both are valid results.
- A forecast block for `Alpha/Team Red | Alpha\Sprint 40`, with `P(full)`, `P(>=80%)`, `expected`, `p10/p50/p90` and three riskiest items, each followed by a `why:` line.
- The files `.sprint-forecast/demo/cache.db`, `model.joblib` and `backtest.csv`.

- [ ] **Step 4: Run the pre-push leak check**

Run each command from the repo root. Every command must print **nothing** unless noted otherwise.

```bash
# 1. The demo output is ignored, not untracked: prints "!! .sprint-forecast/" and no "??" line for it
git status --porcelain --ignored | grep sprint-forecast/

# 2. No data or model artifacts are tracked
git ls-files | grep -E '\.(db|sqlite|jsonl|parquet|joblib)$|(^|/)\.sprint-forecast/|(^|/)\.env$'

# 3. No leftover placeholder markers in shipped files
git grep -nIiE 'TODO|TBD|FIXME|XXX|lorem ipsum|implement later' -- src tests README.md pyproject.toml .github

# 4. No org URL other than the synthetic contoso
git grep -nIE 'dev\.azure\.com/[A-Za-z0-9_.-]+' -- src tests README.md | grep -v 'dev\.azure\.com/contoso'

# 5. No email addresses
git grep -nIE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' -- src tests README.md pyproject.toml LICENSE

# 6. GUIDs: only the public Azure DevOps resource ID (analytics.py) and the all-zero test GUID (test_extract.py) may appear
git grep -nIoE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' -- src tests
```

Expected output:
- Command 1 prints `!! .sprint-forecast/`.
- Command 6 prints three lines and only two distinct GUIDs: `499b84ac-1321-427f-aa17-267ca6975798` once, in `src/sprint_forecast/analytics.py`, and `00000000-0000-0000-0000-000000000001` twice, in `tests/test_extract.py`.
- Every other command prints nothing.

Before pushing to the public `HurleySk/sprint-forecast` repo, the user runs their own grep for real org, project, team and person names. It is not automated here.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: README with quick start, usage, method and privacy notes"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Where it is covered |
|---|---|
| Public-repo rules | Global Constraints, Task 1 (`.gitignore`), Task 15 (leak check) |
| Extraction: entity sets, fields and expands | Task 6 |
| Extraction: types filter, `ge` watermark and dedupe, per-project replace | Task 6 |
| Extraction: 401/403 skip, `["*"]` expanding to REST projects | Task 6 |
| Extraction: UTC normalization and EndDate as an instant | Task 3 |
| Cache schema | Task 4 |
| Sprint = (project, team, dated iteration), with project fallback | Task 8 |
| Item-to-team rule | Task 8 |
| Cutoff C, committed scope, outcome at E | Task 8 |
| Done-category widening, mid-sprint additions | Task 8 |
| Points fallback and imputation, % done by points and by count | Task 8 |
| Dropped empty sprints | Task 8 |
| All features, NaN for missing history, no team identity | Task 9 |
| Model C and team mean | Task 10 |
| Model A: params, 80/20 calibration, isotonic vs Platt | Task 11 |
| LR sanity model, `pred_contrib` explanations | Task 11 |
| Rollup: Gauss–Hermite σ fit on calibration sprints, 10,000 draws, fixed seed | Task 12 |
| Backtest: `min_history`, all-teams training with `end < start`, `retrain_every`, leakage check | Task 13 |
| Backtest: item and sprint metrics, per-team results, LOTO, CSV output | Task 13 (CSV written in Task 14) |
| CLI: all seven commands, `data` report contents, `predict` output | Task 14 |
| `predict` titles fetched on demand | Task 14 |
| Synthetic data | Task 5 |
| Testing list: timeline | Task 7 |
| Testing list: sprints | Task 8 |
| Testing list: features | Task 9 |
| Testing list: baseline and rollup | Tasks 10 and 12 |
| Testing list: model | Task 11 |
| Testing list: backtest | Task 13 |
| Testing list: analytics and extract | Tasks 3 and 6 |
| Testing list: cli and demo | Task 14 |
| CI | Task 1 |

**2. Placeholder scan.** Every code step contains the complete file. The only matches for TODO or TBD are inside the Task 15 grep command itself.

**3. Type consistency.** These names are defined once and used with the same signatures everywhere:
- `CacheData`, `SprintData`, `FEATURE_FRAME_COLUMNS`
- `ItemModel`, `Forecaster`, `BacktestResult`
- `scope_at(..., team=...)`, `build_features_for(...)`, `forecast_sprint(...)`, `run_backtest(...)`

The code in this plan was generated from a prototype in which the full 116-test suite passes.

**4. Review Focus.** Each of the five items has its test in the owning task: Tasks 3, 6 (two tests), 8 and 14.
