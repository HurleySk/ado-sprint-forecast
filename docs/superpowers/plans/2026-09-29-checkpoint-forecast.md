# In-sprint Checkpoint Forecasts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the day-1 item model with one checkpoint model that scores any open item (committed or added) at any time t in a sprint, so a running sprint is forecast from where it stands instead of from day 1.

**Architecture:** A new `checkpoints.py` builds open rows: every item open in a sprint at t, described as of t, with its commit-time state and points, the sprint's committed points done by t, and its outcome. `features.py` turns them into one feature set (the item as of t, the sprint's plan and assignee load at the cutoff, progress at t). The model trains on open rows at 0, 25, 50 and 75% of each past sprint. `forecast.py` scores open rows at any t and rolls the committed ones up on top of the points already done; the backtest, export and CLI score and report per checkpoint.

**Tech Stack:** Python 3.11+, pandas 3, numpy, scipy, scikit-learn, LightGBM, click, joblib; pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-checkpoint-forecast-design.md`

## Global Constraints

- `requires-python = ">=3.11"`. Runtime dependencies stay exactly pandas, numpy, scipy, scikit-learn, lightgbm, click, joblib. Add no dependency.
- The repo is public. Only the synthetic names `contoso`, `Alpha`, `Beta`, `Team Red`, `Team Blue`, `Team Green` and `Team Gray` appear in code, tests, docs and commit messages. No real data and no real names, in fixtures either. Before every push, run the owner's private leak check (a `git grep` for real org, project, team and person names, kept outside this repo) and push only when it prints nothing.
- No comment, discussion or sentiment features (spec non-goal).
- The sprint target is unchanged: the share of **day-1 committed points** done by the sprint's end. Baseline C, the team-mean reference, extraction and the cache schema are unchanged.
- `CHECKPOINTS = (0.0, 0.25, 0.5, 0.75)`. A checkpoint's time is `t = cutoff + f * (end - cutoff)`.
- `FEATURE_VERSION = 2` is stored in the model bundle as `feature_version`. A bundle without it fails `predict` and `export` with exactly: model was trained by an older version; run `sprint-forecast train`
- The model recipe is unchanged: `LGBM_PARAMS`, the 80/20 time split by sprint start, isotonic calibration at >= 1000 calibration rows else Platt, `P_CLIP`.
- Timestamps are UTC. pandas 3 stores them as `datetime64[us, UTC]`, and `Series.to_numpy()` of a tz-aware column returns an object array of Timestamps: build datetime columns by assigning Series with matching (reset) indexes, never from `.to_numpy()`. Cast `item_id` to `int64` before any merge.
- CLI output is ASCII only (Windows consoles may use legacy code pages).
- Tests are offline and deterministic, with the seed written in each test. The whole suite must stay well under a minute (today about 30 s for 178 tests).
- No em dashes in code, docs or commit messages.
- Work on `main`: each task ends with a commit and `git push origin main` (the owner's standing rule for this repo). If the push is rejected: `git fetch origin main && git rebase --autostash origin/main && git push origin main`. Never force-push.
- Commit messages are one sentence ending with a period.

**Commands** (Windows paths; on Linux/macOS use `.venv/bin/python`), from the repo root:
- one file: `.venv/Scripts/python -m pytest tests/test_checkpoints.py -q`
- whole suite: `.venv/Scripts/python -m pytest -q`

**File map**

| File | Change | Task |
|---|---|---|
| `src/sprint_forecast/sprints.py` | helpers shared with `checkpoints.py` lose the underscore; new `iteration_group` | 1 |
| `src/sprint_forecast/checkpoints.py` | new: open rows at t, checkpoint rows, sprint progress per checkpoint | 1 |
| `src/sprint_forecast/features.py` | checkpoint feature set, `build_checkpoint_features`, `build_checkpoint_frame`, `FEATURE_VERSION` | 2 |
| `src/sprint_forecast/model.py` | labels and phrases for the new features; calibration grouped by sprint and checkpoint | 2 (interim), 3 |
| `src/sprint_forecast/rollup.py` | the rollup adds points already done and simulates committed open items only | 3 |
| `src/sprint_forecast/forecast.py` | score an iteration at any t; `check_bundle` | 3 |
| `src/sprint_forecast/backtest.py` | per-checkpoint backtest with the `progress` reference | 4 |
| `src/sprint_forecast/export.py` | running sprints scored now, day-1 columns, added and closed item rows | 5 |
| `src/sprint_forecast/cli.py` | train (3), backtest (3 interim, 4), predict and export (5) | 3-5 |
| `tests/test_checkpoints.py`, `tests/test_forecast.py` | new | 1, 3 |
| `README.md`, `pyproject.toml`, `src/sprint_forecast/__init__.py`, `tests/test_smoke.py` | docs, version 0.2.0 | 6 |

## Review Focus

1. A sprint that has already ended, scored "now" (`predict` after its end, or any caller passing a late t): t clamps to the sprint's end and the forecast is the observed outcome, with done points equal to the sprint's actual. Test: Task 3 `test_an_ended_sprint_is_scored_at_its_end_with_the_observed_outcome`.
2. A running sprint whose committed items are all done or out of the sprint, so it has no open rows: it still gets a forecast (a point mass at its done share), and export still lists its whole committed scope. Tests: Task 3 `test_a_sprint_with_nothing_left_open_is_a_point_mass_at_its_done_share`, Task 5 `test_a_running_sprint_with_nothing_left_open_lists_each_committed_item` and `test_export_lists_committed_items_that_are_done_or_gone`.
3. A model file trained by 0.1 (no `feature_version`): `predict` and `export` stop with the retrain message instead of a KeyError deep inside LightGBM. Tests: Task 3 `test_check_bundle_needs_the_current_feature_version`, Task 5 `test_predict_and_export_with_an_old_model_ask_to_retrain` and `test_export_refuses_a_model_from_an_older_version`.
4. An item added mid-sprint with no estimate: it gets the team's usual item size from earlier sprints, not NaN or 1. Tests: Task 1 `test_an_item_added_after_the_cutoff_is_a_row_with_its_outcome`, Task 3 `test_an_unestimated_added_item_takes_the_teams_usual_size`.
5. Two teams sharing an iteration, with an item moved from one team's area to the other's mid-sprint: it becomes the second team's added item, and the first team's committed points stay as they were on day 1. Test: Task 1 `test_an_area_move_to_another_team_makes_it_that_teams_added_item`.

---

### Task 0: Record the private acceptance baseline (controller only; nothing is committed)

The spec's acceptance check compares the new backtest with today's on real data. Today's numbers must be recorded before any code changes. This runs on the owner's machine against a private workspace outside this repo; `$PRIVATE_ROOT` is the folder whose `.sprint-forecast/` holds the real `cache.db` and `config.toml`. Skip this task (and Task 6 Step 5) when no private workspace exists, and say so in the final report.

- [ ] **Step 1: Run the current backtest on the private data**

Run (repo at the commit before Task 1, package installed editable):
```bash
.venv/Scripts/sprint-forecast --root "$PRIVATE_ROOT" backtest > "$PRIVATE_ROOT/acceptance-baseline.txt"
cp "$PRIVATE_ROOT/.sprint-forecast/backtest.csv" "$PRIVATE_ROOT/acceptance-baseline-backtest.csv"
```
Expected: exit code 0; the text file holds "Sprint-level metrics", "Item-level metrics" and the "Model A ... baseline C on CRPS" line.

- [ ] **Step 2: Note the two baseline numbers**

From `acceptance-baseline.txt`, note model `a`'s overall (`(all)`) CRPS and model `a`'s item log loss. Keep them only in `$PRIVATE_ROOT` and in the controller's notes; never in this repo.

---

### Task 1: Open rows at any time t (`checkpoints.py`)

**Files:**
- Modify: `src/sprint_forecast/sprints.py` (renames at lines 86-133 and every call site; `scope_at` at 292-321; new `iteration_group`)
- Create: `src/sprint_forecast/checkpoints.py`
- Create: `tests/test_checkpoints.py`
- Modify: `tests/test_sprints.py` (one new test)

**Interfaces:**
- Consumes: `sprints.py` as it is today; `timeline.as_of_many(revisions, keys, time_col)`; `tests/helpers.py` (`rev`, `iteration`, `team`, `build_cache`); the `synth16` fixture from `tests/conftest.py`.
- Produces (later tasks rely on these exact names):
  - `sprints.pairs_for(revisions, windows) -> DataFrame` (was `_pairs`), `sprints.carryover(revisions, rows, iteration_end) -> np.ndarray` (was `_carryover`), `sprints.Assigner` (was `_Assigner`), `sprints.iteration_end_map(cache) -> Series` (was `_iteration_end_map`), `sprints.horizon(cache, projects) -> Series` (was `_horizon`).
  - `sprints.iteration_group(cal: DataFrame, iteration: str, team: str | None = None) -> DataFrame`: the calendar rows of `iteration` (only `team`'s when given); raises `ValueError("no team runs a dated iteration with path ...")` or `ValueError("team ... does not run ...; teams: ...")`.
  - `checkpoints.CHECKPOINTS = (0.0, 0.25, 0.5, 0.75)`, `OPEN_ROW_COLUMNS`, `CHECKPOINT_ROW_COLUMNS = ["checkpoint"] + OPEN_ROW_COLUMNS`, `PROGRESS_COLUMNS = ["sprint_id", "checkpoint", "t", "done_points", "committed_points"]`.
  - `checkpoints.checkpoint_time(cutoff: Series, end: Series, f: float) -> Series`.
  - `checkpoints.state_changes(revisions) -> DataFrame[item_id, changed, first]`.
  - `checkpoints.done_points_at(cache, items, t: Series indexed by sprint_id, done_categories) -> Series` (float, indexed by sprint_id; empty when `items` is empty).
  - `checkpoints.open_rows(cache, sd, sprints, t, *, work_item_types, done_categories) -> DataFrame[OPEN_ROW_COLUMNS]`, `t` index-aligned with `sprints` (one row per sprint_id; `sd.sprints` or `sprint_calendar` rows both work).
  - `checkpoints.build_checkpoint_rows(cache, sd, *, work_item_types, done_categories, checkpoints=CHECKPOINTS) -> DataFrame[CHECKPOINT_ROW_COLUMNS]`, sorted by start, sprint_id, checkpoint, item_id; `checkpoint` is a float.
  - `checkpoints.checkpoint_progress(cache, sd, *, done_categories, checkpoints=CHECKPOINTS) -> DataFrame[PROGRESS_COLUMNS]`.

What an open row is (spec, "Checkpoint rows"): at t the item's iteration is the sprint's iteration, its type is a configured type, its state category is not done and not `Removed`, and its area puts it in the sprint's team (`Assigner`, the rule `sprints.py` applies at the cutoff). `y` = in the iteration and in a done category at the sprint's end, NaN while the sprint is open. Past the cutoff, `is_added` = not in the sprint's committed scope (`sd.items`), and rows of sprints absent from `sd.sprints` are dropped. Before the cutoff the open items are the committed scope: `is_added = 0`, commit-time columns equal the values at t, `committed_points` = their points, `done_points_at_t = 0`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_checkpoints.py`:

```python
import pandas as pd
import pytest

from conftest import SYNTH_TYPES
from helpers import build_cache, iteration, rev, team
from sprint_forecast.checkpoints import (
    CHECKPOINT_ROW_COLUMNS,
    OPEN_ROW_COLUMNS,
    PROGRESS_COLUMNS,
    build_checkpoint_rows,
    checkpoint_progress,
    checkpoint_time,
    open_rows,
    state_changes,
)
from sprint_forecast.sprints import build_sprints, sprint_calendar

TYPES = ["User Story", "Bug"]
DONE = ["Completed"]
IT0, IT1, IT2 = "Alpha\\Sprint 0", "Alpha\\Sprint 1", "Alpha\\Sprint 2"
S0 = iteration(IT0, "2024-02-19T05:00:00.000Z", "2024-03-04T04:59:59.999Z")
S1 = iteration(IT1, "2024-03-04T05:00:00.000Z", "2024-03-18T04:59:59.999Z")  # cutoff 2024-03-05T05:00Z
S2 = iteration(IT2, "2024-03-18T05:00:00.000Z", "2024-04-01T04:59:59.999Z")
PRE = "2024-03-01T00:00:00.000Z"    # before Sprint 1 starts
PLAN = "2024-03-04T12:00:00.000Z"   # after the start, before the cutoff
T = "2024-03-11T00:00:00.000Z"      # mid-sprint: the t of most tests
RED = team("Team Red", ["Alpha\\Red"], [IT0, IT1, IT2])
BLUE = team("Team Blue", ["Alpha\\Blue"], [IT1])
SPRINT1 = "Alpha/Team Red|Alpha\\Sprint 1"
BLUE1 = "Alpha/Team Blue|Alpha\\Sprint 1"


def at(tmp_path, revisions, t=T, teams=(RED,), sprint_ids=(SPRINT1,)):
    """open_rows of the given Sprint 1 team sprints at t (hand-built caches have no extraction time, so every
    2024 sprint counts as ended and y is known)."""
    cache = build_cache(tmp_path, revisions, [S0, S1, S2], list(teams))
    sd = build_sprints(cache, work_item_types=TYPES, done_categories=DONE)
    cal = sprint_calendar(cache)
    s = cal[cal["sprint_id"].isin(sprint_ids)]
    rows = open_rows(cache, sd, s, pd.Series(pd.Timestamp(t), index=s.index), work_item_types=TYPES, done_categories=DONE)
    return rows, sd


def test_items_done_before_t_are_not_rows_but_count_as_done(tmp_path):
    rows, _ = at(tmp_path, [
        rev(1, 1, PRE, iteration=IT1),
        rev(1, 2, "2024-03-08T00:00:00.000Z", iteration=IT1, state="Closed", state_category="Completed"),
        rev(2, 1, PRE, iteration=IT1, story_points=5.0),
    ])
    assert list(rows.columns) == OPEN_ROW_COLUMNS
    assert rows["item_id"].tolist() == [2]
    r = rows.iloc[0]
    assert r["done_points_at_t"] == 3.0 and r["committed_points"] == 8.0
    assert r["is_added"] == 0 and r["points_at_commit"] == 5.0 and r["state_category_at_commit"] == "Proposed"
    assert r["y"] == 0.0


def test_an_item_added_after_the_cutoff_is_a_row_with_its_outcome(tmp_path):
    rows, _ = at(tmp_path, [
        rev(1, 1, PRE, iteration=IT1),
        rev(6, 1, "2024-02-20T00:00:00.000Z", iteration=IT0, story_points=5.0),  # the team's prior median size: 5
        rev(2, 1, PRE, story_points=None),
        rev(2, 2, "2024-03-07T00:00:00.000Z", iteration=IT1, story_points=None),
        rev(2, 3, "2024-03-15T00:00:00.000Z", iteration=IT1, story_points=None, state="Closed",
            state_category="Completed"),
    ])
    added = rows.set_index("item_id").loc[2]
    assert added["is_added"] == 1 and added["y"] == 1.0
    assert added["is_unestimated"] and added["points"] == 5.0
    assert pd.isna(added["points_at_commit"]) and pd.isna(added["state_category_at_commit"])
    assert added["committed_points"] == 3.0


def test_committed_items_moved_out_before_t_drop_out_and_after_t_stay_with_y_0(tmp_path):
    rows, _ = at(tmp_path, [
        rev(1, 1, PRE, iteration=IT1),
        rev(1, 2, "2024-03-08T00:00:00.000Z", iteration=IT2),
        rev(2, 1, PRE, iteration=IT1),
        rev(2, 2, "2024-03-14T00:00:00.000Z", iteration=IT2),
        rev(3, 1, PRE, iteration=IT1),
        rev(3, 2, "2024-03-15T00:00:00.000Z", iteration=IT1, state="Closed", state_category="Completed"),
    ])
    r = rows.set_index("item_id")
    assert r.index.tolist() == [2, 3]
    assert r["y"].to_dict() == {2: 0.0, 3: 1.0}
    assert (r["committed_points"] == 9.0).all() and (r["done_points_at_t"] == 0.0).all()


def test_state_changes_and_reassignment_on_a_known_history(tmp_path):
    rows, _ = at(tmp_path, [
        rev(1, 1, PRE, assigned_to_sk="u1"),
        rev(1, 2, PLAN, iteration=IT1, assigned_to_sk="u1"),
        rev(1, 3, "2024-03-06T00:00:00.000Z", iteration=IT1, assigned_to_sk="u1", state="Active",
            state_category="InProgress"),
        rev(1, 4, "2024-03-08T00:00:00.000Z", iteration=IT1, assigned_to_sk="u2", state="Active",
            state_category="InProgress"),
        rev(1, 5, "2024-03-09T00:00:00.000Z", iteration=IT1, assigned_to_sk="u2", state="Review",
            state_category="InProgress"),
        rev(1, 6, "2024-03-12T00:00:00.000Z", iteration=IT1, assigned_to_sk="u2", state="Closed",
            state_category="Completed"),
        rev(2, 1, PRE, iteration=IT1, assigned_to_sk="u1"),
        rev(2, 2, "2024-03-07T00:00:00.000Z", iteration=IT1, assigned_to_sk="u1", story_points=5.0),
        rev(3, 1, PRE, iteration=IT1),
    ])
    r = rows.set_index("item_id")
    assert r.loc[1, "n_state_changes"] == 2 and r.loc[1, "reassigned"] == 1
    assert r.loc[1, "state_changed"] == pd.Timestamp("2024-03-09T00:00:00Z")
    assert r.loc[1, "state_category"] == "InProgress" and r.loc[1, "revisions_so_far"] == 5 and r.loc[1, "y"] == 1.0
    assert r.loc[2, "n_state_changes"] == 0 and r.loc[2, "reassigned"] == 0
    assert r.loc[2, "state_changed"] == pd.Timestamp(PRE)
    assert r.loc[2, "last_changed"] == pd.Timestamp("2024-03-07T00:00:00Z")
    assert r.loc[2, "points"] == 5.0 and r.loc[2, "points_at_commit"] == 3.0
    assert r.loc[3, "reassigned"] == 0


def test_an_area_move_to_another_team_makes_it_that_teams_added_item(tmp_path):
    rows, _ = at(tmp_path, [
        rev(1, 1, PRE, iteration=IT1),
        rev(1, 2, "2024-03-08T00:00:00.000Z", iteration=IT1, area="Alpha\\Blue"),
        rev(2, 1, PRE, iteration=IT1, area="Alpha\\Blue"),
        rev(3, 1, PRE, iteration=IT1),
    ], teams=(RED, BLUE), sprint_ids=(SPRINT1, BLUE1))
    red, blue = rows[rows["sprint_id"] == SPRINT1], rows[rows["sprint_id"] == BLUE1]
    assert red["item_id"].tolist() == [3] and (red["committed_points"] == 6.0).all()
    assert blue.set_index("item_id")["is_added"].to_dict() == {1: 1, 2: 0}
    assert (blue["committed_points"] == 3.0).all()


def test_before_the_cutoff_the_open_items_are_the_committed_scope(tmp_path):
    rows, sd = at(tmp_path, [
        rev(1, 1, PRE, iteration=IT1),
        rev(2, 1, PLAN, iteration=IT1, story_points=5.0),
        rev(3, 1, "2024-03-05T00:00:00.000Z", iteration=IT1),  # joins after t, before the cutoff
    ], t="2024-03-04T18:00:00.000Z")
    r = rows.set_index("item_id")
    assert r.index.tolist() == [1, 2]
    assert (r["is_added"] == 0).all() and (r["done_points_at_t"] == 0.0).all() and (r["reassigned"] == 0).all()
    assert (r["committed_points"] == 8.0).all()
    assert sd.sprints.set_index("sprint_id").loc[SPRINT1, "committed_points"] == 11.0
    assert r["points_at_commit"].tolist() == [3.0, 5.0] and (r["state_category_at_commit"] == "Proposed").all()


def test_rows_at_the_cutoff_are_the_committed_items(synth16):
    sd = synth16.sprints
    rows = build_checkpoint_rows(synth16.cache, sd, work_item_types=SYNTH_TYPES, done_categories=DONE,
                                 checkpoints=(0.0,))
    assert list(rows.columns) == CHECKPOINT_ROW_COLUMNS
    items = sd.items.reset_index(drop=True)
    assert list(zip(rows["sprint_id"], rows["item_id"])) == list(zip(items["sprint_id"], items["item_id"]))
    for col in ("points", "raw_points", "is_unestimated", "carryover_count", "revisions_so_far"):
        pd.testing.assert_series_equal(rows[col], items[col], check_dtype=False, check_names=False)
    assert rows["assigned_to_sk"].fillna("").tolist() == items["assigned_to_sk"].fillna("").tolist()
    assert rows["state_category"].tolist() == items["state_category_at_commit"].tolist()
    assert (rows["is_added"] == 0).all() and (rows["reassigned"] == 0).all() and (rows["done_points_at_t"] == 0).all()
    pd.testing.assert_series_equal(rows["y"], items["done"], check_dtype=False, check_names=False)


def test_checkpoint_progress_counts_committed_points_done_by_t(synth16):
    sd = synth16.sprints
    prog = checkpoint_progress(synth16.cache, sd, done_categories=DONE, checkpoints=(0.0, 1.0))
    assert list(prog.columns) == PROGRESS_COLUMNS and len(prog) == 2 * len(sd.sprints)
    first, last = prog[prog["checkpoint"] == 0.0], prog[prog["checkpoint"] == 1.0].set_index("sprint_id")
    assert (first["done_points"] == 0.0).all()
    closed = sd.sprints[sd.sprints["done_points"].notna()].set_index("sprint_id")
    assert last.loc[closed.index, "done_points"].to_numpy() == pytest.approx(closed["done_points"].to_numpy())
    assert (last["t"] == sd.sprints.set_index("sprint_id").loc[last.index, "end"]).all()


def test_checkpoint_time_is_a_share_of_the_way_from_cutoff_to_end():
    cutoff = pd.Series(pd.to_datetime(["2024-03-05T05:00:00Z"]))
    end = pd.Series(pd.to_datetime(["2024-03-15T05:00:00Z"]))
    assert checkpoint_time(cutoff, end, 0.0).iloc[0] == cutoff.iloc[0]
    assert checkpoint_time(cutoff, end, 0.25).iloc[0] == pd.Timestamp("2024-03-07T17:00:00Z")


def test_state_changes_keep_the_first_revision_and_each_change_of_state():
    revs = pd.DataFrame({
        "item_id": [1, 1, 1, 1, 2],
        "rev": [1, 2, 3, 4, 1],
        "changed": pd.to_datetime(["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04", "2024-03-01"], utc=True),
        "state": ["New", "New", "Active", "New", "Active"],
    })
    ev = state_changes(revs)
    assert list(zip(ev["item_id"], ev["changed"].dt.day, ev["first"])) == [
        (1, 1, True), (1, 3, False), (1, 4, False), (2, 1, True),
    ]
```

Append to `tests/test_sprints.py` (add `iteration_group` to its `from sprint_forecast.sprints import (...)` list):

```python
def test_iteration_group_filters_to_a_team_and_explains_bad_input(tmp_path):
    blue = team("Team Blue", ["Alpha\\Blue"], ["Alpha\\Sprint 1"])
    cache = build_cache(tmp_path, [rev(1, 1, PRE)], [S1], [RED, blue])
    cal = sprint_calendar(cache)
    assert set(iteration_group(cal, "Alpha\\Sprint 1")["team"]) == {"Team Red", "Team Blue"}
    assert iteration_group(cal, "Alpha\\Sprint 1", "Team Blue")["team"].tolist() == ["Team Blue"]
    with pytest.raises(ValueError, match="no team runs"):
        iteration_group(cal, "Alpha\\Nope")
    with pytest.raises(ValueError, match="does not run"):
        iteration_group(cal, "Alpha\\Sprint 1", "Team Pink")
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_checkpoints.py tests/test_sprints.py -q`
Expected: collection errors: `ModuleNotFoundError: No module named 'sprint_forecast.checkpoints'` and `ImportError: cannot import name 'iteration_group'`.

- [ ] **Step 3: Make the shared helpers public and add `iteration_group` in `sprints.py`**

Rename in `src/sprint_forecast/sprints.py`, at the definition and at every call site in the file (nothing outside `sprints.py` uses them today):

| Old | New |
|---|---|
| `class _Assigner` / `_Assigner(` / `assign: _Assigner` | `class Assigner` / `Assigner(` / `assign: Assigner` |
| `def _iteration_end_map` / `_iteration_end_map(` | `def iteration_end_map` / `iteration_end_map(` |
| `def _horizon` / `_horizon(` | `def horizon` / `horizon(` |
| `def _pairs` / `_pairs(` | `def pairs_for` / `pairs_for(` |
| `def _carryover` / `_carryover(` | `def carryover` / `carryover(` |

Check with `git grep -nE "_Assigner|_iteration_end_map|_horizon|_pairs\(|_carryover" -- src` (expected: no output).

Add after `sprint_calendar` (before `class Assigner`):

```python
def iteration_group(cal: pd.DataFrame, iteration: str, team: str | None = None) -> pd.DataFrame:
    """The calendar rows (one per team) of `iteration`, only `team`'s when given. Raises ValueError for an iteration
    no team runs, or a team that does not run it."""
    group = cal[cal["iteration"] == iteration]
    if group.empty:
        raise ValueError(f"no team runs a dated iteration with path {iteration!r}")
    if team is not None and team not in set(group["team"]):
        raise ValueError(f"team {team!r} does not run {iteration!r}; teams: {', '.join(group['team'])}")
    return group if team is None else group[group["team"] == team]
```

In `scope_at`, replace the five lines from `group = cal[cal["iteration"] == iteration]` through the `raise ValueError(f"team {team!r} ...")` line with:

```python
    group = iteration_group(cal, iteration, team)
```

(the next line, `windows = group.iloc[:1][...]`, stays as it is).

- [ ] **Step 4: Create `src/sprint_forecast/checkpoints.py`**

```python
"""Checkpoint rows: every item open in a sprint at an instant t, described as of t, with its outcome at the end.

An open item is in the sprint's iteration, of a configured type, neither done nor removed, and in the sprint's team
by area, all as of t. Past the cutoff a row knows whether the item was in the sprint's day-1 committed scope and how
many committed points were done by t; before the cutoff the open items are the committed scope."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sprint_forecast.cache import CacheData
from sprint_forecast.sprints import (
    CALENDAR_COLUMNS,
    REMOVED,
    Assigner,
    SprintData,
    carryover,
    horizon,
    impute_points,
    iteration_end_map,
    pairs_for,
    raw_points,
    sprint_calendar,
)
from sprint_forecast.timeline import as_of_many

CHECKPOINTS = (0.0, 0.25, 0.5, 0.75)
OPEN_ROW_COLUMNS = CALENDAR_COLUMNS + [
    "t", "item_id", "type", "state_category", "state_category_at_commit", "points_at_commit", "area",
    "assigned_to_sk", "parent_id", "created", "last_changed", "state_changed", "revisions_so_far", "n_state_changes",
    "carryover_count", "raw_points", "points", "is_unestimated", "is_added", "reassigned", "committed_points",
    "done_points_at_t", "y",
]
CHECKPOINT_ROW_COLUMNS = ["checkpoint"] + OPEN_ROW_COLUMNS
PROGRESS_COLUMNS = ["sprint_id", "checkpoint", "t", "done_points", "committed_points"]


def checkpoint_time(cutoff: pd.Series, end: pd.Series, f: float) -> pd.Series:
    """The instant a share f of the way from each sprint's commit cutoff to its end."""
    return cutoff + (end - cutoff) * f


def state_changes(revisions: pd.DataFrame) -> pd.DataFrame:
    """Each item's first revision (first=True) and every later revision whose state differs from the one before."""
    r = revisions.sort_values(["item_id", "changed", "rev"], kind="mergesort").reset_index(drop=True)
    state = r["state"].fillna("")
    prev = state.groupby(r["item_id"], sort=False).shift()
    first = prev.isna()
    keep = first | (state != prev)
    return r.loc[keep, ["item_id", "changed"]].assign(first=first[keep]).reset_index(drop=True)


def _last_state_change(events: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """When each row's item last changed state at or before the row's t (index aligned with `rows`)."""
    left = pd.DataFrame({"item_id": rows["item_id"].astype("int64"), "t": rows["t"]}).reset_index(drop=True)
    left["_row"] = np.arange(len(left))
    left["t"] = left["t"].astype(events["changed"].dtype)
    right = events[["item_id", "changed"]].assign(item_id=events["item_id"].astype("int64"))
    out = pd.merge_asof(
        left.sort_values("t", kind="mergesort"), right.sort_values("changed", kind="mergesort"),
        left_on="t", right_on="changed", by="item_id", direction="backward",
    )
    return out.sort_values("_row")["changed"].set_axis(rows.index)


def _state_changes_between(events: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """State changes of each row's item in (start, t], not counting its first revision (index aligned with `rows`)."""
    left = pd.DataFrame({
        "item_id": rows["item_id"].astype("int64"), "start": rows["start"], "t": rows["t"],
    }).reset_index(drop=True)
    left["_row"] = np.arange(len(left))
    later = events.loc[~events["first"], ["item_id", "changed"]]
    hist = left.merge(later.assign(item_id=later["item_id"].astype("int64")), on="item_id")
    hist = hist[(hist["changed"] > hist["start"]) & (hist["changed"] <= hist["t"])]
    counts = left["_row"].map(hist.groupby("_row").size()).fillna(0).astype("int64")
    return counts.set_axis(rows.index)


def done_points_at(cache: CacheData, items: pd.DataFrame, t: pd.Series, done_categories) -> pd.Series:
    """Per sprint: points of `items` (sprint_id, item_id, iteration, points) that are in their sprint's iteration and
    in a done category as of that sprint's t. `t` is indexed by sprint_id and covers every item's sprint."""
    if items.empty:
        return pd.Series(dtype="float64")
    it = items.reset_index(drop=True)
    keys = pd.DataFrame({"item_id": it["item_id"].astype("int64"), "t": it["sprint_id"].map(t)})
    at = as_of_many(cache.revisions, keys, "t")
    done = (at["iteration"] == it["iteration"]) & at["state_category"].isin(set(done_categories))
    return (it["points"].astype("float64") * done).groupby(it["sprint_id"]).sum()


def open_rows(
    cache: CacheData,
    sd: SprintData,
    sprints: pd.DataFrame,
    t: pd.Series,
    *,
    work_item_types,
    done_categories,
) -> pd.DataFrame:
    """Items open in each sprint row at its own t (`t` index-aligned with `sprints`, one row per sprint_id), with
    their state as of t, commit-time state and points (NaN for items added after the cutoff), the sprint's committed
    points and the committed points done by t, and y: in the iteration and done at the sprint's end (NaN while the
    sprint is open). Before a sprint's cutoff its open items are its committed scope. Past the cutoff, sprints with
    no committed scope in `sd` are dropped."""
    if sprints["sprint_id"].duplicated().any():
        raise ValueError("open_rows needs one row per sprint")
    empty = pd.DataFrame(columns=OPEN_ROW_COLUMNS)
    revs = cache.revisions
    types, done = set(work_item_types), set(done_categories)
    windows = sprints[CALENDAR_COLUMNS].assign(t=t).reset_index(drop=True).rename(columns={"iteration": "target"})
    pairs = pairs_for(revs, windows)
    if pairs.empty:
        return empty
    at_t = as_of_many(revs, pairs[["item_id", "t"]], "t")
    is_open = (
        (at_t["iteration"] == pairs["target"])
        & at_t["type"].isin(types)
        & ~at_t["state_category"].isin(done | {REMOVED})
    )
    pairs, at_t = pairs[is_open], at_t[is_open]
    team = Assigner(cache, sprint_calendar(cache))(at_t["area"], pairs["target"], pairs["project"])
    mine = team == pairs["team"]
    rows = pairs[mine].rename(columns={"target": "iteration"}).reset_index(drop=True)
    st = at_t[mine].reset_index(drop=True)
    if rows.empty:
        return empty
    rows["item_id"] = rows["item_id"].astype("int64")
    for col in ("type", "state_category", "area", "assigned_to_sk", "parent_id", "created"):
        rows[col] = st[col]
    rows["last_changed"] = st["changed"]
    rows["revisions_so_far"] = st["revisions_so_far"].astype("int64")
    rows["raw_points"] = raw_points(st)
    rows = impute_points(rows, sd.items)
    rows["carryover_count"] = carryover(revs, rows.assign(cutoff=rows["t"]), iteration_end_map(cache))

    until_cutoff = rows["t"].where(rows["t"] < rows["cutoff"], rows["cutoff"])
    at_c = as_of_many(revs, pd.DataFrame({"item_id": rows["item_id"], "t": until_cutoff}), "t")
    rows["reassigned"] = (at_c["assigned_to_sk"].fillna("") != rows["assigned_to_sk"].fillna("")).astype("int64")
    events = state_changes(revs[revs["item_id"].isin(set(rows["item_id"]))])
    rows["state_changed"] = _last_state_change(events, rows)
    rows["n_state_changes"] = _state_changes_between(events, rows)

    commit = sd.items[["sprint_id", "item_id", "state_category_at_commit", "points"]].rename(
        columns={"points": "points_at_commit"})
    commit = commit.assign(item_id=commit["item_id"].astype("int64"))
    rows = rows.merge(commit, on=["sprint_id", "item_id"], how="left")
    committed = sd.sprints.set_index("sprint_id")["committed_points"]
    after = rows["t"] >= rows["cutoff"]
    rows = rows[~after | rows["sprint_id"].isin(committed.index)].reset_index(drop=True)
    if rows.empty:
        return empty
    after = rows["t"] >= rows["cutoff"]
    rows["is_added"] = (after & rows["points_at_commit"].isna()).astype("int64")
    rows.loc[~after, "state_category_at_commit"] = rows.loc[~after, "state_category"]
    rows.loc[~after, "points_at_commit"] = rows.loc[~after, "points"]
    open_points = rows.groupby("sprint_id")["points"].sum()
    rows["committed_points"] = (
        rows["sprint_id"].map(committed).where(after, rows["sprint_id"].map(open_points)).astype("float64")
    )
    t_by_sprint = rows.drop_duplicates("sprint_id").set_index("sprint_id")["t"]
    scope = sd.items[sd.items["sprint_id"].isin(set(rows.loc[after, "sprint_id"]))]
    done_points = done_points_at(cache, scope, t_by_sprint, done_categories)
    rows["done_points_at_t"] = rows["sprint_id"].map(done_points).fillna(0.0).where(after, 0.0)

    at_end = as_of_many(revs, pd.DataFrame({"item_id": rows["item_id"], "end": rows["end"]}), "end")
    ended_done = (at_end["iteration"] == rows["iteration"]) & at_end["state_category"].isin(done)
    rows["y"] = ended_done.astype("float64").where(rows["end"] <= horizon(cache, rows["project"]))
    out = rows[OPEN_ROW_COLUMNS].sort_values(["start", "sprint_id", "item_id"], kind="mergesort")
    return out.reset_index(drop=True)


def build_checkpoint_rows(
    cache: CacheData,
    sd: SprintData,
    *,
    work_item_types,
    done_categories,
    checkpoints=CHECKPOINTS,
) -> pd.DataFrame:
    """open_rows for every sprint of `sd` at each checkpoint (a share of the way from its cutoff to its end)."""
    s = sd.sprints
    frames = []
    for f in checkpoints:
        rows = open_rows(
            cache, sd, s, checkpoint_time(s["cutoff"], s["end"], f),
            work_item_types=work_item_types, done_categories=done_categories,
        )
        if len(rows):
            frames.append(rows.assign(checkpoint=float(f)))
    if not frames:
        return pd.DataFrame(columns=CHECKPOINT_ROW_COLUMNS)
    out = pd.concat(frames, ignore_index=True)[CHECKPOINT_ROW_COLUMNS]
    return out.sort_values(["start", "sprint_id", "checkpoint", "item_id"], kind="mergesort").reset_index(drop=True)


def checkpoint_progress(cache: CacheData, sd: SprintData, *, done_categories, checkpoints=CHECKPOINTS) -> pd.DataFrame:
    """Per sprint and checkpoint: t, the sprint's committed points in its iteration and done as of t, and its
    committed points. Sprints with no open item at a checkpoint still get a row here."""
    s = sd.sprints
    if s.empty:
        return pd.DataFrame(columns=PROGRESS_COLUMNS)
    frames = []
    for f in checkpoints:
        t = checkpoint_time(s["cutoff"], s["end"], f)
        done = done_points_at(cache, sd.items, t.set_axis(s["sprint_id"].to_numpy()), done_categories)
        frames.append(pd.DataFrame({
            "sprint_id": s["sprint_id"],
            "checkpoint": float(f),
            "t": t,
            "done_points": s["sprint_id"].map(done).fillna(0.0).astype("float64"),
            "committed_points": s["committed_points"].astype("float64"),
        }))
    return pd.concat(frames, ignore_index=True)[PROGRESS_COLUMNS]
```

- [ ] **Step 5: Run the new tests**

Run: `.venv/Scripts/python -m pytest tests/test_checkpoints.py tests/test_sprints.py -q`
Expected: all pass.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass (178 before this task, plus the 11 new tests).

- [ ] **Step 7: Commit and push**

```bash
git add src/sprint_forecast/sprints.py src/sprint_forecast/checkpoints.py tests/test_checkpoints.py tests/test_sprints.py
git commit -m "Add checkpoint rows: every item open in a sprint at any time t, with its commit-time state, the committed points done by t and its outcome."
git push origin main
```

---

### Task 2: Checkpoint features (`features.py`)

**Files:**
- Modify: `src/sprint_forecast/features.py` (constants at lines 9-23; new functions after `build_features`)
- Modify: `src/sprint_forecast/model.py:16` (interim import, replaced in Task 3)
- Modify: `tests/conftest.py` (`Synth` gains `ckpt` and `progress`)
- Modify: `tests/test_features.py`, `tests/test_model.py:7`

**Interfaces:**
- Consumes (Task 1): `checkpoints.CHECKPOINTS`, `checkpoints.build_checkpoint_rows`, `checkpoints.checkpoint_progress`, `checkpoints.open_rows`, `checkpoints.checkpoint_time`, and the `OPEN_ROW_COLUMNS` / `CHECKPOINT_ROW_COLUMNS` row layout.
- Produces:
  - `features.CATEGORICAL = ["type", "state_category"]`; `features.FEATURES` (29 names: `ITEM_FEATURES + SPRINT_FEATURES + ASSIGNEE_FEATURES + PROGRESS_FEATURES`); `features.NUMERIC`.
  - `features.PROGRESS_FEATURES = ["elapsed", "days_left", "done_share", "days_in_state", "n_state_changes", "is_added", "reassigned"]`.
  - `features.DAY1_FEATURES`: today's 20 feature names, in today's order (the day-1 frame keeps them).
  - `features.FEATURE_FRAME_COLUMNS = ID_COLUMNS + DAY1_FEATURES + ["y"]` (the day-1 frame; unchanged content).
  - `features.CHECKPOINT_FRAME_COLUMNS = ID_COLUMNS + ["checkpoint", "t", "state_category_at_commit", "points_at_commit"] + FEATURES + ["y"]`.
  - `features.FEATURE_VERSION = 2`.
  - `features.build_checkpoint_features(rows, day1, history_sprints) -> DataFrame[CHECKPOINT_FRAME_COLUMNS]`: `rows` from `open_rows` or `build_checkpoint_rows`; `day1` the `build_features_for` frame of the same sprints; `history_sprints` for trailing velocity. Row order follows `rows`.
  - `features.build_checkpoint_frame(cache, sd, *, work_item_types, done_categories, checkpoints=CHECKPOINTS) -> DataFrame[CHECKPOINT_FRAME_COLUMNS]`: the training frame.
  - `tests/conftest.py`: `Synth(cache, sprints, frame, ckpt, progress)`; `ckpt` = `build_checkpoint_frame(..., work_item_types=SYNTH_TYPES, done_categories=["Completed"])`, `progress` = `checkpoint_progress(..., done_categories=["Completed"])`.
  - Interim until Task 3: `model.py` trains on `DAY1_FEATURES` (aliased as `FEATURES`) with `CATEGORICAL = ["type", "state_category_at_commit"]`, so every caller that still passes a day-1 frame keeps working.

Feature definitions (spec, "Features"): item features are recomputed at t; sprint features and `assignee_load_ratio` are the day-1 values (the plan the sprint started with; NaN for added items and for sprints absent from `day1`). `is_unassigned` moves from the assignee group to the item group because it is read at t. `elapsed` = (t - cutoff) / (end - cutoff) clipped to [0, 1], 0 before the cutoff. `days_left` = (end - max(t, cutoff)) in days, never negative. `done_share` = `done_points_at_t / committed_points`, 0 before the cutoff. `days_in_state` = days since the last state change at or before t.

- [ ] **Step 1: Write the failing tests**

In `tests/conftest.py`, replace the imports, `Synth` and `_synth` with:

```python
from sprint_forecast.cache import CacheData, connect, load_cache
from sprint_forecast.checkpoints import checkpoint_progress
from sprint_forecast.features import build_checkpoint_frame, build_features
from sprint_forecast.sprints import SprintData, build_sprints
from sprint_forecast.synth import generate

SYNTH_TYPES = ["User Story", "Product Backlog Item", "Bug"]


@dataclass
class Synth:
    cache: CacheData
    sprints: SprintData
    frame: pd.DataFrame  # day-1 feature frame (baseline C and the team-mean reference read it)
    ckpt: pd.DataFrame  # checkpoint feature frame at CHECKPOINTS (model A trains on it)
    progress: pd.DataFrame  # per sprint and checkpoint: t, committed points done by t, committed points


def _synth(tmp_path_factory, seed: int, n_sprints: int) -> Synth:
    path = generate(tmp_path_factory.mktemp(f"synth{seed}_{n_sprints}") / "cache.db", seed=seed, n_sprints=n_sprints)
    conn = connect(path)
    cache = load_cache(conn)
    conn.close()
    sd = build_sprints(cache, work_item_types=SYNTH_TYPES)
    ckpt = build_checkpoint_frame(cache, sd, work_item_types=SYNTH_TYPES, done_categories=["Completed"])
    progress = checkpoint_progress(cache, sd, done_categories=["Completed"])
    return Synth(cache, sd, build_features(sd), ckpt, progress)
```

In `tests/test_features.py`, replace the imports (lines 1-17) with:

```python
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
```

In `test_revisions_after_cutoff_do_not_change_features`, change `cols = ID_COLUMNS + FEATURES` to `cols = ID_COLUMNS + DAY1_FEATURES`.

Append to `tests/test_features.py`:

```python
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
```

(`age_days`: created 2024-01-01T00:00Z, t 2024-03-04T18:00Z, 63.75 days; `days_in_state`: the only revision is 2024-03-01T00:00Z.)

In `tests/test_model.py`, change line 7 to:

```python
from sprint_forecast.features import DAY1_FEATURES as FEATURES
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_features.py -q`
Expected: collection error `ImportError: cannot import name 'CHECKPOINT_FRAME_COLUMNS' from 'sprint_forecast.features'` (conftest fails the same way on `build_checkpoint_frame`).

- [ ] **Step 3: Add the checkpoint feature set to `features.py`**

Replace lines 1-23 of `src/sprint_forecast/features.py` (docstring through `FEATURE_FRAME_COLUMNS`) with:

```python
"""Feature matrices. The day-1 frame describes committed items at the commit cutoff; the checkpoint frame describes
items open at any time t in a sprint. Team history uses only sprints with end < this sprint's start."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sprint_forecast.cache import CacheData
from sprint_forecast.checkpoints import CHECKPOINTS, build_checkpoint_rows
from sprint_forecast.sprints import SprintData

MISSING = "(missing)"
CATEGORICAL = ["type", "state_category"]
ITEM_FEATURES = [
    "type", "state_category", "points", "points_rel", "is_unestimated", "carryover_count",
    "age_days", "days_since_change", "revisions_so_far", "has_parent", "is_unassigned",
]
SPRINT_FEATURES = [
    "load_ratio", "n_items", "bug_share", "carryover_share", "unestimated_share",
    "team_trailing_completion", "sprint_length_days", "team_sprint_index",
]
ASSIGNEE_FEATURES = ["assignee_load_ratio"]
PROGRESS_FEATURES = ["elapsed", "days_left", "done_share", "days_in_state", "n_state_changes", "is_added", "reassigned"]
FEATURES = ITEM_FEATURES + SPRINT_FEATURES + ASSIGNEE_FEATURES + PROGRESS_FEATURES
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
# The day-1 frame's features (model A before 0.2.0); it still supplies sprint context and assignee load.
DAY1_FEATURES = [
    "type", "state_category_at_commit", "points", "points_rel", "is_unestimated", "carryover_count",
    "age_days", "days_since_change", "revisions_so_far", "has_parent",
] + SPRINT_FEATURES + ["assignee_load_ratio", "is_unassigned"]
ID_COLUMNS = ["sprint_id", "project", "team", "team_key", "iteration", "start", "end", "cutoff", "item_id"]
FEATURE_FRAME_COLUMNS = ID_COLUMNS + DAY1_FEATURES + ["y"]
CHECKPOINT_FRAME_COLUMNS = (
    ID_COLUMNS + ["checkpoint", "t", "state_category_at_commit", "points_at_commit"] + FEATURES + ["y"]
)
FEATURE_VERSION = 2  # stored in the model bundle; bump when FEATURES or their meaning change
```

`checkpoints.py` imports only from `cache`, `sprints` and `timeline`, so importing it here makes no cycle.

Append after `build_features`:

```python
def build_checkpoint_features(rows: pd.DataFrame, day1: pd.DataFrame, history_sprints: pd.DataFrame) -> pd.DataFrame:
    """Model features of open rows (checkpoints.open_rows or build_checkpoint_rows): the item as of each row's t,
    its sprint's plan and its assignee's load at the cutoff (from `day1`, the day-1 frame of the same sprints; NaN
    for items added after the cutoff), and the sprint's progress at t. Row order follows `rows`."""
    rows = rows.reset_index(drop=True)
    if rows.empty:
        return pd.DataFrame(columns=CHECKPOINT_FRAME_COLUMNS)
    targets = rows.drop_duplicates("sprint_id")[["sprint_id", "team_key", "start"]]
    hist = team_history(targets, history_sprints).set_index("sprint_id")
    sid, t, cutoff = rows["sprint_id"], rows["t"], rows["cutoff"]
    velocity = sid.map(hist["trailing_velocity"])
    velocity = velocity.where(velocity > 0)
    out = rows[ID_COLUMNS].copy()
    out["checkpoint"] = rows["checkpoint"].astype("float64") if "checkpoint" in rows else np.nan
    out["t"] = t
    out["state_category_at_commit"] = rows["state_category_at_commit"]
    out["points_at_commit"] = rows["points_at_commit"].astype("float64")
    out["type"] = rows["type"].fillna(MISSING).astype(str)
    out["state_category"] = rows["state_category"].fillna(MISSING).astype(str)
    out["points"] = rows["points"].astype(float)
    out["points_rel"] = out["points"] / velocity
    out["is_unestimated"] = rows["is_unestimated"].astype(float)
    out["carryover_count"] = rows["carryover_count"].astype(float)
    out["age_days"] = (t - rows["created"]) / DAY
    out["days_since_change"] = (t - rows["last_changed"]) / DAY
    out["revisions_so_far"] = rows["revisions_so_far"].astype(float)
    out["has_parent"] = rows["parent_id"].notna().astype(float)
    out["is_unassigned"] = rows["assigned_to_sk"].isna().astype(float)
    plan = day1.drop_duplicates("sprint_id").set_index("sprint_id")
    for col in SPRINT_FEATURES:
        out[col] = sid.map(plan[col]).astype(float)
    load = day1.assign(item_id=day1["item_id"].astype("int64")).set_index(["sprint_id", "item_id"])
    keys = pd.MultiIndex.from_arrays([sid, rows["item_id"].astype("int64")])
    out["assignee_load_ratio"] = load["assignee_load_ratio"].reindex(keys).to_numpy(dtype=float)
    span = rows["end"] - cutoff
    elapsed = (t - cutoff) / span.where(span > pd.Timedelta(0))
    out["elapsed"] = elapsed.clip(0.0, 1.0).fillna((t >= cutoff).astype(float))
    out["days_left"] = ((rows["end"] - t.where(t > cutoff, cutoff)) / DAY).clip(lower=0.0)
    committed = rows["committed_points"].astype(float)
    out["done_share"] = (rows["done_points_at_t"].astype(float) / committed.where(committed > 0)).fillna(0.0)
    out["days_in_state"] = (t - rows["state_changed"]) / DAY
    for col in ("n_state_changes", "is_added", "reassigned"):
        out[col] = rows[col].astype(float)
    out["y"] = rows["y"].astype("float64")
    return out[CHECKPOINT_FRAME_COLUMNS]


def build_checkpoint_frame(
    cache: CacheData,
    sd: SprintData,
    *,
    work_item_types,
    done_categories,
    checkpoints=CHECKPOINTS,
) -> pd.DataFrame:
    """Checkpoint features of every sprint in `sd` at each checkpoint: the frame model A trains and backtests on."""
    rows = build_checkpoint_rows(
        cache, sd, work_item_types=work_item_types, done_categories=done_categories, checkpoints=checkpoints,
    )
    return build_checkpoint_features(rows, build_features(sd), sd.sprints)
```

- [ ] **Step 4: Keep model A on the day-1 features until Task 3**

In `src/sprint_forecast/model.py`, replace line 16 (`from sprint_forecast.features import CATEGORICAL, FEATURES, MISSING, NUMERIC`) with:

```python
from sprint_forecast.features import DAY1_FEATURES as FEATURES, MISSING

CATEGORICAL = ["type", "state_category_at_commit"]  # interim until model A trains on checkpoint rows
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
```

- [ ] **Step 5: Run the feature tests**

Run: `.venv/Scripts/python -m pytest tests/test_features.py -q`
Expected: all pass.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass. The session fixtures now also build the checkpoint frames; the suite should stay under a minute. If it passes but takes over 60 s, report the time in the task summary.

- [ ] **Step 7: Commit and push**

```bash
git add src/sprint_forecast/features.py src/sprint_forecast/model.py tests/conftest.py tests/test_features.py tests/test_model.py
git commit -m "Add checkpoint features: each open item as of t, its sprint's day-1 plan and the sprint's progress at t."
git push origin main
```

---

### Task 3: Model A on checkpoint rows, the rollup on top of work done, and scoring at any t

**Files:**
- Modify: `src/sprint_forecast/model.py` (imports, `FEATURE_LABELS`, `PHRASES`, `_describe`, `train_item_model`'s `calib_frame`)
- Modify: `src/sprint_forecast/rollup.py` (`simulate_pct_done`, `fit_forecaster`, `forecast_sprint`)
- Rewrite: `src/sprint_forecast/forecast.py`
- Modify: `src/sprint_forecast/cli.py` (`_train`, `_backtest`, imports)
- Modify: `tests/test_model.py`, `tests/test_rollup.py`, `tests/test_backtest.py`, `tests/test_export.py` (the `bundle` fixture)
- Create: `tests/test_forecast.py`

**Interfaces:**
- Consumes: Task 1 `open_rows`, `done_points_at`, `iteration_group`; Task 2 `FEATURES`, `CATEGORICAL`, `NUMERIC`, `MISSING`, `FEATURE_VERSION`, `build_checkpoint_features`, `build_checkpoint_frame`, `CHECKPOINT_FRAME_COLUMNS`, and the `synth.ckpt` / `synth.progress` fixtures.
- Produces:
  - `model.ItemModel.calib_frame` columns `["sprint_id", "group", "p", "y"]`; `group` = `sprint_id + "@" + str(checkpoint)` (just `sprint_id` for a frame without `checkpoint`).
  - `rollup.simulate_pct_done(p, points, sigma, n_draws=N_DRAWS, seed=0, *, done_points=0.0, total_points=None) -> np.ndarray`: samples of `(done_points + sum(points * done)) / total_points`; `total_points` defaults to `sum(points)`; NaN samples when the total is not positive; the constant `done_points / total_points` when there are no items.
  - `rollup.forecast_sprint(fc, items, n_draws=N_DRAWS, seed=0, *, done_points=0.0, total_points=None) -> (p, samples)`: `p` for every row of `items` (checkpoint features); samples simulate only rows with `is_added == 0`, weighted by `points_at_commit`.
  - `forecast.ScoredSprint(sprint, items, summary, velocity, done_points, t, committed)`: `items` = open rows at t with checkpoint features plus `p`; `done_points` = committed points done at t; `t` = the instant scored (clamped to the sprint's end); `committed` = the committed items (`SprintData.items` rows, or `scope_at` items before the cutoff).
  - `forecast.score_iteration(fc, cache, history, *, iteration, t, work_item_types, done_categories, commit_grace_days, team=None) -> list[ScoredSprint]` (same signature as today; new meaning of `t`).
  - `forecast.OLD_MODEL` (the retrain message) and `forecast.check_bundle(bundle) -> None` (raises `ValueError(OLD_MODEL)` unless `bundle["feature_version"] == FEATURE_VERSION`).
  - Model bundles written by `train` carry `feature_version`, `n_rows` and `n_items`.

How a sprint is scored at t (spec, "Model and rollup" and "Scoring"): committed items done at t count as done; committed items still open are simulated with their model probabilities and their commit-time points; committed items no longer open count 0; items added after the cutoff get a probability but stay out of the sum. The share is over the sprint's committed points. From the sprint's end on, t is the end and the forecast is the observed outcome (a point mass). Before the cutoff, the committed scope is the iteration's open items at t, as today.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_forecast.py`:

```python
import pandas as pd
import pytest

from conftest import SYNTH_TYPES
from helpers import build_cache, iteration, rev, team
from sprint_forecast.features import FEATURE_VERSION
from sprint_forecast.forecast import check_bundle, score_iteration
from sprint_forecast.rollup import fit_forecaster
from sprint_forecast.sprints import build_sprints, scope_at

DONE = ["Completed"]
RED12 = "Alpha/Team Red|Alpha\\Sprint 12"  # synth16: 2024-06-10 05:00 to 2024-06-24 04:59:59.999 UTC
TYPES = ["User Story", "Bug"]
IT0, IT1 = "Alpha\\Sprint 0", "Alpha\\Sprint 1"
S0 = iteration(IT0, "2024-02-19T05:00:00.000Z", "2024-03-04T04:59:59.999Z")
S1 = iteration(IT1, "2024-03-04T05:00:00.000Z", "2024-03-18T04:59:59.999Z")  # cutoff 2024-03-05T05:00Z
RED = team("Team Red", ["Alpha\\Red"], [IT0, IT1])
PRE = "2024-03-01T00:00:00.000Z"
MID = pd.Timestamp("2024-03-11T00:00:00Z")


@pytest.fixture(scope="module")
def fc(synth16):
    return fit_forecaster(synth16.ckpt, seed=0)


def score(fc, cache, history, t, iteration="Alpha\\Sprint 12", types=SYNTH_TYPES):
    return score_iteration(fc, cache, history, iteration=iteration, t=t, work_item_types=types,
                           done_categories=DONE, commit_grace_days=1.0, team="Team Red")


def test_a_running_sprint_is_scored_at_t_on_top_of_the_points_done(fc, synth16):
    t = pd.Timestamp("2024-06-19T12:00:00Z")
    (sc,) = score(fc, synth16.cache, synth16.sprints, t)
    assert sc.t == t and sc.sprint["sprint_id"] == RED12
    committed = float(sc.sprint["committed_points"])
    assert sc.done_points > 0
    assert sc.summary["p10"] >= sc.done_points / committed
    assert (sc.items["sprint_id"] == RED12).all() and sc.items["p"].between(0, 1).all()
    still_committed = sc.items.loc[sc.items["is_added"] == 0, "item_id"]
    assert set(still_committed) <= set(sc.committed["item_id"])


def test_before_the_cutoff_the_scope_is_the_open_items_and_nothing_is_done(fc, synth16):
    t = pd.Timestamp("2024-06-10T12:00:00Z")
    (sc,) = score(fc, synth16.cache, synth16.sprints, t)
    _, items = scope_at(synth16.cache, synth16.sprints, iteration="Alpha\\Sprint 12", cutoff=t,
                        work_item_types=SYNTH_TYPES, done_categories=DONE, team="Team Red")
    assert sorted(sc.items["item_id"]) == sorted(items["item_id"])
    assert (sc.items["is_added"] == 0).all() and sc.done_points == 0.0 and sc.t == t
    assert sc.sprint["cutoff"] == pd.Timestamp("2024-06-11T05:00:00Z")


def test_an_ended_sprint_is_scored_at_its_end_with_the_observed_outcome(fc, synth16):
    (sc,) = score(fc, synth16.cache, synth16.sprints, pd.Timestamp("2024-06-29T00:00:00Z"))
    actual = synth16.sprints.sprints.set_index("sprint_id").loc[RED12]
    assert sc.t == actual["end"]
    assert sc.done_points == pytest.approx(actual["done_points"])
    assert sc.summary["expected"] == pytest.approx(actual["pct_done"])
    assert sc.summary["p10"] == sc.summary["p90"]
    assert (sc.items["p"] == 0.0).all()


def test_a_sprint_with_nothing_left_open_is_a_point_mass_at_its_done_share(fc, tmp_path):
    cache = build_cache(tmp_path, [
        rev(1, 1, PRE, iteration=IT1),
        rev(1, 2, "2024-03-08T00:00:00.000Z", iteration=IT1, state="Closed", state_category="Completed"),
        rev(2, 1, PRE, iteration=IT1, story_points=5.0),
        rev(2, 2, "2024-03-09T00:00:00.000Z", iteration="Alpha"),
    ], [S0, S1], [RED])
    history = build_sprints(cache, work_item_types=TYPES, done_categories=DONE)
    (sc,) = score(fc, cache, history, MID, iteration=IT1, types=TYPES)
    assert sc.items.empty and sc.done_points == 3.0
    assert sc.summary["p10"] == sc.summary["p90"] == pytest.approx(3 / 8)
    assert sorted(sc.committed["item_id"]) == [1, 2]


def test_an_unestimated_added_item_takes_the_teams_usual_size(fc, tmp_path):
    cache = build_cache(tmp_path, [
        rev(6, 1, "2024-02-20T00:00:00.000Z", iteration=IT0, story_points=5.0),  # the team's prior median size: 5
        rev(1, 1, PRE, iteration=IT1),
        rev(2, 1, PRE, story_points=None),
        rev(2, 2, "2024-03-07T00:00:00.000Z", iteration=IT1, story_points=None),
    ], [S0, S1], [RED])
    history = build_sprints(cache, work_item_types=TYPES, done_categories=DONE)
    (sc,) = score(fc, cache, history, MID, iteration=IT1, types=TYPES)
    added = sc.items.set_index("item_id").loc[2]
    assert added["is_added"] == 1.0 and added["is_unestimated"] == 1.0 and added["points"] == 5.0
    assert 0.0 < added["p"] < 1.0


def test_check_bundle_needs_the_current_feature_version():
    with pytest.raises(ValueError, match="older version; run `sprint-forecast train`"):
        check_bundle({"forecaster": None})
    with pytest.raises(ValueError, match="older version"):
        check_bundle({"feature_version": FEATURE_VERSION - 1})
    check_bundle({"feature_version": FEATURE_VERSION})
```

In `tests/test_rollup.py`, replace `test_forecaster_on_synth` with:

```python
def test_forecaster_on_synth(synth40):
    frame = synth40.ckpt
    prog = synth40.progress.set_index(["sprint_id", "checkpoint"])
    last = frame["start"].max()
    fc = fit_forecaster(frame[frame["end"] < last], seed=0)
    assert 0.0 <= fc.sigma <= 3.0
    target = frame[(frame["start"] == last) & (frame["checkpoint"] == 0.5)]
    rows = target[target["sprint_id"] == target["sprint_id"].iloc[0]]
    done, total = prog.loc[(rows["sprint_id"].iloc[0], 0.5), ["done_points", "committed_points"]]
    p, samples = forecast_sprint(fc, rows, n_draws=2000, seed=0, done_points=done, total_points=total)
    s = summarize(samples)
    assert done / total <= s["p10"] <= s["p50"] <= s["p90"] <= 1.0
    assert 0.0 <= s["p_full"] <= s["p_80"] <= 1.0
    assert len(p) == len(rows)
```

and append:

```python
def test_rollup_adds_the_points_already_done():
    all_done = simulate_pct_done(np.ones(3), np.array([1.0, 2.0, 3.0]), sigma=0.5, n_draws=2000, seed=0,
                                 done_points=4.0, total_points=10.0)
    assert all_done.mean() == pytest.approx(1.0, abs=1e-3)
    nothing_open = simulate_pct_done(np.array([]), np.array([]), 0.5, n_draws=10, done_points=3.0, total_points=8.0)
    assert np.all(nothing_open == 3 / 8)
    assert np.isnan(simulate_pct_done(np.array([0.5]), np.array([1.0]), 0.5, n_draws=10, total_points=0.0)).all()
```

In `tests/test_model.py`:
- line 7: `from sprint_forecast.features import FEATURES`
- add `ISOTONIC_MIN_ITEMS,` to the `from sprint_forecast.model import (...)` list
- replace every `synth40.frame` with `synth40.ckpt` (lines 27, 59, 67, 74, 145)
- replace `test_calibration_uses_last_sprints_and_platt_when_small` with:

```python
def test_calibration_uses_last_sprints_and_groups_by_checkpoint(trained):
    model, train, _ = trained
    assert model.calibration == ("isotonic" if len(model.calib_frame) >= ISOTONIC_MIN_ITEMS else "platt")
    fit_ids, cal_ids = split_by_time(train)
    assert len(cal_ids) == round(0.2 * (len(fit_ids) + len(cal_ids)))
    starts = train.drop_duplicates("sprint_id").set_index("sprint_id")["start"]
    assert starts[cal_ids].min() >= starts[fit_ids].max()
    cf = model.calib_frame
    assert list(cf.columns) == ["sprint_id", "group", "p", "y"]
    assert set(cf["sprint_id"]) == set(cal_ids) and cf["group"].nunique() > len(cal_ids)
    assert (cf["group"].str.split("@").str[0] == cf["sprint_id"]).all()
    assert cf["p"].between(P_CLIP, 1 - P_CLIP).all()
```

- in `test_unknown_category_and_empty_frame_predict`, change `state_category_at_commit="Resolved"` to `state_category="Resolved"`
- in the `test_describe_drivers_reads_as_a_sentence` parameter list, replace `("state_category_at_commit", "Proposed", "state at commit was Proposed"),` with:

```python
    ("state_category", "InProgress", "state is InProgress"),
    ("elapsed", 0.6, "60% of the sprint has passed"),
    ("days_left", 3.2, "3 days left"),
    ("done_share", 0.42, "42% of committed points done"),
    ("days_in_state", 6.0, "in the same state for 6 days"),
    ("n_state_changes", 1.0, "1 state change this sprint"),
    ("is_added", 1.0, "added after day 1"),
    ("reassigned", 1.0, "reassigned since day 1"),
```

- at the end of `test_describe_drivers_says_why_a_value_is_missing`, append:

```python
    added = values.copy()
    added["is_added"] = 1.0
    assert describe_drivers(contrib, added, k=1) == ["added after day 1"]
```

- in `test_missing_categories_train_and_predict`, change both `state_category_at_commit` to `state_category`.

In `tests/test_backtest.py`, add a module-level helper after the imports and use it in the three `run_backtest` calls (lines 56, 78, 85):

```python
def day1(synth):
    """Checkpoint rows at the cutoff: the frame the backtest scores until it learns checkpoints (Task 4)."""
    return synth.ckpt[synth.ckpt["checkpoint"] == 0.0]
```

e.g. `run_backtest(synth16.sprints, day1(synth16), min_history=8, retrain_every=4, n_draws=2000, seed=0)`.

In `tests/test_export.py`, replace the `bundle` fixture with:

```python
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
```

and add `from sprint_forecast.features import FEATURE_VERSION` to its imports.

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_forecast.py tests/test_rollup.py tests/test_model.py -q`
Expected: `test_forecast.py` fails to import `check_bundle`; `test_rollup.py` fails with `TypeError: forecast_sprint() got an unexpected keyword argument 'done_points'` and the same for `simulate_pct_done`; in `test_model.py`, the contribution columns are the day-1 features, `calib_frame` has no `group`, and `describe_drivers` raises `KeyError`, because model A still reads the day-1 features.

- [ ] **Step 3: Train model A on the checkpoint features (`model.py`)**

Replace the interim import from Task 2 (the `DAY1_FEATURES as FEATURES` line and the two lines defining `CATEGORICAL` and `NUMERIC`) with:

```python
from sprint_forecast.features import CATEGORICAL, FEATURES, MISSING, NUMERIC
```

Replace `FEATURE_LABELS` with:

```python
FEATURE_LABELS = {
    "type": "work item type",
    "state_category": "state",
    "points": "item size in points",
    "points_rel": "item size vs team velocity",
    "is_unestimated": "item is unestimated",
    "carryover_count": "sprints already carried over",
    "age_days": "item age in days",
    "days_since_change": "days since last change",
    "revisions_so_far": "number of edits so far",
    "has_parent": "has a parent item",
    "is_unassigned": "item is unassigned",
    "load_ratio": "sprint load vs team velocity",
    "n_items": "items committed to the sprint",
    "bug_share": "share of bugs in the sprint",
    "carryover_share": "share of carried-over items in the sprint",
    "unestimated_share": "share of unestimated items in the sprint",
    "team_trailing_completion": "team's recent completion rate",
    "sprint_length_days": "sprint length in days",
    "team_sprint_index": "team's number of earlier sprints",
    "assignee_load_ratio": "assignee load vs their recent delivery",
    "elapsed": "share of the sprint passed",
    "days_left": "days left in the sprint",
    "done_share": "share of committed points done",
    "days_in_state": "days in the current state",
    "n_state_changes": "state changes this sprint",
    "is_added": "added after day 1",
    "reassigned": "reassigned since day 1",
}
```

In `train_item_model`, change the `ItemModel(...)` line and the `calib_frame` block to:

```python
    model = ItemModel(lgbm, lr, None, "none", categories, pd.DataFrame(columns=["sprint_id", "group", "p", "y"]))
```

```python
    if len(cal):
        model.calib_frame = pd.DataFrame({
            "sprint_id": cal["sprint_id"].to_numpy(),
            "group": _shock_groups(cal),
            "p": predict_proba(model, cal),
            "y": cal["y"].astype(int).to_numpy(),
        })
```

and add above `train_item_model`:

```python
def _shock_groups(frame: pd.DataFrame) -> np.ndarray:
    """Rows that share one sprint shock: a sprint at one checkpoint."""
    sid = frame["sprint_id"].astype(str)
    if "checkpoint" not in frame:
        return sid.to_numpy()
    return (sid + "@" + frame["checkpoint"].astype(str)).to_numpy()
```

Replace the `PHRASES` dict with:

```python
PHRASES = {
    "type": lambda v: f"item is a {v}",
    "state_category": lambda v: f"state is {v}",
    "points": lambda v: f"item size is {v:g} point{'' if v == 1 else 's'}",
    "points_rel": lambda v: f"item size is {_times(v)} team velocity",
    "is_unestimated": lambda v: "item is unestimated" if v else "item is estimated",
    "carryover_count": lambda v: f"already carried over {_count(v, 'sprint')}",
    "age_days": lambda v: f"item is {_count(v, 'day')} old",
    "days_since_change": lambda v: f"no change in {_count(v, 'day')}",
    "revisions_so_far": lambda v: f"{_count(v, 'edit')} so far",
    "has_parent": lambda v: "item has a parent" if v else "item has no parent",
    "is_unassigned": lambda v: "item is unassigned" if v else "item is assigned",
    "load_ratio": lambda v: f"sprint load is {_times(v)} team velocity",
    "n_items": lambda v: f"{_count(v, 'item')} committed to the sprint",
    "bug_share": lambda v: f"{_pct(v)} of the sprint's items are bugs",
    "carryover_share": lambda v: f"{_pct(v)} of the sprint's items were carried over",
    "unestimated_share": lambda v: f"{_pct(v)} of the sprint's items are unestimated",
    "team_trailing_completion": lambda v: f"team recently finished {_pct(v)} of committed points",
    "sprint_length_days": lambda v: f"sprint is {_count(v, 'day')} long",
    "team_sprint_index": lambda v: f"team has {_count(v, 'earlier sprint')}",
    "assignee_load_ratio": lambda v: f"assignee load is {_times(v)} their recent delivery",
    "elapsed": lambda v: f"{_pct(v)} of the sprint has passed",
    "days_left": lambda v: f"{_count(v, 'day')} left",
    "done_share": lambda v: f"{_pct(v)} of committed points done",
    "days_in_state": lambda v: f"in the same state for {_count(v, 'day')}",
    "n_state_changes": lambda v: f"{_count(v, 'state change')} this sprint",
    "is_added": lambda v: "added after day 1" if v else "committed on day 1",
    "reassigned": lambda v: "reassigned since day 1" if v else "same assignee as on day 1",
}
```

Replace `_describe` with:

```python
def _describe(feature: str, values: pd.Series) -> str:
    if not _is_missing(values[feature]):
        return PHRASES[feature](values[feature])
    if feature == "assignee_load_ratio":
        if values.get("is_unassigned") == 1:
            return "item is unassigned"
        if values.get("is_added") == 1:
            return "added after day 1"
    return MISSING_PHRASES.get(feature, f"{FEATURE_LABELS[feature]} unknown")
```

- [ ] **Step 4: Roll up on top of the points already done (`rollup.py`)**

Replace `simulate_pct_done`:

```python
def simulate_pct_done(
    p: np.ndarray, points: np.ndarray, sigma: float, n_draws: int = N_DRAWS, seed: int = 0, *,
    done_points: float = 0.0, total_points: float | None = None,
) -> np.ndarray:
    """Samples of (done_points + sum(points * done)) / total_points, with one shared shock z ~ N(0, sigma^2) per
    draw; each item's mean done rate stays p (see shifted_logit). total_points defaults to sum(points). NaN when
    the total is not positive; the constant done_points / total_points when there are no items."""
    p = np.asarray(p, dtype=float)
    w = np.asarray(points, dtype=float)
    total = float(w.sum()) if total_points is None else float(total_points)
    if not total > 0:
        return np.full(n_draws, np.nan)
    if len(p) == 0:
        return np.full(n_draws, done_points / total)
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, sigma, size=(n_draws, 1)) if sigma > 0 else np.zeros((n_draws, 1))
    prob = expit(shifted_logit(p, sigma)[None, :] + z)
    done = rng.random((n_draws, len(p))) < prob
    return (done_points + done @ w) / total
```

In `fit_forecaster`, fit sigma per sprint and checkpoint:

```python
    sigma = fit_sigma(cf["p"].to_numpy(), cf["y"].to_numpy(), cf["group"].to_numpy()) if len(cf) else 0.0
```

Replace `forecast_sprint`:

```python
def forecast_sprint(
    fc: Forecaster, items: pd.DataFrame, n_draws: int = N_DRAWS, seed: int = 0, *,
    done_points: float = 0.0, total_points: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Item probabilities for every row of one sprint's checkpoint features, and samples of the share of its
    committed points done by the end: rows still committed (is_added == 0) are simulated with their commit-time
    points on top of `done_points`, out of `total_points` (default: those rows' points)."""
    p = predict_proba(fc.item_model, items)
    committed = (items["is_added"] == 0).to_numpy()
    w = items["points_at_commit"].to_numpy(dtype=float)[committed]
    samples = simulate_pct_done(
        p[committed], w, fc.sigma, n_draws, seed, done_points=done_points, total_points=total_points,
    )
    return p, samples
```

- [ ] **Step 5: Score an iteration at any t (`forecast.py`)**

Replace `src/sprint_forecast/forecast.py` with:

```python
"""Score an iteration at any time t with a trained forecaster (shared by `predict` and `export`)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sprint_forecast.cache import CacheData
from sprint_forecast.checkpoints import done_points_at, open_rows
from sprint_forecast.features import FEATURE_VERSION, build_checkpoint_features, build_features_for, team_history
from sprint_forecast.rollup import Forecaster, forecast_sprint, simulate_pct_done, summarize
from sprint_forecast.sprints import SprintData, iteration_group, scope_at, sprint_calendar
from sprint_forecast.timeline import to_utc

OLD_MODEL = "model was trained by an older version; run `sprint-forecast train`"


@dataclass
class ScoredSprint:
    sprint: pd.Series  # one summarize_sprints row of the committed scope; outcome columns empty while it runs
    items: pd.DataFrame  # open rows at t (checkpoint features) plus "p", each item's calibrated probability of done
    summary: dict[str, float]  # summarize() of the simulated share of committed points done by the sprint's end
    velocity: float  # team trailing velocity in points; NaN without history
    done_points: float  # committed points in the iteration and done at t; 0 before the cutoff
    t: pd.Timestamp  # the instant scored: the requested t, or the sprint's end once that has passed
    committed: pd.DataFrame  # the committed items (SprintData.items rows) the summary is a share of


def check_bundle(bundle: dict) -> None:
    """Raise ValueError unless `bundle` was trained on this version's features."""
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError(OLD_MODEL)


def score_iteration(
    fc: Forecaster,
    cache: CacheData,
    history: SprintData,
    *,
    iteration: str,
    t,
    work_item_types: list[str],
    done_categories: list[str],
    commit_grace_days: float,
    team: str | None = None,
) -> list[ScoredSprint]:
    """Forecast each team sprint of `iteration` as it stands at `t` (its end, once that has passed). Committed
    items done by t count as done, committed items still open are scored and simulated, and items added after the
    cutoff are scored but stay out of the summary. Before the cutoff the committed scope is the iteration's open
    items at t. Raises ValueError for an iteration no team runs, or a team that does not run it."""
    group = iteration_group(sprint_calendar(cache, commit_grace_days), iteration, team)
    cutoff, end = group["cutoff"].iloc[0], group["end"].iloc[0]
    t = min(to_utc(t), end)
    if t < cutoff:
        sprints, items = scope_at(
            cache, history, iteration=iteration, cutoff=t, work_item_types=work_item_types,
            done_categories=done_categories, commit_grace_days=commit_grace_days, team=team,
        )
        sprints, items = sprints.assign(cutoff=cutoff), items.assign(cutoff=cutoff)
    else:
        sprints = history.sprints[history.sprints["sprint_id"].isin(group["sprint_id"])]
        items = history.items[history.items["sprint_id"].isin(sprints["sprint_id"])]
    if items.empty:
        return []
    day1 = build_features_for(sprints, items, history.sprints, history.items)
    windows = group[group["sprint_id"].isin(sprints["sprint_id"])]
    rows = open_rows(
        cache, history, windows, pd.Series(t, index=windows.index),
        work_item_types=work_item_types, done_categories=done_categories,
    )
    feats = build_checkpoint_features(rows, day1, history.sprints)
    done = pd.Series(dtype="float64")
    if t >= cutoff:
        done = done_points_at(cache, items, pd.Series(t, index=sprints["sprint_id"].to_numpy()), done_categories)
    velocity = team_history(sprints, history.sprints).set_index("sprint_id")["trailing_velocity"]
    scored = []
    for _, sprint in sprints.iterrows():
        sid = sprint["sprint_id"]
        mine = feats[feats["sprint_id"] == sid]
        done_points = float(done.get(sid, 0.0))
        total = float(sprint["committed_points"])
        if t >= end:  # over: whatever is still open did not finish
            p = np.zeros(len(mine))
            samples = simulate_pct_done(p[:0], p[:0], fc.sigma, done_points=done_points, total_points=total)
        else:
            p, samples = forecast_sprint(fc, mine, seed=0, done_points=done_points, total_points=total)
        scored.append(ScoredSprint(
            sprint, mine.assign(p=p), summarize(samples), float(velocity.get(sid, math.nan)), done_points, t,
            items[items["sprint_id"] == sid],
        ))
    return scored
```

- [ ] **Step 6: Train and backtest on checkpoint rows (`cli.py`)**

In `src/sprint_forecast/cli.py`, replace `from sprint_forecast.features import build_features` with:

```python
from sprint_forecast.features import FEATURE_VERSION, build_checkpoint_frame
```

Add a helper after `_build`:

```python
def _checkpoint_frame(cache: CacheData, sd: SprintData, s: Settings) -> pd.DataFrame:
    return build_checkpoint_frame(cache, sd, work_item_types=s.work_item_types, done_categories=s.done_categories)
```

Replace the first two lines of `_backtest`'s body with (the backtest still scores the cutoff only; Task 4 adds the other checkpoints):

```python
    cache = _load_cache(workdir)
    sd = _build(cache, s)
    frame = _checkpoint_frame(cache, sd, s)
    result = run_backtest(sd, frame[frame["checkpoint"] == 0.0], **kwargs)
```

Replace `_train` with:

```python
def _train(workdir: Path, s: Settings) -> dict:
    cache = _load_cache(workdir)
    sd = _build(cache, s)
    frame = _checkpoint_frame(cache, sd, s)
    try:
        fc = fit_forecaster(frame, seed=0)
    except ValueError as e:
        raise click.ClickException(f"cannot train: {e}") from None
    known = frame[frame["y"].notna()]
    bundle = {
        "forecaster": fc,
        "feature_version": FEATURE_VERSION,
        "work_item_types": s.work_item_types,
        "done_categories": s.done_categories,
        "commit_grace_days": s.commit_grace_days,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_sprints": int(len(sd.sprints)),
        "n_items": int(len(known.drop_duplicates(["sprint_id", "item_id"]))),
        "n_rows": int(len(known)),
        "version": __version__,
    }
    path = workdir / MODEL_FILE
    joblib.dump(bundle, path)
    click.echo(
        f"Trained on {bundle['n_sprints']} sprints / {bundle['n_items']} items / {bundle['n_rows']} checkpoint rows "
        f"(calibration: {fc.item_model.calibration}, sprint shock sigma = {fc.sigma:.2f}) -> {path}"
    )
    per_checkpoint = known.groupby("checkpoint").size()
    click.echo("  rows per checkpoint: " + ", ".join(f"{f:.0%} {n}" for f, n in per_checkpoint.items()))
    return bundle
```

- [ ] **Step 7: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_forecast.py tests/test_rollup.py tests/test_model.py tests/test_backtest.py tests/test_export.py -q`
Expected: all pass.

- [ ] **Step 8: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass (the CLI demo still reports models a, c and team_mean; `train` now prints "checkpoint rows" and "rows per checkpoint").

- [ ] **Step 9: Commit and push**

```bash
git add src/sprint_forecast/model.py src/sprint_forecast/rollup.py src/sprint_forecast/forecast.py src/sprint_forecast/cli.py tests/test_forecast.py tests/test_model.py tests/test_rollup.py tests/test_backtest.py tests/test_export.py
git commit -m "Train model A on checkpoint rows and score a sprint at any time t on top of the committed points already done."
git push origin main
```

---

### Task 4: Backtest every checkpoint against the progress reference

**Files:**
- Modify: `src/sprint_forecast/backtest.py`
- Modify: `src/sprint_forecast/cli.py` (`_backtest`, the `backtest` command's `--model` mapping and docstring, imports)
- Modify: `tests/test_backtest.py`, `tests/test_cli.py` (`test_demo_runs_end_to_end`, `test_backtest_writes_csv`)

**Interfaces:**
- Consumes: Task 1 `CHECKPOINTS`, `checkpoint_progress` (`PROGRESS_COLUMNS`); Task 2 checkpoint frame (`checkpoint`, `is_added`, `points_at_commit`, `team_trailing_completion`, `y`); Task 3 `forecast_sprint(..., done_points=, total_points=)`, `simulate_pct_done(..., done_points=, total_points=)`, the `_checkpoint_frame` helper in `cli.py`.
- Produces:
  - `backtest.SPRINT_MODELS = ("a", "progress", "c", "team_mean")`.
  - `backtest.ITEM_ROW_COLUMNS = ["sprint_id", "team_key", "checkpoint", "item_id", "is_added", "y", "p_a", "p_lr"]`.
  - `backtest.run_backtest(sd, frame, progress, *, models=SPRINT_MODELS, min_history=8, retrain_every=4, n_draws=N_DRAWS, seed=0, project=None, team=None, loto=True) -> BacktestResult`; raises `ValueError("the progress reference needs model a")` when `"progress"` is asked for without `"a"`.
  - `BacktestResult(sprint_rows, item_rows, summary, item_summary, calibration, calibration_pooled, loto, skipped)`: `sprint_rows` gain `checkpoint`; `summary` columns `["model", "checkpoint", "team_key", "n_sprints"] + METRICS`; `item_summary` columns `["model", "checkpoint", "scope", "n_items", "brier", "log_loss", "auc"]` with `scope` "committed" or "added"; `calibration` is model A at the cutoff, `calibration_pooled` over every checkpoint.
  - `format_report` lines "At the commit cutoff, model A {beats|does NOT beat} baseline C on CRPS (x vs y)." and, per checkpoint after the cutoff, "At 50% of the sprint, model A vs progress + day-1 odds: MAE x vs y, CRPS x vs y."
  - `backtest.csv` gains `checkpoint`; its rows at 0 keep today's meaning.

The models per checkpoint (spec, "Backtest"): `a` at every checkpoint; `progress` after the cutoff, which is the points done so far plus the cutoff probabilities of the committed items still open, with the same sigma (what a running sprint showed before this change); `c` and `team_mean` at the cutoff only. Training rows are the checkpoint rows of sprints that ended before the target started. Leave-one-team-out scores at the cutoff.

- [ ] **Step 1: Write the failing tests**

In `tests/test_backtest.py`:
- delete the `day1` helper added in Task 3;
- add `ITEM_ROW_COLUMNS,` to the `from sprint_forecast.backtest import (...)` list, and add `from sprint_forecast.checkpoints import CHECKPOINTS` below it;
- replace the `result` fixture and everything after it with:

```python
@pytest.fixture(scope="module")
def result(synth16):
    return run_backtest(
        synth16.sprints, synth16.ckpt, synth16.progress, min_history=8, retrain_every=4, n_draws=2000, seed=0,
    )


def test_backtest_has_no_leakage(result):
    a = result.sprint_rows[result.sprint_rows["model"] == "a"]
    assert len(a) > 0
    assert (a["train_end_max"] < a["start"]).all()


def test_backtest_metrics_are_finite(result):
    overall = result.summary[result.summary["team_key"] == "(all)"]
    assert set(zip(overall["model"], overall["checkpoint"])) == (
        {("a", f) for f in CHECKPOINTS} | {("progress", f) for f in CHECKPOINTS[1:]}
        | {("c", 0.0), ("team_mean", 0.0)}
    )
    assert np.isfinite(overall[METRICS].to_numpy(dtype=float)).all()
    assert set(result.summary["team_key"]) == {"(all)", "Alpha/Team Red", "Alpha/Team Blue", "Beta/Team Green"}
    s = result.item_summary
    assert set(s.loc[s["checkpoint"] == 0.0, "scope"]) == {"committed"}
    assert set(s.loc[s["checkpoint"] > 0, "scope"]) == {"committed", "added"}
    assert np.isfinite(s[["brier", "log_loss"]].to_numpy(dtype=float)).all()
    assert np.isfinite(s.loc[s["scope"] == "committed", "auc"].to_numpy(dtype=float)).all()
    assert result.calibration["n"].sum() == (result.item_rows["checkpoint"] == 0.0).sum()
    assert result.calibration_pooled["n"].sum() == len(result.item_rows)
    assert len(result.loto) == 3 and np.isfinite(result.loto[["crps", "mae"]].to_numpy(dtype=float)).all()
    text = format_report(result)
    assert "At the commit cutoff, model A" in text and "baseline C on CRPS" in text
    assert "At 75% of the sprint, model A vs progress + day-1 odds: MAE" in text
    assert "Leave-one-team-out" in text


def test_model_a_beats_the_progress_reference_late_in_the_sprint(result):
    overall = result.summary[result.summary["team_key"] == "(all)"].set_index(["model", "checkpoint"])
    assert overall.loc[("a", 0.75), "mae"] < overall.loc[("progress", 0.75), "mae"]


def test_later_checkpoints_start_from_the_points_done(synth16, result):
    prog = synth16.progress.set_index(["sprint_id", "checkpoint"])
    rows = result.sprint_rows[result.sprint_rows["checkpoint"] > 0]
    assert set(rows["model"]) == {"a", "progress"}
    floor = np.array([
        prog.loc[(r.sprint_id, r.checkpoint), "done_points"] / r.committed_points for r in rows.itertuples()
    ])
    assert (rows["p10"].to_numpy() >= floor - 1e-12).all()


def test_item_rows_carry_the_checkpoint_and_added_items(result):
    items = result.item_rows
    assert list(items.columns) == ITEM_ROW_COLUMNS
    assert (items.loc[items["checkpoint"] == 0.0, "is_added"] == 0).all()
    assert (items.loc[items["checkpoint"] > 0, "is_added"] == 1).any()
    n_committed = result.sprint_rows.drop_duplicates("sprint_id").set_index("sprint_id")["n_items"]
    at_cutoff = items[items["checkpoint"] == 0.0].groupby("sprint_id").size()
    assert (at_cutoff == n_committed[at_cutoff.index]).all()


def test_the_progress_reference_needs_model_a(synth16):
    with pytest.raises(ValueError, match="needs model a"):
        run_backtest(synth16.sprints, synth16.ckpt, synth16.progress, models=("progress",))


def test_backtest_model_filter_and_team_filter(synth16):
    res = run_backtest(
        synth16.sprints, synth16.ckpt, synth16.progress, models=("c", "team_mean"), team="Team Green", n_draws=500,
    )
    assert set(res.sprint_rows["model"]) == {"c", "team_mean"}
    assert set(res.sprint_rows["checkpoint"]) == {0.0}
    assert set(res.sprint_rows["team"]) == {"Team Green"}
    assert res.item_rows.empty and res.loto.empty


def test_backtest_with_no_targets(synth16):
    res = run_backtest(synth16.sprints, synth16.ckpt, synth16.progress, min_history=99, n_draws=100)
    assert res.sprint_rows.empty
    assert "No sprints qualified" in format_report(res)
```

In `tests/test_cli.py`, in `test_demo_runs_end_to_end` replace `assert set(rows["model"]) == {"a", "c", "team_mean"}` with:

```python
    assert set(rows["model"]) == {"a", "progress", "c", "team_mean"}
    assert set(rows["checkpoint"]) == {0.0, 0.25, 0.5, 0.75}
```

and in `test_backtest_writes_csv` append:

```python
    assert set(rows["checkpoint"]) == {0.0}
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_backtest.py tests/test_cli.py -q`
Expected: `test_backtest.py` fails to collect with `ImportError: cannot import name 'ITEM_ROW_COLUMNS' from 'sprint_forecast.backtest'`; in `test_cli.py`, `test_demo_runs_end_to_end` fails on the model set and `test_backtest_writes_csv` with `KeyError: 'checkpoint'`.

- [ ] **Step 3: Score each target at every checkpoint (`backtest.py`)**

Change the module docstring and imports to:

```python
"""Expanding-window backtest at each checkpoint of a sprint: model A (plus LR item metrics), the progress reference,
model C and the team-mean reference."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from sprint_forecast.baseline import team_mean_samples, velocity_bootstrap
from sprint_forecast.features import team_history
from sprint_forecast.model import P_CLIP, predict_proba, predict_proba_lr
from sprint_forecast.rollup import FULL_TOL, N_DRAWS, fit_forecaster, forecast_sprint, simulate_pct_done, summarize
from sprint_forecast.sprints import SprintData

SPRINT_MODELS = ("a", "progress", "c", "team_mean")
METRICS = ["crps", "pinball_10", "pinball_50", "pinball_90", "brier_full", "brier_80", "coverage", "mae"]
TARGET_COLUMNS = ["sprint_id", "project", "team", "team_key", "iteration", "start", "n_items", "committed_points"]
ITEM_ROW_COLUMNS = ["sprint_id", "team_key", "checkpoint", "item_id", "is_added", "y", "p_a", "p_lr"]
SUMMARY_COLUMNS = ["model", "checkpoint", "team_key", "n_sprints"] + METRICS
ITEM_SUMMARY_COLUMNS = ["model", "checkpoint", "scope", "n_items", "brier", "log_loss", "auc"]
CALIBRATION_COLUMNS = ["range", "n", "mean_p", "frac_done"]
LOTO_COLUMNS = ["team_key", "n_sprints", "crps", "mae", "coverage", "item_auc"]
```

Replace `BacktestResult` with:

```python
@dataclass
class BacktestResult:
    sprint_rows: pd.DataFrame  # one row per target sprint, model and checkpoint
    item_rows: pd.DataFrame  # ITEM_ROW_COLUMNS: model A's and LR's item probabilities at each checkpoint
    summary: pd.DataFrame  # SUMMARY_COLUMNS: mean metrics by model and checkpoint, overall ("(all)") and per team
    item_summary: pd.DataFrame  # ITEM_SUMMARY_COLUMNS: by checkpoint, committed and added items apart
    calibration: pd.DataFrame  # model A at the commit cutoff
    calibration_pooled: pd.DataFrame  # model A over every checkpoint
    loto: pd.DataFrame
    skipped: dict = field(default_factory=dict)
```

Replace `_item_block` and `_summaries` with:

```python
def _item_block(fc, items: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "sprint_id": items["sprint_id"].to_numpy(),
        "team_key": items["team_key"].to_numpy(),
        "checkpoint": items["checkpoint"].to_numpy(dtype=float),
        "item_id": items["item_id"].to_numpy(),
        "is_added": items["is_added"].to_numpy(dtype=int),
        "y": items["y"].to_numpy(),
        "p_a": p,
        "p_lr": predict_proba_lr(fc.item_model, items),
    })[ITEM_ROW_COLUMNS]


def _summaries(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    keys = ["model", "checkpoint"]
    overall = rows.groupby(keys)[METRICS].mean().assign(team_key="(all)")
    overall["n_sprints"] = rows.groupby(keys).size()
    per_team = rows.groupby(keys + ["team_key"])[METRICS].mean()
    per_team["n_sprints"] = rows.groupby(keys + ["team_key"]).size()
    out = pd.concat([overall.reset_index(), per_team.reset_index()], ignore_index=True)
    return out[SUMMARY_COLUMNS]
```

Add after `item_metrics`:

```python
def _item_summaries(item_rows: pd.DataFrame) -> pd.DataFrame:
    """Item metrics of model A and LR per checkpoint, for committed items and items added after the cutoff apart."""
    out = []
    for (f, added), block in item_rows.groupby(["checkpoint", "is_added"], sort=True):
        for name, col in (("a", "p_a"), ("lr", "p_lr")):
            out.append({
                "model": name, "checkpoint": f, "scope": "added" if added else "committed",
                **item_metrics(block["y"].to_numpy(), block[col].to_numpy()),
            })
    return pd.DataFrame(out, columns=ITEM_SUMMARY_COLUMNS)
```

In `calibration_table`, change the last line to `return table[CALIBRATION_COLUMNS]`.

Replace `leave_one_team_out` with:

```python
def leave_one_team_out(
    sprints: pd.DataFrame, frame: pd.DataFrame, held_out: list[str], n_draws: int, seed: int,
) -> pd.DataFrame:
    """Cross-team generalization check (not time-ordered): train on the other teams' checkpoint rows, predict the
    held-out team's sprints at the commit cutoff."""
    rows = []
    known = sprints[sprints["pct_done"].notna()]
    for team_key in held_out:
        try:
            fc = fit_forecaster(frame[frame["team_key"] != team_key], seed=seed)
        except ValueError:
            continue
        ys, ps, metrics = [], [], []
        for t in known[known["team_key"] == team_key].itertuples(index=False):
            items = frame[(frame["sprint_id"] == t.sprint_id) & (frame["checkpoint"] == 0.0)]
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
    return pd.DataFrame(rows, columns=LOTO_COLUMNS)
```

Replace `run_backtest` with:

```python
def run_backtest(
    sd: SprintData,
    frame: pd.DataFrame,
    progress: pd.DataFrame,
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
    """Score each target sprint at every checkpoint of `frame` (checkpoint rows; `progress` holds each sprint's
    committed points done by each checkpoint): model A at every checkpoint, the progress reference after the cutoff,
    model C and the team mean at the cutoff. Model A trains on the rows of sprints that ended before the target
    started."""
    if "progress" in models and "a" not in models:
        raise ValueError("the progress reference needs model a")
    frame = frame[frame["y"].notna()]
    done_by = progress.set_index(["sprint_id", "checkpoint"])["done_points"]
    checkpoints = sorted(frame["checkpoint"].unique())
    targets = select_targets(sd.sprints, min_history, project, team)
    sprint_rows, item_blocks = [], []
    skipped = {m: 0 for m in SPRINT_MODELS}
    fc, train = None, None
    for i, t in enumerate(targets.itertuples(index=False)):
        rows = frame[frame["sprint_id"] == t.sprint_id]
        day1 = rows[rows["checkpoint"] == 0.0]
        base = {c: getattr(t, c) for c in TARGET_COLUMNS}
        total = float(t.committed_points)
        if "a" in models:
            if i % max(1, retrain_every) == 0:
                train = frame[frame["end"] < t.start]
                try:
                    fc = fit_forecaster(train, seed=seed)
                except ValueError:
                    fc = None
            if fc is None:
                skipped["a"] += 1
                skipped["progress"] += int("progress" in models)
            else:
                check_no_leakage(train, t.start)
                p0 = pd.Series(predict_proba(fc.item_model, day1), index=day1["item_id"].to_numpy())
                for f in checkpoints:
                    items = rows[rows["checkpoint"] == f]
                    done = float(done_by.get((t.sprint_id, f), 0.0))
                    p, samples = forecast_sprint(fc, items, n_draws, seed, done_points=done, total_points=total)
                    sprint_rows.append({
                        **base, "checkpoint": f, "model": "a", **sprint_metrics(samples, t.pct_done),
                        "sigma": fc.sigma, "train_sprints": train["sprint_id"].nunique(),
                        "train_end_max": train["end"].max(),
                    })
                    if len(items):
                        item_blocks.append(_item_block(fc, items, p))
                    if "progress" in models and f > 0:
                        still = items[items["is_added"] == 0]
                        samples = simulate_pct_done(
                            still["item_id"].map(p0).to_numpy(dtype=float),
                            still["points_at_commit"].to_numpy(dtype=float),
                            fc.sigma, n_draws, seed, done_points=done, total_points=total,
                        )
                        sprint_rows.append({
                            **base, "checkpoint": f, "model": "progress", **sprint_metrics(samples, t.pct_done),
                        })
        if "c" in models:
            samples = velocity_bootstrap(sd.sprints, pd.Series(t._asdict()), n_draws, seed)
            if samples is None:
                skipped["c"] += 1
            else:
                sprint_rows.append({**base, "checkpoint": 0.0, "model": "c", **sprint_metrics(samples, t.pct_done)})
        if "team_mean" in models:
            samples = team_mean_samples(float(day1["team_trailing_completion"].iloc[0]))
            if samples is None:
                skipped["team_mean"] += 1
            else:
                sprint_rows.append({
                    **base, "checkpoint": 0.0, "model": "team_mean", **sprint_metrics(samples, t.pct_done),
                })
    rows = pd.DataFrame(sprint_rows)
    item_rows = pd.concat(item_blocks, ignore_index=True) if item_blocks else pd.DataFrame(columns=ITEM_ROW_COLUMNS)
    at_cutoff = item_rows[item_rows["checkpoint"] == 0.0]
    empty_cal = pd.DataFrame(columns=CALIBRATION_COLUMNS)
    calibration = calibration_table(at_cutoff["y"], at_cutoff["p_a"]) if len(at_cutoff) else empty_cal
    pooled = calibration_table(item_rows["y"], item_rows["p_a"]) if len(item_rows) else empty_cal
    held_out = sorted(targets["team_key"].unique())
    loto_rows = pd.DataFrame(columns=LOTO_COLUMNS)
    if loto and "a" in models and sd.sprints["team_key"].nunique() >= 2 and held_out:
        loto_rows = leave_one_team_out(sd.sprints, frame, held_out, n_draws, seed)
    return BacktestResult(
        rows, item_rows, _summaries(rows), _item_summaries(item_rows), calibration, pooled, loto_rows, skipped,
    )
```

Replace `format_report` with:

```python
def format_report(result: BacktestResult) -> str:
    """Plain-text report: sprint metrics by model and checkpoint (overall, then per team), the verdicts, item metrics
    by checkpoint, calibration at the cutoff and pooled, LOTO."""
    if result.sprint_rows.empty:
        return "No sprints qualified for the backtest (try a smaller --min-history)."

    def fmt(v):
        return f"{v:.3f}"

    lines = [
        "Sprint-level metrics by checkpoint (share of the way from the commit cutoff to the sprint's end; "
        "lower is better, except coverage: target 0.80)",
        result.summary.to_string(index=False, float_format=fmt),
    ]
    overall = result.summary[result.summary["team_key"] == "(all)"].set_index(["model", "checkpoint"])
    if ("a", 0.0) in overall.index and ("c", 0.0) in overall.index:
        a, c = overall.loc[("a", 0.0), "crps"], overall.loc[("c", 0.0), "crps"]
        verdict = "beats" if a < c else "does NOT beat"
        lines.append(f"\nAt the commit cutoff, model A {verdict} baseline C on CRPS ({a:.3f} vs {c:.3f}).")
    later = sorted(f for m, f in overall.index if m == "progress")
    if later:
        lines.append("")
    for f in later:
        a, p = overall.loc[("a", f)], overall.loc[("progress", f)]
        lines.append(
            f"At {f:.0%} of the sprint, model A vs progress + day-1 odds: "
            f"MAE {a['mae']:.3f} vs {p['mae']:.3f}, CRPS {a['crps']:.3f} vs {p['crps']:.3f}."
        )
    if len(result.item_summary):
        lines.append("\nItem-level metrics by checkpoint (committed items, and items added after the cutoff)")
        lines.append(result.item_summary.to_string(index=False, float_format=fmt))
        lines.append("\nCalibration of model A at the commit cutoff (10 bins)")
        lines.append(result.calibration.to_string(index=False, float_format=fmt))
        lines.append("\nCalibration of model A over every checkpoint (10 bins)")
        lines.append(result.calibration_pooled.to_string(index=False, float_format=fmt))
    if len(result.loto):
        lines.append("\nLeave-one-team-out at the commit cutoff (cross-team generalization check, not time-ordered)")
        lines.append(result.loto.to_string(index=False, float_format=fmt))
    if any(result.skipped.values()):
        lines.append(f"\nSkipped targets per model: {result.skipped}")
    return "\n".join(lines)
```

- [ ] **Step 4: Pass the checkpoint rows and progress from the CLI (`cli.py`)**

Add the import:

```python
from sprint_forecast.checkpoints import checkpoint_progress
```

Replace the four body lines of `_backtest` that Task 3 wrote (through `result = run_backtest(...)`) with:

```python
    cache = _load_cache(workdir)
    sd = _build(cache, s)
    progress = checkpoint_progress(cache, sd, done_categories=s.done_categories)
    result = run_backtest(sd, _checkpoint_frame(cache, sd, s), progress, **kwargs)
```

In the `backtest` command, change the docstring and the model mapping to:

```python
    """Expanding-window backtest at each checkpoint of a sprint; writes .sprint-forecast/backtest.csv."""
    models = {"a": ("a", "progress", "team_mean"), "c": ("c", "team_mean"), "all": SPRINT_MODELS}[model_choice]
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_backtest.py tests/test_cli.py -q`
Expected: all pass.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass, in under a minute.

- [ ] **Step 7: Commit and push**

```bash
git add src/sprint_forecast/backtest.py src/sprint_forecast/cli.py tests/test_backtest.py tests/test_cli.py
git commit -m "Backtest every checkpoint: model A at each one, the progress reference after the cutoff, C and the team mean at the cutoff."
git push origin main
```

---

### Task 5: Export and predict score running sprints now, with the day-1 forecast beside them

**Files:**
- Modify: `src/sprint_forecast/export.py` (column lists, `_item_rows`, `_sprint_row`, `export_forecasts`; new `_item_frame`)
- Modify: `src/sprint_forecast/cli.py` (`_load_model`, `_predict`, the `predict` command's `--as-of`, the data report's added-items line)
- Modify: `tests/test_export.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Task 3 `ScoredSprint.done_points`, `.t`, `.committed`, `.items["is_added"]`, `.items["points_at_commit"]`; `forecast.check_bundle` and `OLD_MODEL`; `score_iteration` at any t.
- Produces:
  - `export.SPRINT_FORECAST_COLUMNS` = today's list + `["day1_expected", "day1_p10", "day1_p90"]` (appended, so existing Power BI column positions hold); `basis` is always "now"; `scored_as_of` is the run time.
  - `export.ITEM_FORECAST_COLUMNS` = today's list + `["is_added"]`; per sprint: every open row at the run (`p_done` from the model, risk factors), then every committed item no longer open (`p_done` 1 when done, 0 when out of the sprint, no risk factors). `points` is the commit-time size for committed items (they sum to `committed_points`) and the current size for added ones; `state_category_at_commit` is empty for added items.
  - `export_forecasts(...)` raises `ValueError(OLD_MODEL)` for a bundle without `feature_version == FEATURE_VERSION`, before writing anything.
  - CLI: `predict --as-of now|commit` (default `now`); `_predict(workdir, *, iteration, team, as_of, top, title_lookup, now=None)` keeps its signature; `predict` and `export` fail with the retrain message on an old bundle.

- [ ] **Step 1: Write the failing tests**

In `tests/test_export.py`:
- change `from helpers import build_cache, rev` to `from helpers import build_cache, iteration, rev, team`;
- in `test_export_forecasts_running_and_upcoming_sprints`, replace the two `basis` lines and the `scored_as_of` line with:

```python
    assert (sprints["basis"] == "now").all()
    day1 = ["day1_expected", "day1_p10", "day1_p90"]
    assert sprints.loc[sprints["status"] == "running", day1].notna().all().all()
    assert sprints.loc[sprints["status"] == "upcoming", day1].isna().all().all()
    assert (sprints["day1_p10"].dropna() <= sprints["day1_p90"].dropna()).all()
```

  and, after the line that sets `red`:

```python
    assert red["scored_as_of"] == "2024-06-12T12:00:00Z"
```

- replace `test_export_item_rows_match_their_sprint` with:

```python
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
```

- add after `test_export_item_rows_match_their_sprint`:

```python
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
```

In `tests/test_cli.py`:
- add `import shutil` and `import joblib` to the imports;
- in `test_predict_past_sprint_as_of_commit_with_titles`, change the `predict` call to
  `run(workspace, "predict", "--iteration", "Alpha\\Sprint 12", "--team", "Team Blue", "--as-of", "commit", "--top", "2")`;
- in `test_export_lists_each_forecast`, change the expected text to `"Alpha/Team Red | Alpha\Sprint 12: running, scored as of now"`;
- replace `test_predict_auto_uses_now_before_cutoff` with:

```python
def _predict_output(workspace, capsys, now, as_of="now", team="Team Red", top=0):
    results = cli._predict(
        workspace / ".sprint-forecast", iteration="Alpha\\Sprint 12", team=team, as_of=as_of, top=top,
        title_lookup=None, now=pd.Timestamp(now),
    )
    return results, capsys.readouterr().out


def test_predict_before_the_cutoff_scores_the_scope_now(workspace, capsys):
    results, out = _predict_output(workspace, capsys, "2024-06-10T12:00:00Z")  # cutoff: 2024-06-11 05:00 UTC
    assert "scored as of now (2024-06-10 12:00 UTC)" in out
    assert "done so far" not in out and "day-1 forecast" not in out
    assert len(results) == 1 and 0.0 <= results[0]["p_full"] <= results[0]["p_80"] <= 1.0


def test_predict_mid_sprint_shows_done_so_far_and_the_day1_forecast(workspace, capsys):
    results, out = _predict_output(workspace, capsys, "2024-06-14T12:00:00Z")
    assert "scored as of now (2024-06-14 12:00 UTC)" in out
    assert "done so far:" in out and "day-1 forecast: expected" in out
    _, commit = _predict_output(workspace, capsys, "2024-06-14T12:00:00Z", as_of="commit")
    assert "scored as of commit cutoff (2024-06-11 05:00 UTC)" in commit and "day-1 forecast" not in commit


def test_predict_marks_added_items(workspace, capsys):
    # seed 5: item 1326 joined Team Blue's Sprint 12 on 2024-06-13 and stays open past the sprint's end
    _, out = _predict_output(workspace, capsys, "2024-06-14T12:00:00Z", team="Team Blue", top=100)
    line = next(text for text in out.splitlines() if "#1326  " in text)
    assert line.rstrip().endswith("pts  added")


def test_predict_after_the_sprint_scores_it_at_its_end(workspace, capsys):
    results, out = _predict_output(workspace, capsys, "2024-07-01T00:00:00Z")
    assert "scored as of sprint end (2024-06-24 04:59 UTC)" in out
    assert results[0]["p10"] == results[0]["p90"]


def test_predict_and_export_with_an_old_model_ask_to_retrain(workspace, tmp_path):
    workdir = tmp_path / ".sprint-forecast"
    workdir.mkdir()
    shutil.copy(workspace / ".sprint-forecast" / "cache.db", workdir / "cache.db")
    bundle = joblib.load(workspace / ".sprint-forecast" / "model.joblib")
    bundle.pop("feature_version")
    joblib.dump(bundle, workdir / "model.joblib")
    for args in (("predict", "--iteration", "Alpha\\Sprint 12", "--no-titles"), ("export", "--out", str(tmp_path))):
        result = CliRunner().invoke(cli.main, ["--root", str(tmp_path), *args])
        assert result.exit_code != 0 and "older version; run `sprint-forecast train`" in result.output
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/Scripts/python -m pytest tests/test_export.py tests/test_cli.py -q`
Expected: in `test_export.py`, failures on the new columns (`KeyError: 'is_added'`, `day1_expected` missing), the basis ("commit cutoff" where "now" is expected), `scored_as_of`, the nothing-left-open sprint (no closed rows yet) and the old model (no error raised). In `test_cli.py`, `test_predict_mid_sprint_...`, `test_predict_marks_added_items`, `test_predict_after_the_sprint_...` and `test_predict_and_export_with_an_old_model_...` fail on the missing text or exit code; `test_predict_before_the_cutoff_scores_the_scope_now` already passes (it guards today's before-cutoff behavior, which does not change).

- [ ] **Step 3: Export running sprints as of now (`export.py`)**

Change the imports:

```python
from sprint_forecast.forecast import ScoredSprint, check_bundle, score_iteration
```

Append the new columns to the two lists:

```python
SPRINT_FORECAST_COLUMNS = [
    "run_id", "run_at", "forecast_key", "sprint_id", "project", "team", "team_key", "iteration",
    "start", "end", "cutoff", "status", "basis", "scored_as_of", "elapsed_share",
    "n_items", "committed_points", "n_unestimated", "trailing_velocity", "load_ratio",
    "p_full", "p_80", "expected", "p10", "p50", "p90",
    "items_done_so_far", "points_done_so_far", "pct_done_so_far", "data_as_of", "model_trained_at",
    "day1_expected", "day1_p10", "day1_p90",
]
```

```python
ITEM_FORECAST_COLUMNS = [
    "run_id", "forecast_key", "sprint_id", "item_id", "type", "state_category_at_commit", "points",
    "is_unestimated", "carryover_count", "p_done", *RISK_FACTOR_COLUMNS,
    "state_category_now", "in_sprint_now", "done_now", "is_added",
]
```

Replace `_item_rows` with:

```python
def _item_frame(items, *, points, p_done, is_added, factors, progress, run_id, key) -> pd.DataFrame:
    frame = pd.DataFrame({
        "run_id": run_id,
        "forecast_key": key,
        "sprint_id": items["sprint_id"],
        "item_id": items["item_id"].astype("int64"),
        "type": items["type"],
        "state_category_at_commit": items["state_category_at_commit"].where(~is_added),
        "points": points,
        "is_unestimated": items["is_unestimated"].astype(bool),
        "carryover_count": items["carryover_count"].astype("int64"),
        "p_done": p_done,
        "is_added": is_added,
    }, index=items.index)
    for k, col in enumerate(RISK_FACTOR_COLUMNS):
        frame[col] = [f[k] if k < len(f) else None for f in factors]
    return frame.join(progress)[ITEM_FORECAST_COLUMNS]


def _item_rows(fc, scored: ScoredSprint, cache: CacheData, done_categories, run_id: str, key: str) -> pd.DataFrame:
    """Every open row at scored.t with its p_done and risk factors, then every committed item no longer open: done
    (p_done 1) or out of the sprint (p_done 0), without risk factors."""
    rows = scored.items
    still_open = set(rows.loc[rows["is_added"] == 0, "item_id"].astype("int64"))
    closed = scored.committed[~scored.committed["item_id"].astype("int64").isin(still_open)]
    parts = []
    if len(rows):
        added = rows["is_added"] == 1
        contrib = contributions(fc.item_model, rows)
        parts.append(_item_frame(
            rows,
            points=rows["points_at_commit"].where(~added, rows["points"]).astype("float64"),
            p_done=rows["p"].astype("float64"),
            is_added=added,
            factors=[describe_drivers(contrib.loc[i], rows.loc[i], k=N_RISK_FACTORS) for i in rows.index],
            progress=progress_at(cache, rows, scored.t, done_categories),
            run_id=run_id, key=key,
        ))
    if len(closed):
        progress = progress_at(cache, closed, scored.t, done_categories)
        parts.append(_item_frame(
            closed,
            points=closed["points"].astype("float64"),
            p_done=progress["done_now"].astype("float64"),
            is_added=pd.Series(False, index=closed.index),
            factors=[[] for _ in closed.index],
            progress=progress,
            run_id=run_id, key=key,
        ))
    if not parts:
        return pd.DataFrame(columns=ITEM_FORECAST_COLUMNS)
    return pd.concat(parts, ignore_index=True)
```

Replace `_sprint_row` with:

```python
def _sprint_row(scored: ScoredSprint, items: pd.DataFrame, *, run_id, now, key, status, day1, data_as_of, trained_at):
    s = scored.sprint
    committed = float(s["committed_points"])
    v = scored.velocity
    length = (s["end"] - s["start"]) / pd.Timedelta(days=1)
    elapsed = (now - s["start"]) / pd.Timedelta(days=1)
    return {
        "run_id": run_id, "run_at": now, "forecast_key": key,
        **{c: s[c] for c in ("sprint_id", "project", "team", "team_key", "iteration", "start", "end", "cutoff")},
        "status": status, "basis": "now", "scored_as_of": scored.t,
        "elapsed_share": float(np.clip(elapsed / length, 0.0, 1.0)) if length > 0 else math.nan,
        "n_items": int(s["n_items"]), "committed_points": committed, "n_unestimated": int(s["n_unestimated"]),
        "trailing_velocity": v, "load_ratio": committed / v if v > 0 else math.nan,
        **scored.summary,
        "items_done_so_far": int(items["done_now"].sum()), "points_done_so_far": scored.done_points,
        "pct_done_so_far": scored.done_points / committed if committed > 0 else math.nan,
        "data_as_of": data_as_of, "model_trained_at": trained_at,
        "day1_expected": day1["expected"] if day1 else math.nan,
        "day1_p10": day1["p10"] if day1 else math.nan,
        "day1_p90": day1["p90"] if day1 else math.nan,
    }
```

In `export_forecasts`, change the docstring's first sentence to "Forecast every running team sprint and each team's next sprint as they stand now, with the forecast at the commit cutoff beside each running sprint that has passed it.", add `check_bundle(bundle)` as the first line of the body (before `out = Path(out)`), and replace the `for iteration, group in chosen.groupby(...)` loop with:

```python
    for iteration, group in chosen.groupby("iteration", sort=False):
        status = group.set_index("sprint_id")["status"]
        cutoff = group["cutoff"].iloc[0]
        day1 = {}
        if now >= cutoff:
            day1 = {
                sc.sprint["sprint_id"]: sc.summary
                for sc in score_iteration(fc, cache, history, iteration=iteration, t=cutoff, **settings)
            }
        for scored in score_iteration(fc, cache, history, iteration=iteration, t=now, **settings):
            sid = scored.sprint["sprint_id"]
            if sid not in status.index:
                continue
            key = f"{run_id}|{sid}"
            items = _item_rows(fc, scored, cache, settings["done_categories"], run_id, key)
            item_frames.append(items)
            sprint_rows.append(_sprint_row(
                scored, items, run_id=run_id, now=now, key=key, status=status[sid], day1=day1.get(sid),
                data_as_of=cache.extracted_at.get(scored.sprint["project"], pd.NaT), trained_at=trained_at,
            ))
```

- [ ] **Step 4: Predict as of now by default (`cli.py`)**

Change the forecast import to:

```python
from sprint_forecast.forecast import check_bundle, score_iteration
```

Replace `_load_model` with:

```python
def _load_model(workdir: Path) -> dict:
    path = workdir / MODEL_FILE
    if not path.exists():
        raise click.ClickException(f"no model at {path}; run `sprint-forecast train` first")
    bundle = joblib.load(path)
    try:
        check_bundle(bundle)
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    return bundle
```

In `_predict`, replace the block from `cutoff = group["cutoff"].iloc[0]` through the end of the `except ValueError` clause with:

```python
    cutoff, end = group["cutoff"].iloc[0], group["end"].iloc[0]
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    t = cutoff if as_of == "commit" else now
    kwargs = dict(
        iteration=iteration, work_item_types=s.work_item_types, done_categories=s.done_categories,
        commit_grace_days=s.commit_grace_days, team=team,
    )
    try:
        scored = score_iteration(fc, cache, history, t=t, **kwargs)
        day1 = {}
        if as_of == "now" and now >= cutoff:
            day1 = {sc.sprint["sprint_id"]: sc.summary for sc in score_iteration(fc, cache, history, t=cutoff, **kwargs)}
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    label = "commit cutoff" if as_of == "commit" else ("sprint end" if now >= end else "now")
    t = min(t, end)
```

(the `No committed items in ... as of {label} ({t:...})` message below stays as it is). Then replace the body of the `for sc in scored:` loop, up to and including the `P(full)` echo, with:

```python
        srow, summary, v = sc.sprint, sc.summary, sc.velocity
        committed = float(srow["committed_points"])
        click.echo(f"\n{srow['team_key']} | {iteration}")
        click.echo(
            f"  window {srow['start']:%Y-%m-%d} .. {srow['end']:%Y-%m-%d} UTC; "
            f"scored as of {label} ({sc.t:%Y-%m-%d %H:%M} UTC)"
        )
        click.echo(
            f"  committed: {int(srow['n_items'])} items, {committed:.1f} points "
            f"({int(srow['n_unestimated'])} unestimated)"
        )
        if sc.t >= cutoff:
            share = sc.done_points / committed if committed > 0 else math.nan
            click.echo(f"  done so far: {sc.done_points:.1f} of {committed:.1f} points ({_pct(share)})")
        if v > 0:
            click.echo(f"  load: {committed / v:.2f}x trailing velocity ({v:.1f} points)")
        else:
            click.echo("  load: n/a (no velocity history for this team)")
        click.echo(
            f"  P(full) {_pct(summary['p_full'])}   P(>=80%) {_pct(summary['p_80'])}   "
            f"expected {_pct(summary['expected'])}   p10/p50/p90 "
            f"{_pct(summary['p10'])} / {_pct(summary['p50'])} / {_pct(summary['p90'])}"
        )
        d1 = day1.get(srow["sprint_id"])
        if d1:
            click.echo(
                f"  day-1 forecast: expected {_pct(d1['expected'])}, p10/p50/p90 "
                f"{_pct(d1['p10'])} / {_pct(d1['p50'])} / {_pct(d1['p90'])}"
            )
```

and in the riskiest-items loop, replace the item line with:

```python
                added = "  added" if r["is_added"] == 1 else ""
                click.echo(f"    #{int(r['item_id'])}  p={r['p']:.2f}  {r['points']:.1f} pts{added}  {title}".rstrip())
```

Change the `--as-of` option of the `predict` command to:

```python
@click.option(
    "--as-of", "as_of", type=click.Choice(["now", "commit"]), default="now", show_default=True,
    help="now: the sprint as it stands (done so far plus the forecast for what is still open; its end once over). "
         "commit: the day-1 view at the commit cutoff.",
)
```

In `_data`, change the added-items line to:

```python
        f"Items added mid-sprint (after the commit cutoff; scored per item, not part of committed scope): "
        f"{report['added_mid_sprint']}",
```

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/test_export.py tests/test_cli.py -q`
Expected: all pass.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass, in under a minute.

- [ ] **Step 7: Commit and push**

```bash
git add src/sprint_forecast/export.py src/sprint_forecast/cli.py tests/test_export.py tests/test_cli.py
git commit -m "Score running sprints as of now in export and predict, with done so far, the day-1 forecast beside it and added items marked."
git push origin main
```

---

### Task 6: Docs, version 0.2.0 and the private acceptance run

**Files:**
- Modify: `README.md` (intro, the `--as-of` paragraph, the Power BI section, "How it works")
- Modify: `pyproject.toml:7`, `src/sprint_forecast/__init__.py:3`, `tests/test_smoke.py:5`

**Interfaces:**
- Consumes: everything from Tasks 1-5; the Task 0 baseline in `$PRIVATE_ROOT`.
- Produces: version 0.2.0 (breaking: the model bundle and the export columns).

- [ ] **Step 1: Write the failing test**

In `tests/test_smoke.py`, change line 5 to:

```python
    assert sprint_forecast.__version__ == "0.2.0"
```

- [ ] **Step 2: Run it to see it fail**

Run: `.venv/Scripts/python -m pytest tests/test_smoke.py -q`
Expected: FAIL with `AssertionError: assert '0.1.0' == '0.2.0'`.

- [ ] **Step 3: Bump the version**

`pyproject.toml` line 7: `version = "0.2.0"`. `src/sprint_forecast/__init__.py` line 3: `__version__ = "0.2.0"`.

Run: `.venv/Scripts/python -m pip install -e ".[dev]" -q && .venv/Scripts/python -m pytest tests/test_smoke.py -q`
Expected: PASS (the reinstall refreshes the installed metadata that `sprint-forecast --version` reads).

- [ ] **Step 4: Update the README**

`README.md` has CRLF line endings in the working tree; keep them (edit in place, do not rewrite the file with LF).

Replace line 3 with:

```markdown
Forecast how much of an Azure DevOps sprint's **day-1 committed scope** a team will deliver, from its first day
to its last.
```

Replace the paragraph that starts "The output is a distribution" (lines 5-8) with:

```markdown
The output is a distribution over "% of committed points reaching Done": P(full scope), P(>=80%),
the expected %, and a p10/p50/p90 band. Before a sprint it is a planning-time sanity check ("is this sprint
overloaded?"). While it runs, the forecast starts from the points already done and scores every open item,
including items added after day 1, on where it stands now. It is also an honest ML exercise: every model is
backtested against a dumb velocity baseline, and losing to the baseline is reported as a result.
```

Replace the paragraph that starts "`--done-categories Resolved,Completed`" (lines 35-37) with:

```markdown
`--done-categories Resolved,Completed` (on `data`, `backtest` and `train`) widens what counts as done
when your process closes work in a Resolved state. `predict` scores the sprint as it stands now: the points
done so far, the forecast for the committed points still open, the day-1 forecast beside it, and the riskiest
open items, with items added after day 1 marked `added`. Once the sprint is over it reports the outcome.
`predict --as-of commit` shows the day-1 view at the commit cutoff. After upgrading from 0.1, run `train`
again: older model files are refused with a message saying so.
```

Replace the paragraph that starts "`sprint-forecast export` forecasts" (lines 41-42) with:

```markdown
`sprint-forecast export` forecasts every running team sprint as it stands at the run (the points done so far
plus the forecast for the committed items still open, with the forecast made at the commit cutoff in the
`day1_*` columns) and each team's next sprint (scored on the scope loaded so far), then writes:
```

In the table, replace the `sprint_forecasts`, `item_forecasts` and `backtest.csv` rows with:

```markdown
| `sprint_forecasts/<run_id>.csv` | run x team sprint | status, committed items and points, load vs velocity, P(full), P(>=80%), expected, p10/p50/p90 as of the run, share done so far, and the day-1 expected, p10 and p90 |
| `item_forecasts/<run_id>.csv` | run x item open in the sprint or committed to it | work item ID, points, probability of done (1 or 0 for committed items already done or out of the sprint), top 3 risk factors for open items, state now, done so far, added after day 1 or not |
```

```markdown
| `backtest.csv` | backtested sprint, checkpoint and model | predicted band vs actual at that point of the sprint, copied from the last `backtest` run |
```

Replace the paragraph that starts "Forecast files are added per run" (lines 54-55) with:

```markdown
Forecast files are added per run and never rewritten, so the history (how the forecast moved through the sprint,
next to the day-1 forecast) builds up; `sprints.csv`, `items.csv`, `cycle.csv`, `cycle_states.csv` and
`backtest.csv` are replaced.

Changed in 0.2.0: a running sprint used to be scored at its commit cutoff, and `item_forecasts` listed committed
items only. A report that rebuilt a projected finish from `points_done_so_far` plus the open items' `p_done`
should read `expected` instead, or keep the rebuild and filter `is_added = False`. A report on `backtest.csv`
that wants the day-1 view filters `checkpoint = 0`.
```

In "How it works", replace the **Features**, **Models** and **Backtest** bullets with:

```markdown
- **Features** describe an item open in the sprint at a time t: its type, size, state, days in that state,
  state changes this sprint, age, edits and carry-overs, whether it was added after day 1 or reassigned since,
  the sprint's day-1 plan (load vs velocity, bug, carry-over and unestimated shares, the team's recent
  completion, the assignee's load) and its progress at t (share of the sprint passed, days left, share of
  committed points done). Training uses t at the commit cutoff and at 25%, 50% and 75% of the way to the end;
  scoring uses any t. Everything reads revisions up to t and sprints that ended before the sprint started.
  Team identity is never a feature.
- **Models**: C, a velocity bootstrap baseline; a team-mean reference; A, a LightGBM item classifier trained
  on those checkpoint rows, with time-split calibration (plus a logistic-regression sanity check). At t, the
  sprint's distribution is the committed points done by t plus a Monte Carlo over the committed items still
  open, with a shared per-sprint shock whose size is fitted by maximum likelihood; the shock widens the spread
  without moving any item's calibrated probability. Items added after day 1 are scored but stay outside the
  committed scope the target measures.
- **Backtest**: expanding window, retrained every few sprints, with a leakage check. Each sprint is scored at
  the commit cutoff and at 25%, 50% and 75% of the way to the end: CRPS, pinball loss, Brier scores, coverage
  and MAE per model, checkpoint and team. Model A meets baseline C at the cutoff and, later on, a progress
  reference (points done so far plus the day-1 odds of the committed items still open). Item metrics are split
  by checkpoint and into committed and added items, and a leave-one-team-out check covers generalization.
```

Check: `git grep -n "scored at its commit cutoff, so the number stays" -- README.md` prints nothing, and `git diff --stat` shows only the files of this task so far.

- [ ] **Step 5: Private acceptance run (controller only; nothing is committed)**

Skip when Task 0 was skipped, and say so in the final report. Otherwise, on the owner's machine:

```bash
.venv/Scripts/sprint-forecast --root "$PRIVATE_ROOT" train
.venv/Scripts/sprint-forecast --root "$PRIVATE_ROOT" backtest > "$PRIVATE_ROOT/acceptance-after.txt"
```

Expected: both exit 0; `train` prints "checkpoint rows" and "rows per checkpoint". From `acceptance-after.txt`, against the Task 0 numbers:
- the `(all)` row of model `a` at checkpoint 0.000: CRPS within 0.005 of the baseline's model `a` CRPS;
- the item metrics row `a`, checkpoint 0.000, committed: log loss within 0.01 of the baseline's model `a` log loss;
- the lines "At 50% of the sprint, ..." and "At 75% of the sprint, ...": model A's MAE is below the progress reference's.

If a check fails, stop and report the numbers to the owner (in the conversation, never in the repo); do not tune the model inside this plan. Delete `acceptance-*.txt` and `acceptance-baseline-backtest.csv` from `$PRIVATE_ROOT` once the owner has seen the result.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q`
Expected: all pass, in under a minute.

- [ ] **Step 7: Commit and push**

```bash
git add README.md pyproject.toml src/sprint_forecast/__init__.py tests/test_smoke.py
git commit -m "Release 0.2.0: in-sprint checkpoint forecasts, with the README covering scoring as of now, the backtest checkpoints and the new export columns."
git push origin main
```

---

## Decisions on spec ambiguities

Rulings this plan makes where the spec is silent or loose. Each names what it costs if wrong.

1. **`is_added` is a model feature, and the frame also carries `points_at_commit`.** The spec lists `is_added` as a frame column next to `FEATURES`; here it lives in `FEATURES` (the progress group) only, so it is not duplicated, and `CHECKPOINT_FRAME_COLUMNS` adds `points_at_commit`, which the rollup needs to weigh committed items by their day-1 size. Cost if wrong: one column moves.
2. **`build_checkpoint_features(rows, day1, history_sprints)` takes a third argument.** `points_rel` divides by the team's trailing velocity, which needs sprint history the spec's two-argument signature does not pass. Cost if wrong: a signature change.
3. **From a sprint's end on, t is its end and the forecast is a point mass at the observed outcome.** The spec only says the forecast is "as of t"; scoring a finished sprint any other way would forecast the past. Cost if wrong: `predict` after the end shows the outcome instead of a band.
4. **Before the cutoff, commit-time fields are the values at t** (`state_category_at_commit` = `state_category`, `points_at_commit` = `points`, `is_added` = 0, `reassigned` = 0), so an upcoming sprint scores as today. Cost if wrong: none for the target; only feature values before the cutoff.
5. **`reassigned` for an item created after the cutoff compares its assignee at t with no assignee**, since it had none at the cutoff. Cost if wrong: added items that were assigned when created read as reassigned.
6. **An item moved to another team's area mid-sprint becomes that team's added item.** The first team keeps its day-1 committed points and counts the item 0 unless it is done in the iteration. Cost if wrong: two teams sharing an iteration see a moved item differently than the owner expects.
7. **Task 2 aliases the day-1 features into `model.py`** so the suite stays green between Tasks 2 and 3. Task 3 removes the alias. Cost if wrong: none after Task 3.
8. **`simulate_pct_done`'s and `forecast_sprint`'s new parameters are keyword-only**, so no positional caller changes meaning. Cost if wrong: none.
9. **`ScoredSprint` gains `t` and `committed`** besides the spec's `done_points`: export needs the instant scored and the committed items no longer open. Cost if wrong: two fields.
10. **The progress reference runs only after the cutoff, and `--model a` runs a, progress and the team mean.** At the cutoff the progress reference equals model A. Cost if wrong: one extra identical row per target.
11. **Leave-one-team-out trains on the other teams' rows at every checkpoint and scores at the cutoff.** The spec says it "stays at f = 0". Cost if wrong: LOTO numbers move slightly against 0.1.
12. **Export appends the new columns at the end of both files** (`day1_*` in `sprint_forecasts`, `is_added` in `item_forecasts`) so Power BI reports that address columns by position keep working. Cost if wrong: column order only.
13. **In `item_forecasts`, `points` is the commit-time size for committed items and the current size for added items**, so committed rows still sum to `committed_points`. Cost if wrong: re-estimated committed items show their day-1 size.
14. **Committed items no longer open get `p_done` = `done_now`** (1 when done in the sprint, 0 when out of it or removed) and no risk factors. Cost if wrong: none; this is the spec's rule written as a formula.
15. **A running sprint before its cutoff has empty `day1_*` columns**, like an upcoming one: its cutoff forecast does not exist yet. Cost if wrong: three empty cells for about a day per sprint.
16. **`predict --as-of` drops `auto`** (spec: `now` and `commit`). A script that passes `--as-of auto` gets a usage error. Cost if wrong: one choice to restore as an alias of `now`.
17. **After the sprint's end, `predict` labels the view "sprint end" and still prints the day-1 forecast** next to the outcome. Cost if wrong: one line of output.
