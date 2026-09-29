# In-sprint (checkpoint) forecasts design

## Goal

Forecast a running sprint from where it stands now, not from day 1. Today every running sprint is scored at its
commit cutoff, so its forecast never moves: an item still untouched on day 9 of 10 keeps its day-1 chance, and work
added after day 1 is never scored. This change replaces the day-1 item model with one **checkpoint model** that scores
any open item at any time t in the sprint, committed or added.

The sprint target is unchanged: the share of **day-1 committed points** done by the sprint's end.

## Evidence

A throwaway spike on a private dataset (266 ended sprints, 55 time blocks, expanding-window backtest, 3 seeds, block
bootstrap CIs) scored items at checkpoints 0, 25, 50 and 75% of the way from cutoff to end:

| Checkpoint | Log loss, committed items still open: day-1 model -> checkpoint model | Sprint MAE of expected %done: progress + day-1 odds -> checkpoint model |
|---|---|---|
| 0% | 0.567 -> 0.570 (within noise) | 0.197 -> 0.198 |
| 25% | 0.568 -> 0.555 | 0.192 -> 0.187 |
| 50% | 0.585 -> 0.547 | 0.182 -> 0.173 |
| 75% | 0.651 -> 0.574 | 0.167 -> 0.146 |

Gains at 50% and 75% were significant. Added items were 13-32% of open items mid-sprint; the model ranked them at AUC
0.70-0.73. Separate models per checkpoint scored about the same as this one pooled model, which is simpler and scores
any t. Comment activity and local sentiment (VADER, RoBERTa) added nothing, so this design reads no comments.

## Non-goals

- No comment, discussion or sentiment features.
- No change to the sprint target, baseline C, the team-mean reference, extraction, or the cache schema.
- Added work is scored per item but does not enter the committed-scope sprint distribution.

## Checkpoint rows (new `checkpoints.py`)

A checkpoint row is one (sprint, item, t) where the item is **open in the sprint at t**:

- its iteration as of t is the sprint's iteration,
- its type is a configured work-item type,
- its state category as of t is not a done category and not `Removed`,
- its team, by area as of t, is the sprint's team (`_Assigner`, as `sprints.py` does at the cutoff).

For t before the cutoff (upcoming sprints) the same rule applies at t.

`y` = the item is in the sprint's iteration **and** in a done category as of the sprint's end (NaN while the sprint
is open), exactly as for committed items today. An item moved out or removed after t has `y = 0`.

Training rows: every sprint in `SprintData.sprints` at each checkpoint in `CHECKPOINTS = (0.0, 0.25, 0.5, 0.75)`,
with `t = cutoff + f * (end - cutoff)`. Rows at f = 0 are the committed items (up to area-at-t team assignment).
Rows whose sprint is not in `SprintData.sprints` (no committed scope) are dropped.

Per row it also records `is_added` (the item is not in the sprint's committed scope) and the sprint's
`done_points_at_t`: points of committed items that are in the iteration and done as of t.

Interfaces:

```python
CHECKPOINTS = (0.0, 0.25, 0.5, 0.75)

def checkpoint_time(cutoff: pd.Series, end: pd.Series, f: float) -> pd.Series: ...

def open_rows(cache: CacheData, sd: SprintData, sprints: pd.DataFrame, t: pd.Series, *,
              work_item_types, done_categories) -> pd.DataFrame:
    """Open items of each sprint row at its own t (index-aligned with `sprints`), with state as of t,
    is_added, the committed scope's done_points_at_t, and y."""

def build_checkpoint_rows(cache: CacheData, sd: SprintData, *, work_item_types, done_categories,
                          checkpoints=CHECKPOINTS) -> pd.DataFrame:
    """open_rows for every sprint at every checkpoint, with a `checkpoint` column."""
```

`checkpoints.py` reuses `sprints.py` helpers (`_pairs`, `as_of_many`, `_Assigner`, `_carryover`, `raw_points`,
`impute_points`); they become public (no leading underscore) since two modules share them.

## Features (`features.py`)

One feature set for every checkpoint. Item features are recomputed at t; sprint context and assignee load stay at
the cutoff, because they describe the plan the sprint started with.

| Group | Features | As of |
|---|---|---|
| Item | `type`, `state_category`, `points`, `points_rel`, `is_unestimated`, `carryover_count`, `age_days`, `days_since_change`, `revisions_so_far`, `has_parent`, `is_unassigned` | t |
| Sprint | `load_ratio`, `n_items`, `bug_share`, `carryover_share`, `unestimated_share`, `team_trailing_completion`, `sprint_length_days`, `team_sprint_index` | cutoff (committed scope) |
| Assignee | `assignee_load_ratio` (NaN for added items) | cutoff |
| Progress | `elapsed`, `days_left`, `done_share`, `days_in_state`, `n_state_changes`, `is_added`, `reassigned` | t |

Definitions:

- `state_category` replaces `state_category_at_commit` as the model feature (they are equal at f = 0).
  `state_category_at_commit` stays an ID column for exports.
- `elapsed` = (t - cutoff) / (end - cutoff), clipped to [0, 1]. Before the cutoff it is 0.
- `days_left` = (end - max(t, cutoff)) in days.
- `done_share` = `done_points_at_t` / committed points of the sprint; 0 before the cutoff.
- `days_in_state` = days since the item's state last changed, as of t.
- `n_state_changes` = state changes in (start, t].
- `reassigned` = assignee as of t differs from assignee as of the cutoff (unassigned counts as a value); 0 before
  the cutoff.
- `points`, `points_rel`, `is_unestimated`, `carryover_count` use the existing definitions evaluated at t
  (`impute_points` against history, `carryover` counting iterations ended before start, revisions up to t).

`build_features_for` (the day-1 frame) stays: it supplies the sprint context and assignee load, and baseline C and
the team-mean reference still read it. A new `build_checkpoint_features(rows, day1_frame) -> DataFrame` returns
`ID_COLUMNS + ["checkpoint", "t", "is_added", "state_category_at_commit"] + FEATURES + ["y"]`.

`FEATURE_VERSION = 2` is stored in the model bundle.

## Model and rollup

- `model.py`: same recipe (LightGBM params, 80/20 time split by sprint start, isotonic at >= 1000 calibration rows
  else Platt, `P_CLIP`), trained on checkpoint rows with `y` known. `FEATURE_LABELS`, `PHRASES` and
  `MISSING_PHRASES` gain the new features, e.g. "in the same state for 6 days", "3 days left", "added after day 1",
  "reassigned since day 1", "42% of committed points done". `calib_frame` gains `group` = sprint_id + checkpoint.
- `rollup.py`: sigma is fitted on the calibration rows grouped by `group`. `simulate_pct_done` gains
  `done_points=0.0` and `total_points=None` (default: sum of `points`), and returns
  `(done_points + draws @ points) / total_points`. With the defaults it behaves as today.
- A sprint's distribution at t simulates its **committed items still open at t** with `done_points = done_points_at_t`
  and `total_points = committed_points`. Committed items that left the sprint count 0. Items done at t count as
  done; a later reopen or move-out is ignored (the spike did the same).

## Scoring (`forecast.py`)

`score_iteration(fc, cache, history, *, iteration, t, ...)` builds open rows of the iteration's team sprints at t,
scores every row, and returns per sprint:

- `items`: every open row with `p` and `is_added`;
- `summary`: the committed-scope distribution above;
- `done_points`, plus the existing `sprint` and `velocity`.

Before the cutoff the committed scope is the open items at t, so an upcoming sprint scores as it does today.

## Export (`export.py`)

- Running sprints are scored at **now** (`basis = "now"`), then again at the cutoff for the day-1 reference.
  Upcoming sprints are unchanged.
- `sprint_forecasts` gains `day1_expected`, `day1_p10`, `day1_p90` (the same model at the cutoff; empty for upcoming
  sprints). `p_full` .. `p90` and `expected` are now as of the run.
- `item_forecasts` gains `is_added`, and holds, per running sprint:
  - every open row at now with its `p_done` and risk factors;
  - every committed item already done (`p_done = 1`) or out of the sprint (`p_done = 0`), with no risk factors,
    so the file still lists the whole committed scope.
  `state_category_at_commit` is empty for added items.
- Consumers that rebuilt a projected finish from `points_done_so_far` plus open items' `p_done` should read
  `expected` instead, or keep the rebuild and filter `is_added = false`.
- A bundle whose `feature_version` is not 2 fails `export` and `predict` with "model was trained by an older version;
  run `sprint-forecast train`".

## Backtest (`backtest.py`)

- Same targets, training window and leakage check. Training rows are checkpoint rows of sprints with
  `end < target.start`.
- Each target is scored at every checkpoint. Sprint rows gain `checkpoint`; item rows gain `checkpoint` and
  `is_added`.
- Models per checkpoint:
  - `a` at every checkpoint;
  - `progress` at f > 0: done so far plus the f = 0 probabilities of the committed items still open, same sigma
    (what a running sprint showed before this change);
  - `c` and `team_mean` at f = 0 only.
- Report: sprint metrics by model and checkpoint, item metrics by checkpoint (committed and added separately), the
  calibration table at f = 0 and pooled, the "A vs C" verdict at f = 0, and one line per f > 0 comparing `a` with
  `progress` on MAE and CRPS. Leave-one-team-out stays at f = 0.
- `backtest.csv` gains `checkpoint`. Rows at 0 keep today's meaning, so a consumer filters `checkpoint = 0` to keep
  its current view.

## CLI

- `train`: fits on checkpoint rows and prints row counts per checkpoint.
- `predict --as-of`: choices `now` (default) and `commit` (the day-1 view). It prints done so far, the forecast as of
  t, and the day-1 forecast for running sprints. Riskiest items include added ones, marked "added".
- `backtest`: prints the per-checkpoint report.

## Testing

Hand-built timelines (`tests/helpers.py` style), one test per rule:

1. An item done before t is not a row; it counts in `done_points_at_t` when committed.
2. An item added after the cutoff and open at t is a row with `is_added = 1` and `y` from its end state.
3. A committed item moved out before t is not a row and counts 0; moved out after t, it is a row with `y = 0`.
4. `reassigned`, `days_in_state` and `n_state_changes` on a known revision history.
5. No leakage: appending a revision after t leaves every feature of that row unchanged.
6. At f = 0, item features of committed rows equal today's `build_features_for` values.
7. `simulate_pct_done` with `done_points`: all open p = 1 gives `total/total`; no open items gives a constant.
8. Synthetic data (`synth.py` already moves finishing items to Active early and leaves some Proposed): the backtest's
   `a` beats `progress` on MAE at f = 0.75, and every training sprint ended before its target started.
9. Export: a running sprint has `basis = "now"`, filled `day1_*` columns, added rows with `is_added`, done committed
   items with `p_done = 1`.
10. CLI: a bundle without `feature_version = 2` fails with the retrain message.

Acceptance on real data (run privately, not committed): at f = 0 the backtest's CRPS is within 0.005 and item log
loss within 0.01 of the pre-change backtest; at f = 0.5 and 0.75, `a` beats `progress` on MAE.

## Docs and release

- README: "How it works" gains the checkpoint model; the Power BI section lists the new and changed columns.
- Version 0.2.0 (breaking: model bundle and export columns).
