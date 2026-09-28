# sprint-forecast design

## Goal

Predict how much of a sprint's **day-1 committed scope** an Azure DevOps team will deliver. Output is a **distribution** over "% of committed points reaching Done", which gives:
- P(full scope), meaning 100%
- P(≥80%)
- the expected %
- a p10/p50/p90 band

The primary use is a **planning-time sanity check** ("is this sprint overloaded?"). It also has to be a fun, honest ML project: every model is backtested against a dumb baseline, and losing to the baseline is reported as a result.

ADO data is messy:
- items bounce between iterations
- items go unestimated
- iterations have no dates
- teams run no sprints
- "done" states vary by process

That's why a learned model might add value over velocity arithmetic. The pipeline surfaces the mess explicitly (the `data` report) instead of silently dropping it.

## Public-repo rules

This repo is public, so it contains code only:
- `.gitignore`: `.sprint-forecast/`, `*.db`, `*.sqlite`, `*.jsonl`, `*.parquet`, `*.joblib`, `.env`
- No real org, project, team, person, or work-item text in code, tests, docs, or examples. Tests and the demo use synthetic data from `synth.py`.
- Local cache and model artifacts live only in the gitignored `.sprint-forecast/` directory, so model files can contain real category values.
- Assignees are stored and used only as opaque Analytics `UserSK` GUIDs, and only as numeric load features.

## Architecture

```
ADO Analytics OData ──extract──▶ .sprint-forecast/cache.db (SQLite)
                                   │
                         timeline ─┤ as-of reconstruction per item
                          sprints ─┤ (project, team, iteration) → committed scope + outcome
                         features ─┤ day-1 item + sprint/team features (no leakage)
               baseline │ model   ─┤ C: velocity bootstrap   A: LightGBM + calibration
                          rollup  ─┤ sprint-shock Monte Carlo → %done distribution
                        backtest ─┘ expanding-window evaluation
```

Package `src/sprint_forecast/`. Each module has one job and plain-data interfaces (pandas DataFrames / dataclasses):

| Module | Responsibility |
|---|---|
| `config.py` | Load `.sprint-forecast/config.toml` (org URL, projects, auth method, work-item types, done categories). `ADO_PAT` env var overrides the PAT. |
| `analytics.py` | Minimal Analytics OData client. Auth is either PAT (Basic) or az-cli bearer token (`az account get-access-token --resource 499b84ac-…`). It pages by following `@odata.nextLink`, otherwise by `$skip` while pages are full. It retries on 429/5xx with backoff. |
| `extract.py` | Pulls revisions, iterations, and teams per project into `cache.db`, with incremental watermarks. It's the only module that talks to ADO. |
| `cache.py` | SQLite schema and read helpers for `cache.db`. |
| `timeline.py` | As-of lookup: an item's state at instant t is its last revision with `changed ≤ t`. |
| `sprints.py` | Builds sprints, team assignment, committed scope, and outcomes. |
| `features.py` | Feature matrix at the commit cutoff. |
| `baseline.py` | Model C. |
| `model.py` | Model A (LightGBM) plus a logistic-regression sanity model, and calibration. |
| `rollup.py` | Sprint-shock σ fit and Monte Carlo simulation. |
| `backtest.py` | Expanding-window backtest and metrics. |
| `synth.py` | Synthetic `cache.db` generator with planted effects. |
| `cli.py` | Click CLI. |

## Extraction (`extract.py`)

Queries run against `https://analytics.dev.azure.com/{org}/{project}/_odata/v4.0-preview/`:

- **WorkItemRevisions**, filtered to the configured types (default `User Story, Product Backlog Item, Requirement, Bug`) and `ChangedDate ge <watermark>`:
  - Selected fields: `WorkItemId, Revision, ChangedDate, RevisedDate, WorkItemType, State, StateCategory, StoryPoints, Effort, ParentWorkItemId, CreatedDate`
  - Expanded: `Iteration($select=IterationPath)`, `Area($select=AreaPath)`, `AssignedTo($select=UserSK)`
- **Iterations**: `IterationSK, IterationPath, IterationName, StartDate, EndDate, IsEnded`
- **Teams**: `TeamSK, TeamName`, expanding `Areas($select=AreaPath)` and `Iterations($select=IterationPath)`

Projects come from config (`["*"]` means REST `_apis/projects`).

How the data is stored and refreshed:
- Timestamps are normalized to UTC ISO-8601 (`…Z`). Iteration `EndDate` is 23:59:59.999 local on the last day, and it's kept as that instant.
- Revisions are upserted on `(id, rev)` and are immutable once written.
- The watermark is the max `changed` seen per project, queried with `ge`, so boundary rows re-arrive and are deduped.
- Iterations and teams are replaced per project on every run.
- A 401/403 from Analytics for one project is reported and skipped. It doesn't abort the run.

`cache.db` schema:
```sql
revisions(item_id, rev, project, changed, revised, created, type, state, state_category,
          iteration, area, assigned_to_sk, story_points, effort, parent_id, PRIMARY KEY(item_id, rev))
iterations(project, iteration_sk PRIMARY KEY, path, name, start_date, end_date, is_ended)
teams(project, team_sk PRIMARY KEY, name)
team_areas(team_sk, area_path)          team_iterations(team_sk, iteration_path)
meta(key PRIMARY KEY, value)            -- schema_version, per-project watermarks, last_extract
```

## Sprint reconstruction (`sprints.py`)

- **Sprint** = (project, team, iteration), for iterations with both dates.
  - Teams come from `team_iterations`: a team runs a sprint when it subscribes to that iteration.
  - If no team in a project subscribes to any dated iteration, the fallback is project-level sprints (team = `<project>`).
- **Item → team.** Among the teams subscribed to the item's iteration:
  - exact area match first
  - then the longest area-path prefix match
  - then the single subscribed team, if there's only one
  - otherwise the item is *unassigned*: counted in the data report and excluded
- **Commit cutoff** `C = start + commit_grace` (default 1 day), so items added on planning day count as committed.
- **Committed scope** is items as of C where:
  - `iteration == sprint iteration`
  - the type is in the configured types
  - `state_category ∉ done_categories ∪ {Removed}`
- **Outcome** at `E = end_date`: done ⇔ as of E, `iteration == sprint iteration` and `state_category ∈ done_categories`. The default done set is `{Completed}`, and `--done-categories Resolved,Completed` widens it.
  - Moved out of the sprint → not done.
  - Removed → not done.
  - Items added after C are excluded from committed scope. They're counted in the report as "added mid-sprint".
- **Points** = `story_points`, falling back to `effort`, as of C. Missing points are imputed with the team's trailing median item size (then the global median, then 1), and `is_unestimated` is set.
- **% done** = Σ points(done) / Σ points(committed). A count-based % is also reported.
- Sprints with zero committed items are dropped and counted in the report.

## Features (`features.py`)

Everything is computed as of C. Team history uses only sprints with `end < this sprint's start`, the same ordering the backtest uses.

- **Item**:
  - `type` (categorical)
  - `state_category_at_commit` (Proposed / InProgress / Resolved)
  - `points`, and `points_rel` = points ÷ team trailing velocity
  - `is_unestimated`
  - `carryover_count`: distinct earlier iterations the item was in, taken from revisions before C whose iteration ended before this start
  - `age_days` (C − created)
  - `days_since_change`
  - `revisions_so_far`
  - `has_parent`
- **Sprint/team**:
  - `load_ratio` = committed points ÷ trailing-3 mean velocity
  - `n_items`
  - `bug_share`
  - `carryover_share`
  - `unestimated_share`
  - `team_trailing_completion` (mean %done of the last 5)
  - `sprint_length_days`
  - `team_sprint_index`
- **Assignee**:
  - `assignee_load_ratio` = the assignee's committed points this sprint ÷ their trailing mean delivered points (last 3 sprints they appear in)
  - `is_unassigned`

Here velocity means delivered committed points. Trailing stats with no history are NaN. LightGBM handles NaN natively; logistic regression imputes the median and adds a missing flag. Team identity is never a feature, so the model generalizes to teams it hasn't seen.

## Models

- **C — velocity bootstrap (baseline)**:
  - %done = min(1, V / committed), where V is drawn from the team's last 8 delivered velocities.
  - With fewer than 3 prior team sprints, it draws completion ratios from the project's last 20 sprints instead.
- **Team mean (reference)**: a point mass at `team_trailing_completion`.
- **A — item model**:
  - `LGBMClassifier` (num_leaves 15, min_child_samples 20, learning_rate 0.03, n_estimators 300, subsample 0.8, colsample 0.8)
  - Calibration: split the training sprints by time, 80% to fit and the last 20% to calibrate. Use isotonic regression if the calibration set has ≥1000 items, otherwise Platt scaling.
  - Logistic regression runs in parallel as a sanity check.
  - Explanations use LightGBM `pred_contrib=True` (SHAP values), with no `shap` dependency.
- **Rollup**:
  - Items in a sprint aren't independent (bad weeks, outages), so there's a shared sprint shock z ~ N(0, σ²) on the logit scale.
  - σ is fitted by maximizing Σ_sprints log ∫ Π_i Bern(y_i | sigmoid(logit p_i + z)) N(z; 0, σ²) dz, using 32-node Gauss–Hermite quadrature, bounded to σ ∈ [0, 3], over out-of-sample predictions on the calibration sprints.
  - Simulation uses 10,000 draws per sprint and a fixed seed.

## Backtest (`backtest.py`)

- Sprints are ordered by start date. A target sprint needs ≥ `min_history` (default 8) prior sprints for its team.
- For each target, the model is trained on items from **all** teams' sprints with `end < target.start`, and retrained every `retrain_every` (default 4) targets. A leakage assertion checks every training sprint ended before the target started.
- Metrics:
  - **Item** (A, LR): Brier, log loss, ROC AUC, and a 10-bin calibration table.
  - **Sprint** (A, C, team mean):
    - CRPS, from samples: E|X−y| − ½E|X−X′|
    - pinball loss at 0.1, 0.5 and 0.9
    - Brier score for P(full) and for P(≥80%)
    - p10–p90 coverage
    - MAE of the mean
- Reported overall and per team. With ≥2 teams there's also **leave-one-team-out**: train on the other teams and predict all of the held-out team's sprints. It's labeled as a cross-team generalization check, since it isn't time-ordered.
- Output: the table is printed and per-sprint rows are written to `.sprint-forecast/backtest.csv`.

## CLI

```
sprint-forecast init --org URL --projects "A,B" | "*" [--auth pat|az-cli]
sprint-forecast extract [--project P ...] [--full]
sprint-forecast data                     # data-quality report
sprint-forecast backtest [--project P] [--team T] [--model a|c|all] [--min-history N] [--retrain-every K]
sprint-forecast train                    # → .sprint-forecast/model.joblib
sprint-forecast predict --iteration PATH [--team T] [--as-of now|commit]
sprint-forecast demo                     # synthetic data, full pipeline, no ADO needed
```

`data` reports:
- projects, teams (running sprints or not), and dated vs undated iterations
- sprints reconstructed and their committed items
- unassigned items and unestimated share
- items added mid-sprint
- the done-state mix at end (what share of committed items end in Resolved rather than Completed)

`predict`:
- It scores the iteration's contents as of the **commit cutoff** when the sprint has passed it, otherwise as of **now**, which is the planning-time check. `--as-of now` forces current contents.
- Output:
  - committed items and points (unestimated count)
  - load vs velocity
  - P(full), P(≥80%), expected %, and the p10/p50/p90 band
  - the riskiest items (ID, truncated title, p, points), each with its top two negative drivers from SHAP contributions, labeled in plain words

Titles are fetched on demand for display and never cached.

## Synthetic data (`synth.py`)

It generates a `cache.db` with 2 projects and 3 teams (plus one team that runs no sprints), with about 40 two-week sprints per team.

Each item has a latent completion probability:
```
logit p = b0 − 0.7·carryover − 1.3·(load_ratio − 1) − 0.35·log(points) + team_effect + sprint_shock
```

Unfinished items carry over to the next sprint (or are occasionally Removed). It also plants:
- items added mid-sprint
- about 15% unpointed items
- a few undated iterations
- iteration bounces

Names are synthetic (`Alpha`, `Beta`, `Team Red`, and so on), and the output is deterministic per seed. It backs the tests and `demo`.

## Testing

pytest, run in CI on ubuntu and windows, Python 3.11–3.13.

- `timeline`:
  - as-of at exact boundaries
  - before creation → none
- `sprints`: hand-built revision fixtures, one per case:
  - committed vs added after C
  - moved out mid-sprint
  - Removed
  - done in Resolved vs Completed under both done sets
  - carryover counting
  - team assignment (exact, prefix, single-team, unassigned)
  - project-level fallback
- `features`: no-leakage checks. Trailing stats ignore sprints ending after start, and features are unchanged by revisions after C.
- `baseline` and `rollup`:
  - σ = 0 matches the independent Poisson-binomial mean
  - the σ fit recovers a planted σ on simulated outcomes within tolerance
- `model` on synth: item AUC > 0.65, and the mean SHAP contribution of `carryover_count` is negative.
- `backtest`: the leakage assertion holds, and metrics come out finite on synth.
- `analytics` and `extract`, with mocked HTTP: both paging modes, timestamp normalization, watermark `ge` + dedupe, per-project replace, 401 skip.
- `cli`: `demo` runs end to end on a temp dir.

## CI/CD

- `.github/workflows/ci.yml`: pytest matrix on push and PR.
- No package publishing for now.
