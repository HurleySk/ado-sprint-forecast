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
sprint-forecast export --out <folder>   # CSVs for Power BI (see below)
```

`--done-categories Resolved,Completed` (on `data`, `backtest` and `train`) widens what counts as done
when your process closes work in a Resolved state. `predict --as-of now` scores the iteration's current
contents; by default it uses the contents at the commit cutoff once that has passed.

## Power BI

`sprint-forecast export` forecasts every running team sprint (scored at its commit cutoff, so the number stays
put for the rest of the sprint) and each team's next sprint (scored on the scope loaded so far), then writes:

| File | One row per | Contents |
|---|---|---|
| `sprint_forecasts/<run_id>.csv` | run x team sprint | status, committed items and points, load vs velocity, P(full), P(>=80%), expected, p10/p50/p90, share done so far |
| `item_forecasts/<run_id>.csv` | run x committed item | work item ID, points, probability of done, top 3 risk factors, state now, done so far |
| `sprints.csv` | reconstructed sprint | committed and done points, % done (empty until the sprint ends) |
| `backtest.csv` | backtested sprint and model | predicted band vs actual, copied from the last `backtest` run |

Forecast files are added per run and never rewritten, so the history (progress against the day-1 forecast)
builds up; `sprints.csv` and `backtest.csv` are replaced. Files are written to a temp name and renamed, so a
sync client or a refresh never reads half a file. Timestamps are UTC (`2024-06-10T05:00:00Z`).

The files carry no titles or people; join `item_id` to Azure DevOps (e.g. the Analytics OData `WorkItems` feed)
for titles and assignees. A typical schedule: `extract` then `export` daily, `train` and `backtest` weekly,
with `--out` pointing at a synced SharePoint or OneDrive folder that Power BI reads with its folder connector.

## How it works

- **Extract**: Analytics OData `WorkItemRevisions`, `Iterations` and `Teams` per project into a local
  SQLite cache, with per-project watermarks (each run re-reads the last day to catch late revisions).
  A project that returns 401/403 is skipped; fields its process lacks (e.g. `StoryPoints`) are dropped
  from the query; any other failure is reported, the remaining projects still run, and the exit code is 1.
  A missing `StateCategory` is inferred from the state name.
- **Sprint** = (project, team, iteration with dates). Items are assigned to the subscribed team whose area
  path matches exactly, else by longest prefix, else the only subscribed team; otherwise they are
  reported as unassigned. Committed scope is what sits in the iteration at start + 1 day (the commit
  cutoff); an item is done if at the end date it is still in the iteration and in a done category.
  Sprints that had not ended when the data was extracted have no outcome yet and are never trained on.
- **Features** are computed as of the cutoff and use only sprints that ended before the sprint started.
  Team identity is never a feature.
- **Models**: C, a velocity bootstrap baseline; a team-mean reference; A, a LightGBM item classifier with
  time-split calibration (plus a logistic-regression sanity check). Item probabilities roll up to a sprint
  distribution through a Monte Carlo with a shared per-sprint shock whose size is fitted by maximum
  likelihood; the shock widens the spread without moving any item's calibrated probability.
- **Backtest**: expanding window, retrained every few sprints, with a leakage check, CRPS, pinball loss,
  Brier scores, coverage and MAE per model and team, plus a leave-one-team-out generalization check.

## Privacy

This repository contains code only. The cache, trained models and backtest output live in
`.sprint-forecast/`, which is git-ignored along with `*.db`, `*.sqlite`, `*.jsonl`, `*.parquet`,
`*.joblib` and `.env`. Assignees are kept only as opaque Analytics user keys and used only as numeric
load features. Work item titles are fetched on demand for display and never stored. `export` writes to the folder you
name: project, team and iteration names, work item IDs and numbers, no titles or people.

## Development

```bash
.venv/Scripts/python -m pytest
```

Tests are offline and deterministic: HTTP is mocked through an injected `fetch_json(url)` function, and
model tests run on synthetic data from `sprint_forecast.synth`. CI runs on Ubuntu and Windows with
Python 3.11-3.13.

## License

MIT
