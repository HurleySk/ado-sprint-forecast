# sprint-forecast

Forecast how much of an Azure DevOps sprint's **day-1 committed scope** a team will deliver, from its first day
to its last.

The output is a distribution over "% of committed points reaching Done": P(full scope), P(>=80%),
the expected %, and a p10/p50/p90 band. Before a sprint it is a planning-time sanity check ("is this sprint
overloaded?"). While it runs, the forecast starts from the points already done and scores every open item,
including items added after day 1, on where it stands now. It is also an honest ML exercise: every model is
backtested against a dumb velocity baseline, and losing to the baseline is reported as a result.

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
when your process closes work in a Resolved state. `predict` scores the sprint as it stands now: the points
done so far, the forecast for the committed points still open, the day-1 forecast beside it, and the riskiest
open items, with items added after day 1 marked `added`. Once the sprint is over it reports the outcome.
`predict --as-of commit` shows the day-1 view at the commit cutoff. After upgrading, run `train` again: model
files from an older version are refused with a message saying so.

Two settings in `.sprint-forecast/config.toml`, under `[model]`:

```toml
[model]
close_grace_hours = 12                      # done also counts items closed up to 12 h after the end, still in the sprint
exclude_title_pattern = "placeholder|^tracking item"   # regex, case-insensitive
```

`close_grace_hours` is for teams that close the last items the morning after a sprint ends. Those items count as done
for training, outcomes and exports, and `sprints.csv` and `items.csv` keep the strict numbers beside them
(`pct_done_strict`, `done_strict`). The data report shows how much of the done work the grace added.
`exclude_title_pattern` leaves out placeholder items (a standing "tracker" story, say) from everything. `extract`
matches titles against it and stores only the IDs of the items that match, never the titles. Change it and run
`extract` again, then `train`.

## Power BI

`sprint-forecast export` forecasts every running team sprint as it stands at the run (the points done so far
plus the forecast for the committed items still open, with the forecast made at the commit cutoff in the
`day1_*` columns) and each team's next sprint (scored on the scope loaded so far), then writes:

| File | One row per | Contents |
|---|---|---|
| `sprint_forecasts/<run_id>.csv` | run x team sprint | status, committed items and points, load vs velocity, P(full), P(>=80%), expected, p10/p50/p90 as of the run, share done so far, and the day-1 expected, p10 and p90 |
| `item_forecasts/<run_id>.csv` | run x item open in the sprint or committed to it | work item ID, points, probability of done from the rank model (1 or 0 for committed items already done or out of the sprint), top 3 risk factors for open items, state now, done so far, added after day 1 or not, sprints in a row without a state change |
| `sprints.csv` | reconstructed sprint | committed and done points, % done (empty until the sprint ends), and the same without the close grace |
| `items.csv` | reconstructed sprint x item in it | points, sprints already carried over, added after the commit cutoff or not, assignee at the sprint's end (display name), done and done without the close grace (empty until the sprint ends), where it went, state changes in the sprint |
| `cycle.csv` | finished item | team, type, points when work started and when done, re-estimated or not, started, done, business days between, moves back into a state it had left |
| `cycle_states.csv` | finished item x state | business days the item spent in that state between starting and done |
| `golive.csv` | active parent item with open children | team, children open and done, open points (unestimated ones at the team's median), recent burn, P(all done by the running sprint's end), the first sprint end with P >= 50% and >= 85% |
| `golive_curve.csv` | parent x coming sprint | P(every open child done) by that sprint's end, for the running sprint and the next 12 |
| `backtest.csv` | backtested sprint, checkpoint and model | predicted band vs actual at that point of the sprint, copied from the last `backtest` run |

Forecast files are added per run and never rewritten, so the history (how the forecast moved through the sprint,
next to the day-1 forecast) builds up; `sprints.csv`, `items.csv`, `cycle.csv`, `cycle_states.csv`, the go-live
files and `backtest.csv` are replaced.

Changed in 0.3.0: `item_forecasts` `p_done` for open items comes from the rank model, which also reads the exact
state name (a story waiting on review ranks apart from one in development). It orders items better; the sprint
forecast keeps model A, since the state name did not make it more accurate. `items.csv` `fate` says where each
item went once the sprint ended: `done`, `done_in_grace` (closed within the close grace), `closed_later` (closed
after that, still in the sprint), `carried` (moved to a later sprint), `backlog` (moved out to an undated
iteration), `removed` or `open`.

The go-live forecast covers parents (a feature, say) with an open child of the counted types and a child either
open in a running or coming sprint or finished in the last 16 weeks. Its team is the one holding most of its open
children's points. Children open in any team's running sprint are drawn with the item model; each later sprint of
the parent's team burns one of its last 8 ended sprints (points of the children finished in it, from the first
sprint one was started). Under 2 such sprints, or no burn in them, gets
no curve. Work added to the parent later is not foreseen, so the dates are a floor.

Changed in 0.2.0: a running sprint used to be scored at its commit cutoff, and `item_forecasts` listed committed
items only. A report that rebuilt a projected finish from `points_done_so_far` plus the open items' `p_done`
should read `expected` instead, or keep the rebuild and filter `is_added = False`. A report on `backtest.csv`
that wants the day-1 view filters `checkpoint = 0`.

`cycle.csv` answers how long work took, for checking estimates against it. An item's cycle runs from its first
active state (InProgress, or Resolved when that isn't done) to the first time it reached a done category;
reopening later is ignored, and an item that went straight to done has no cycle. Business days skip Saturdays
and Sundays and keep fractions (UTC). Its team is the one running the iteration it finished in that owns its
area, else any team in the project that owns the area. `cycle_states.csv` splits that time by state name, so a
report can group states into phases (development, waiting for test, test, approval) for its own process. Files are written to a temp name and renamed, so a
sync client or a refresh never reads half a file. Timestamps are UTC (`2024-06-10T05:00:00Z`).

The files carry no titles; join `item_id` to Azure DevOps (e.g. the Analytics OData `WorkItems` feed) for them.
`items.csv` lists every item committed at the cutoff and, flagged `added_mid`, every item added after it that
was still in the sprint at the end (unless it was already done before the sprint), so throughput counts all the
work a sprint finished. Committed rows match `sprints.csv`, which covers committed scope only. Each row names
who held the item at the sprint's end, for delivery by person; items with no assignee are left blank, and a user
key Analytics has no name for reads `Unknown user`. A typical schedule: `extract` then `export` daily, `train` and `backtest` weekly,
with `--out` pointing at a synced SharePoint or OneDrive folder that Power BI reads with its folder connector.

## How it works

- **Extract**: Analytics OData `WorkItemRevisions`, `Iterations`, `Teams` and `Users` per project into a local
  SQLite cache, with per-project watermarks (each run re-reads the last day to catch late revisions).
  A project that returns 401/403 is skipped; fields its process lacks (e.g. `StoryPoints`) are dropped
  from the query; any other failure is reported, the remaining projects still run, and the exit code is 1.
  A missing `StateCategory` is inferred from the state name. User names are optional: if `Users` cannot be
  read, the extract carries on and says so.
- **Sprint** = (project, team, iteration with dates). Items are assigned to the subscribed team whose area
  path matches exactly, else by longest prefix, else the only subscribed team; otherwise they are
  reported as unassigned. Committed scope is what sits in the iteration at start + 1 day (the commit
  cutoff); an item is done if at the end date it is still in the iteration and in a done category.
  Sprints that had not ended when the data was extracted have no outcome yet and are never trained on.
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
  committed scope the target measures. A second LightGBM, the rank model, also reads the exact state name and
  orders items for `predict` and `item_forecasts`.
- **Backtest**: expanding window, retrained every few sprints, with a leakage check. Each sprint is scored at
  the commit cutoff and at 25%, 50% and 75% of the way to the end: CRPS, pinball loss, Brier scores, coverage
  and MAE per model, checkpoint and team. Model A meets baseline C at the cutoff and, later on, a progress
  reference (points done so far plus the day-1 odds of the committed items still open). Item metrics are split
  by checkpoint and into committed and added items, and a leave-one-team-out check covers generalization.

## Privacy

This repository contains code only. The cache, trained models and backtest output live in
`.sprint-forecast/`, which is git-ignored along with `*.db`, `*.sqlite`, `*.jsonl`, `*.parquet`,
`*.joblib` and `.env`. The model sees assignees only as numeric load features computed from opaque Analytics
user keys; the cache also keeps each key's display name so `export` can name assignees in `items.csv` and
`cycle.csv`. Work item titles are fetched on demand for display and never stored. `export` writes to the folder
you name: project, team, iteration and state names, work item IDs and numbers, and assignee names in `items.csv`
and `cycle.csv`; no titles.

## Development

```bash
.venv/Scripts/python -m pytest
```

Tests are offline and deterministic: HTTP is mocked through an injected `fetch_json(url)` function, and
model tests run on synthetic data from `sprint_forecast.synth`. CI runs on Ubuntu and Windows with
Python 3.11-3.13.

## License

MIT
