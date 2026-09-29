import io
import os
import shutil
import sys
import tomllib
from importlib.metadata import entry_points

import joblib
import pandas as pd
import pytest
from click.testing import CliRunner

from sprint_forecast import cli
from sprint_forecast.analytics import HttpError
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
    assert set(rows["model"]) == {"a", "progress", "c", "team_mean"}
    assert set(rows["checkpoint"]) == {0.0, 0.25, 0.5, 0.75}
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
    assert set(rows["checkpoint"]) == {0.0}


def test_predict_past_sprint_as_of_commit_with_titles(workspace, monkeypatch):
    run(workspace, "init", "--org", "https://dev.azure.com/contoso", "--projects", "Alpha,Beta", "--force")
    try:
        monkeypatch.setenv("ADO_PAT", "not-a-real-token")
        monkeypatch.setattr(cli, "fetch_titles", lambda fetch, org, ids: {i: f"Synthetic item {i}" for i in ids})
        result = run(
            workspace, "predict", "--iteration", "Alpha\\Sprint 12", "--team", "Team Blue", "--as-of", "commit",
            "--top", "2",
        )
    finally:
        (workspace / ".sprint-forecast" / "config.toml").unlink()
    assert result.exit_code == 0, result.output
    assert "Alpha/Team Blue | Alpha\\Sprint 12" in result.output
    assert "scored as of commit cutoff" in result.output
    assert "Synthetic item" in result.output
    assert result.output.count("p=") == 2


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


def test_data_on_empty_cache_reports_zero(tmp_path):
    connect(tmp_path / ".sprint-forecast" / "cache.db").close()
    result = run(tmp_path, "data")
    assert result.exit_code == 0, result.output
    assert "Sprints reconstructed: 0" in result.output


def test_extract_reports_failures_without_traceback(tmp_path, monkeypatch):
    monkeypatch.setenv("ADO_PAT", "not-a-real-token")

    def fetch(url):
        if "/Beta/" in url:
            raise HttpError(500, url, "boom")
        if "_apis/projects" in url:
            return {"value": [{"name": "Alpha"}, {"name": "Beta"}]}
        return {"value": []}

    monkeypatch.setattr(cli, "make_fetch_json", lambda method, pat: fetch)
    run(tmp_path, "init", "--org", "https://dev.azure.com/contoso", "--projects", "*")
    result = run(tmp_path, "extract")
    assert result.exit_code == 1
    assert "Beta: failed (HTTP 500: boom)" in result.output and "1 failed" in result.output


def test_extract_cannot_list_projects_is_a_clean_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ADO_PAT", "not-a-real-token")

    def fetch(url):
        raise HttpError(401, url, "")

    monkeypatch.setattr(cli, "make_fetch_json", lambda method, pat: fetch)
    run(tmp_path, "init", "--org", "https://dev.azure.com/contoso", "--projects", "*")
    result = run(tmp_path, "extract")
    assert result.exit_code == 1
    assert "Error: cannot list projects: HTTP 401" in result.output


def console_entry():
    """The function the installed `sprint-forecast` command calls."""
    (ep,) = entry_points(group="console_scripts", name="sprint-forecast")
    return ep.load()


@pytest.mark.skipif(os.name != "nt", reason="click expands wildcards in argv only on Windows")
def test_console_command_keeps_star_literal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "argv", ["sprint-forecast", "init", "--org", "https://dev.azure.com/contoso", "--projects", "*"])
    with pytest.raises(SystemExit) as exc:
        console_entry()()
    assert exc.value.code == 0
    doc = tomllib.loads((tmp_path / ".sprint-forecast" / "config.toml").read_text(encoding="utf-8"))
    assert doc["ado"]["projects"] == ["*"]


def test_console_command_replaces_characters_the_terminal_cannot_encode(tmp_path, monkeypatch):
    root = tmp_path / "café → plan"
    out, err = io.BytesIO(), io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(out, encoding="cp1252"))
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(err, encoding="cp1252"))
    monkeypatch.setattr(sys, "argv", [
        "sprint-forecast", "--root", str(root), "init", "--org", "https://dev.azure.com/contoso", "--projects", "Alpha",
    ])
    with pytest.raises(SystemExit) as exc:
        console_entry()()
    sys.stdout.flush()
    assert exc.value.code == 0
    assert "café ? plan" in out.getvalue().decode("cp1252")


def test_export_command_writes_power_bi_tables(workspace, tmp_path):
    out = tmp_path / "bi"
    result = run(workspace, "export", "--out", str(out))
    assert result.exit_code == 0, result.output
    assert "No running or upcoming team sprints" in result.output  # synthetic sprints all ended in 2024
    assert (out / "sprints.csv").exists() and len(list((out / "sprint_forecasts").glob("*.csv"))) == 1


def test_export_lists_each_forecast(workspace, tmp_path, capsys):
    now = pd.Timestamp("2024-06-12T12:00:00Z")  # Alpha Sprint 12 is running
    result = cli._export(workspace / ".sprint-forecast", tmp_path, now=now)
    output = capsys.readouterr().out
    assert "Alpha/Team Red | Alpha\Sprint 12: running, scored as of now" in output
    assert f"Wrote {len(result.files)} files to {tmp_path}" in output


def test_export_without_a_model_explains(tmp_path):
    result = CliRunner().invoke(cli.main, ["--root", str(tmp_path), "export"])
    assert result.exit_code != 0 and "sprint-forecast train" in result.output
