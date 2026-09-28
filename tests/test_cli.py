import tomllib

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
