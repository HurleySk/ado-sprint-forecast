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
