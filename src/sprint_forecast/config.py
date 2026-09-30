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
    close_grace_hours: float = 0.0  # count items closed this long after the sprint's end, still in it, as done
    exclude_title_pattern: str = ""  # regex (case-insensitive); items whose title matches are left out entirely


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
    pattern = str(model.get("exclude_title_pattern", ""))
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValueError(f"model.exclude_title_pattern is not a valid regular expression: {e}") from None
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
        close_grace_hours=float(model.get("close_grace_hours", 0.0)),
        exclude_title_pattern=pattern,
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
            "close_grace_hours": float(cfg.close_grace_hours),
            "exclude_title_pattern": cfg.exclude_title_pattern,
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
