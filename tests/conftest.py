"""Shared, session-scoped synthetic data so the expensive pipeline runs once per test session."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

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


@pytest.fixture(scope="session")
def synth40(tmp_path_factory) -> Synth:
    """Seed 0, 40 sprints per team: ~1700 committed items; used for model-quality tests."""
    return _synth(tmp_path_factory, seed=0, n_sprints=40)


@pytest.fixture(scope="session")
def synth16(tmp_path_factory) -> Synth:
    """Seed 1, 16 sprints per team: small and fast; used for backtest and plumbing tests."""
    return _synth(tmp_path_factory, seed=1, n_sprints=16)
