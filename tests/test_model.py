import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from sprint_forecast import model as model_module
from sprint_forecast.features import FEATURES
from sprint_forecast.model import (
    P_CLIP,
    contributions,
    describe_drivers,
    predict_proba,
    predict_proba_lr,
    split_by_time,
    train_item_model,
)


def _time_split(frame):
    """Train on sprints that ended before the median sprint start; test on sprints starting at or after it."""
    split = frame.drop_duplicates("sprint_id")["start"].median()
    return frame[frame["end"] < split], frame[frame["start"] >= split]


@pytest.fixture(scope="module")
def trained(synth40):
    train, test = _time_split(synth40.frame)
    return train_item_model(train, seed=0), train, test


def test_item_auc_beats_065_on_synth(trained):
    model, _, test = trained
    assert roc_auc_score(test["y"], predict_proba(model, test)) > 0.65
    assert roc_auc_score(test["y"], predict_proba_lr(model, test)) > 0.6


def test_carryover_shap_is_negative_for_carried_items(trained):
    model, _, test = trained
    contrib = contributions(model, test)
    assert list(contrib.columns) == FEATURES + ["bias"]
    carried = test["carryover_count"] > 0
    assert contrib.loc[carried, "carryover_count"].mean() < 0
    assert contrib.loc[carried, "carryover_count"].mean() < contrib.loc[~carried, "carryover_count"].mean()


def test_calibration_uses_last_sprints_and_platt_when_small(trained):
    model, train, _ = trained
    assert model.calibration == "platt"
    fit_ids, cal_ids = split_by_time(train)
    assert len(cal_ids) == round(0.2 * (len(fit_ids) + len(cal_ids)))
    starts = train.drop_duplicates("sprint_id").set_index("sprint_id")["start"]
    assert starts[cal_ids].min() >= starts[fit_ids].max()
    assert set(model.calib_frame["sprint_id"]) == set(cal_ids)
    assert model.calib_frame["p"].between(P_CLIP, 1 - P_CLIP).all()


def test_isotonic_when_calibration_set_is_large(synth40, monkeypatch):
    monkeypatch.setattr(model_module, "ISOTONIC_MIN_ITEMS", 50)
    train, test = _time_split(synth40.frame)
    model = train_item_model(train, seed=0)
    assert model.calibration == "isotonic"
    p = predict_proba(model, test)
    assert np.all((p >= P_CLIP) & (p <= 1 - P_CLIP))


def test_training_is_deterministic(synth40):
    train, test = _time_split(synth40.frame)
    a = predict_proba(train_item_model(train, seed=0), test)
    b = predict_proba(train_item_model(train, seed=0), test)
    assert np.array_equal(a, b)


def test_single_class_training_raises(synth40):
    frame = synth40.frame.assign(y=1.0)
    with pytest.raises(ValueError, match="both"):
        train_item_model(frame)


def test_unknown_category_and_empty_frame_predict(trained):
    model, _, test = trained
    odd = test.head(3).assign(type="Requirement", state_category_at_commit="Resolved")
    p = predict_proba(model, odd)
    assert p.shape == (3,) and np.isfinite(p).all()
    assert predict_proba_lr(model, odd).shape == (3,)
    assert predict_proba(model, test.iloc[0:0]).shape == (0,)


def test_describe_drivers_in_plain_words():
    contrib = pd.Series({f: 0.0 for f in FEATURES} | {"carryover_count": -0.9, "load_ratio": -0.4, "points": 0.3})
    values = pd.Series({f: np.nan for f in FEATURES} | {"carryover_count": 2.0, "load_ratio": 1.456})
    assert describe_drivers(contrib, values) == [
        "sprints already carried over = 2",
        "sprint load vs team velocity = 1.46",
    ]
    assert describe_drivers(contrib.clip(lower=0), values) == []
    flags = pd.Series({f: 0.0 for f in FEATURES} | {"is_unestimated": -0.5, "team_trailing_completion": -0.2})
    flag_values = pd.Series({f: np.nan for f in FEATURES} | {"is_unestimated": 1.0})
    assert describe_drivers(flags, flag_values) == [
        "item is unestimated = yes",
        "team has no earlier sprints",
    ]


def test_describe_drivers_says_why_a_value_is_missing():
    contrib = pd.Series({f: 0.0 for f in FEATURES} | {"assignee_load_ratio": -0.9, "load_ratio": -0.5,
                                                      "points_rel": -0.4, "age_days": -0.1})
    values = pd.Series({f: np.nan for f in FEATURES} | {"is_unassigned": 0.0})
    assert describe_drivers(contrib, values, k=3) == [
        "assignee has no recent delivery on record",
        "team has no recent velocity",
        "item age in days unknown",
    ]
    unassigned = values.copy()
    unassigned["is_unassigned"] = 1.0
    assert describe_drivers(contrib, unassigned, k=1) == ["item is unassigned"]


def test_missing_categories_train_and_predict(synth40):
    frame = synth40.frame.copy()
    frame.loc[frame.index[::7], "state_category_at_commit"] = np.nan
    frame.loc[frame.index[::11], "type"] = np.nan
    model = train_item_model(frame, seed=0)
    assert "(missing)" in model.categories["state_category_at_commit"]
    p = predict_proba(model, frame.head(50))
    assert np.isfinite(p).all() and np.isfinite(predict_proba_lr(model, frame.head(50))).all()
