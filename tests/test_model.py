import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from sprint_forecast import model as model_module
from sprint_forecast.features import FEATURES
from sprint_forecast.model import (
    ISOTONIC_MIN_ITEMS,
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
    train, test = _time_split(synth40.ckpt)
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


def test_calibration_uses_last_sprints_and_groups_by_checkpoint(trained):
    model, train, _ = trained
    assert model.calibration == ("isotonic" if len(model.calib_frame) >= ISOTONIC_MIN_ITEMS else "platt")
    fit_ids, cal_ids = split_by_time(train)
    assert len(cal_ids) == round(0.2 * (len(fit_ids) + len(cal_ids)))
    starts = train.drop_duplicates("sprint_id").set_index("sprint_id")["start"]
    assert starts[cal_ids].min() >= starts[fit_ids].max()
    cf = model.calib_frame
    assert list(cf.columns) == ["sprint_id", "group", "p", "y"]
    assert set(cf["sprint_id"]) == set(cal_ids) and cf["group"].nunique() > len(cal_ids)
    assert (cf["group"].str.split("@").str[0] == cf["sprint_id"]).all()
    assert cf["p"].between(P_CLIP, 1 - P_CLIP).all()


def test_isotonic_when_calibration_set_is_large(synth40, monkeypatch):
    monkeypatch.setattr(model_module, "ISOTONIC_MIN_ITEMS", 50)
    train, test = _time_split(synth40.ckpt)
    model = train_item_model(train, seed=0)
    assert model.calibration == "isotonic"
    p = predict_proba(model, test)
    assert np.all((p >= P_CLIP) & (p <= 1 - P_CLIP))


def test_training_is_deterministic(synth40):
    train, test = _time_split(synth40.ckpt)
    a = predict_proba(train_item_model(train, seed=0), test)
    b = predict_proba(train_item_model(train, seed=0), test)
    assert np.array_equal(a, b)


def test_single_class_training_raises(synth40):
    frame = synth40.ckpt.assign(y=1.0)
    with pytest.raises(ValueError, match="both"):
        train_item_model(frame)


def test_unknown_category_and_empty_frame_predict(trained):
    model, _, test = trained
    odd = test.head(3).assign(type="Requirement", state_category="Resolved")
    p = predict_proba(model, odd)
    assert p.shape == (3,) and np.isfinite(p).all()
    assert predict_proba_lr(model, odd).shape == (3,)
    assert predict_proba(model, test.iloc[0:0]).shape == (0,)


def test_describe_drivers_in_plain_words():
    contrib = pd.Series({f: 0.0 for f in FEATURES} | {"carryover_count": -0.9, "load_ratio": -0.4, "points": 0.3})
    values = pd.Series({f: np.nan for f in FEATURES} | {"carryover_count": 2.0, "load_ratio": 1.456})
    assert describe_drivers(contrib, values) == [
        "already carried over 2 sprints",
        "sprint load is 1.5× team velocity",
    ]
    assert describe_drivers(contrib.clip(lower=0), values) == []
    flags = pd.Series({f: 0.0 for f in FEATURES} | {"is_unestimated": -0.5, "team_trailing_completion": -0.2})
    flag_values = pd.Series({f: np.nan for f in FEATURES} | {"is_unestimated": 1.0})
    assert describe_drivers(flags, flag_values) == [
        "item is unestimated",
        "team has no earlier sprints",
    ]


@pytest.mark.parametrize("feature, value, text", [
    ("unestimated_share", 0.75, "75% of the sprint's items are unestimated"),
    ("bug_share", 0.5, "50% of the sprint's items are bugs"),
    ("carryover_share", 0.696, "70% of the sprint's items were carried over"),
    ("team_trailing_completion", 0.322, "team recently finished 32% of committed points"),
    ("assignee_load_ratio", 4.5, "assignee load is 4.5× their recent delivery"),
    ("points_rel", 0.173, "item size is 0.17× team velocity"),
    ("days_since_change", 32.8, "no change in 33 days"),
    ("age_days", 1.0, "item is 1 day old"),
    ("sprint_length_days", 38.0, "sprint is 38 days long"),
    ("carryover_count", 1.0, "already carried over 1 sprint"),
    ("points", 0.5, "item size is 0.5 points"),
    ("n_items", 40.0, "40 items committed to the sprint"),
    ("revisions_so_far", 12.0, "12 edits so far"),
    ("team_sprint_index", 7.0, "team has 7 earlier sprints"),
    ("has_parent", 0.0, "item has no parent"),
    ("is_unassigned", 1.0, "item is unassigned"),
    ("type", "Bug", "item is a Bug"),
    ("state_category", "InProgress", "state is InProgress"),
    ("elapsed", 0.6, "60% of the sprint has passed"),
    ("days_left", 3.2, "3 days left"),
    ("done_share", 0.42, "42% of committed points done"),
    ("days_in_state", 6.0, "in the same state for 6 days"),
    ("n_state_changes", 1.0, "1 state change this sprint"),
    ("is_added", 1.0, "added after day 1"),
    ("reassigned", 1.0, "reassigned since day 1"),
])
def test_describe_drivers_reads_as_a_sentence(feature, value, text):
    contrib = pd.Series({f: 0.0 for f in FEATURES} | {feature: -1.0})
    values = pd.Series({f: np.nan for f in FEATURES} | {feature: value}, dtype=object)
    assert describe_drivers(contrib, values, k=1) == [text]


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
    added = values.copy()
    added["is_added"] = 1.0
    assert describe_drivers(contrib, added, k=1) == ["added after day 1"]


def test_missing_categories_train_and_predict(synth40):
    frame = synth40.ckpt.copy()
    frame.loc[frame.index[::7], "state_category"] = np.nan
    frame.loc[frame.index[::11], "type"] = np.nan
    model = train_item_model(frame, seed=0)
    assert "(missing)" in model.categories["state_category"]
    p = predict_proba(model, frame.head(50))
    assert np.isfinite(p).all() and np.isfinite(predict_proba_lr(model, frame.head(50))).all()
