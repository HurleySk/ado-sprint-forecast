"""Model A: LightGBM item classifier with time-split calibration, plus a logistic-regression sanity model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sprint_forecast.features import DAY1_FEATURES as FEATURES, MISSING

CATEGORICAL = ["type", "state_category_at_commit"]  # interim until model A trains on checkpoint rows
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]

LGBM_PARAMS = {
    "num_leaves": 15,
    "min_child_samples": 20,
    "learning_rate": 0.03,
    "n_estimators": 300,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_jobs": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
CALIBRATION_SHARE = 0.2
ISOTONIC_MIN_ITEMS = 1000
P_CLIP = 1e-4

FEATURE_LABELS = {
    "type": "work item type",
    "state_category_at_commit": "state at commit",
    "points": "item size in points",
    "points_rel": "item size vs team velocity",
    "is_unestimated": "item is unestimated",
    "carryover_count": "sprints already carried over",
    "age_days": "item age in days",
    "days_since_change": "days since last change",
    "revisions_so_far": "number of edits so far",
    "has_parent": "has a parent item",
    "load_ratio": "sprint load vs team velocity",
    "n_items": "items committed to the sprint",
    "bug_share": "share of bugs in the sprint",
    "carryover_share": "share of carried-over items in the sprint",
    "unestimated_share": "share of unestimated items in the sprint",
    "team_trailing_completion": "team's recent completion rate",
    "sprint_length_days": "sprint length in days",
    "team_sprint_index": "team's number of earlier sprints",
    "assignee_load_ratio": "assignee load vs their recent delivery",
    "is_unassigned": "item is unassigned",
}


@dataclass
class ItemModel:
    lgbm: LGBMClassifier
    lr: Pipeline
    calibrator: IsotonicRegression | LogisticRegression | None
    calibration: str
    categories: dict[str, list[str]]
    calib_frame: pd.DataFrame


def split_by_time(frame: pd.DataFrame, share: float = CALIBRATION_SHARE) -> tuple[list[str], list[str]]:
    """Split sprint ids by start: the first ~80% fit the model, the last ~20% calibrate it."""
    sprints = frame.drop_duplicates("sprint_id").sort_values(["start", "sprint_id"], kind="mergesort")
    ids = list(sprints["sprint_id"])
    n_cal = max(1, int(round(len(ids) * share))) if len(ids) >= 2 else 0
    return ids[: len(ids) - n_cal], ids[len(ids) - n_cal:]


def _category_values(col: pd.Series) -> pd.Series:
    return col.fillna(MISSING).astype(str)


def to_matrix(frame: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    X = frame[FEATURES].copy()
    for col in CATEGORICAL:
        values = _category_values(X[col])
        X[col] = pd.Categorical(values.where(values.isin(categories[col])), categories=categories[col])
    for col in NUMERIC:
        X[col] = X[col].astype("float64")
    return X


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, P_CLIP, 1 - P_CLIP)
    return np.log(p / (1 - p))


def _make_lr() -> Pipeline:
    pre = ColumnTransformer([
        ("num", make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler()
        ), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=2000))


def _lr_input(frame: pd.DataFrame) -> pd.DataFrame:
    X = frame[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = _category_values(X[col]).astype(object)
    for col in NUMERIC:
        X[col] = X[col].astype("float64")
    return X


def _calibrate(model: ItemModel, raw: np.ndarray) -> np.ndarray:
    if model.calibration == "isotonic":
        p = model.calibrator.predict(raw)
    elif model.calibration == "platt":
        p = model.calibrator.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
    else:
        p = raw
    return np.clip(p, P_CLIP, 1 - P_CLIP)


def train_item_model(frame: pd.DataFrame, seed: int = 0) -> ItemModel:
    frame = frame[frame["y"].notna()]
    if frame["y"].nunique() < 2:
        raise ValueError("training data needs both delivered and undelivered items")
    categories = {c: sorted(_category_values(frame[c]).unique()) for c in CATEGORICAL}
    fit_ids, cal_ids = split_by_time(frame)
    fit = frame[frame["sprint_id"].isin(fit_ids)]
    cal = frame[frame["sprint_id"].isin(cal_ids)]
    if fit["y"].nunique() < 2:
        fit, cal = frame, frame.iloc[0:0]
    lgbm = LGBMClassifier(**LGBM_PARAMS, random_state=seed)
    lgbm.fit(to_matrix(fit, categories), fit["y"].astype(int))
    lr = _make_lr().fit(_lr_input(fit), fit["y"].astype(int))
    model = ItemModel(lgbm, lr, None, "none", categories, pd.DataFrame(columns=["sprint_id", "p", "y"]))
    if len(cal) and cal["y"].nunique() == 2:
        raw = lgbm.predict_proba(to_matrix(cal, categories))[:, 1]
        y = cal["y"].astype(int).to_numpy()
        if len(cal) >= ISOTONIC_MIN_ITEMS:
            model.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y)
            model.calibration = "isotonic"
        else:
            model.calibrator = LogisticRegression().fit(_logit(raw).reshape(-1, 1), y)
            model.calibration = "platt"
    if len(cal):
        model.calib_frame = pd.DataFrame({
            "sprint_id": cal["sprint_id"].to_numpy(),
            "p": predict_proba(model, cal),
            "y": cal["y"].astype(int).to_numpy(),
        })
    return model


def predict_proba(model: ItemModel, frame: pd.DataFrame) -> np.ndarray:
    if len(frame) == 0:
        return np.array([], dtype=float)
    raw = model.lgbm.predict_proba(to_matrix(frame, model.categories))[:, 1]
    return _calibrate(model, raw)


def predict_proba_lr(model: ItemModel, frame: pd.DataFrame) -> np.ndarray:
    if len(frame) == 0:
        return np.array([], dtype=float)
    return model.lr.predict_proba(_lr_input(frame))[:, 1]


def contributions(model: ItemModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Per-item SHAP contributions (log-odds scale, uncalibrated) from LightGBM pred_contrib; last column 'bias'."""
    values = model.lgbm.predict(to_matrix(frame, model.categories), pred_contrib=True)
    return pd.DataFrame(np.asarray(values), columns=FEATURES + ["bias"], index=frame.index)


def _pct(v) -> str:
    return f"{round(100 * v)}%"


def _times(v) -> str:
    return f"{v:.0f}×" if v >= 10 else f"{v:.2g}×"


def _count(v, unit: str) -> str:
    n = round(v)
    return f"{n} {unit}{'' if n == 1 else 's'}"


# A driver's value in plain words: shares as percentages, ratios as multiples, days rounded, flags as statements.
PHRASES = {
    "type": lambda v: f"item is a {v}",
    "state_category_at_commit": lambda v: f"state at commit was {v}",
    "points": lambda v: f"item size is {v:g} point{'' if v == 1 else 's'}",
    "points_rel": lambda v: f"item size is {_times(v)} team velocity",
    "is_unestimated": lambda v: "item is unestimated" if v else "item is estimated",
    "carryover_count": lambda v: f"already carried over {_count(v, 'sprint')}",
    "age_days": lambda v: f"item is {_count(v, 'day')} old",
    "days_since_change": lambda v: f"no change in {_count(v, 'day')}",
    "revisions_so_far": lambda v: f"{_count(v, 'edit')} so far",
    "has_parent": lambda v: "item has a parent" if v else "item has no parent",
    "load_ratio": lambda v: f"sprint load is {_times(v)} team velocity",
    "n_items": lambda v: f"{_count(v, 'item')} committed to the sprint",
    "bug_share": lambda v: f"{_pct(v)} of the sprint's items are bugs",
    "carryover_share": lambda v: f"{_pct(v)} of the sprint's items were carried over",
    "unestimated_share": lambda v: f"{_pct(v)} of the sprint's items are unestimated",
    "team_trailing_completion": lambda v: f"team recently finished {_pct(v)} of committed points",
    "sprint_length_days": lambda v: f"sprint is {_count(v, 'day')} long",
    "team_sprint_index": lambda v: f"team has {_count(v, 'earlier sprint')}",
    "assignee_load_ratio": lambda v: f"assignee load is {_times(v)} their recent delivery",
    "is_unassigned": lambda v: "item is unassigned" if v else "item is assigned",
}


# Why a history feature is NaN, said plainly (see features.team_history and features.assignee_load).
MISSING_PHRASES = {
    "assignee_load_ratio": "assignee has no recent delivery on record",
    "team_trailing_completion": "team has no earlier sprints",
    "load_ratio": "team has no recent velocity",
    "points_rel": "team has no recent velocity",
}


def _is_missing(value) -> bool:
    return isinstance(value, (float, np.floating)) and np.isnan(value)


def _describe(feature: str, values: pd.Series) -> str:
    if not _is_missing(values[feature]):
        return PHRASES[feature](values[feature])
    if feature == "assignee_load_ratio" and values.get("is_unassigned") == 1:
        return "item is unassigned"
    return MISSING_PHRASES.get(feature, f"{FEATURE_LABELS[feature]} unknown")


def describe_drivers(contrib: pd.Series, values: pd.Series, k: int = 2) -> list[str]:
    """The k most negative contributions, in plain words, e.g. 'already carried over 2 sprints'.
    Features that read the same (both velocity ratios when the team has no velocity) are listed once."""
    neg = contrib[FEATURES]
    out: list[str] = []
    for f in neg[neg < 0].sort_values().index:
        if len(out) >= k:
            break
        text = _describe(f, values)
        if text not in out:
            out.append(text)
    return out
