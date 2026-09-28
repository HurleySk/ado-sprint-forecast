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

from sprint_forecast.features import CATEGORICAL, FEATURES, NUMERIC

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


def to_matrix(frame: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    X = frame[FEATURES].copy()
    for col in CATEGORICAL:
        values = X[col].astype(str)
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
        X[col] = X[col].astype(str).astype(object)
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
    categories = {c: sorted(frame[c].astype(str).unique()) for c in CATEGORICAL}
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


BOOLEAN_FEATURES = {"is_unestimated", "has_parent", "is_unassigned"}


def _fmt(feature: str, value) -> str:
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "missing"
    if feature in BOOLEAN_FEATURES:
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3g}"
    return str(value)


def describe_drivers(contrib: pd.Series, values: pd.Series, k: int = 2) -> list[str]:
    """The k most negative contributions, in plain words, e.g. 'sprints already carried over = 2'."""
    neg = contrib[FEATURES]
    neg = neg[neg < 0].sort_values().head(k)
    return [f"{FEATURE_LABELS[f]} = {_fmt(f, values[f])}" for f in neg.index]
