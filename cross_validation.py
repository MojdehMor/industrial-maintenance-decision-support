"""cross-validation protocols for categorization and regression.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from modules.categorization import classification_preprocessor
from modules.regression import regression_preprocessor


def classification_cv(
    frame: pd.DataFrame, target: str,
    *,
    categorical: Sequence[str],
    numeric: Sequence[str],
    models: Mapping[str, object],
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-target stratified CV; rare classes trigger KFold rather than data loss."""
    features = list(categorical) + list(numeric)
    if target in features:
        raise ValueError("Target must not be part of the input features")
    data = frame.dropna(subset=[target]).reset_index(drop=True)
    X, y = data[features], data[target].astype(str)
    if len(y) < n_folds or y.nunique() < 2:
        raise ValueError("Insufficient labelled records or unique targets")
    min_class = int(y.value_counts().min())
    if min_class >= n_folds:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = splitter.split(X, y)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = splitter.split(X)
    prepared_folds = list(folds)
    results = []
    for name, model in models.items():
        metrics = []
        for ti, vi in prepared_folds:
            if y.iloc[ti].nunique() < 2:
                raise ValueError("A training fold contains only one target class")
            pipeline = Pipeline([
                ("preprocessor", classification_preprocessor(categorical, numeric)),
                ("model", clone(model)),
            ])
            pipeline.fit(X.iloc[ti], y.iloc[ti])
            pred = pipeline.predict(X.iloc[vi])
            metrics.append({
                "Accuracy": accuracy_score(y.iloc[vi], pred),
                "Macro_F1": f1_score(y.iloc[vi], pred, average="macro", zero_division=0),
                "Weighted_F1": f1_score(y.iloc[vi], pred, average="weighted", zero_division=0),
            })
        results.append({"Target": target, "Model": name, "Folds": n_folds,
                        "Splitter": type(splitter).__name__,
                        **{metric + "_mean": float(np.mean([v[metric] for v in metrics]))
                           for metric in metrics[0]},
                        **{metric + "_std": float(np.std([v[metric] for v in metrics]))
                           for metric in metrics[0]}})
    return pd.DataFrame(results)


def regression_cv(
    frame: pd.DataFrame, target: str,
    *,
    predictors: Sequence[str],
    models: Mapping[str, object],
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Fold-fitted preprocessing with KFold and aggregate MAE/RMSE/R²."""
    if target in predictors:
        raise ValueError("Target leakage: outcome supplied among predictors")
    if len(set(predictors)) != len(predictors):
        raise ValueError("Duplicate feature columns")
    data = frame.loc[:, list(predictors) + [target]].copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data[np.isfinite(data[target])].reset_index(drop=True)
    if len(data) < n_folds:
        raise ValueError("Insufficient numeric outcome observations")
    X, y = data[list(predictors)], data[target]
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = []
    for name, model in models.items():
        stats = []
        for train_idx, test_idx in splitter.split(X):
            train, test = X.iloc[train_idx], X.iloc[test_idx]
            prep = regression_preprocessor(train)
            pipeline = Pipeline([("features", prep), ("model", clone(model))])
            pipeline.fit(train, y.iloc[train_idx])
            estimated = pipeline.predict(test)
            stats.append({"MAE": mean_absolute_error(y.iloc[test_idx], estimated),
                          "RMSE": float(np.sqrt(mean_squared_error(y.iloc[test_idx], estimated))),
                          "R2": (r2_score(y.iloc[test_idx], estimated)
                                 if len(test_idx) >= 2 else float("nan"))})
        results.append({"Target": target, "Model": name, "Folds": n_folds,
                        "Valid_records": len(y),
                        **{metric + "_mean": float(np.nanmean([v[metric] for v in stats]))
                           for metric in stats[0]},
                        **{metric + "_std": float(np.nanstd([v[metric] for v in stats]))
                           for metric in stats[0]}})
    return pd.DataFrame(results).sort_values("RMSE_mean")
