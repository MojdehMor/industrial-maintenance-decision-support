"""Regression model definitions for numerical maintenance outcomes."""
from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


def extract_numeric_value(value: object) -> float:
    
    if pd.isna(value):
        return float("nan")
    cleaned = str(value).strip().replace(" ", "")
    for unit in ("€", "$", "usd", "eur", "euro"):
        cleaned = cleaned.replace(unit, "")
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    result = re.search(r"[-+]?(?:\d*\.\d+|\d+)", cleaned)
    return float(result.group(0)) if result else float("nan")


def regression_preprocessor(
    frame: pd.DataFrame,
    *,
    numeric_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Generate a fold-compatible feature transformer.

    Column selection must be defined independently of target/outcome variables
    and audited for leakage by the researcher before evaluation.
    """
    if numeric_columns is None:
        numeric_columns = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])
                           and frame[c].notna().any()]
    if categorical_columns is None:
        categorical_columns = [c for c in frame if c not in numeric_columns]
    transforms = []
    if numeric_columns:
        transforms.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), list(numeric_columns)))
    if categorical_columns:
        transforms.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), list(categorical_columns)))
    if not transforms:
        raise ValueError("At least one feature is necessary")
    return ColumnTransformer(transforms)


def regression_models(seed: int = 42) -> Mapping[str, object]:
    """Core regressors and notebook hyperparameters (without optional add-ons)."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "SVR": SVR(kernel="rbf", C=10, epsilon=0.1),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=seed),
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "MLP Regressor": MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            max_iter=1000, random_state=seed),
    }
