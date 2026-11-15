"""Intervention categorization and classification model definitions.

Industry-specific mapping dictionaries and labels are intentionally NOT bundled.
Callers must supply approved keyword rules and their own tabular features.
"""
from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def normalize_text(value: object) -> str:
    """Return case-folded, whitespace-normalized text; missing values become empty."""
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def ordered_keyword_map(value: object,
                        rules: Sequence[tuple[str, str]],
                        default: str) -> str:
    """Assign first matching class in a caller-approved ordered rule list.

    Rule order is significant, matching the rule-based workflow of the source
    research. No project-specific labels or keywords are embedded here.
    """
    text = normalize_text(value)
    if not text:
        return default
    for keyword, category in rules:
        token = normalize_text(keyword)
        if token and token in text:
            return str(category)
    return default


def categorize_interventions(
    frame: pd.DataFrame,
    primary_rules: Sequence[tuple[str, str]],
    secondary_rules: Sequence[tuple[str, str]],
    *,
    action_column: str = "action",
    details_column: str = "details",
    primary_default: str = "other/unspecified",
    secondary_default: str = "no secondary intervention specified",
) -> pd.DataFrame:
    """Return a copy with primary, secondary and secondary-required labels.

    The caller is responsible for ensuring that the rule taxonomy is approved
    for public use. No examples or source records are stored by the module.
    """
    missing = {action_column, details_column} - set(frame.columns)
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(sorted(missing)))
    out = frame.copy()
    out["Primary_Class"] = out[action_column].map(
        lambda x: ordered_keyword_map(x, primary_rules, primary_default))
    out["Secondary_Class"] = out[details_column].map(
        lambda x: ordered_keyword_map(x, secondary_rules, secondary_default))
    out["Secondary_Binary"] = out["Secondary_Class"].ne(secondary_default)
    return out


def classification_preprocessor(
    categorical_columns: Sequence[str], numeric_columns: Sequence[str]
) -> ColumnTransformer:
    """Imputation + one-hot encoding for categories, scaling for numbers."""
    transforms = []
    if categorical_columns:
        transforms.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), list(categorical_columns)))
    if numeric_columns:
        transforms.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), list(numeric_columns)))
    if not transforms:
        raise ValueError("At least one feature column must be supplied")
    return ColumnTransformer(transforms)


def classification_models(seed: int = 42) -> Mapping[str, object]:
    """Core classifier family from the research categorization study."""
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=seed),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=0.7, class_weight="balanced", solver="lbfgs"),
        "Linear SVM": SVC(kernel="linear", C=1.0, class_weight="balanced"),
        "RBF SVM": SVC(kernel="rbf", C=1.0, class_weight="balanced"),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                             early_stopping=True, random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=10, class_weight="balanced_subsample",
            random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=seed),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250, max_depth=None, class_weight="balanced", random_state=seed),
    }
