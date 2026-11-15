"""Content-based, case-based and hybrid maintenance-action recommendation.

Pure algorithmic components: nothing is loaded from disk or uploaded at import.
No private action labels, industrial cases, model weights or part IDs are stored.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_WEIGHTS = {"ML": 0.80, "CBR": 0.15, "Content": 0.05}
KNOWN_CONTENT_FEATURES = frozenset({"age", "business", "affected_component"})
MISSING_LABELS = frozenset({"", "none", "unknown", "nan", "n/a", "na",
                            "not provided", "unspecified", "missing", "null"})


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def known_category(value: object) -> bool:
    return clean_text(value) not in MISSING_LABELS


def normvec(values: Sequence[float]) -> np.ndarray:
    x = np.maximum(np.nan_to_num(np.asarray(values, dtype=float), nan=0.0,
                                  posinf=0.0, neginf=0.0), 0.0)
    return x / x.sum() if x.sum() > 0 else np.zeros_like(x)


def weighted_vote(
    indices: Sequence[int], similarities: Sequence[float],
    target_labels: Sequence[str], classes: Sequence[str],
) -> np.ndarray:
    class_index = {label: idx for idx, label in enumerate(classes)}
    out = np.zeros(len(classes), dtype=float)
    for idx, score in zip(indices, similarities):
        if score > 0:
            out[class_index[target_labels[int(idx)]]] += float(score)
    return normvec(out)


def popularity_scores(labels: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    counts = Counter(labels)
    return normvec([counts[a] for a in classes])


def content_scores(
    query_age: float,
    query_business: str,
    query_component: str,
    train_ages: Sequence[float],
    train_businesses: Sequence[str],
    train_components: Sequence[str],
    labels: Sequence[str],
    classes: Sequence[str],
    *,
    enabled_features: Sequence[str] = ("age", "business"),
    neighbors: int = 15,
) -> np.ndarray:
    """Top-neighbor content voting with equal weights among observed features.

    Numeric age uses inverse absolute distance; categorical attributes use exact
    match. Missing query features are skipped. If no selected signal is observed,
    content abstains with an all-zero vector; no popularity fallback is applied.
    """
    feature_set = tuple(enabled_features)
    if not feature_set or len(feature_set) != len(set(feature_set)) or not set(feature_set) <= KNOWN_CONTENT_FEATURES:
        raise ValueError("Unrecognized, duplicate, or empty enabled_features")
    n = len(labels)
    if any(len(seq) != n for seq in (train_ages, train_businesses, train_components)):
        raise ValueError("Training feature lengths do not match labels")
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    components = []
    ages = np.asarray(train_ages, dtype=float)
    businesses = np.asarray(train_businesses, dtype=object)
    part_types = np.asarray(train_components, dtype=object)
    if "age" in feature_set and np.isfinite(query_age):
        s = np.zeros(n, float)
        observed = np.isfinite(ages)
        s[observed] = 1.0 / (1.0 + np.abs(ages[observed] - float(query_age)))
        components.append(s)
    if "business" in feature_set and known_category(query_business):
        observed = np.array([known_category(v) for v in businesses], bool)
        components.append((observed & (businesses == query_business)).astype(float))
    if "affected_component" in feature_set and known_category(query_component):
        observed = np.array([known_category(v) for v in part_types], bool)
        components.append((observed & (part_types == query_component)).astype(float))
    if not components:
        return np.zeros(len(classes), float)
    scores = np.mean(components, axis=0)
    candidates = np.flatnonzero(scores > 0)
    if len(candidates) == 0:
        return np.zeros(len(classes), float)
    nearest = candidates[np.argsort(-scores[candidates], kind="stable")[:neighbors]]
    return weighted_vote(nearest, scores[nearest], labels, classes)


def sbert_embeddings(texts: Sequence[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """Encode incident descriptions using a pretrained, label-independent encoder.

    Requires the optional `sentence-transformers` dependency. Embeddings remain
    in memory and MUST NOT be added to a public source repository.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return np.asarray(model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


def cbr_scores(
    similarities: Sequence[float],
    train_businesses: Sequence[str],
    query_business: str,
    labels: Sequence[str],
    classes: Sequence[str],
    *,
    neighbors: int = 15,
) -> np.ndarray:
    """Business-gated top-neighbor similarity-weighted case-based retrieval."""
    sim = np.clip(np.asarray(similarities, dtype=float), 0.0, 1.0)
    if len(sim) != len(labels) or len(train_businesses) != len(labels):
        raise ValueError("Similarity, business, and target sizes differ")
    business = np.asarray(train_businesses, dtype=object)
    peers = np.flatnonzero(business == query_business)
    if not len(peers):
        peers = np.arange(len(labels))
    nearest = peers[np.argsort(-sim[peers], kind="stable")[:neighbors]]
    return weighted_vote(nearest, sim[nearest], labels, classes)


def make_internal_ml(seed: int = 42) -> Pipeline:
    """TF-IDF text + business one-hot + scaled age, logistic regression.

    This is the internal ML component of the hybrid, not an additional separately
    reported recommendation baseline in the manuscript.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformer = ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), "incident_text"),
        ("business", categorical, ["business"]),
        ("age", numeric, ["age"]),
    ])
    return Pipeline([
        ("features", transformer),
        ("model", LogisticRegression(max_iter=3000, random_state=seed)),
    ])


def fuse_scores(
    blocks: Mapping[str, Sequence[float]],
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> np.ndarray:
    if not weights or not all(k in blocks for k in weights):
        raise ValueError("Every weighted component must be present in blocks")
    if any(not np.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Fusion weights must be finite and nonnegative")
    arrays = [np.asarray(blocks[name], dtype=float) * weight
              for name, weight in weights.items()]
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("Score vectors have inconsistent lengths")
    return normvec(np.sum(arrays, axis=0))


def rank_actions(scores: Sequence[float], classes: Sequence[str]) -> list[str]:
    vector = np.asarray(scores, dtype=float)
    if len(vector) != len(classes):
        raise ValueError("Score and action-vector lengths differ")
    if not np.any(vector > 0):
        return []
    index = {action: i for i, action in enumerate(classes)}
    return sorted(classes, key=lambda action: (-vector[index[action]], action))
