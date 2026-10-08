"""Reusable training and inference helpers for the Iris showcase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FEATURE_NAMES = ["sepal length", "sepal width", "petal length", "petal width"]


@dataclass(frozen=True)
class Evaluation:
    """Evaluation results that are easy to print or render in the app."""

    accuracy: float
    report: dict[str, Any]
    confusion: np.ndarray


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return features, labels, and class names from the canonical Iris dataset."""
    dataset = load_iris()
    return dataset.data, dataset.target, list(dataset.target_names)


def build_pipeline() -> Pipeline:
    """Build the model pipeline with preprocessing kept inside the estimator."""
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, random_state=RANDOM_STATE)),
        ]
    )


def train_model() -> tuple[Pipeline, Evaluation, list[str]]:
    """Train on a stratified split and return the fitted model and metrics."""
    features, labels, class_names = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    model = build_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    evaluation = Evaluation(
        accuracy=float(accuracy_score(y_test, predictions)),
        report=classification_report(
            y_test, predictions, target_names=class_names, output_dict=True, zero_division=0
        ),
        confusion=confusion_matrix(y_test, predictions),
    )
    return model, evaluation, class_names


def predict_one(model: Pipeline, measurements: list[float]) -> tuple[str, np.ndarray]:
    """Predict a single flower and return the class name plus class probabilities."""
    _, _, class_names = load_data()
    values = np.asarray(measurements, dtype=float).reshape(1, -1)
    probabilities = model.predict_proba(values)[0]
    return class_names[int(model.predict(values)[0])], probabilities
