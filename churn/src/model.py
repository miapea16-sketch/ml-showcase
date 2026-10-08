"""Customer churn modeling with preprocessing, comparison, and explainability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv"
TARGET = "Churn"
DROP_COLUMNS = ["customerID"]


@dataclass(frozen=True)
class ModelResult:
    name: str
    model: Pipeline
    metrics: dict[str, float]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and clean the public IBM Telco Customer Churn sample."""
    frame = pd.read_csv(path)
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame["TotalCharges"] = frame["TotalCharges"].fillna(frame["MonthlyCharges"])
    frame[TARGET] = (frame[TARGET] == "Yes").astype(int)
    return frame


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create transparent domain features without using the target."""
    features = frame.drop(columns=[TARGET, *DROP_COLUMNS], errors="ignore").copy()
    features["tenure_band"] = pd.cut(
        features["tenure"], bins=[-1, 6, 24, 48, np.inf], labels=["0-6", "7-24", "25-48", "49+"]
    ).astype(str)
    features["avg_monthly_spend"] = features["TotalCharges"] / features["tenure"].clip(lower=1)
    return features


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = features.select_dtypes(exclude=["object", "category"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def build_models(features: pd.DataFrame) -> dict[str, Pipeline]:
    """Return two strong baselines with identical preprocessing."""
    preprocessor = make_preprocessor(features)
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(features)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def train_and_compare() -> tuple[dict[str, ModelResult], pd.DataFrame, pd.Series]:
    """Train both models on one stratified split and return comparable metrics."""
    frame = load_data()
    features = make_features(frame)
    labels = frame[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    results: dict[str, ModelResult] = {}
    for name, model in build_models(features).items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions)),
            "recall": float(recall_score(y_test, predictions)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }
        results[name] = ModelResult(name, model, metrics)
    return results, x_test, y_test


def best_result(results: dict[str, ModelResult]) -> ModelResult:
    """Select the model with the best ROC AUC on the holdout set."""
    return max(results.values(), key=lambda result: result.metrics["roc_auc"])


def predict_customer(model: Pipeline, customer: dict[str, Any]) -> tuple[int, float]:
    """Return a churn prediction and probability for one customer."""
    row = pd.DataFrame([customer])
    row["tenure_band"] = pd.cut(
        row["tenure"], bins=[-1, 6, 24, 48, np.inf], labels=["0-6", "7-24", "25-48", "49+"]
    ).astype(str)
    row["avg_monthly_spend"] = row["TotalCharges"] / row["tenure"].clip(lower=1)
    probability = float(model.predict_proba(row)[:, 1][0])
    return int(probability >= 0.5), probability


def feature_importance(
    result: ModelResult, x_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Estimate global feature importance using permutation importance."""
    importance = permutation_importance(
        result.model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return pd.DataFrame(
        {"feature": x_test.columns, "importance": importance.importances_mean}
    ).sort_values("importance", ascending=False)


def classification_report_for(
    result: ModelResult, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, Any]:
    return classification_report(y_test, result.model.predict(x_test), output_dict=True)
