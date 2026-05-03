from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FallPrediction:
    is_fall: bool
    confidence: float
    mode: str


@dataclass
class TrainedImuFallModel:
    model: Any
    feature_columns: list[str]
    mode: str

    def predict(self, features: dict[str, float]) -> FallPrediction:
        row = pd.DataFrame([{column: features[column] for column in self.feature_columns}])

        if self.mode == "supervised":
            probabilities = self.model.predict_proba(row)[0]
            classes = list(self.model.classes_)
            fall_index = classes.index(1)
            confidence = float(probabilities[fall_index])
            return FallPrediction(is_fall=confidence >= 0.5, confidence=round(confidence, 3), mode=self.mode)

        anomaly = int(self.model.predict(row)[0]) == -1
        raw_score = -float(self.model.decision_function(row)[0])
        confidence = 1.0 / (1.0 + np.exp(-8.0 * raw_score))
        return FallPrediction(is_fall=anomaly, confidence=round(float(confidence), 3), mode=self.mode)


def train_model(feature_table: pd.DataFrame) -> TrainedImuFallModel:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.model_selection import train_test_split

    if "label" in feature_table.columns and feature_table["label"].nunique() >= 2:
        x = feature_table.drop(columns=["label"])
        y = feature_table["label"].astype(int)
        stratify = y if min(y.value_counts()) >= 2 else None
        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42, stratify=stratify)
        model = RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced")
        model.fit(x_train, y_train)
        return TrainedImuFallModel(model=model, feature_columns=list(x.columns), mode="supervised")

    x = feature_table.drop(columns=["label"], errors="ignore")
    model = IsolationForest(n_estimators=180, contamination=0.08, random_state=42)
    model.fit(x)
    return TrainedImuFallModel(model=model, feature_columns=list(x.columns), mode="anomaly")


def save_model(model: TrainedImuFallModel, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)


def load_model(path: str | Path) -> TrainedImuFallModel:
    return joblib.load(path)
