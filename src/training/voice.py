"""Training utilities for the voice authentication model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from formative_demo.paths import DATA_DIR, MODEL_DIR

MODEL_PATH = MODEL_DIR / "voice_auth_model_v2.pkl"
SCALER_PATH = MODEL_DIR / "voice_scaler_v2.pkl"
ENCODER_PATH = MODEL_DIR / "speaker_encoder_v2.pkl"
METRICS_PATH = MODEL_DIR / "voice_model_v2_metrics.json"


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Audio features CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in {"filename", "speaker", "phrase"}]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["speaker"].astype(str).str.strip().str.lower().to_numpy()
    return X, y, feature_cols


def train_voice_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
):
    classifier = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=random_state,
        verbose=True,
    )

    classifier.fit(X_train, y_train)
    val_pred = classifier.predict(X_val)
    report = classification_report(
        y_val,
        val_pred,
        output_dict=True,
        zero_division=0,
    )
    return classifier, report


def run_voice_training(test_size: float = 0.2, random_state: int = 42) -> dict:
    X, y, feature_cols = load_dataset(DATA_DIR / "audio_features.csv")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled,
        y_encoded,
        test_size=test_size,
        stratify=y_encoded,
        random_state=random_state,
    )

    classifier, report = train_voice_classifier(X_train, y_train, X_val, y_val, random_state)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    metrics = {
        "val_accuracy": float(report.get("accuracy", 0.0)),
        "per_class": {
            label: metrics.get("f1-score", 0.0)
            for label, metrics in report.items()
            if label in label_encoder.classes_
        },
        "feature_columns": feature_cols,
        "iterations": int(getattr(classifier, "n_iter_", 0)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train voice authentication model (dense)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    metrics = run_voice_training(test_size=args.test_size, random_state=args.random_state)

    print("Validation accuracy:", metrics["val_accuracy"])
    for cls, score in metrics["per_class"].items():
        print(f" - {cls}: F1={score:.3f}")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":  # noqa: D401
    main()
