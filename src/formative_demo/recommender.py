"""Recommender utilities for the Formative demo CLI."""
from __future__ import annotations

from typing import List, Sequence

import joblib
import numpy as np
import pandas as pd

from .paths import MODEL_DIR


def recommend_stub() -> List[dict]:
    return [
        {"product_id": "demo_prod_1", "name": "Essential Pack", "score": 0.9},
        {"product_id": "demo_prod_2", "name": "Upgrade Bundle", "score": 0.7},
        {"product_id": "demo_prod_3", "name": "Accessory Kit", "score": 0.5},
    ]


def load_recommender_pipeline():
    pipeline_path = MODEL_DIR / "recommender_pipeline.pkl"
    if not pipeline_path.exists():
        return None
    try:
        return joblib.load(str(pipeline_path))
    except Exception:
        return None


def generate_recommendations(
    pipeline,
    merged_csv: str,
    customer_id: str,
    *,
    top_k: int = 3,
) -> Sequence[dict]:
    merged = pd.read_csv(merged_csv)
    key_cols = [c for c in merged.columns if "customer" in c.lower() or "id" in c.lower()]
    if key_cols:
        key = key_cols[0]
        row = merged[merged[key].astype(str) == str(customer_id)]
    else:
        first_col = merged.columns[0]
        row = merged[merged[first_col].astype(str) == str(customer_id)]

    if row.shape[0] == 0:
        return recommend_stub()

    feat_cols = pipeline.get("feature_columns")
    X_row = row[feat_cols]
    X_row = X_row.fillna(0)
    X_proc = pipeline["preprocessor"].transform(X_row)
    probs = pipeline["model"].predict_proba(X_proc)
    top_idx = np.argsort(probs, axis=1)[:, ::-1][:, :top_k][0]
    recommendations = []
    for idx in top_idx:
        label = pipeline["label_encoder"].inverse_transform([idx])[0]
        score = float(probs[0, idx])
        recommendations.append({"product_id": label, "name": label, "score": score})
    return recommendations
