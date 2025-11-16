"""Metadata helpers for the demo (class names, etc.)."""
from __future__ import annotations

import json
from typing import Optional, Sequence

import pandas as pd

from .paths import DATA_DIR, BASE_DIR


def load_class_names() -> Optional[Sequence[str]]:
    path = BASE_DIR / "class_names.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass

    img_csv = DATA_DIR / "image_features.csv"
    if img_csv.exists():
        df = pd.read_csv(img_csv)
        if "label" in df.columns:
            labels = sorted(df["label"].unique().tolist())
            return labels
    return None
