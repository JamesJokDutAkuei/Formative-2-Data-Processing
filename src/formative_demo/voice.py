"""Voice authentication helpers for the Formative demo CLI."""
from __future__ import annotations

import json
from typing import Dict, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from tensorflow.keras.models import load_model  # type: ignore[import]
except Exception:  # pragma: no cover - tensorflow optional
    load_model = None

from .paths import DATA_DIR, MODEL_DIR

VOICE_PROB_THRESHOLD = 0.6
VOICE_MODEL_V2_PATH = MODEL_DIR / "voice_auth_model_v2.pkl"
VOICE_SCALER_V2_PATH = MODEL_DIR / "voice_scaler_v2.pkl"
VOICE_ENCODER_V2_PATH = MODEL_DIR / "speaker_encoder_v2.pkl"
VOICE_METRICS_V2_PATH = MODEL_DIR / "voice_model_v2_metrics.json"


def load_audio_features() -> pd.DataFrame:
    csv_path = DATA_DIR / "audio_features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Audio features CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_voice_model():
    vm_path = MODEL_DIR / "voice_auth_model_5class.h5"
    if not vm_path.exists():
        raise FileNotFoundError(f"Voice model not found: {vm_path}")
    if load_model is None:
        raise RuntimeError("TensorFlow/Keras not available to load voice model")
    return load_model(str(vm_path))


def load_voice_label_encoder():
    encoder_path = MODEL_DIR / "speaker_encoder_5.pkl"
    if not encoder_path.exists():
        return None
    try:
        return joblib.load(str(encoder_path))
    except Exception:
        return None


def _load_v2_feature_columns() -> Optional[Sequence[str]]:
    if not VOICE_METRICS_V2_PATH.exists():
        return None
    try:
        metrics = json.loads(VOICE_METRICS_V2_PATH.read_text())
    except Exception:
        return None
    feature_cols = metrics.get("feature_columns")
    if isinstance(feature_cols, list):
        return feature_cols
    return None


def load_voice_assets() -> Dict[str, object]:
    if VOICE_MODEL_V2_PATH.exists():
        try:
            model = joblib.load(str(VOICE_MODEL_V2_PATH))
        except Exception as exc:  # pragma: no cover - IO errors
            raise RuntimeError(f"Failed to load voice model v2: {exc}") from exc

        scaler = None
        if VOICE_SCALER_V2_PATH.exists():
            try:
                scaler = joblib.load(str(VOICE_SCALER_V2_PATH))
            except Exception as exc:  # pragma: no cover - IO errors
                raise RuntimeError(f"Failed to load voice scaler v2: {exc}") from exc

        encoder = None
        if VOICE_ENCODER_V2_PATH.exists():
            try:
                encoder = joblib.load(str(VOICE_ENCODER_V2_PATH))
            except Exception as exc:  # pragma: no cover - IO errors
                raise RuntimeError(f"Failed to load voice encoder v2: {exc}") from exc
        if encoder is None:
            encoder = load_voice_label_encoder()

        return {
            "model": model,
            "scaler": scaler,
            "encoder": encoder,
            "mode": "mlp",
            "feature_columns": _load_v2_feature_columns(),
        }

    model = load_voice_model()
    voice_encoder = load_voice_label_encoder()
    return {
        "model": model,
        "scaler": None,
        "encoder": voice_encoder,
        "mode": "conv",
        "feature_columns": None,
    }


def predict_voice(
    voice_model,
    audio_df: pd.DataFrame,
    filename: str,
    *,
    scaler=None,
    mode: str = "conv",
    feature_columns: Optional[Sequence[str]] = None,
) -> Tuple[int, float]:
    row = audio_df[audio_df["filename"] == filename]
    if row.shape[0] == 0:
        raise FileNotFoundError(f"Audio features for '{filename}' not found in CSV")

    drop_cols = [c for c in ["filename", "speaker", "phrase"] if c in row.columns]
    feature_values = row.drop(columns=drop_cols).fillna(0.0)

    if feature_columns:
        missing = [c for c in feature_columns if c not in feature_values.columns]
        if missing:
            raise ValueError(f"Missing audio feature columns: {missing}")
        feature_values = feature_values[feature_columns]

    if mode == "mlp":
        feats = feature_values.to_numpy(dtype=np.float32)
        if scaler is not None:
            feats = scaler.transform(feats)
        if hasattr(voice_model, "predict_proba"):
            probs = voice_model.predict_proba(feats)[0]
        else:  # pragma: no cover - fallback path
            preds = voice_model.predict(feats)
            probs = np.asarray(preds).reshape(-1)
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        return top_idx, top_prob

    flat = feature_values.to_numpy(dtype=np.float32).reshape(-1)
    if flat.size % 94 != 0:
        raise ValueError(f"Unexpected audio feature length {flat.size}; cannot reshape to 94-frame tensor")
    frame_width = flat.size // 94
    spec = flat.reshape(94, frame_width)
    spec = spec[:, :13]
    spec = spec.reshape(1, 94, 13, 1)
    preds = voice_model.predict(spec)
    speaker_probs = preds[0] if isinstance(preds, (list, tuple)) else preds
    speaker_probs = np.asarray(speaker_probs).reshape(-1)
    top_idx = int(np.argmax(speaker_probs))
    top_prob = float(speaker_probs[top_idx])
    return top_idx, top_prob
