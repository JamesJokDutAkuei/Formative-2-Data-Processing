"""Centralised filesystem paths used across the demo."""
from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
ASSET_DIR = BASE_DIR / "assets"
