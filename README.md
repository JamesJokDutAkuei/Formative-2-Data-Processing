# Formative 2 – Multimodal Access Control Demo

This project demonstrates an authentication flow that combines face recognition,
voice verification, and product recommendations. The latest refactor organizes
runtime code into a reusable `formative_demo` package and moves training logic
into `training`, making it easier to run the CLI demo or retrain individual
models.

The repository ships with precomputed features and trained models so the demo
can run out of the box. You can regenerate artefacts with the included training
scripts when new data becomes available.

---

## Repository Layout

```
.
├── assets/
│   └── demo_faces/                 # sample face images for the CLI
├── data/
│   └── processed/
│       ├── audio_features.csv      # MFCC-style tabular voice features
│       └── image_features.csv      # image feature metadata / class labels
├── models/
│   ├── face_authentication_model.keras
│   ├── voice_auth_model_5class.h5        # legacy CNN fallback
│   ├── voice_auth_model_v2.pkl           # preferred scikit-learn MLP
│   ├── voice_scaler_v2.pkl               # StandardScaler for the MLP
│   ├── speaker_encoder_5.pkl             # legacy label encoder
│   ├── speaker_encoder_v2.pkl            # label encoder for the MLP
│   ├── voice_model_v2_metrics.json       # feature order + validation metrics
│   ├── product_recommender.pkl           # best traditional ML recommender
│   └── recommender_pipeline.pkl          # preprocessing + classifier bundle
├── notebooks/
│   ├── Voiceprint.ipynb
│   └── xgb_model.ipynb
├── requirements.txt
├── src/
│   ├── formative_demo/
│   │   ├── __init__.py
│   │   ├── cli.py                 # command-line orchestration
│   │   ├── face.py                # face model loader + predictor
│   │   ├── metadata.py            # class-name helpers
│   │   ├── paths.py               # central path constants
│   │   ├── recommender.py         # recommender utilities
│   │   └── voice.py               # voice model loader + predictor
│   ├── training/
│   │   ├── __init__.py
│   │   ├── recommender.py         # retrains recommender pipeline
│   │   └── voice.py               # retrains MLP voice classifier
│   ├── demo_cli.py                # thin wrapper calling formative_demo.cli.main
│   ├── train_recommender.py       # wrapper calling training.recommender.main
│   └── train_voice_model_v2.py    # wrapper calling training.voice.main
└── README.md
```

---

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

TensorFlow is required to run the face model and the legacy voice CNN. The CLI
detects missing TensorFlow imports and will raise a clear error if you attempt
to authenticate faces without it.

---

## Running the Demo CLI

```bash
python src/demo_cli.py \
  --face assets/demo_faces/demo-2.jpeg \
  --voice James_confirm_1.m4a
```

The CLI will:

- rebuild the MobileNetV2-based face head and score the supplied image against
  known classes (from `class_names.json` or the labels in
  `data/processed/image_features.csv`)
- load `data/processed/audio_features.csv`, apply the saved scaler, and predict
  a speaker label with `voice_auth_model_v2.pkl` (falling back to the legacy
  CNN if the new model is absent)
- deny access if face/voice predictions disagree—unless the voice result is the
  explicit `unknown` class
- attempt to produce real product recommendations via
  `models/recommender_pipeline.pkl`, otherwise emit deterministic stub items

### Optional Recommender Parameters

```
--recommender-customer-id ID    Customer identifier to score.
--merged-csv PATH_OR_URL        CSV containing the features expected by the pipeline.
```

Both flags are required for live recommendations. The CLI looks up the supplied
customer row inside the merged dataset, transforms the features via the stored
pipeline, and reports the top predictions. If either flag is omitted, the CLI
still grants access but falls back to the stub recommendations.

---

## Regenerating Models

### Voice Authentication (MLP v2)

```
python src/train_voice_model_v2.py --test-size 0.2 --random-state 42
```

This delegates to `training.voice`. It reads
`data/processed/audio_features.csv`, standardizes the features, trains an
`MLPClassifier`, and writes the classifier, scaler, label encoder, and metrics
JSON. The metrics file records the feature column order, which the CLI enforces
during inference.

### Product Recommender

```
python src/train_recommender.py \
  --transactions path/to/transactions.csv \
  --social path/to/social.csv \
  [--tune]
```

The training module merges customer data, builds a preprocessing pipeline
(numeric scaling + categorical one-hot encoding), evaluates Random Forest,
Logistic Regression, and optionally XGBoost, and persists the best model along
with its preprocessing steps.

---

## Data & Artefact Contracts

- `data/processed/audio_features.csv` must include a `filename` column alongside
  numeric features—these names are used on the CLI via `--voice`.
- `models/voice_model_v2_metrics.json` lists the feature column order for the
  MLP; do not reorder the columns without updating the file.
- `class_names.json` (if present at the repo root) should align with the face
  model output indices. When absent, the CLI derives class names from the
  `label` column in `image_features.csv`.
- `models/recommender_pipeline.pkl` should contain the keys
  `preprocessor`, `model`, `label_encoder`, and `feature_columns` so the CLI can
  reproduce the exact feature engineering used at training time.

---

## Troubleshooting Tips

- **TensorFlow import errors:** install the version pinned in
  `requirements.txt`, or modify `formative_demo.face` to use an alternative
  embedding extractor that matches your environment.
- **Voice filename not found:** confirm the string passed to `--voice`
  exactly matches a row in `audio_features.csv`.
- **Identity mismatch despite same person:** check for casing/spacing
  differences between face labels and voice encoder classes. The CLI lowercases
  both before comparison but cannot map entirely different spellings.
- **Stub recommendations appearing:** supply both `--recommender-customer-id`
  and `--merged-csv` so the pipeline can look up the correct row.

---

Feel free to open an issue or continue refactoring—this structure is intended to
be a solid baseline for further multimodal experiments.
