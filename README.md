# Formative 2 — Multimodal Data Preprocessing & Product Recommendation

**Ready-to-copy `README.md`** for your GitHub repository. Paste this directly into your repo.

---

# Formative 2 — Multimodal Data Preprocessing & Product Recommendation

**Course:** Data Preprocessing (Formative 2)
**Due:** Sunday by 11:59 PM
**Points:** 40

## Project Overview

This project implements a *User Identity and Product Recommendation System Flow* that authenticates a user using **facial recognition** and **voice verification** before allowing the **product recommendation** model to run. The pipeline merges tabular customer profiles and transaction data, processes image and audio samples, engineers multimodal features, trains three models (face recognition, voiceprint verification, product recommendation), and includes a command-line mini-app to simulate authorized and unauthorized transactions.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── customer_social_profiles.csv
│   │   ├── customer_transactions.csv
│   │   ├── images/                 # raw images per member: memberID_img1.jpg ...
│   │   └── audio/                  # raw audio per member: memberID_phrase1.wav ...
│   ├── processed/
│   │   ├── merged_customers.csv
│   │   ├── image_features.csv
│   │   └── audio_features.csv
├── notebooks/
│   └── EDA_and_Feature_Engineering.ipynb
├── src/
│   ├── preprocess_merge.py         # merges tabular data & feature engineering
│   ├── image_processing.py         # augmentations & feature extraction
│   ├── audio_processing.py         # augmentations & feature extraction
│   ├── train_models.py             # trains 3 models + saves artifacts
│   ├── evaluate_models.py          # evaluation metrics & plots
│   └── demo_cli.py                 # command-line simulator (auth + predict)
├── models/
│   ├── face_recognition_model.pkl
│   ├── voiceprint_model.pkl
│   └── product_recommender.pkl
├── reports/
│   ├── Final_Report.pdf
│   └── contributions.md
└── assets/
    └── system_flow_screenshot.png
```

---

## Quick Setup

1. Clone the repo:

```bash
git clone <YOUR_REPO_URL>
cd <REPO_DIR>
```

2. Create Python environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

3. `requirements.txt` (recommended)

```
numpy
pandas
scikit-learn
opencv-python
matplotlib
librosa
soundfile
python_speech_features
joblib
face_recognition         # optional, for quick face embeddings (dlib based)
tensorflow               # or torch if using deep embeddings
xgboost
seaborn
jupyterlab
```

> *Pick either `tensorflow` or `torch` depending on your embedding approach. If `face_recognition` / `dlib` is problematic to install in CI, use a lightweight embedding approach (OpenCV + pretrained nets).*

---

## Datasets & Files (placeholders you must replace with your files)

* `data/raw/customer_social_profiles.csv` — user profile features (user_id, name, age, location, interests, social_activity_metrics, etc.)
* `data/raw/customer_transactions.csv` — historical transactions (user_id, product_id, category, timestamp, amount, device, purchase_context, etc.)
* `data/raw/images/` — each group member must upload ≥3 images: `neutral`, `smile`, `surprised` (filenames like `user123_neutral.jpg`).
* `data/raw/audio/` — each member must upload ≥2 voice samples: short phrases (e.g., `user123_confirm.wav`, `user123_yes_approve.wav`).

---

## Task Walkthrough (what each script does)

### 1. `src/preprocess_merge.py`

* Loads `customer_social_profiles.csv` and `customer_transactions.csv`.
* Merges on `user_id` (left join or inner depending on instructions).
* Cleans columns: handles missing values, converts timestamps to datetime, encodes categorical variables (one-hot or label encoding), aggregates transaction-level features (e.g., avg_amount, recency, frequency).
* Saves merged and engineered dataset to `data/processed/merged_customers.csv`.
* Example run:

```bash
python src/preprocess_merge.py --profiles data/raw/customer_social_profiles.csv \
    --transactions data/raw/customer_transactions.csv \
    --out data/processed/merged_customers.csv
```

**Typical engineered columns to include**

* `user_id`, `age`, `location`, `num_posts`, `avg_session_time`, `total_spent`, `purchase_count`, `fav_category`, `days_since_last_purchase`, `device_pref`, target `most_likely_product` (or product category).

---

### 2. `src/image_processing.py`

* Loads raw images from `data/raw/images/`.
* For each image:

  * Display sample images (matplotlib).
  * Apply augmentations (example: rotation ±15°, horizontal flip, slight zoom, convert to grayscale).
  * Extract features:

    * Option A (recommended): Use a pre-trained face embedding model (e.g., FaceNet, `face_recognition` face_encodings) → produces 128-D or 512-D embeddings.
    * Option B: color histograms (per channel), HOG descriptors, or a small CNN feature vector.
* Saves `data/processed/image_features.csv` with columns like:

  * `user_id`, `image_id`, `augmentation`, `embedding_0`, ..., `embedding_n`, `hist_bin_0`, ...
* Example run:

```bash
python src/image_processing.py --img_dir data/raw/images --out data/processed/image_features.csv
```

---

### 3. `src/audio_processing.py`

* Loads raw audio files from `data/raw/audio/`.
* For each sample:

  * Display waveform and spectrogram (matplotlib).
  * Apply augmentations: pitch shift (±2 semitones), time-stretch (0.9x, 1.1x), add background noise.
  * Extract audio features:

    * MFCCs (mean & std over frames)
    * Spectral roll-off
    * Zero-crossing rate
    * Energy (RMS)
  * Save `data/processed/audio_features.csv` with columns:

    * `user_id`, `audio_id`, `augmentation`, `mfcc_0_mean`, ..., `mfcc_12_mean`, `spectral_rolloff`, `rms_energy`, ...
* Example run:

```bash
python src/audio_processing.py --audio_dir data/raw/audio --out data/processed/audio_features.csv
```

---

### 4. `src/train_models.py`

* Loads `data/processed/merged_customers.csv`, `image_features.csv`, and `audio_features.csv`.
* Joins multimodal features by `user_id` into a single dataset for the product recommendation model.
* **Model 1 — Facial Recognition**

  * Train a classifier (e.g., Logistic Regression / Random Forest / SVM) on face embeddings to map embeddings -> `user_id` (or a binary authorized/not-authorized label).
  * Save model `models/face_recognition_model.pkl`.
* **Model 2 — Voiceprint Verification**

  * Train classifier on extracted audio features to map audio features -> `user_id` (or binary authorized).
  * Save model `models/voiceprint_model.pkl`.
* **Model 3 — Product Recommendation**

  * Use merged tabular + aggregated multi-modal signals to predict `most_likely_product` or `product_category`.
  * Models: RandomForest, XGBoost or Logistic Regression (for categories).
  * Evaluate with Accuracy, F1-Score, and Loss (cross entropy for multi-class).
  * Save model `models/product_recommender.pkl`.
* Example run:

```bash
python src/train_models.py --merged data/processed/merged_customers.csv \
    --image_feats data/processed/image_features.csv \
    --audio_feats data/processed/audio_features.csv \
    --out_dir models/
```

---

### 5. `src/evaluate_models.py`

* Loads saved models and test splits.
* Produces metrics table: Accuracy, Precision, Recall, F1-score, Confusion matrix.
* Saves evaluation outputs in `reports/` and optional plots in `assets/`.

Run:

```bash
python src/evaluate_models.py --models_dir models --test_data data/processed/merged_customers.csv
```

---

### 6. `src/demo_cli.py` — Command-line mini-app (system demonstration)

* Simulates a full transaction:

  1. Input face image path → face model returns `user_id` or `unauthorized`.
  2. If authorized, the app prompts for voice sample path → voice model returns `approve` or `deny`.
  3. If both pass, call the product recommender to show predicted product(s).
* Also includes an option to simulate unauthorized attempt with images/audio from unknown users or deliberately altered samples.
* Example usage:

```bash
# authorized demo
python src/demo_cli.py --face data/demo/user123_neutral.jpg --voice data/demo/user123_confirm.wav

# unauthorized demo
python src/demo_cli.py --face data/demo/unknown.jpg --voice data/demo/noise.wav
```

**Expected CLI output (example):**

```
[Face Auth] user123 — MATCH (confidence: 0.92)
[Voice Auth] user123 — APPROVED (confidence: 0.88)
[Recommender] Predicted product: 'EcoBottle 750ml' (prob=0.63)
```

---

## Feature Engineering Notes (recommended)

* **Tabular**

  * Fill missing numerical with median; categorical with mode or new category `Unknown`.
  * Time-based features: `days_since_last_login`, `hour_of_day`, `weekday`.
  * Aggregations per user: `total_spend`, `avg_purchase_value`, `purchase_frequency`.
* **Image**

  * Use facial alignment before embedding extraction.
  * L2-normalize embeddings.
  * Store augmentation metadata so you can avoid data leakage across train/test.
* **Audio**

  * Normalize volume (RMS).
  * Trim silence at ends.
  * Use delta + delta-delta MFCC stats if helpful.

---

## Evaluation & Metrics

* **Face & Voice models:** accuracy, F1-score, per-class recall/precision (or ROC-AUC for binary).
* **Product Recommender:** accuracy, macro/micro F1-score, top-3 accuracy (since product prediction can be multi-class).
* Save metrics CSV and confusion matrices in `reports/`.

---

## How to Submit

* **Report:** `reports/Final_Report.pdf` — includes methodology, preprocessing steps, augmentation details, feature lists, model configs, and results.
* **Video:** Include a public or unlisted video link (e.g., YouTube) demonstrating:

  * Data merge & EDA
  * Image & audio processing (show waveforms, spectrograms, sample augmentations)
  * Model training & evaluation
  * CLI demo: authorized & unauthorized attempts
* **GitHub:** include all scripts, Jupyter notebook, `data/processed/*.csv` (if allowed), and model artifacts (or instructions to regenerate).
* **Contributions:** `reports/contributions.md` — who did what (3–4 sentences per member).

---

## Team Contribution Template (copy to `reports/contributions.md`)

```
# Team contributions

- Member A (Name, Student ID)
  - Role: Data merging, tabular feature engineering, EDA notebook.
  - Contributions: Wrote `src/preprocess_merge.py`, prepared aggregated features, contributed to the final report.

- Member B (Name, Student ID)
  - Role: Image data collection & processing.
  - Contributions: Collected images, implemented augmentations, wrote `src/image_processing.py`, produced `image_features.csv`.

- Member C (Name, Student ID)
  - Role: Audio collection & processing, CLI demo.
  - Contributions: Collected audio samples, implemented augmentations, wrote `src/audio_processing.py`, developed `src/demo_cli.py`.

- Member D (Name, Student ID)
  - Role: Model training & evaluation.
  - Contributions: Implemented `src/train_models.py`, `src/evaluate_models.py`, generated metrics and plots, prepared the video demo.

# End
```

---

## Reproducibility checklist (to include in the repo)

* [ ] `requirements.txt` present and installable
* [ ] Scripts include `--seed` option for deterministic splits
* [ ] `README.md` includes dataset sources & expected file names
* [ ] Jupyter notebook `notebooks/EDA_and_Feature_Engineering.ipynb` documents explorations
* [ ] Video link and GitHub URL included in final report

---

## Helpful Tips & Troubleshooting

* If `face_recognition` fails to install on some machines, fallback to OpenCV + pre-trained `resnet` embeddings (TensorFlow/Keras) or use HOG descriptors with an SVM.
* Ensure audio sample rates are consistent (e.g., 16 kHz).
* When training with augmented images/audio, ensure test set contains only original (non-augmented) samples to avoid inflated metrics.
* Keep privacy in mind: if real user images/audio used, ensure consent and do not publish original files publicly.

---

## Placeholders you must replace

* `<YOUR_REPO_URL>` — replace with your GitHub URL.
* `reports/Final_Report.pdf` — attach final PDF.
* `VIDEO_LINK` — paste your system simulation video link.
* Team member names & student IDs in `reports/contributions.md`.

---

## Contact / Questions

If anything is unclear, open an issue in the repo or contact the team lead listed in `reports/contributions.md`.

---

### Short License (optional)

```
MIT License
```

---

Good luck — paste this `README.md` in your repository and update the placeholders (video link, repo URL, model names, and team contributions). If you want, I can also generate the `requirements.txt`, a starter `demo_cli.py`, or Jupyter notebook skeleton next. Which one should I produce now?
