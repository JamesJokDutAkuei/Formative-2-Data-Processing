"""Training utilities for the product recommender."""
from __future__ import annotations

import argparse
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, top_k_accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:  # pragma: no cover - optional dependency missing
    HAS_XGB = False

from formative_demo.paths import MODEL_DIR

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path_or_url: str) -> pd.DataFrame:
    return pd.read_csv(path_or_url)


def build_dataset(transactions: pd.DataFrame, social: pd.DataFrame) -> pd.DataFrame:
    if "customer_id" in transactions.columns and "customer_id" in social.columns:
        transactions["customer_id"] = transactions["customer_id"].astype(str)
        social["customer_id"] = social["customer_id"].astype(str)
        merged = transactions.merge(social, on="customer_id", how="left")
    else:
        t0 = transactions.columns[0]
        s0 = social.columns[0]
        transactions[t0] = transactions[t0].astype(str)
        social[s0] = social[s0].astype(str)
        merged = transactions.merge(social, left_on=t0, right_on=s0, how="left")
    return merged


def preprocess_and_train(merged: pd.DataFrame, tune: bool = False):
    target_col = "product_category"
    if target_col not in merged.columns:
        raise RuntimeError(f"Expected target column `{target_col}` in merged data")
    merged = merged.dropna(subset=[target_col])
    y_raw = merged[target_col].astype(str)

    exclude_keywords = ["customer", "id", "filename"]
    exclude_cols = [c for c in merged.columns if any(k in c.lower() for k in exclude_keywords)]
    exclude_cols = list(set(exclude_cols + [target_col]))

    numeric_cols = [c for c in merged.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    categorical_cols = [c for c in merged.select_dtypes(include=["object", "category"]).columns if c not in exclude_cols]

    merged[numeric_cols] = merged[numeric_cols].fillna(0.0)
    merged[categorical_cols] = merged[categorical_cols].fillna("NA")

    X_raw = merged[numeric_cols + categorical_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers.append(("cat", ohe, categorical_cols))

    preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)

    le_y = LabelEncoder()
    y = le_y.fit_transform(y_raw)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor.fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    candidates = {
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
        "lr": LogisticRegression(max_iter=2000),
    }
    if HAS_XGB:
        candidates["xgb"] = XGBClassifier(
            eval_metric="mlogloss",
            use_label_encoder=False,
            n_estimators=500,
            random_state=42,
        )

    if tune:
        print("Running RandomizedSearchCV for RandomForest (lightweight)...")
        param_dist_rf = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }
        rs_rf = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_distributions=param_dist_rf,
            n_iter=20,
            cv=3,
            scoring="f1_weighted",
            n_jobs=-1,
            random_state=42,
        )
        rs_rf.fit(X_train, y_train)
        print("RandomizedSearchCV RF best params:", rs_rf.best_params_)
        candidates["rf"] = rs_rf.best_estimator_

    best_model = None
    best_score = -1.0
    results = {}
    for name, model in candidates.items():
        if name == "xgb" and HAS_XGB:
            xgb_eval_x = X_train[: int(0.1 * X_train.shape[0])]
            xgb_eval_y = y_train[: int(0.1 * len(y_train))]
            try:
                model.fit(X_train, y_train, eval_set=[(xgb_eval_x, xgb_eval_y)], early_stopping_rounds=10, verbose=False)
            except TypeError:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        ll = log_loss(y_test, probs) if probs is not None else float("nan")
        top3 = top_k_accuracy_score(y_test, probs, k=3) if probs is not None else float("nan")
        results[name] = {"accuracy": acc, "f1_weighted": f1, "log_loss": ll, "top3": top3}
        score = f1
        if score > best_score:
            best_score = score
            best_model = model

    out_model_path = MODEL_DIR / "product_recommender.pkl"
    out_pipeline_path = MODEL_DIR / "recommender_pipeline.pkl"
    joblib.dump(best_model, out_model_path)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": best_model,
            "label_encoder": le_y,
            "feature_columns": numeric_cols + categorical_cols,
        },
        out_pipeline_path,
    )
    return results, out_model_path, out_pipeline_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train recommender from transactions + social CSVs")
    parser.add_argument("--transactions", required=True, help="Path or URL to transactions CSV")
    parser.add_argument("--social", required=True, help="Path or URL to social CSV")
    parser.add_argument("--tune", action="store_true", help="Run lightweight hyperparameter tuning for RandomForest")
    args = parser.parse_args(argv)

    print("Loading transactions...")
    tx = load_csv(args.transactions)
    print("Loading social...")
    so = load_csv(args.social)
    print("Merging...")
    merged = build_dataset(tx, so)

    print("Computing customer aggregates (RFM) when possible...")
    cust_keys = [c for c in tx.columns if "customer" in c.lower() or "id" in c.lower()]
    date_candidates = [c for c in tx.columns if any(k in c.lower() for k in ["date", "time", "created"])]
    amount_candidates = [c for c in tx.columns if any(k in c.lower() for k in ["amount", "price", "total", "value"])]
    if cust_keys and date_candidates and amount_candidates:
        ck = cust_keys[0]
        dtc = date_candidates[0]
        amt = amount_candidates[0]
        try:
            tx[dtc] = pd.to_datetime(tx[dtc], errors="coerce")
            snapshot = tx[dtc].max()
            rfm = tx.groupby(ck).agg(
                recency_days=(dtc, lambda s: (snapshot - s.max()).days if pd.notnull(s.max()) else 9999),
                frequency=(dtc, "count"),
                monetary=(amt, "sum"),
            ).reset_index()
            product_cols = [c for c in tx.columns if any(k in c.lower() for k in ["product", "item", "sku"])]
            if product_cols:
                pcol = product_cols[0]
                prod_counts = (
                    tx.groupby(ck)[pcol]
                    .nunique()
                    .reset_index()
                    .rename(columns={pcol: "unique_product_count"})
                )
                last_prod = (
                    tx.sort_values(dtc)
                    .groupby(ck)[pcol]
                    .last()
                    .reset_index()
                    .rename(columns={pcol: "last_product"})
                )
                rfm = rfm.merge(prod_counts, on=ck, how="left")
                rfm = rfm.merge(last_prod, on=ck, how="left")
            merge_keys = [c for c in merged.columns if ck.lower() in c.lower() or "customer" in c.lower()]
            if merge_keys:
                mk = merge_keys[0]
                merged = merged.merge(rfm, left_on=mk, right_on=ck, how="left")
            else:
                merged = merged.merge(rfm, left_on=ck, right_on=ck, how="left")
            print(
                "RFM merged. Added columns:",
                [c for c in ["recency_days", "frequency", "monetary"] if c in merged.columns],
            )
        except Exception as exc:
            print("RFM computation failed:", exc)
    else:
        print("RFM not computed: needed customer/date/amount columns not found.")

    print("Training candidate models...")
    results, model_path, pipeline_path = preprocess_and_train(merged, tune=args.tune)
    print("Training results:")
    for name, metrics in results.items():
        print(name, metrics)
    print("Saved recommender model to", model_path)
    print("Saved recommender pipeline to", pipeline_path)


if __name__ == "__main__":  # noqa: D401
    main()
