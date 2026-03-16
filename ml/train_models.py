"""
EcomML – Model Training Pipeline
Dataset : eCommerce behavior data from multi-category store (2019-Oct.csv)
Source  : https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store

Models trained
--------------
1. Logistic Regression  (sklearn)
2. Random Forest        (sklearn)
3. K-Means Clustering   (sklearn)

Run
---
    pip install -r requirements.txt
    python train_models.py --data 2019-Oct.csv

Output
------
    models/logistic_regression.pkl
    models/random_forest.pkl
    models/kmeans.pkl
    models/scaler.pkl
    models/feature_columns.json
    reports/metrics_report.json
    reports/classification_report_rf.txt
    reports/classification_report_lr.txt
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix, f1_score,
    precision_score, recall_score
)

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
MODELS_DIR  = "models"
REPORTS_DIR = "reports"
RANDOM_SEED = 42
TEST_SIZE   = 0.20
N_CLUSTERS  = 4
RF_TUNING_ITERS = 18
RF_TUNING_MAX_SAMPLES = 120_000

# ──────────────────────────────────────────────
#  FEATURE ENGINEERING
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw event-level rows into one row per user_session.

    Raw columns expected (from 2019-Oct.csv):
        event_time, event_type, product_id, category_id,
        category_code, brand, price, user_id, user_session

    Engineered features:
        view_count        – number of 'view' events
        cart_count        – number of 'cart' events
        purchase_count    – number of 'purchase' events  (TARGET)
        session_duration  – max-min event_time in minutes
        avg_price_viewed  – mean price across all viewed products
        unique_categories – distinct category_code values seen
        unique_brands     – distinct brands interacted with
        brand_loyalty     – max brand frequency / total events (0–1)
        cart_to_view_ratio – cart_count / (view_count + 1)
        price_range       – max price – min price of products seen
        label             – 1 if user purchased in this session else 0
    """
    print("[feature_eng] Parsing event_time …")
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_time"])

    print("[feature_eng] Aggregating per user_session …")
    grp = df.groupby(["user_id", "user_session"])

    feat = pd.DataFrame()
    feat["view_count"]        = grp.apply(lambda x: (x["event_type"] == "view").sum())
    feat["cart_count"]        = grp.apply(lambda x: (x["event_type"] == "cart").sum())
    feat["purchase_count"]    = grp.apply(lambda x: (x["event_type"] == "purchase").sum())
    feat["session_duration"]  = grp["event_time"].apply(
        lambda x: (x.max() - x.min()).total_seconds() / 60
    )
    feat["avg_price_viewed"]  = grp["price"].mean().fillna(0)
    feat["unique_categories"] = grp["category_code"].nunique()
    feat["unique_brands"]     = grp["brand"].nunique()
    feat["brand_loyalty"]     = grp["brand"].apply(
        lambda x: x.dropna().value_counts().iloc[0] / len(x.dropna()) if len(x.dropna()) > 0 else 0
    )
    feat["cart_to_view_ratio"] = feat["cart_count"] / (feat["view_count"] + 1)
    feat["price_range"]        = grp["price"].apply(lambda x: x.max() - x.min()).fillna(0)

    feat = feat.reset_index()
    feat["label"] = (feat["purchase_count"] > 0).astype(int)

    print(f"[feature_eng] Shape after aggregation: {feat.shape}")
    print(f"[feature_eng] Purchase rate: {feat['label'].mean()*100:.2f}%")
    return feat


FEATURE_COLS = [
    "view_count", "cart_count", "session_duration",
    "avg_price_viewed", "unique_categories", "unique_brands",
    "brand_loyalty", "cart_to_view_ratio", "price_range"
]

# ──────────────────────────────────────────────
#  GENERATE SYNTHETIC DATA (used when CSV absent)
# ──────────────────────────────────────────────
def generate_synthetic_data(n: int = 100_000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Creates a realistic synthetic dataset that mirrors the 2019-Oct.csv
    feature distributions for demo / CI runs.
    """
    print(f"[synthetic] Generating {n:,} synthetic sessions …")
    rng = np.random.default_rng(seed)

    view_count        = rng.negative_binomial(5, 0.25, n).clip(0, 100).astype(float)
    cart_count        = (rng.negative_binomial(2, 0.5, n) * (view_count > 0)).clip(0, 30).astype(float)
    purchase_count    = (rng.negative_binomial(1, 0.85, n) * (cart_count > 0)).clip(0, 10).astype(float)
    session_duration  = rng.exponential(8, n).clip(0.1, 120)
    avg_price_viewed  = rng.lognormal(4.2, 1.0, n).clip(1, 2000)
    unique_categories = rng.integers(1, 9, n).astype(float)
    unique_brands     = rng.integers(1, 8, n).astype(float)
    brand_loyalty     = rng.beta(2, 5, n)
    cart_to_view_ratio= cart_count / (view_count + 1)
    price_range       = (avg_price_viewed * rng.uniform(0, 0.8, n)).clip(0)

    label = (purchase_count > 0).astype(int)

    df = pd.DataFrame({
        "view_count": view_count,
        "cart_count": cart_count,
        "session_duration": session_duration,
        "avg_price_viewed": avg_price_viewed,
        "unique_categories": unique_categories,
        "unique_brands": unique_brands,
        "brand_loyalty": brand_loyalty,
        "cart_to_view_ratio": cart_to_view_ratio,
        "price_range": price_range,
        "label": label,
    })
    print(f"[synthetic] Purchase rate: {label.mean()*100:.2f}%")
    return df


# ──────────────────────────────────────────────
#  TRAINING
# ──────────────────────────────────────────────
def train_logistic_regression(X_train, y_train):
    print("[LR] Training Logistic Regression …")
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    print("[RF] Hyperparameter tuning Random Forest (accuracy-focused) …")

    X_tune, y_tune = X_train, y_train
    if len(X_train) > RF_TUNING_MAX_SAMPLES:
        X_tune, _, y_tune, _ = train_test_split(
            X_train,
            y_train,
            train_size=RF_TUNING_MAX_SAMPLES,
            random_state=RANDOM_SEED,
            stratify=y_train,
        )
        print(f"[RF] Using stratified sample for tuning: {len(X_tune):,}")

    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 12, 18, 24],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.7],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "bootstrap": [True],
    }

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        param_distributions=param_dist,
        n_iter=RF_TUNING_ITERS,
        scoring="accuracy",
        cv=3,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_tune, y_tune)

    print(f"[RF] Best CV accuracy: {search.best_score_:.4f}")
    print(f"[RF] Best params: {search.best_params_}")

    model = search.best_estimator_
    model.fit(X_train, y_train)
    return model


def train_kmeans(X_scaled):
    print(f"[KM] Training K-Means with k={N_CLUSTERS} …")
    model = KMeans(
        n_clusters=N_CLUSTERS,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=RANDOM_SEED,
    )
    model.fit(X_scaled)
    return model


# ──────────────────────────────────────────────
#  EVALUATION
# ──────────────────────────────────────────────
def evaluate_classifier(name, model, X_test, y_test, scaler=None):
    X_in = scaler.transform(X_test) if scaler else X_test
    y_pred  = model.predict(X_in)
    y_proba = model.predict_proba(X_in)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, zero_division=0)

    metrics = {
        "model": name,
        "accuracy":  round(acc,  4),
        "auc_roc":   round(auc,  4),
        "f1_score":  round(f1,   4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "confusion_matrix": cm,
    }

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\n{report}")
    return metrics, report


def evaluate_kmeans(model, X_scaled):
    from sklearn.metrics import silhouette_score
    labels   = model.labels_
    inertia  = model.inertia_
    sil      = silhouette_score(X_scaled, labels, sample_size=10_000, random_state=RANDOM_SEED)
    counts   = pd.Series(labels).value_counts().sort_index().to_dict()

    print(f"\n{'='*50}")
    print(f"  K-Means (k={N_CLUSTERS})")
    print(f"{'='*50}")
    print(f"  Inertia         : {inertia:,.1f}")
    print(f"  Silhouette Score: {sil:.4f}")
    for k, v in counts.items():
        print(f"  Cluster {k}       : {v:,} samples ({v/len(labels)*100:.1f}%)")

    return {
        "model":            "K-Means",
        "k":                N_CLUSTERS,
        "inertia":          round(inertia, 2),
        "silhouette_score": round(sil, 4),
        "cluster_sizes":    {str(k): int(v) for k, v in counts.items()},
    }


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
def main(data_paths: list[str]):
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Load / generate data ─────────────────
    valid_paths = [p for p in data_paths if p and os.path.exists(p)]
    if valid_paths:
        frames = []
        for p in valid_paths:
            print(f"[data] Loading {p} …")
            chunk = pd.read_csv(p)
            print(f"[data]   Shape: {chunk.shape}")
            frames.append(chunk)
        raw = pd.concat(frames, ignore_index=True)
        print(f"[data] Combined shape: {raw.shape}")
        df = engineer_features(raw)
    else:
        print("[data] No valid CSV paths supplied – using synthetic data for demo.")
        df = generate_synthetic_data(n=100_000)

    X = df[FEATURE_COLS].astype(float).fillna(0).values
    y = df["label"].values

    # ── Split ────────────────────────────────
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Fit scaler on train only to prevent test-set leakage.
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)
    X_scaled_all = scaler.transform(X)

    print(f"\n[split] Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    # ── Train ────────────────────────────────
    lr  = train_logistic_regression(X_tr, y_tr)
    rf  = train_random_forest(X_tr, y_tr)
    km  = train_kmeans(X_scaled_all)

    # ── Evaluate ─────────────────────────────
    lr_metrics, lr_report = evaluate_classifier("Logistic Regression", lr, X_te, y_te)
    rf_metrics, rf_report = evaluate_classifier("Random Forest",        rf, X_te, y_te)
    km_metrics            = evaluate_kmeans(km, X_scaled_all)

    # ── Save models ──────────────────────────
    joblib.dump(lr,     os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    joblib.dump(rf,     os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(km,     os.path.join(MODELS_DIR, "kmeans.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    # ── Save reports ─────────────────────────
    report_data = {
        "logistic_regression": lr_metrics,
        "random_forest":       rf_metrics,
        "kmeans":              km_metrics,
    }
    with open(os.path.join(REPORTS_DIR, "metrics_report.json"), "w") as f:
        json.dump(report_data, f, indent=2)

    with open(os.path.join(REPORTS_DIR, "classification_report_lr.txt"), "w") as f:
        f.write(lr_report)

    with open(os.path.join(REPORTS_DIR, "classification_report_rf.txt"), "w") as f:
        f.write(rf_report)

    print("\n✓ All models saved to ./models/")
    print("✓ Reports saved to ./reports/")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ecomml models")
    parser.add_argument("--data", type=str, nargs="+",
                        default=["../data_sample/2019-Oct.csv", "../data_sample/2019-Nov.csv"],
                        help="One or more paths to dataset CSV files (space-separated)")
    args = parser.parse_args()
    main(args.data)
