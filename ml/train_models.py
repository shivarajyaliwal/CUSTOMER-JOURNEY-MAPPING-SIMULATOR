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
        "user_id": np.arange(1, n + 1, dtype=int),
        "view_count": view_count,
        "cart_count": cart_count,
        "purchase_count": purchase_count,
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


def build_dashboard_report(raw_df: pd.DataFrame | None, feat_df: pd.DataFrame, data_paths: list[str]):
    if raw_df is not None and not raw_df.empty:
        event_type = raw_df["event_type"].fillna("unknown")
        total_events = int(len(raw_df))
        unique_users = int(raw_df["user_id"].nunique()) if "user_id" in raw_df.columns else 0
        view_count = int((event_type == "view").sum())
        cart_count = int((event_type == "cart").sum())
        purchase_count = int((event_type == "purchase").sum())

        if {"user_id", "user_session"}.issubset(raw_df.columns):
            sess = raw_df.groupby(["user_id", "user_session"]) ["event_type"]
            has_purchase = sess.apply(lambda x: (x == "purchase").any())
            has_cart = sess.apply(lambda x: (x == "cart").any())
            purchase_rate = float(has_purchase.mean()) if len(has_purchase) else 0.0
            total_sessions = int(len(has_purchase))
            purchased_sessions = int(has_purchase.sum())
            cart_sessions = int(has_cart.sum())
            carts_total = int(has_cart.sum())
            carts_no_purchase = int((has_cart & ~has_purchase).sum())
            cart_abandonment = (carts_no_purchase / carts_total) if carts_total else 0.0
            abandoned_cart_sessions = carts_no_purchase
        else:
            purchase_rate = float(feat_df["label"].mean()) if "label" in feat_df.columns else 0.0
            cart_abandonment = 0.0
            total_sessions = int(len(feat_df))
            purchased_sessions = int(feat_df["label"].sum()) if "label" in feat_df.columns else 0
            cart_sessions = 0
            abandoned_cart_sessions = 0

        ts = pd.to_datetime(raw_df["event_time"], utc=True, errors="coerce") if "event_time" in raw_df.columns else None
        if ts is not None:
            daily = (
                raw_df.assign(_ts=ts)
                .dropna(subset=["_ts"])
                .assign(day=lambda d: d["_ts"].dt.strftime("%Y-%m-%d"))
                .groupby(["day", "event_type"])
                .size()
                .unstack(fill_value=0)
                .sort_index()
            )
            if len(daily) > 31:
                daily = daily.tail(31)
            day_labels = daily.index.tolist()
            day_views = daily.get("view", pd.Series([0] * len(day_labels), index=day_labels)).astype(int).tolist()
            day_carts = daily.get("cart", pd.Series([0] * len(day_labels), index=day_labels)).astype(int).tolist()
            day_purchases = daily.get("purchase", pd.Series([0] * len(day_labels), index=day_labels)).astype(int).tolist()
        else:
            day_labels, day_views, day_carts, day_purchases = [], [], [], []

        if "category_code" in raw_df.columns:
            top_categories = raw_df["category_code"].fillna("unknown").astype(str).value_counts(normalize=True).head(6)
            cat_labels = [c.split(".")[-1] for c in top_categories.index.tolist()]
            cat_values = [round(float(v * 100), 2) for v in top_categories.tolist()]
        else:
            cat_labels, cat_values = [], []

        hourly_labels = [f"{h}h" for h in range(24)]
        if "event_time" in raw_df.columns:
            hour_series = pd.to_datetime(raw_df["event_time"], utc=True, errors="coerce").dt.hour
            hour_counts = hour_series.value_counts().sort_index()
            hourly_values = [int(hour_counts.get(h, 0)) for h in range(24)]
        else:
            hourly_values = [0] * 24

        if {"brand", "price", "event_type"}.issubset(raw_df.columns):
            purchase_df = raw_df[raw_df["event_type"] == "purchase"].copy()
            purchase_df["brand"] = purchase_df["brand"].fillna("unknown").astype(str)
            purchase_df["price"] = pd.to_numeric(purchase_df["price"], errors="coerce").fillna(0.0)
            brand_rev = purchase_df.groupby("brand")["price"].sum().sort_values(ascending=False).head(10)
            total_rev = float(brand_rev.sum())
            brand_labels = brand_rev.index.tolist()
            if total_rev > 0:
                brand_values = [round(float(v / total_rev * 100), 2) for v in brand_rev.tolist()]
            else:
                brand_values = [0.0] * len(brand_labels)
        else:
            brand_labels, brand_values = [], []

        price_bins = [-1, 10, 25, 50, 100, 250, 500, float("inf")]
        price_labels = ["<$10", "$10-25", "$25-50", "$50-100", "$100-250", "$250-500", ">$500"]
        if "price" in raw_df.columns:
            prices = pd.to_numeric(raw_df["price"], errors="coerce").dropna()
            if len(prices):
                price_bucket = pd.cut(prices, bins=price_bins, labels=price_labels)
                price_counts = price_bucket.value_counts().reindex(price_labels, fill_value=0)
                price_values = [round(float((count / len(prices)) * 100), 2) for count in price_counts.tolist()]
            else:
                price_values = [0.0] * len(price_labels)
        else:
            price_values = [0.0] * len(price_labels)

        session_labels = ["<1", "1-3", "3-5", "5-10", "10-20", "20-30", ">30"]
        if {"user_id", "user_session", "event_time"}.issubset(raw_df.columns):
            ts = pd.to_datetime(raw_df["event_time"], utc=True, errors="coerce")
            sess_df = raw_df.assign(_ts=ts).dropna(subset=["_ts"])
            sess_duration = sess_df.groupby(["user_id", "user_session"])["_ts"].agg(lambda x: (x.max() - x.min()).total_seconds() / 60)
            if len(sess_duration):
                dur_bins = [-1, 1, 3, 5, 10, 20, 30, float("inf")]
                dur_bucket = pd.cut(sess_duration, bins=dur_bins, labels=session_labels)
                dur_counts = dur_bucket.value_counts().reindex(session_labels, fill_value=0)
                session_values = [round(float((count / len(sess_duration)) * 100), 2) for count in dur_counts.tolist()]
            else:
                session_values = [0.0] * len(session_labels)
        else:
            session_values = [0.0] * len(session_labels)
    else:
        total_events = int((feat_df.get("view_count", 0) + feat_df.get("cart_count", 0) + feat_df.get("purchase_count", 0)).sum())
        unique_users = int(feat_df.get("user_id", pd.Series(dtype="object")).nunique()) if "user_id" in feat_df.columns else 0
        view_count = int(feat_df.get("view_count", pd.Series(dtype="float64")).sum())
        cart_count = int(feat_df.get("cart_count", pd.Series(dtype="float64")).sum())
        purchase_count = int(feat_df.get("purchase_count", pd.Series(dtype="float64")).sum())
        purchase_rate = float(feat_df["label"].mean()) if "label" in feat_df.columns else 0.0
        with_cart = feat_df.get("cart_count", pd.Series(dtype="float64")) > 0
        no_purchase = feat_df.get("purchase_count", pd.Series(dtype="float64")) == 0
        denom = int(with_cart.sum())
        cart_abandonment = float((with_cart & no_purchase).sum() / denom) if denom else 0.0
        day_labels = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=14, freq="D").strftime("%Y-%m-%d").tolist()
        trend_weights = np.array([0.82, 0.86, 0.90, 0.95, 0.97, 1.00, 1.04, 1.08, 1.12, 1.09, 1.05, 1.02, 0.98, 0.94])
        trend_weights = trend_weights / trend_weights.sum()
        day_views = np.round(view_count * trend_weights).astype(int).tolist()
        day_carts = np.round(cart_count * trend_weights).astype(int).tolist()
        day_purchases = np.round(purchase_count * trend_weights).astype(int).tolist()
        cat_labels = ["electronics", "appliances", "computers", "smartphones", "accessories", "lifestyle"]
        cat_values = [24.5, 19.8, 17.1, 15.3, 12.2, 11.1]
        hourly_labels = [f"{h}h" for h in range(24)]
        hourly_profile = np.array([18, 12, 8, 6, 6, 8, 14, 24, 38, 54, 63, 69, 74, 78, 82, 85, 88, 92, 84, 70, 56, 43, 31, 23], dtype=float)
        hourly_scale = max(total_events, 1) / hourly_profile.sum()
        hourly_values = np.round(hourly_profile * hourly_scale).astype(int).tolist()
        brand_labels = ["samsung", "apple", "xiaomi", "huawei", "lg", "lenovo", "sony", "bosch"]
        brand_values = [21.4, 18.6, 14.9, 12.7, 10.4, 8.6, 7.1, 6.3]
        price_labels = ["<$10", "$10-25", "$25-50", "$50-100", "$100-250", "$250-500", ">$500"]
        price_bins = [-1, 10, 25, 50, 100, 250, 500, float("inf")]
        prices = feat_df.get("avg_price_viewed", pd.Series(dtype="float64")).astype(float)
        if len(prices):
            price_bucket = pd.cut(prices, bins=price_bins, labels=price_labels)
            price_counts = price_bucket.value_counts().reindex(price_labels, fill_value=0)
            price_values = [round(float((count / len(prices)) * 100), 2) for count in price_counts.tolist()]
        else:
            price_values = [0.0] * len(price_labels)
        session_labels = ["<1", "1-3", "3-5", "5-10", "10-20", "20-30", ">30"]
        dur_bins = [-1, 1, 3, 5, 10, 20, 30, float("inf")]
        durations = feat_df.get("session_duration", pd.Series(dtype="float64")).astype(float)
        if len(durations):
            dur_bucket = pd.cut(durations, bins=dur_bins, labels=session_labels)
            dur_counts = dur_bucket.value_counts().reindex(session_labels, fill_value=0)
            session_values = [round(float((count / len(durations)) * 100), 2) for count in dur_counts.tolist()]
        else:
            session_values = [0.0] * len(session_labels)
        total_sessions = int(len(feat_df))
        purchased_sessions = int(feat_df["label"].sum()) if "label" in feat_df.columns else 0
        cart_sessions = int((feat_df.get("cart_count", pd.Series(dtype="float64")) > 0).sum())
        abandoned_cart_sessions = int(((feat_df.get("cart_count", pd.Series(dtype="float64")) > 0) & (feat_df.get("purchase_count", pd.Series(dtype="float64")) == 0)).sum())

    return {
        "source_data_paths": data_paths,
        "overview": {
            "total_events": total_events,
            "unique_users": unique_users,
            "purchase_rate": round(float(purchase_rate), 6),
            "cart_abandonment": round(float(cart_abandonment), 6),
        },
        "session_stats": {
            "total_sessions": int(total_sessions),
            "purchased_sessions": int(purchased_sessions),
            "cart_sessions": int(cart_sessions),
            "abandoned_cart_sessions": int(abandoned_cart_sessions),
        },
        "funnel": {
            "view": int(view_count),
            "cart": int(cart_count),
            "purchase": int(purchase_count),
        },
        "daily_event_volume": {
            "labels": day_labels,
            "view": day_views,
            "cart": day_carts,
            "purchase": day_purchases,
        },
        "top_categories": {
            "labels": cat_labels,
            "values": cat_values,
        },
        "hourly_activity": {
            "labels": hourly_labels,
            "values": hourly_values,
        },
        "top_brands": {
            "labels": brand_labels,
            "values": brand_values,
        },
        "price_distribution": {
            "labels": price_labels,
            "values": price_values,
        },
        "session_duration_distribution": {
            "labels": session_labels,
            "values": session_values,
        },
    }


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
def main(data_paths: list[str], synthetic_sessions: int = 100_000, rf_tuning_iters: int = RF_TUNING_ITERS):
    global RF_TUNING_ITERS
    RF_TUNING_ITERS = rf_tuning_iters

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
        raw = None
        df = generate_synthetic_data(n=synthetic_sessions)

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

    dashboard_report = build_dashboard_report(raw, df, valid_paths)
    with open(os.path.join(REPORTS_DIR, "dashboard_report.json"), "w") as f:
        json.dump(dashboard_report, f, indent=2)

    print("\n✓ All models saved to ./models/")
    print("✓ Reports saved to ./reports/")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ecomml models")
    parser.add_argument("--data", type=str, nargs="+",
                        default=["../data_sample/2019-Oct.csv", "../data_sample/2019-Nov.csv"],
                        help="One or more paths to dataset CSV files (space-separated)")
    parser.add_argument("--synthetic-sessions", type=int, default=100_000,
                        help="Synthetic sessions to generate when CSV files are unavailable")
    parser.add_argument("--rf-tuning-iters", type=int, default=RF_TUNING_ITERS,
                        help="Random Forest tuning iterations for demo builds")
    args = parser.parse_args()
    main(args.data, synthetic_sessions=args.synthetic_sessions, rf_tuning_iters=args.rf_tuning_iters)
