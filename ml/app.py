import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.metrics import precision_recall_curve

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DASHBOARD_DIR = PROJECT_DIR / "dashboard"

# Ensure predict.py can resolve model paths correctly.
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

from predict import CLUSTER_BUY_PROB, CLUSTER_NAMES, load_models, score_row, transform_for_kmeans  # noqa: E402
from train_models import build_dashboard_report, generate_synthetic_data  # noqa: E402

app = Flask(__name__)
CORS(app)

MODELS = None
MODEL_LOAD_ERROR = None
PREDICTIONS_CACHE = None
PREDICTIONS_CACHE_KEY = None


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def _build_dashboard_data_from_raw(df):
    event_type = df.get("event_type", pd.Series(dtype="object")).fillna("unknown")
    views = int((event_type == "view").sum())
    carts = int((event_type == "cart").sum())
    purchases = int((event_type == "purchase").sum())
    total_events = int(len(df))

    unique_users = int(df.get("user_id", pd.Series(dtype="object")).nunique())

    session_purchase_rate = 0.0
    cart_abandonment = 0.0
    total_sessions = 0
    purchased_sessions = 0
    cart_sessions = 0
    abandoned_cart_sessions = 0
    if {"user_id", "user_session"}.issubset(df.columns):
        sess = df.groupby(["user_id", "user_session"]) ["event_type"]
        has_purchase = sess.apply(lambda x: (x == "purchase").any())
        has_cart = sess.apply(lambda x: (x == "cart").any())
        session_purchase_rate = float(has_purchase.mean()) if len(has_purchase) else 0.0
        total_sessions = int(len(has_purchase))
        purchased_sessions = int(has_purchase.sum())
        cart_sessions = int(has_cart.sum())

        carts_total = int(has_cart.sum())
        carts_no_purchase = int((has_cart & ~has_purchase).sum())
        cart_abandonment = (carts_no_purchase / carts_total) if carts_total else 0.0
        abandoned_cart_sessions = carts_no_purchase

    if "event_time" in df.columns:
        ts = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
        df_time = df.assign(_ts=ts).dropna(subset=["_ts"])
        daily = (
            df_time.assign(day=df_time["_ts"].dt.strftime("%Y-%m-%d"))
            .groupby(["day", "event_type"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
    else:
        daily = pd.DataFrame()

    if len(daily) > 31:
        daily = daily.tail(31)

    day_labels = daily.index.tolist()
    day_views = daily.get("view", pd.Series([0] * len(day_labels), index=day_labels)).astype(int).tolist()
    day_carts = daily.get("cart", pd.Series([0] * len(day_labels), index=day_labels)).astype(int).tolist()
    day_purchases = daily.get("purchase", pd.Series([0] * len(day_labels), index=day_labels)).astype(int).tolist()

    if "category_code" in df.columns:
        top_categories = (
            df["category_code"].fillna("unknown").astype(str).value_counts(normalize=True).head(6)
        )
        category_labels = [c.split(".")[-1] for c in top_categories.index.tolist()]
        category_values = [round(float(v * 100), 2) for v in top_categories.tolist()]
    else:
        category_labels = []
        category_values = []

    # Hourly activity distribution
    if "event_time" in df.columns:
        ts = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
        hour_counts = ts.dt.hour.value_counts().sort_index()
        hourly_labels = [f"{h}h" for h in range(24)]
        hourly_values = [int(hour_counts.get(h, 0)) for h in range(24)]
    else:
        hourly_labels = [f"{h}h" for h in range(24)]
        hourly_values = [0] * 24

    # Brand revenue share from purchase events.
    if {"brand", "price", "event_type"}.issubset(df.columns):
        purchase_df = df[df["event_type"] == "purchase"].copy()
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
        brand_labels = []
        brand_values = []

    # Price distribution by buckets (% of all events with valid prices).
    price_bins = [-1, 10, 25, 50, 100, 250, 500, float("inf")]
    price_labels = ["<$10", "$10-25", "$25-50", "$50-100", "$100-250", "$250-500", ">$500"]
    if "price" in df.columns:
        prices = pd.to_numeric(df["price"], errors="coerce").dropna()
        if len(prices):
            price_bucket = pd.cut(prices, bins=price_bins, labels=price_labels)
            price_counts = price_bucket.value_counts().reindex(price_labels, fill_value=0)
            price_values = [round(float((count / len(prices)) * 100), 2) for count in price_counts.tolist()]
        else:
            price_values = [0.0] * len(price_labels)
    else:
        price_values = [0.0] * len(price_labels)

    # Session duration distribution (% of sessions)
    session_labels = ["<1", "1-3", "3-5", "5-10", "10-20", "20-30", ">30"]
    if {"user_id", "user_session", "event_time"}.issubset(df.columns):
        ts = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
        sess_df = df.assign(_ts=ts).dropna(subset=["_ts"])
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

    return {
        "overview": {
            "total_events": total_events,
            "unique_users": unique_users,
            "purchase_rate": round(session_purchase_rate, 6),
            "cart_abandonment": round(cart_abandonment, 6),
        },
        "session_stats": {
            "total_sessions": total_sessions,
            "purchased_sessions": purchased_sessions,
            "cart_sessions": cart_sessions,
            "abandoned_cart_sessions": abandoned_cart_sessions,
        },
        "funnel": {
            "view": views,
            "cart": carts,
            "purchase": purchases,
        },
        "daily_event_volume": {
            "labels": day_labels,
            "view": day_views,
            "cart": day_carts,
            "purchase": day_purchases,
        },
        "top_categories": {
            "labels": category_labels,
            "values": category_values,
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


def _load_dashboard_data():
    dashboard_path = BASE_DIR / "reports" / "dashboard_report.json"
    if dashboard_path.exists():
        with open(dashboard_path, encoding="utf-8") as f:
            payload = json.load(f)
        return payload, dashboard_path

    candidate_files = sorted((PROJECT_DIR / "data_sample").glob("*.csv"))
    if not candidate_files:
        demo_feat = generate_synthetic_data(n=25_000)
        payload = build_dashboard_report(None, demo_feat, ["synthetic-demo"])
        payload["source"] = "synthetic-demo"
        return payload, None

    latest = max(candidate_files, key=lambda p: p.stat().st_mtime)
    raw = pd.read_csv(latest)
    payload = _build_dashboard_data_from_raw(raw)
    payload["source"] = latest.name
    return payload, latest


def _engineer_session_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    frame = raw_df.copy()
    frame["event_time"] = pd.to_datetime(frame.get("event_time"), utc=True, errors="coerce")
    frame = frame.dropna(subset=["event_time"])

    if "price" in frame.columns:
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce").fillna(0.0)
    else:
        frame["price"] = 0.0

    if "brand" not in frame.columns:
        frame["brand"] = "unknown"
    if "category_code" not in frame.columns:
        frame["category_code"] = "unknown"

    group_cols = ["user_id", "user_session"]
    grp = frame.groupby(group_cols)

    feat = pd.DataFrame(index=grp.size().index)
    feat["view_count"] = grp["event_type"].apply(lambda x: (x == "view").sum())
    feat["cart_count"] = grp["event_type"].apply(lambda x: (x == "cart").sum())
    feat["purchase_count"] = grp["event_type"].apply(lambda x: (x == "purchase").sum())
    feat["session_duration"] = grp["event_time"].apply(lambda x: (x.max() - x.min()).total_seconds() / 60)
    feat["avg_price_viewed"] = grp["price"].mean().fillna(0.0)
    feat["unique_categories"] = grp["category_code"].nunique()
    feat["unique_brands"] = grp["brand"].nunique()
    feat["brand_loyalty"] = grp["brand"].apply(
        lambda x: x.dropna().value_counts().iloc[0] / len(x.dropna()) if len(x.dropna()) > 0 else 0.0
    )
    feat["cart_to_view_ratio"] = feat["cart_count"] / (feat["view_count"] + 1)
    feat["price_range"] = grp["price"].apply(lambda x: x.max() - x.min()).fillna(0.0)
    feat["label"] = (feat["purchase_count"] > 0).astype(int)

    feat = feat.reset_index()
    return feat


def _compress_pr_curve(recall_vals, precision_vals, max_points=30):
    if len(recall_vals) <= max_points:
        return [round(float(v), 4) for v in recall_vals], [round(float(v), 4) for v in precision_vals]
    idx = np.linspace(0, len(recall_vals) - 1, max_points).astype(int)
    r = [round(float(recall_vals[i]), 4) for i in idx]
    p = [round(float(precision_vals[i]), 4) for i in idx]
    return r, p


def _build_predictions_payload_from_features(feat: pd.DataFrame, models):
    lr_model, rf_model, km_model, scaler, feature_cols = models
    if feat.empty:
        return {
            "users_scored": 0,
            "high_propensity": 0,
            "mid_propensity": 0,
            "low_propensity": 0,
            "score_distribution": {"labels": [f"{i/10:.1f}" for i in range(11)], "values": [0] * 11},
            "pr_curve": {"rf": {"recall": [0.0, 1.0], "precision": [1.0, 0.0]}, "lr": {"recall": [0.0, 1.0], "precision": [1.0, 0.0]}},
            "top_users": [],
        }

    if "user_id" not in feat.columns:
        feat = feat.copy()
        feat["user_id"] = np.arange(1, len(feat) + 1, dtype=int)
    if "purchase_count" not in feat.columns:
        feat = feat.copy()
        feat["purchase_count"] = feat.get("label", pd.Series([0] * len(feat))).astype(int)

    x = feat[feature_cols].astype(float).fillna(0.0)
    x_scaled = scaler.transform(x.values)

    lr_probs = lr_model.predict_proba(x_scaled)[:, 1]
    rf_probs = rf_model.predict_proba(x_scaled)[:, 1]
    clusters = km_model.predict(transform_for_kmeans(x_scaled)).astype(int)
    km_probs = np.array([CLUSTER_BUY_PROB.get(int(c), 0.3) for c in clusters], dtype=float)
    ensemble = (lr_probs * 0.30) + (rf_probs * 0.50) + (km_probs * 0.20)

    users_scored = int(len(feat))
    high_propensity = int((rf_probs > 0.7).sum())
    mid_propensity = int(((rf_probs >= 0.3) & (rf_probs <= 0.7)).sum())
    low_propensity = int((rf_probs < 0.3).sum())

    hist, _ = np.histogram(rf_probs, bins=np.linspace(0.0, 1.0, 12))
    distribution = {
        "labels": [f"{i/10:.1f}" for i in range(11)],
        "values": [int(v) for v in hist.tolist()],
    }

    y_true = feat["label"].values
    if len(np.unique(y_true)) > 1:
        rf_precision, rf_recall, _ = precision_recall_curve(y_true, rf_probs)
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, lr_probs)
        rf_r, rf_p = _compress_pr_curve(rf_recall, rf_precision)
        lr_r, lr_p = _compress_pr_curve(lr_recall, lr_precision)
    else:
        rf_r, rf_p = [0.0, 1.0], [1.0, 1.0]
        lr_r, lr_p = [0.0, 1.0], [1.0, 1.0]

    scored = feat[["user_id", "view_count", "cart_count", "purchase_count", "avg_price_viewed"]].copy()
    scored["cluster"] = clusters
    scored["cluster_name"] = [CLUSTER_NAMES.get(int(c), "Unknown") for c in clusters]
    scored["rf_score"] = rf_probs
    scored["lr_score"] = lr_probs
    scored["ensemble_score"] = ensemble
    scored["propensity"] = np.where(rf_probs > 0.7, "high", np.where(rf_probs >= 0.3, "med", "low"))

    top = scored.sort_values("rf_score", ascending=False).head(25)
    top_users = []
    for _, row in top.iterrows():
        top_users.append({
            "id": str(row["user_id"]),
            "views": int(row["view_count"]),
            "carts": int(row["cart_count"]),
            "purchases": int(row["purchase_count"]),
            "avgPrice": f"${int(round(float(row['avg_price_viewed'])))}",
            "cluster": int(row["cluster"]),
            "cluster_name": row["cluster_name"],
            "rfS": round(float(row["rf_score"]), 3),
            "lrS": round(float(row["lr_score"]), 3),
            "prop": row["propensity"],
        })

    return {
        "users_scored": users_scored,
        "high_propensity": high_propensity,
        "mid_propensity": mid_propensity,
        "low_propensity": low_propensity,
        "score_distribution": distribution,
        "pr_curve": {
            "rf": {"recall": rf_r, "precision": rf_p},
            "lr": {"recall": lr_r, "precision": lr_p},
        },
        "top_users": top_users,
    }


def _build_predictions_payload(raw_df: pd.DataFrame, models):
    feat = _engineer_session_features(raw_df)
    return _build_predictions_payload_from_features(feat, models)


def _build_predictions_payload_heuristic_from_features(feat: pd.DataFrame):
    if feat.empty:
        return {
            "users_scored": 0,
            "high_propensity": 0,
            "mid_propensity": 0,
            "low_propensity": 0,
            "score_distribution": {"labels": [f"{i/10:.1f}" for i in range(11)], "values": [0] * 11},
            "pr_curve": {"rf": {"recall": [0.0, 1.0], "precision": [1.0, 0.0]}, "lr": {"recall": [0.0, 1.0], "precision": [1.0, 0.0]}},
            "top_users": [],
            "inference_mode": "heuristic",
        }

    if "user_id" not in feat.columns:
        feat = feat.copy()
        feat["user_id"] = np.arange(1, len(feat) + 1, dtype=int)
    if "purchase_count" not in feat.columns:
        feat = feat.copy()
        feat["purchase_count"] = feat.get("label", pd.Series([0] * len(feat))).astype(int)

    views = feat["view_count"].astype(float).values
    carts = feat["cart_count"].astype(float).values
    purchases = feat["purchase_count"].astype(float).values
    duration = feat["session_duration"].astype(float).values
    avg_price = feat["avg_price_viewed"].astype(float).values
    cats = feat["unique_categories"].astype(float).values
    loyalty = feat["brand_loyalty"].astype(float).values

    rf_probs = (
        np.minimum(views / 40.0, 1.0) * 0.28
        + np.minimum(carts / 10.0, 1.0) * 0.24
        + np.minimum(purchases / 5.0, 1.0) * 0.10
        + np.minimum(duration / 30.0, 1.0) * 0.18
        + np.minimum(avg_price / 300.0, 1.0) * 0.06
        + np.minimum(cats / 8.0, 1.0) * 0.06
        + np.minimum(loyalty, 1.0) * 0.05
    )
    rf_probs += np.where(views > 0, (carts / np.maximum(views, 1.0)) * 0.08, 0.0)
    rf_probs = np.clip(rf_probs, 0.01, 0.99)

    z = (
        -2.1
        + views * 0.04
        + carts * 0.18
        + purchases * 0.25
        + duration * 0.04
        + (avg_price / 100.0) * 0.12
        + cats * 0.08
        + loyalty * 1.4
    )
    lr_probs = 1.0 / (1.0 + np.exp(-z * 0.7))
    lr_probs = np.clip(lr_probs, 0.01, 0.99)

    centroids = np.array([
        [28.0, 1.2, 0.1, 60.0, 2.0],
        [15.0, 4.8, 3.2, 120.0, 6.0],
        [52.0, 3.1, 0.8, 85.0, 3.0],
        [18.0, 6.2, 5.8, 280.0, 8.0],
    ])
    points = np.column_stack([
        views,
        carts,
        purchases,
        avg_price,
        loyalty * 10.0,
    ])
    diffs = points[:, None, :] - centroids[None, :, :]
    weights = np.array([40.0, 10.0, 6.0, 200.0, 10.0])
    dists = ((diffs / weights) ** 2).sum(axis=2)
    clusters = dists.argmin(axis=1)
    km_probs = np.array([CLUSTER_BUY_PROB.get(int(c), 0.3) for c in clusters], dtype=float)

    ensemble = (lr_probs * 0.30) + (rf_probs * 0.50) + (km_probs * 0.20)

    users_scored = int(len(feat))
    high_propensity = int((rf_probs > 0.7).sum())
    mid_propensity = int(((rf_probs >= 0.3) & (rf_probs <= 0.7)).sum())
    low_propensity = int((rf_probs < 0.3).sum())

    hist, _ = np.histogram(rf_probs, bins=np.linspace(0.0, 1.0, 12))
    distribution = {
        "labels": [f"{i/10:.1f}" for i in range(11)],
        "values": [int(v) for v in hist.tolist()],
    }

    y_true = feat["label"].values
    if len(np.unique(y_true)) > 1:
        rf_precision, rf_recall, _ = precision_recall_curve(y_true, rf_probs)
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, lr_probs)
        rf_r, rf_p = _compress_pr_curve(rf_recall, rf_precision)
        lr_r, lr_p = _compress_pr_curve(lr_recall, lr_precision)
    else:
        rf_r, rf_p = [0.0, 1.0], [1.0, 1.0]
        lr_r, lr_p = [0.0, 1.0], [1.0, 1.0]

    scored = feat[["user_id", "view_count", "cart_count", "purchase_count", "avg_price_viewed"]].copy()
    scored["cluster"] = clusters
    scored["cluster_name"] = [CLUSTER_NAMES.get(int(c), "Unknown") for c in clusters]
    scored["rf_score"] = rf_probs
    scored["lr_score"] = lr_probs
    scored["ensemble_score"] = ensemble
    scored["propensity"] = np.where(rf_probs > 0.7, "high", np.where(rf_probs >= 0.3, "med", "low"))

    top = scored.sort_values("rf_score", ascending=False).head(25)
    top_users = []
    for _, row in top.iterrows():
        top_users.append({
            "id": str(row["user_id"]),
            "views": int(row["view_count"]),
            "carts": int(row["cart_count"]),
            "purchases": int(row["purchase_count"]),
            "avgPrice": f"${int(round(float(row['avg_price_viewed'])))}",
            "cluster": int(row["cluster"]),
            "cluster_name": row["cluster_name"],
            "rfS": round(float(row["rf_score"]), 3),
            "lrS": round(float(row["lr_score"]), 3),
            "prop": row["propensity"],
        })

    return {
        "users_scored": users_scored,
        "high_propensity": high_propensity,
        "mid_propensity": mid_propensity,
        "low_propensity": low_propensity,
        "score_distribution": distribution,
        "pr_curve": {
            "rf": {"recall": rf_r, "precision": rf_p},
            "lr": {"recall": lr_r, "precision": lr_p},
        },
        "top_users": top_users,
        "inference_mode": "heuristic",
    }


def _build_predictions_payload_heuristic(raw_df: pd.DataFrame):
    feat = _engineer_session_features(raw_df)
    return _build_predictions_payload_heuristic_from_features(feat)


def _load_predictions_data():
    global PREDICTIONS_CACHE, PREDICTIONS_CACHE_KEY

    models = get_models()

    predictions_report_path = BASE_DIR / "reports" / "predictions_report.json"
    if predictions_report_path.exists():
        cache_key = (str(predictions_report_path), predictions_report_path.stat().st_mtime)
        if PREDICTIONS_CACHE is not None and PREDICTIONS_CACHE_KEY == cache_key:
            return PREDICTIONS_CACHE, predictions_report_path, None
        with open(predictions_report_path, encoding="utf-8") as f:
            payload = json.load(f)
        PREDICTIONS_CACHE = payload
        PREDICTIONS_CACHE_KEY = cache_key
        return payload, predictions_report_path, None

    candidate_files = sorted((PROJECT_DIR / "data_sample").glob("*.csv"))
    if not candidate_files:
        mode = "ml" if models is not None else "heuristic"
        cache_key = ("synthetic-demo", mode)
        if PREDICTIONS_CACHE is not None and PREDICTIONS_CACHE_KEY == cache_key:
            return PREDICTIONS_CACHE, None, None

        demo_feat = generate_synthetic_data(n=25_000)
        if models is None:
            payload = _build_predictions_payload_heuristic_from_features(demo_feat)
            payload["model_error"] = MODEL_LOAD_ERROR
        else:
            payload = _build_predictions_payload_from_features(demo_feat, models)
            payload["inference_mode"] = "ml"
        payload["source"] = "synthetic-demo"
        PREDICTIONS_CACHE = payload
        PREDICTIONS_CACHE_KEY = cache_key
        return payload, None, None

    latest = max(candidate_files, key=lambda p: p.stat().st_mtime)
    mode = "ml" if models is not None else "heuristic"
    cache_key = (str(latest), latest.stat().st_mtime, mode)
    if PREDICTIONS_CACHE is not None and PREDICTIONS_CACHE_KEY == cache_key:
        return PREDICTIONS_CACHE, latest, None

    raw = pd.read_csv(latest)
    if models is None:
        payload = _build_predictions_payload_heuristic(raw)
        payload["model_error"] = MODEL_LOAD_ERROR
    else:
        payload = _build_predictions_payload(raw, models)
        payload["inference_mode"] = "ml"
    payload["source"] = latest.name
    PREDICTIONS_CACHE = payload
    PREDICTIONS_CACHE_KEY = cache_key
    return payload, latest, None


def _load_metrics_report():
    metrics_path = BASE_DIR / "reports" / "metrics_report.json"
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f), metrics_path


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_models():
    global MODELS, MODEL_LOAD_ERROR
    if MODELS is not None:
        return MODELS
    if MODEL_LOAD_ERROR is not None:
        return None

    try:
        MODELS = load_models()
        return MODELS
    except SystemExit as exc:
        MODEL_LOAD_ERROR = str(exc)
    except Exception as exc:
        MODEL_LOAD_ERROR = f"Unexpected model load error: {exc}"
    return None


@app.get("/")
def index():
    return send_from_directory(str(DASHBOARD_DIR), "index.html")


@app.get("/healthz")
def healthz():
    models_ready = get_models() is not None
    return jsonify({
        "status": "ok",
        "models_ready": models_ready,
        "model_error": MODEL_LOAD_ERROR,
    })


@app.get("/api/metrics")
def metrics():
    report, _ = _load_metrics_report()
    return jsonify(report)


@app.get("/api/dashboard-summary")
def dashboard_summary():
    report, metrics_path = _load_metrics_report()
    models = get_models()

    lr_metrics = report.get("logistic_regression", {})
    rf_metrics = report.get("random_forest", {})
    km_metrics = report.get("kmeans", {})

    rf_cm = rf_metrics.get("confusion_matrix") or [[0, 0], [0, 0]]
    rf_test_samples = int(sum(sum(row) for row in rf_cm if isinstance(row, list)))

    feature_columns = []
    if models is not None:
        feature_columns = models[4]
    else:
        feature_cols_path = BASE_DIR / "models" / "feature_columns.json"
        if feature_cols_path.exists():
            with open(feature_cols_path, encoding="utf-8") as f:
                feature_columns = json.load(f)

    feature_importance = []
    rf_hyperparams = {}
    if models is not None:
        _, rf_model, _, _, _ = models
        importances = getattr(rf_model, "feature_importances_", None)
        rf_hyperparams = {
            "n_estimators": getattr(rf_model, "n_estimators", None),
            "max_depth": getattr(rf_model, "max_depth", None),
            "min_samples_split": getattr(rf_model, "min_samples_split", None),
            "min_samples_leaf": getattr(rf_model, "min_samples_leaf", None),
        }
        if importances is not None:
            pairs = []
            for idx, col in enumerate(feature_columns):
                if idx < len(importances):
                    pairs.append({
                        "feature": col,
                        "importance": round(float(importances[idx]), 6),
                    })
            feature_importance = sorted(
                pairs,
                key=lambda item: item["importance"],
                reverse=True,
            )

    cluster_sizes = km_metrics.get("cluster_sizes", {})
    cluster_total = sum(int(v) for v in cluster_sizes.values()) if cluster_sizes else 0
    cluster_distribution = []
    for cluster_id, size in sorted(cluster_sizes.items(), key=lambda item: int(item[0])):
        count = int(size)
        pct = round((count / cluster_total) * 100, 2) if cluster_total else 0.0
        cluster_distribution.append({
            "cluster": int(cluster_id),
            "count": count,
            "percentage": pct,
        })

    best_model = "random_forest"
    best_accuracy = _to_float(rf_metrics.get("accuracy"))
    if _to_float(lr_metrics.get("accuracy")) > best_accuracy:
        best_model = "logistic_regression"
        best_accuracy = _to_float(lr_metrics.get("accuracy"))

    last_updated = datetime.fromtimestamp(
        metrics_path.stat().st_mtime,
        tz=timezone.utc,
    ).isoformat()

    return jsonify({
        "status": "ok",
        "models_ready": models is not None,
        "model_error": MODEL_LOAD_ERROR,
        "last_updated": last_updated,
        "best_model": best_model,
        "best_accuracy": round(best_accuracy, 4),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "rf_test_samples": rf_test_samples,
        "logistic_regression": lr_metrics,
        "random_forest": rf_metrics,
        "random_forest_hyperparams": rf_hyperparams,
        "kmeans": km_metrics,
        "cluster_distribution": cluster_distribution,
        "feature_importance": feature_importance,
    })


@app.get("/api/dashboard-data")
def dashboard_data():
    payload, source_path = _load_dashboard_data()
    last_updated = None
    if source_path is not None:
        last_updated = datetime.fromtimestamp(
            source_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()

    return jsonify({
        "status": "ok",
        "last_updated": last_updated,
        **payload,
    })


@app.get("/api/predictions-data")
def predictions_data():
    payload, source_path, err = _load_predictions_data()

    last_updated = None
    if source_path is not None:
        last_updated = datetime.fromtimestamp(
            source_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()

    return jsonify({
        "status": "ok",
        "last_updated": last_updated,
        **payload,
    })


@app.post("/api/predict/existing")
def predict_existing():
    models = get_models()
    if models is None:
        return jsonify({
            "error": "Models are not available on this deployment.",
            "details": MODEL_LOAD_ERROR,
        }), 503

    lr_model, rf_model, km_model, scaler, feature_cols = models
    body = request.get_json(force=True) or {}

    views = int(body.get("views", 0))
    carts = int(body.get("carts", 0))
    session_duration = float(body.get("session_duration", 0))
    avg_price = float(body.get("avg_price", 50))
    unique_categories = int(body.get("unique_categories", 1))
    # UI uses 0-10 slider, model expects 0-1 range.
    brand_loyalty = float(body.get("brand_loyalty", 0)) / 10.0

    row = {
        "view_count": views,
        "cart_count": carts,
        "session_duration": session_duration,
        "avg_price_viewed": avg_price,
        "unique_categories": unique_categories,
        "unique_brands": max(1, unique_categories // 2),
        "brand_loyalty": max(0.0, min(1.0, brand_loyalty)),
        "cart_to_view_ratio": carts / (views + 1),
        "price_range": 0.0,
    }

    result = score_row(row, lr_model, rf_model, km_model, scaler, feature_cols)
    return jsonify(result)


@app.post("/api/predict/new")
def predict_new():
    models = get_models()
    if models is None:
        return jsonify({
            "error": "Models are not available on this deployment.",
            "details": MODEL_LOAD_ERROR,
        }), 503

    lr_model, rf_model, km_model, scaler, feature_cols = models
    body = request.get_json(force=True) or {}

    views = int(body.get("views", 0))
    carts = int(body.get("carts", 0))
    session_duration = float(body.get("duration", 0))
    avg_price = float(body.get("avg_price", 50))
    unique_categories = int(body.get("unique_categories", 1))
    brand_affinity = float(body.get("brand_affinity", 0)) / 10.0

    row = {
        "view_count": views,
        "cart_count": carts,
        "session_duration": session_duration,
        "avg_price_viewed": avg_price,
        "unique_categories": unique_categories,
        "unique_brands": max(1, unique_categories // 2),
        "brand_loyalty": max(0.0, min(1.0, brand_affinity)),
        "cart_to_view_ratio": carts / (views + 1),
        "price_range": 0.0,
    }

    result = score_row(row, lr_model, rf_model, km_model, scaler, feature_cols)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
