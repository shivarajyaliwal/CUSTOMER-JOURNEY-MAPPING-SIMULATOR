"""
EcomML – Prediction / Inference
================================
Use trained models to score a single customer or a batch CSV.

Usage
-----
# Single prediction (existing customer)
python predict.py --mode single \
    --views 22 --carts 5 --purchases 3 \
    --session_duration 12 --avg_price 89 \
    --unique_categories 3 --unique_brands 2 \
    --brand_loyalty 0.4 --cart_to_view_ratio 0.23 \
    --price_range 60

# Batch prediction
python predict.py --mode batch --input customers.csv --output scored.csv

# New customer (no purchase history)
python predict.py --mode new \
    --views 6 --carts 2 --session_duration 8 \
    --avg_price 75 --unique_categories 2
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

MODELS_DIR   = "models"
FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "feature_columns.json")
CLUSTER_BUY_PROB_PATH = os.path.join(MODELS_DIR, "cluster_buy_prob.json")
CLUSTER_NAMES_PATH = os.path.join(MODELS_DIR, "cluster_names.json")

DEFAULT_CLUSTER_NAMES = {
    0: "Browsers",
    1: "Active Buyers",
    2: "Researchers",
    3: "High-Value",
}

# Purchase probability per cluster (learned from training)
DEFAULT_CLUSTER_BUY_PROB = {0: 0.08, 1: 0.75, 2: 0.22, 3: 0.92}

CLUSTER_NAMES = dict(DEFAULT_CLUSTER_NAMES)
CLUSTER_BUY_PROB = dict(DEFAULT_CLUSTER_BUY_PROB)


def _load_cluster_metadata():
    global CLUSTER_NAMES, CLUSTER_BUY_PROB

    CLUSTER_NAMES = dict(DEFAULT_CLUSTER_NAMES)
    CLUSTER_BUY_PROB = dict(DEFAULT_CLUSTER_BUY_PROB)

    if os.path.exists(CLUSTER_NAMES_PATH):
        with open(CLUSTER_NAMES_PATH) as f:
            raw_names = json.load(f)
        CLUSTER_NAMES.update({int(k): str(v) for k, v in raw_names.items()})

    if os.path.exists(CLUSTER_BUY_PROB_PATH):
        with open(CLUSTER_BUY_PROB_PATH) as f:
            raw_probs = json.load(f)
        CLUSTER_BUY_PROB.update({int(k): float(v) for k, v in raw_probs.items()})


def load_models():
    required = ["logistic_regression.pkl", "random_forest.pkl", "kmeans.pkl", "scaler.pkl"]
    for fname in required:
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            sys.exit(
                f"[ERROR] Model file not found: {path}\n"
                "Run train_models.py first to generate models."
            )
    lr     = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    rf     = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    km     = joblib.load(os.path.join(MODELS_DIR, "kmeans.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)

    _load_cluster_metadata()

    return lr, rf, km, scaler, feature_cols


def score_row(row_dict: dict, lr, rf, km, scaler, feature_cols) -> dict:
    """Score a single feature dictionary."""
    row = pd.DataFrame([row_dict])[feature_cols].fillna(0)
    X   = scaler.transform(row.values)

    lr_prob  = float(lr.predict_proba(X)[0, 1])
    rf_prob  = float(rf.predict_proba(X)[0, 1])
    cluster  = int(km.predict(X)[0])
    km_prob  = CLUSTER_BUY_PROB.get(cluster, 0.3)
    ensemble = lr_prob * 0.30 + rf_prob * 0.50 + km_prob * 0.20

    def verdict(p):
        if p >= 0.70: return "Will Likely Purchase"
        if p >= 0.40: return "Moderate Chance"
        return "Low Likelihood"

    return {
        "lr_score":        round(lr_prob,  4),
        "rf_score":        round(rf_prob,  4),
        "kmeans_score":    round(km_prob,  4),
        "ensemble_score":  round(ensemble, 4),
        "cluster":         cluster,
        "cluster_name":    CLUSTER_NAMES.get(cluster, "Unknown"),
        "verdict":         verdict(ensemble),
    }


def print_result(result: dict):
    print("\n" + "=" * 52)
    print("  PREDICTION RESULT")
    print("=" * 52)
    print(f"  Ensemble Score  : {result['ensemble_score']*100:.1f}%")
    print(f"  Verdict         : {result['verdict']}")
    print(f"  Cluster         : {result['cluster']} – {result['cluster_name']}")
    print("-" * 52)
    print(f"  Random Forest   : {result['rf_score']*100:.1f}%")
    print(f"  Logistic Reg.   : {result['lr_score']*100:.1f}%")
    print(f"  K-Means         : {result['kmeans_score']*100:.1f}%")
    print("=" * 52)


def predict_single(args, lr, rf, km, scaler, feature_cols):
    row = {
        "view_count":         args.views,
        "cart_count":         args.carts,
        "session_duration":   args.session_duration,
        "avg_price_viewed":   args.avg_price,
        "unique_categories":  args.unique_categories,
        "unique_brands":      getattr(args, "unique_brands", 1),
        "brand_loyalty":      getattr(args, "brand_loyalty", 0.0),
        "cart_to_view_ratio": getattr(args, "cart_to_view_ratio",
                                      args.carts / (args.views + 1)),
        "price_range":        getattr(args, "price_range", 0),
    }
    print("\n[input]", row)
    result = score_row(row, lr, rf, km, scaler, feature_cols)
    print_result(result)


def predict_batch(args, lr, rf, km, scaler, feature_cols):
    if not os.path.exists(args.input):
        sys.exit(f"[ERROR] Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    print(f"[batch] Loaded {len(df):,} rows from {args.input}")

    results = []
    for _, row in df.iterrows():
        r = score_row(row.to_dict(), lr, rf, km, scaler, feature_cols)
        results.append(r)

    result_df = pd.DataFrame(results)
    out_df = pd.concat([df.reset_index(drop=True), result_df], axis=1)

    out_path = args.output or "scored_output.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[batch] Saved {len(out_df):,} scored rows → {out_path}")
    print(f"[batch] High propensity (>70%): {(result_df['ensemble_score']>=0.7).sum():,}")
    print(f"[batch] Mid  propensity (40-70%): {((result_df['ensemble_score']>=0.4) & (result_df['ensemble_score']<0.7)).sum():,}")
    print(f"[batch] Low  propensity (<40%): {(result_df['ensemble_score']<0.4).sum():,}")


def main():
    parser = argparse.ArgumentParser(description="EcomML Predict")
    parser.add_argument("--mode", choices=["single", "batch", "new"], default="single")

    # single / new customer fields
    parser.add_argument("--views",              type=float, default=5)
    parser.add_argument("--carts",              type=float, default=1)
    parser.add_argument("--purchases",          type=float, default=0)
    parser.add_argument("--session_duration",   type=float, default=8)
    parser.add_argument("--avg_price",          type=float, default=60)
    parser.add_argument("--unique_categories",  type=float, default=2)
    parser.add_argument("--unique_brands",      type=float, default=1)
    parser.add_argument("--brand_loyalty",      type=float, default=0.0)
    parser.add_argument("--cart_to_view_ratio", type=float, default=None)
    parser.add_argument("--price_range",        type=float, default=0)

    # batch
    parser.add_argument("--input",  type=str, default=None)
    parser.add_argument("--output", type=str, default="scored_output.csv")

    args = parser.parse_args()

    lr, rf, km, scaler, feature_cols = load_models()

    if args.mode in ("single", "new"):
        predict_single(args, lr, rf, km, scaler, feature_cols)
    elif args.mode == "batch":
        predict_batch(args, lr, rf, km, scaler, feature_cols)


if __name__ == "__main__":
    main()
