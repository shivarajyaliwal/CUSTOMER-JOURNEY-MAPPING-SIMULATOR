"""
EcomML – Feature Engineering Utilities
========================================
Reusable functions for transforming raw event-level data
into ML-ready feature vectors.
"""

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "view_count",
    "cart_count",
    "session_duration",
    "avg_price_viewed",
    "unique_categories",
    "unique_brands",
    "brand_loyalty",
    "cart_to_view_ratio",
    "price_range",
]


# ──────────────────────────────────────────────
#  PER-SESSION AGGREGATION
# ──────────────────────────────────────────────
def aggregate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  raw event DataFrame (one row = one event)
    Output: one row per (user_id, user_session) with engineered features
    """
    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_time", "user_id", "user_session"])

    grp = df.groupby(["user_id", "user_session"])

    feat                       = pd.DataFrame(index=grp.groups.keys())
    feat["view_count"]         = grp.apply(lambda x: (x["event_type"] == "view").sum())
    feat["cart_count"]         = grp.apply(lambda x: (x["event_type"] == "cart").sum())
    feat["purchase_count"]     = grp.apply(lambda x: (x["event_type"] == "purchase").sum())
    feat["session_duration"]   = grp["event_time"].apply(
        lambda x: (x.max() - x.min()).total_seconds() / 60
    )
    feat["avg_price_viewed"]   = grp["price"].mean().fillna(0)
    feat["unique_categories"]  = grp["category_code"].nunique()
    feat["unique_brands"]      = grp["brand"].nunique()
    feat["brand_loyalty"]      = grp["brand"].apply(
        lambda x: x.value_counts().iloc[0] / len(x) if len(x) > 0 else 0.0
    )
    feat["cart_to_view_ratio"] = (
        feat["cart_count"] / (feat["view_count"] + 1)
    )
    feat["price_range"]        = grp["price"].apply(
        lambda x: x.max() - x.min()
    ).fillna(0)

    feat = feat.reset_index()
    feat["label"] = (feat["purchase_count"] > 0).astype(int)
    return feat


# ──────────────────────────────────────────────
#  PER-USER AGGREGATION (lifetime features)
# ──────────────────────────────────────────────
def aggregate_users(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all sessions for each user_id into a single
    lifetime feature row. Useful for scoring existing customers.
    """
    sessions = aggregate_sessions(df)
    grp = sessions.groupby("user_id")

    user_feat = pd.DataFrame()
    user_feat["total_sessions"]    = grp["user_session"].nunique()
    user_feat["total_views"]       = grp["view_count"].sum()
    user_feat["total_carts"]       = grp["cart_count"].sum()
    user_feat["total_purchases"]   = grp["purchase_count"].sum()
    user_feat["avg_session_dur"]   = grp["session_duration"].mean()
    user_feat["avg_price_viewed"]  = grp["avg_price_viewed"].mean()
    user_feat["unique_categories"] = grp["unique_categories"].max()
    user_feat["unique_brands"]     = grp["unique_brands"].max()
    user_feat["brand_loyalty"]     = grp["brand_loyalty"].mean()
    user_feat["cart_to_view_ratio"]= grp["cart_to_view_ratio"].mean()
    user_feat["price_range"]       = grp["price_range"].mean()
    user_feat["ever_purchased"]    = (grp["purchase_count"].sum() > 0).astype(int)

    return user_feat.reset_index()


# ──────────────────────────────────────────────
#  SINGLE-ROW DICT → FEATURE ARRAY
# ──────────────────────────────────────────────
def dict_to_feature_array(d: dict) -> np.ndarray:
    """
    Convert a dictionary of raw inputs (from the dashboard or CLI)
    to a 1-D numpy array aligned to FEATURE_COLS.

    Missing keys default to 0. cart_to_view_ratio is auto-computed
    if not supplied.
    """
    if "cart_to_view_ratio" not in d:
        d["cart_to_view_ratio"] = d.get("cart_count", 0) / (d.get("view_count", 0) + 1)

    return np.array([float(d.get(col, 0)) for col in FEATURE_COLS], dtype=np.float64)


# ──────────────────────────────────────────────
#  LABEL ENCODING
# ──────────────────────────────────────────────
CATEGORY_MAP = {
    "electronics": 0, "clothing": 1, "furniture": 2, "sports": 3,
    "beauty": 4, "computers": 5, "phones": 6, "garden": 7,
}

BRAND_MAP = {
    "samsung": 0, "apple": 1, "huawei": 2, "xiaomi": 3, "lg": 4,
    "sony": 5, "nike": 6, "adidas": 7, "bosch": 8, "philips": 9,
}


def encode_category(cat: str) -> int:
    return CATEGORY_MAP.get(str(cat).lower(), -1)


def encode_brand(brand: str) -> int:
    return BRAND_MAP.get(str(brand).lower(), -1)


# ──────────────────────────────────────────────
#  VALIDATION
# ──────────────────────────────────────────────
VALID_RANGES = {
    "view_count":         (0, 500),
    "cart_count":         (0, 100),
    "session_duration":   (0, 300),
    "avg_price_viewed":   (0, 5000),
    "unique_categories":  (0, 20),
    "unique_brands":      (0, 20),
    "brand_loyalty":      (0.0, 1.0),
    "cart_to_view_ratio": (0.0, 1.0),
    "price_range":        (0, 5000),
}


def validate_features(d: dict) -> list[str]:
    """Return list of warning strings for out-of-range values."""
    warnings = []
    for col, (lo, hi) in VALID_RANGES.items():
        val = d.get(col)
        if val is not None and not (lo <= float(val) <= hi):
            warnings.append(f"  {col}={val} is outside expected range [{lo}, {hi}]")
    return warnings
