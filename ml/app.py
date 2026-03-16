import json
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DASHBOARD_DIR = PROJECT_DIR / "dashboard"

# Ensure predict.py can resolve model paths correctly.
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

from predict import load_models, score_row  # noqa: E402

app = Flask(__name__)
CORS(app)

MODELS = None
MODEL_LOAD_ERROR = None


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
    metrics_path = BASE_DIR / "reports" / "metrics_report.json"
    with open(metrics_path, encoding="utf-8") as f:
        return jsonify(json.load(f))


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
