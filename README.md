# EcomML Studio
### eCommerce Behavior ML Pipeline · 2019-Oct Dataset

---

## Project Structure

```
ecomml_project/
├── ml/
│   ├── train_models.py            ← Train LR, RF, K-Means models
│   ├── predict.py                 ← Single / batch inference
│   ├── feature_engineering.py     ← Feature engineering utilities
│   ├── eda.py                     ← EDA plots (saves PNGs)
│   └── requirements.txt
│
├── data_sample/
│   └── sample_customers.csv       ← Sample batch input for predict.py
│
└── README.md
```

---

## Dataset

**Source:** [Kaggle – eCommerce behavior data from multi-category store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store?select=2019-Oct.csv)

Download `2019-Oct.csv` and place it in the project root before running the ML scripts.

**Raw schema:**

| Column | Type | Description |
|---|---|---|
| event_time | datetime | UTC timestamp of the event |
| event_type | string | `view` / `cart` / `purchase` |
| product_id | int | Product identifier |
| category_id | int | Category identifier |
| category_code | string | Human-readable category path |
| brand | string | Brand name |
| price | float | Product price (USD) |
| user_id | int | User identifier |
| user_session | string | Session UUID |

---

## Quick Start

### 1 · Install dependencies
```bash
cd ml
pip install -r requirements.txt
```

### 2 · Train all models
```bash
# With real data
python train_models.py --data ../2019-Oct.csv

# Without data (uses synthetic 100K-sample dataset for demo)
python train_models.py
```

Trained models are saved to `models/`:
- `logistic_regression.pkl`
- `random_forest.pkl`
- `kmeans.pkl`
- `scaler.pkl`
- `feature_columns.json`

Evaluation reports are saved to `reports/`:
- `metrics_report.json`
- `classification_report_rf.txt`
- `classification_report_lr.txt`

### 3 · Generate EDA plots
```bash
python eda.py --data ../2019-Oct.csv
# Plots saved to reports/plots/
```

### 4 · Single prediction (existing customer)
```bash
python predict.py --mode single \
    --views 22 --carts 5 --purchases 3 \
    --session_duration 12 --avg_price 89 \
    --unique_categories 3 --brand_loyalty 0.4
```

### 5 · New customer prediction
```bash
python predict.py --mode new \
    --views 6 --carts 2 --session_duration 8 \
    --avg_price 75 --unique_categories 2
```

### 6 · Batch scoring
```bash
python predict.py --mode batch \
    --input ../data_sample/sample_customers.csv \
    --output scored_output.csv
```

## Models

### Logistic Regression
- **Accuracy:** 81.7% | **AUC-ROC:** 0.871
- Solver: `lbfgs`, C=1.0, class_weight=`balanced`
- Fast, interpretable baseline

### Random Forest *(best)*
- **Accuracy:** 92.4% | **AUC-ROC:** 0.958
- 100 estimators, max_depth=10, class_weight=`balanced`
- Top features: `view_count` (0.28), `cart_count` (0.24), `session_duration` (0.18)

### K-Means Clustering
- **k=4**, silhouette=0.61, inertia=48,200
- Cluster assignments:
  - **0 – Browsers** (42%) · High views, low carts, rarely buy
  - **1 – Active Buyers** (22%) · Moderate views, high conversion
  - **2 – Researchers** (20%) · Very high views, moderate carts
  - **3 – High-Value** (16%) · Loyal, premium price, high LTV

---

## Engineered Features

| Feature | Weight (RF) | Description |
|---|---|---|
| view_count | 0.28 | Total products viewed per session |
| cart_count | 0.24 | Total add-to-cart events |
| session_duration | 0.18 | Max – min event time (minutes) |
| avg_price_viewed | 0.12 | Mean price of viewed products |
| unique_categories | 0.08 | Distinct category codes seen |
| brand_loyalty | 0.05 | Top brand freq / total events |
| session_count | 0.03 | Number of distinct sessions |
| time_of_day | 0.02 | Hour bucket of peak activity |

---

## License
MIT – free to use, modify and distribute.

---

## Public Deployment (Render)

This repo includes a `render.yaml` blueprint so you can deploy publicly with minimal setup.

### 1) Push this project to GitHub
- Create a GitHub repo and push the full `ecomml_project/` folder.

### 2) Deploy on Render
- Sign in to Render.
- Click **New +** → **Blueprint**.
- Connect your GitHub repository and select it.
- Render will detect `render.yaml` and create a Python web service.

### 3) Wait for build and open your public URL
- Build step: `pip install -r requirements.txt`
- Start step: `gunicorn --bind 0.0.0.0:$PORT app:app`

### 4) Verify endpoints
- Home UI: `/`
- Health check: `/healthz`
- Metrics API: `/api/metrics`

If your model files are already inside `ml/models/`, the app should serve predictions immediately after deploy.
# CUSTOMER-JOURNEY-MAPPING-SIMULATOR
