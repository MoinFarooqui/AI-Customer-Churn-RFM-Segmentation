import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# AI-POWERED CUSTOMER CHURN & RFM SEGMENTATION
# CHURN PREDICTION USING RANDOM FOREST
# ============================================================

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths
DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_shopping_data_clean.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "customer_churn_predictions.csv"
)

MODEL_PATH = (
    MODEL_DIR
    / "random_forest_churn_model.joblib"
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CHURN PREDICTION USING RANDOM FOREST")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

df["purchase_date"] = pd.to_datetime(
    df["purchase_date"]
)

print(f"\nTransaction Records: {len(df):,}")
print(f"Unique Customers: {df['customer_id'].nunique():,}")

# ============================================================
# CUSTOMER-LEVEL FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER FEATURE ENGINEERING")
print("=" * 60)

reference_date = (
    df["purchase_date"].max()
    + pd.Timedelta(days=1)
)

customer_data = (
    df.groupby("customer_id")
    .agg(
        customer_age=(
            "customer_age",
            "first"
        ),

        total_purchases=(
            "purchase_date",
            "count"
        ),

        total_spending=(
            "total_purchase_amount",
            "sum"
        ),

        average_order_value=(
            "total_purchase_amount",
            "mean"
        ),

        total_quantity=(
            "quantity",
            "sum"
        ),

        total_returns=(
            "returns",
            "sum"
        ),

        recency=(
            "purchase_date",
            lambda x: (
                reference_date - x.max()
            ).days
        ),

        churn=(
            "churn",
            "first"
        )
    )
    .reset_index()
)

# Calculate return rate
customer_data["return_rate"] = (
    customer_data["total_returns"]
    / customer_data["total_purchases"]
)

print(f"Customer Profiles Created: {len(customer_data):,}")

# ============================================================
# CHURN DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CHURN DISTRIBUTION")
print("=" * 60)

print(customer_data["churn"].value_counts())

# ============================================================
# PREPARE FEATURES
# ============================================================

print("\n" + "=" * 60)
print("PREPARING FEATURES")
print("=" * 60)

features = [
    "customer_age",
    "total_purchases",
    "total_spending",
    "average_order_value",
    "total_quantity",
    "total_returns",
    "recency",
    "return_rate"
]

X = customer_data[features]
y = customer_data["churn"]

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining Records: {len(X_train):,}")
print(f"Testing Records: {len(X_test):,}")

# ============================================================
# RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST MODEL")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)

print("Model training completed.")

# ============================================================
# MODEL PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

feature_importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(
    by="importance",
    ascending=False
)

print(feature_importance)

# ============================================================
# PREDICT CHURN RISK FOR ALL CUSTOMERS
# ============================================================

customer_data["churn_probability"] = (
    model.predict_proba(X)[:, 1]
)

customer_data["churn_risk"] = pd.cut(
    customer_data["churn_probability"],
    bins=[0, 0.30, 0.60, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"],
    include_lowest=True
)

# ============================================================
# SAVE RESULTS
# ============================================================

customer_data.to_csv(
    OUTPUT_PATH,
    index=False
)

joblib.dump(
    model,
    MODEL_PATH
)

print("\n" + "=" * 60)
print("CHURN PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFiles Created:")

print(f"Customer Predictions: {OUTPUT_PATH}")
print(f"Random Forest Model: {MODEL_PATH}")