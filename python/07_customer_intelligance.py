import pandas as pd
from pathlib import Path

# ============================================================
# AI-POWERED CUSTOMER CHURN & RFM SEGMENTATION
# CUSTOMER INTELLIGENCE DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# File paths
SEGMENTS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_segments.csv"
)

CHURN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_churn_predictions.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_intelligence.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CUSTOMER INTELLIGENCE DATA INTEGRATION")
print("=" * 60)

segments_df = pd.read_csv(SEGMENTS_PATH)
churn_df = pd.read_csv(CHURN_PATH)

print(f"\nSegmentation Records: {len(segments_df):,}")
print(f"Churn Prediction Records: {len(churn_df):,}")

# ============================================================
# SELECT REQUIRED COLUMNS
# ============================================================

segments_df = segments_df[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "cluster"
    ]
]

# ============================================================
# MERGE DATASETS
# ============================================================

customer_intelligence = pd.merge(
    segments_df,
    churn_df,
    on="customer_id",
    how="inner",
    suffixes=("_rfm", "_churn")
)

# ============================================================
# DATA VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("MERGED DATASET VALIDATION")
print("=" * 60)

print(f"\nTotal Customer Profiles: {len(customer_intelligence):,}")

print("\nMissing Values:")
print(customer_intelligence.isnull().sum())

print("\nFirst Five Records:")
print(customer_intelligence.head())

# ============================================================
# SAVE DATASET
# ============================================================

customer_intelligence.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("CUSTOMER INTELLIGENCE DATASET CREATED")
print("=" * 60)

print(f"\nSaved to:\n{OUTPUT_PATH}")