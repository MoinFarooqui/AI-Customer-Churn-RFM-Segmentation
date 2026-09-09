import pandas as pd
from pathlib import Path

# ============================================================
# AI-POWERED CUSTOMER CHURN & RFM SEGMENTATION
# RFM ANALYSIS
# ============================================================

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input and output paths
DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_shopping_data_clean.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

RFM_OUTPUT_PATH = (
    OUTPUT_DIR
    / "customer_rfm.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

# Convert purchase date to datetime
df["purchase_date"] = pd.to_datetime(df["purchase_date"])

print("=" * 60)
print("RFM ANALYSIS")
print("=" * 60)

print(f"\nTransaction Records: {len(df):,}")
print(f"Unique Customers: {df['customer_id'].nunique():,}")

# ============================================================
# DEFINE REFERENCE DATE
# ============================================================

# One day after the latest purchase in the dataset
reference_date = df["purchase_date"].max() + pd.Timedelta(days=1)

print(f"\nReference Date: {reference_date.date()}")

# ============================================================
# CALCULATE RFM
# ============================================================

rfm = (
    df.groupby("customer_id")
    .agg(
        recency=(
            "purchase_date",
            lambda x: (reference_date - x.max()).days
        ),
        frequency=(
            "purchase_date",
            "count"
        ),
        monetary=(
            "total_purchase_amount",
            "sum"
        )
    )
    .reset_index()
)

# ============================================================
# RFM SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("RFM SUMMARY")
print("=" * 60)

print(rfm.describe().round(2))

print("\nFirst 10 Customers:")
print(rfm.head(10))

# ============================================================
# VALIDATION CHECKS
# ============================================================

print("\n" + "=" * 60)
print("RFM VALIDATION")
print("=" * 60)

print(f"Total Customers: {len(rfm):,}")

print(f"Missing Values: {rfm.isnull().sum().sum()}")

print(f"Duplicate Customers: {rfm['customer_id'].duplicated().sum()}")

# ============================================================
# SAVE RFM DATASET
# ============================================================

rfm.to_csv(
    RFM_OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("RFM ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nRFM dataset saved to:")
print(RFM_OUTPUT_PATH)