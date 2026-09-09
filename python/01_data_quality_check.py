

import pandas as pd
from pathlib import Path

# ============================================================
# AI-POWERED CUSTOMER CHURN & RFM SEGMENTATION
# DATA QUALITY ASSESSMENT
# ============================================================

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "raw" / "customer_shopping_data.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")

# ============================================================
# DATASET OVERVIEW
# ============================================================

print("\n" + "=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"\nRows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

print("\nColumn Names:")
print(df.columns.tolist())

# ============================================================
# FIRST FIVE ROWS
# ============================================================

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)

print(df.head())

# ============================================================
# DATA TYPES
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())

# ============================================================
# DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

print(f"Duplicate rows: {df.duplicated().sum():,}")

# ============================================================
# UNIQUE CUSTOMERS
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER INFORMATION")
print("=" * 60)

print(f"Unique Customers: {df['Customer ID'].nunique():,}")

# ============================================================
# PURCHASES PER CUSTOMER
# ============================================================

print("\n" + "=" * 60)
print("PURCHASES PER CUSTOMER")
print("=" * 60)

purchase_counts = df.groupby("Customer ID").size()

print(purchase_counts.describe())

# ============================================================
# CHURN DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CHURN DISTRIBUTION")
print("=" * 60)

print(df["Churn"].value_counts())

print("\nChurn Percentage:")

churn_percentage = (
    df["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(churn_percentage)

# ============================================================
# CHURN CONSISTENCY PER CUSTOMER
# ============================================================

print("\n" + "=" * 60)
print("CHURN CONSISTENCY CHECK")
print("=" * 60)

churn_consistency = (
    df.groupby("Customer ID")["Churn"]
    .nunique()
)

inconsistent_customers = (
    churn_consistency > 1
).sum()

print(
    f"Customers with inconsistent churn values: "
    f"{inconsistent_customers:,}"
)

# ============================================================
# NUMERIC SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("NUMERIC SUMMARY")
print("=" * 60)

print(df.describe())

# ============================================================
# DATA QUALITY CHECK COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("DATA QUALITY CHECK COMPLETED")
print("=" * 60)