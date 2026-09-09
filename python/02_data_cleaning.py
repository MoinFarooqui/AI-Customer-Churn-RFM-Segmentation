import pandas as pd
from pathlib import Path

# ============================================================
# AI-POWERED CUSTOMER CHURN & RFM SEGMENTATION
# DATA CLEANING
# ============================================================

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths
RAW_DATA_PATH = (
    BASE_DIR / "data" / "raw" / "customer_shopping_data.csv"
)

PROCESSED_DATA_DIR = (
    BASE_DIR / "data" / "processed"
)

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR / "customer_shopping_data_clean.csv"
)

# Create processed folder if it does not exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(RAW_DATA_PATH)

print("=" * 60)
print("DATA CLEANING STARTED")
print("=" * 60)

print(f"\nOriginal Rows: {df.shape[0]:,}")
print(f"Original Columns: {df.shape[1]}")

# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

print("\n" + "=" * 60)
print("STANDARDIZING COLUMN NAMES")
print("=" * 60)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nUpdated Column Names:")
print(df.columns.tolist())

# ============================================================
# CONVERT PURCHASE DATE
# ============================================================

print("\n" + "=" * 60)
print("CONVERTING PURCHASE DATE")
print("=" * 60)

df["purchase_date"] = pd.to_datetime(
    df["purchase_date"],
    errors="coerce"
)

invalid_dates = df["purchase_date"].isnull().sum()

print(f"Invalid or missing dates: {invalid_dates:,}")

# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES CHECK")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

# ============================================================
# REMOVE DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE ROW CHECK")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows found: {duplicate_count:,}")

if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"Duplicate rows removed: {duplicate_count:,}")
else:
    print("No duplicate rows found.")

# ============================================================
# VALIDATE NUMERIC VALUES
# ============================================================

print("\n" + "=" * 60)
print("NUMERIC DATA VALIDATION")
print("=" * 60)

numeric_columns = [
    "customer_age",
    "product_price",
    "quantity",
    "total_purchase_amount",
    "returns",
    "churn"
]

for column in numeric_columns:

    if column in df.columns:

        invalid_values = (df[column] < 0).sum()

        print(
            f"{column}: "
            f"{invalid_values:,} negative values"
        )

# ============================================================
# REMOVE INVALID RECORDS
# ============================================================

print("\n" + "=" * 60)
print("REMOVING INVALID RECORDS")
print("=" * 60)

rows_before = len(df)

# Remove records without Customer ID
df = df.dropna(subset=["customer_id"])

# Remove records with invalid purchase dates
df = df.dropna(subset=["purchase_date"])

rows_after = len(df)

print(
    f"Rows removed during validation: "
    f"{rows_before - rows_after:,}"
)

# ============================================================
# FINAL DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print(f"Final Rows: {df.shape[0]:,}")
print(f"Final Columns: {df.shape[1]}")

print("\nRemaining Missing Values:")
print(df.isnull().sum().sum())

# ============================================================
# SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    PROCESSED_DATA_PATH,
    index=False
)

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nCleaned dataset saved to:")
print(PROCESSED_DATA_PATH)