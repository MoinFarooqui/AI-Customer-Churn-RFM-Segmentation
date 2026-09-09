import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================================
# AI-POWERED CUSTOMER CHURN & RFM SEGMENTATION
# CUSTOMER SEGMENTATION USING K-MEANS
# ============================================================

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths
RFM_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_rfm.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_segments.csv"
)

# ============================================================
# LOAD RFM DATA
# ============================================================

rfm = pd.read_csv(RFM_DATA_PATH)

print("=" * 60)
print("CUSTOMER SEGMENTATION USING K-MEANS")
print("=" * 60)

print(f"\nTotal Customers: {len(rfm):,}")

# ============================================================
# SELECT RFM FEATURES
# ============================================================

features = [
    "recency",
    "frequency",
    "monetary"
]

X = rfm[features]

# ============================================================
# FEATURE SCALING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SCALING")
print("=" * 60)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("RFM features scaled successfully.")

# ============================================================
# FIND OPTIMAL NUMBER OF CLUSTERS
# ============================================================

print("\n" + "=" * 60)
print("SILHOUETTE SCORE ANALYSIS")
print("=" * 60)

silhouette_scores = {}

for k in range(2, 7):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        cluster_labels
    )

    silhouette_scores[k] = score

    print(
        f"K = {k} | "
        f"Silhouette Score = {score:.4f}"
    )

# ============================================================
# SELECT BEST K
# ============================================================

best_k = max(
    silhouette_scores,
    key=silhouette_scores.get
)

print("\nBest Number of Clusters:", best_k)

# ============================================================
# FINAL K-MEANS MODEL
# ============================================================

print("\n" + "=" * 60)
print("FINAL K-MEANS CLUSTERING")
print("=" * 60)

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

rfm["cluster"] = final_kmeans.fit_predict(
    X_scaled
)

print("\nCluster Distribution:")

print(
    rfm["cluster"]
    .value_counts()
    .sort_index()
)

# ============================================================
# CLUSTER ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CLUSTER CHARACTERISTICS")
print("=" * 60)

cluster_summary = (
    rfm.groupby("cluster")
    .agg(
        customers=("customer_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean")
    )
    .round(2)
)

print(cluster_summary)

# ============================================================
# SAVE RESULTS
# ============================================================

rfm.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("CUSTOMER SEGMENTATION COMPLETED")
print("=" * 60)

print(f"\nResults saved to:")
print(OUTPUT_PATH)