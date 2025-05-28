"""
customer_segmentation_per_user.py

This module loads a cleaned transactions dataset, aggregates features per customer (card_id),
clusters customers into 4 segments, assigns each customer a segment label, and visualizes
the segments in 2D via PCA.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def load_data(path: str) -> pd.DataFrame:
    """Load transactions CSV into a DataFrame."""
    return pd.read_csv(path)


def aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data into customer-level features.
    - Numeric features: mean, sum, std of transaction_amount_kzt and merchant_city_code
    - Categorical features: one-hot encode and sum counts per card_id
    """
    # Numeric features to aggregate
    num_cols = ["transaction_amount_kzt", "merchant_city_code"]
    # Categorical columns to one-hot then sum
    cat_cols = [
        "transaction_type",
        "pos_entry_mode",
        "wallet_type",
        "time_of_day",
        "transaction_currency_code",
    ]
    # One-hot encode categoricals
    df_ohe = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, dtype=int)
    # Build aggregation dict: for numeric -> ['sum','mean','std'], for dummy cols -> 'sum'
    agg_dict = {}
    for col in num_cols:
        agg_dict[col] = ["sum", "mean", "std"]
    for col in df_ohe.columns:
        if any(col.startswith(prefix + "_") for prefix in cat_cols):
            agg_dict[col] = "sum"
    # Group by customer
    grouped = df_ohe.groupby("card_id").agg(agg_dict)
    # Flatten MultiIndex columns
    grouped.columns = [f"{feat}_{stat}" for feat, stat in grouped.columns]
    # Fill any NaNs that result from std of single observations
    return grouped.fillna(0)


def cluster_and_assign(
    features: pd.DataFrame, n_clusters: int = 4, random_state: int = 42
):
    """
    Fit a KMeans clusterer to customer features and assign segment labels.
    Returns features with a 'segment' column and the fitted pipeline.
    """
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clusterer", KMeans(n_clusters=n_clusters, random_state=random_state)),
        ]
    )
    labels = pipeline.fit_predict(features)
    features = features.copy()
    features["segment"] = labels
    return features, pipeline


def plot_segments(features: pd.DataFrame):
    """
    Reduce features to 2 dimensions with PCA and scatter-plot customers colored by segment.
    """
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(features.drop(columns="segment"))
    features["pc1"], features["pc2"] = coords[:, 0], coords[:, 1]
    plt.figure(figsize=(8, 6))
    for seg in sorted(features["segment"].unique()):
        mask = features["segment"] == seg
        plt.scatter(
            features.loc[mask, "pc1"],
            features.loc[mask, "pc2"],
            label=f"Segment {seg}",
            alpha=0.6,
        )
    plt.legend(title="Cluster")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Customer Segments (PCA projection)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster customers into 4 segments and visualize."
    )
    # Load and aggregate
    df = pd.read_csv(
        "/Users/lithrur/Desktop/Coding projects/hackathon_2025_ai/dataset/dataset_cleaned.csv"
    )
    cust_features = aggregate_features(df)

    # Cluster and assign
    cust_segmented, _ = cluster_and_assign(cust_features)
    cust_segmented["segment"].to_csv("customer_segments.csv")

    # Visualize
    plot_segments(cust_segmented)
