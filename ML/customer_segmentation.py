"""Group wholesale customers by their annual product-category spending."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Data source: Cardoso, M. (2013), Wholesale customers,
# UCI Machine Learning Repository. DOI: 10.24432/C5030X
# https://archive.ics.uci.edu/dataset/292/wholesale+customers
# The dataset contains 440 wholesale customers and their annual spending in
# monetary units across six product categories. It has no missing values.
DATA_FILE = Path(__file__).with_name("wholesale_customers.csv")
ELBOW_PLOT_FILE = Path(__file__).with_name("elbow_plot.png")
SEGMENTS_FILE = Path(__file__).with_name("customer_segments.csv")
RANDOM_STATE = 42

SPENDING_FEATURES = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen",
]
REQUIRED_COLUMNS = ["Channel", "Region"] + SPENDING_FEATURES
CHANNEL_NAMES = {1: "Horeca", 2: "Retail"}
REGION_NAMES = {1: "Lisbon", 2: "Oporto", 3: "Other"}


def load_customer_data(file_path=DATA_FILE):
    """Load the wholesale dataset and validate the fields used in clustering."""
    data = pd.read_csv(file_path)

    missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"The dataset is missing these columns: {missing_text}")

    data = data[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()

    if len(data) < 100:
        raise ValueError("The assignment requires at least 100 valid records.")
    if (data[SPENDING_FEATURES] < 0).any().any():
        raise ValueError("Annual spending values cannot be negative.")

    data["channel_name"] = data["Channel"].map(CHANNEL_NAMES)
    data["region_name"] = data["Region"].map(REGION_NAMES)
    data["annual_spending"] = data[SPENDING_FEATURES].sum(axis=1)
    return data


def scale_spending_features(data):
    """Reduce spending skew and give every product category equal scale."""
    log_spending = np.log1p(data[SPENDING_FEATURES])
    scaler = StandardScaler()
    return scaler.fit_transform(log_spending)


def calculate_inertia(scaled_features, k_values):
    """Calculate K-Means inertia for every candidate number of clusters."""
    inertia_values = []
    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
        )
        model.fit(scaled_features)
        inertia_values.append(model.inertia_)
    return inertia_values


def select_elbow_k(k_values, inertia_values):
    """Select the curve point farthest from the first-to-last reference line."""
    normalized_k = (k_values - k_values.min()) / (
        k_values.max() - k_values.min()
    )
    inertia_array = np.asarray(inertia_values)
    normalized_inertia = (inertia_array - inertia_array.min()) / (
        inertia_array.max() - inertia_array.min()
    )

    x1, y1 = normalized_k[0], normalized_inertia[0]
    x2, y2 = normalized_k[-1], normalized_inertia[-1]
    numerator = np.abs(
        (y2 - y1) * normalized_k
        - (x2 - x1) * normalized_inertia
        + x2 * y1
        - y2 * x1
    )
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distances = numerator / denominator
    return int(k_values[np.argmax(distances)])


def save_elbow_plot(k_values, inertia_values, optimal_k):
    """Save the elbow curve and mark the selected value of K."""
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia_values, marker="o", color="#1f77b4")
    plt.axvline(
        optimal_k,
        color="#d62728",
        linestyle="--",
        label=f"Selected K = {optimal_k}",
    )
    plt.xticks(k_values)
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Customer Segmentation")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(ELBOW_PLOT_FILE, dpi=150)
    plt.close()


def print_cluster_analysis(data, cluster_summary):
    """Explain each cluster and suggest a matching marketing strategy."""
    overall_spending = data["annual_spending"].mean()

    print("\nCluster analysis and marketing strategies:")
    for cluster_id, summary in cluster_summary.iterrows():
        cluster_rows = data[data["cluster"] == cluster_id]
        top_category = summary[SPENDING_FEATURES].idxmax()
        dominant_channel = cluster_rows["channel_name"].mode().iloc[0]
        dominant_region = cluster_rows["region_name"].mode().iloc[0]

        print(f"\nCluster {cluster_id} ({int(summary['customer_count'])} customers)")
        print(
            f"  Average annual spending: {summary['annual_spending']:,.0f} "
            "monetary units"
        )
        print(f"  Largest average category: {top_category}")
        print(f"  Most common channel: {dominant_channel}")
        print(f"  Most common region: {dominant_region}")

        if summary["annual_spending"] >= overall_spending * 1.25:
            strategy = (
                "Offer account-based service, loyalty rewards, and early access "
                "to bulk-purchase promotions."
            )
        elif top_category in {"Grocery", "Milk", "Detergents_Paper"}:
            strategy = (
                "Promote recurring grocery bundles and scheduled reordering for "
                "routine stock needs."
            )
        else:
            strategy = (
                "Offer fresh and frozen product volume discounts aimed at "
                "restaurants, hotels, and cafes."
            )
        print(f"  Suggested strategy: {strategy}")


def main():
    """Run the elbow method, create clusters, and save the results."""
    data = load_customer_data()
    scaled_features = scale_spending_features(data)
    print(f"Valid customer records loaded: {len(data):,}")

    k_values = np.arange(1, 11)
    inertia_values = calculate_inertia(scaled_features, k_values)
    optimal_k = select_elbow_k(k_values, inertia_values)

    print("\nElbow-method results:")
    for k, inertia in zip(k_values, inertia_values):
        print(f"  K = {k}: inertia = {inertia:,.2f}")
    print(f"\nSelected K: {optimal_k}")
    print(
        "Justification: The curve bends most strongly at this point. Adding "
        "more clusters after it produces smaller improvements in inertia."
    )

    save_elbow_plot(k_values, inertia_values, optimal_k)

    final_model = KMeans(
        n_clusters=optimal_k,
        random_state=RANDOM_STATE,
        n_init=20,
    )
    data["cluster"] = final_model.fit_predict(scaled_features)

    cluster_summary = data.groupby("cluster")[
        SPENDING_FEATURES + ["annual_spending"]
    ].mean()
    cluster_summary.insert(0, "customer_count", data["cluster"].value_counts())
    cluster_summary = cluster_summary.round(2)

    print("\nAverage cluster characteristics:")
    print(cluster_summary.to_string())
    print_cluster_analysis(data, cluster_summary)

    data.to_csv(SEGMENTS_FILE, index=False)
    print(f"\nCluster assignments saved to: {SEGMENTS_FILE.name}")
    print(f"Elbow plot saved to: {ELBOW_PLOT_FILE.name}")


if __name__ == "__main__":
    main()
