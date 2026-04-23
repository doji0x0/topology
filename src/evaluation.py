import os
import pandas as pd

from noise import generate_noisy_datasets
from stochastic import run_logistic_regression, run_gmm
from topology import run_topological_pipeline
from topology_distance import compute_distances

from data_loader import load_moons_data, load_mnist_data


def evaluate(dataset="moons"):
    # =========================================
    # 1. Load Data
    # =========================================
    if dataset == "moons":
        X, y = load_moons_data()

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        X_full = X
        y_full = y

    elif dataset == "mnist":
        X_full, y_full, X_train, X_test, y_train, y_test = load_mnist_data()

    elif dataset == "student":
        from data_loader import load_student_data

        X, y = load_student_data()

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.3, random_state=42
        )

        X_full = X
        y_full = y

    else:
        raise ValueError("Unknown dataset")

    # =========================================
    # 2. Generate noisy datasets
    # =========================================
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    noisy_data = generate_noisy_datasets(X_full, y_full, noise_levels)

    results = []

    # =========================================
    # 3. Loop over noise levels
    # =========================================
    for level, data in noisy_data.items():
        X_noisy = data["X"]
        y_labels = data["y"]

        from sklearn.model_selection import train_test_split
        X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
            X_noisy, y_labels, test_size=0.3, random_state=42
        )

        # =========================================
        # ML Models
        # =========================================
        logistic = run_logistic_regression(
            X_train_n, X_test_n, y_train_n, y_test_n
        )

        # 🔥 FIXED (no data leakage)
        gmm = run_gmm(
            X_train_n, X_test_n, y_train_n, y_test_n
        )

        # =========================================
        # Topology
        # =========================================
        topo = run_topological_pipeline(X_noisy)

        # =========================================
        # Distances
        # =========================================
        dist = compute_distances(X_full, X_noisy)

        # =========================================
        # Store results
        # =========================================
        row = {
            "noise_level": level,

            # Logistic
            "logistic_accuracy": logistic["accuracy"],
            "logistic_confidence": logistic["confidence"],
            "logistic_time": logistic["execution_time"],

            # GMM
            "gmm_accuracy": gmm["accuracy"],
            "gmm_confidence": gmm["confidence"],
            "gmm_time": gmm["execution_time"],

            # Topology
            "topology_h1_count": topo["h1_feature_count"],
            "topology_max_h1_persistence": topo["max_h1_persistence"],
            "topology_mean_h1_persistence": topo["mean_h1_persistence"],
            "topology_total_h1_persistence": topo["total_h1_persistence"],
            "topology_score": topo["topological_score"],
            "persistence_variance": topo["persistence_variance"],  # 🔥 new
            "topology_time": topo["execution_time"],

            # Distances
            "bottleneck_distance": dist["bottleneck_distance"],
            "wasserstein_distance": dist["wasserstein_distance"],
        }

        results.append(row)

    df = pd.DataFrame(results)

    # =========================================
    # 4. Save results
    # =========================================
    os.makedirs(f"results/{dataset}/tables", exist_ok=True)

    path = f"results/{dataset}/tables/results.csv"
    df.to_csv(path, index=False)

    print(f"Results saved to: {path}")
    return df


if __name__ == "__main__":
    dataset = "mnist"   # غيريها إلى "moons" أو "student"  "mnist"
    evaluate(dataset)