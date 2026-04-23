import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================================
# Load results
# =========================================
def load_results(dataset="moons"):
    path = f"results/{dataset}/tables/results.csv"
    return pd.read_csv(path)


# =========================================
# Save plot
# =========================================
def save_plot(fig, filename, dataset):
    output_dir = f"results/{dataset}/plots"
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")


# =========================================
# 1. Accuracy
# =========================================
def plot_accuracy(df, dataset):
    fig = plt.figure()

    plt.plot(df["noise_level"], df["logistic_accuracy"], marker="o", label="Logistic")
    plt.plot(df["noise_level"], df["gmm_accuracy"], marker="o", label="GMM")

    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Noise")
    plt.legend()
    plt.grid(True)

    save_plot(fig, "accuracy.png", dataset)
    plt.show()


# =========================================
# 2. Confidence
# =========================================
def plot_confidence(df, dataset):
    fig = plt.figure()

    plt.plot(df["noise_level"], df["logistic_confidence"], marker="o", label="Logistic")
    plt.plot(df["noise_level"], df["gmm_confidence"], marker="o", label="GMM")

    plt.xlabel("Noise Level")
    plt.ylabel("Confidence")
    plt.title("Confidence vs Noise")
    plt.legend()
    plt.grid(True)

    save_plot(fig, "confidence.png", dataset)
    plt.show()


# =========================================
# 3. Topology (H1 count)
# =========================================
def plot_topology(df, dataset):
    fig = plt.figure()

    plt.plot(df["noise_level"], df["topology_h1_count"], marker="o")

    plt.xlabel("Noise Level")
    plt.ylabel("H1 Count")
    plt.title("Topology (H1) vs Noise")

    plt.grid(True)
    save_plot(fig, "topology.png", dataset)
    plt.show()


# =========================================
# 4. Topological Score
# =========================================
def plot_topological_score(df, dataset):
    fig = plt.figure()

    plt.plot(df["noise_level"], df["topology_score"], marker="o")

    plt.xlabel("Noise Level")
    plt.ylabel("Topological Score")
    plt.title("Topological Score vs Noise")

    plt.grid(True)
    save_plot(fig, "topo_score.png", dataset)
    plt.show()


# =========================================
# 5. Execution Time
# =========================================
def plot_time(df, dataset):
    fig = plt.figure()

    plt.plot(df["noise_level"], df["logistic_time"], label="Logistic")
    plt.plot(df["noise_level"], df["gmm_time"], label="GMM")
    plt.plot(df["noise_level"], df["topology_time"], label="Topology")

    plt.xlabel("Noise Level")
    plt.ylabel("Time (s)")
    plt.title("Execution Time")

    plt.legend()
    plt.grid(True)

    save_plot(fig, "time.png", dataset)
    plt.show()


# =========================================
# 🔥 6. Topological Distance (IMPORTANT)
# =========================================
def plot_topology_distance(df, dataset):
    fig = plt.figure()

    plt.plot(df["noise_level"], df["bottleneck_distance"], marker="o", label="Bottleneck")
    plt.plot(df["noise_level"], df["wasserstein_distance"], marker="o", label="Wasserstein")

    plt.xlabel("Noise Level")
    plt.ylabel("Distance")
    plt.title("Topological Distance vs Noise")

    plt.legend()
    plt.grid(True)

    save_plot(fig, "topology_distance.png", dataset)
    plt.show()


# =========================================
# Run all plots
# =========================================
if __name__ == "__main__":
    dataset = "mnist"  # 🔥 غيريها إلى "student" "mnist" moons 

    df = load_results(dataset)

    plot_accuracy(df, dataset)
    plot_confidence(df, dataset)
    plot_topology(df, dataset)
    plot_topological_score(df, dataset)
    plot_time(df, dataset)
    plot_topology_distance(df, dataset)