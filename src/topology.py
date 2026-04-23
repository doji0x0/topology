import time
import numpy as np
from ripser import ripser


# =========================================
# 1. Remove infinite values
# =========================================
def _filter_finite(diagram):
    """
    Remove infinite death times from persistence diagram.
    """
    if diagram is None or len(diagram) == 0:
        return np.empty((0, 2))

    diagram = np.array(diagram)

    # Keep only finite death values
    mask = np.isfinite(diagram[:, 1])
    return diagram[mask]


# =========================================
# 2. Compute persistence values
# =========================================
def _persistence_values(diagram):
    """
    Compute persistence (death - birth) for each feature.
    """
    d = _filter_finite(diagram)

    if len(d) == 0:
        return np.array([])

    return d[:, 1] - d[:, 0]


# =========================================
# 3. Compute persistence diagrams
# =========================================
def compute_diagrams(X, maxdim=1):
    """
    Compute persistence diagrams using ripser.
    """
    result = ripser(X, maxdim=maxdim)
    return result["dgms"]


# =========================================
# 4. Extract topological metrics
# =========================================
def extract_metrics(diagrams):
    """
    Extract H1-based topological features.
    """

    # H0 (connected components) – not used but kept for completeness
    h0 = diagrams[0] if len(diagrams) > 0 else np.empty((0, 2))

    # H1 (loops)
    h1 = diagrams[1] if len(diagrams) > 1 else np.empty((0, 2))

    h1_vals = _persistence_values(h1)

    # Metrics
    h1_count = len(h1_vals)

    max_persistence = float(np.max(h1_vals)) if h1_count > 0 else 0.0
    mean_persistence = float(np.mean(h1_vals)) if h1_count > 0 else 0.0
    total_persistence = float(np.sum(h1_vals)) if h1_count > 0 else 0.0

    # Normalized topological score
    topological_score = total_persistence / (h1_count + 1e-8)

    # Stability measure
    persistence_variance = float(np.var(h1_vals)) if h1_count > 0 else 0.0

    return {
        "h1_count": h1_count,
        "max_h1_persistence": max_persistence,
        "mean_h1_persistence": mean_persistence,
        "total_h1_persistence": total_persistence,
        "topological_score": topological_score,
        "persistence_variance": persistence_variance,
    }


# =========================================
# 5. Full Topological Pipeline
# =========================================
def run_topological_pipeline(X, maxdim=1):
    """
    Run full topological analysis pipeline.
    """
    start_time = time.time()

    diagrams = compute_diagrams(X, maxdim=maxdim)
    metrics = extract_metrics(diagrams)

    exec_time = time.time() - start_time

    return {
        "model_name": "Topology",
        "h1_feature_count": metrics["h1_count"],
        "max_h1_persistence": metrics["max_h1_persistence"],
        "mean_h1_persistence": metrics["mean_h1_persistence"],
        "total_h1_persistence": metrics["total_h1_persistence"],
        "topological_score": metrics["topological_score"],
        "persistence_variance": metrics["persistence_variance"],  # 🔥 new
        "execution_time": exec_time,
    }