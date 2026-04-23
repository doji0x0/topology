import numpy as np
from ripser import ripser
from persim import bottleneck, wasserstein


# =========================================
# 1. Compute persistence diagrams
# =========================================
def compute_diagrams(X):
    result = ripser(X, maxdim=1)
    return result["dgms"]


# =========================================
# 2. Extract H1 only (loops)
# =========================================
def get_h1(diagrams):
    if len(diagrams) > 1:
        return diagrams[1]
    return np.empty((0, 2))


# =========================================
# 3. Safe distance computation
# =========================================
def safe_distance(func, dgm1, dgm2):
    # if no loops in both diagrams, distance is zero
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0

    # if one diagram has loops and the other doesn't, distance is infinite or undefined
    if len(dgm1) == 0 or len(dgm2) == 0:
        return np.nan  

    return func(dgm1, dgm2)


# =========================================
# 4. Compute distances
# =========================================
def compute_distances(X_clean, X_noisy):
    # Compute diagrams
    dgms_clean = compute_diagrams(X_clean)
    dgms_noisy = compute_diagrams(X_noisy)

    # Extract loops
    h1_clean = get_h1(dgms_clean)
    h1_noisy = get_h1(dgms_noisy)
 # compute distances
    bottleneck_dist = safe_distance(bottleneck, h1_clean, h1_noisy)
    wasserstein_dist = safe_distance(wasserstein, h1_clean, h1_noisy)

    return {
        "bottleneck_distance": bottleneck_dist,
        "wasserstein_distance": wasserstein_dist
    }