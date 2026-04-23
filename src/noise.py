import numpy as np


def add_gaussian_noise(X, noise_level=0.1, random_state=42):
    """
    Add Gaussian noise to data.
    """
    rng = np.random.default_rng(random_state)

    noise = rng.normal(
        loc=0.0,
        scale=noise_level,
        size=X.shape
    )

    return X + noise


def generate_noisy_datasets(X, y, noise_levels=None, seeds=None):
    """
    Generate noisy datasets using multiple seeds,
    then average them into ONE dataset per noise level.
    """

    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    datasets = {}

    for level in noise_levels:
        noisy_versions = []

        for seed in seeds:
            X_noisy = add_gaussian_noise(
                X,
                noise_level=level,
                random_state=seed
            )
            noisy_versions.append(X_noisy)

        # 🔥 الحل: نأخذ المتوسط (average)
        X_avg = np.mean(noisy_versions, axis=0)

        datasets[level] = {
            "X": X_avg,
            "y": y
        }

    return datasets