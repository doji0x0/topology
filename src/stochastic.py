import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture


# =========================================
# 1. Logistic Regression
# =========================================
def run_logistic_regression(X_train, X_test, y_train, y_test):
    start_time = time.time()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # Confidence = average max probability
    confidence = np.mean(np.max(y_prob, axis=1))

    exec_time = time.time() - start_time

    return {
        "model_name": "Logistic Regression",
        "accuracy": accuracy,
        "confidence": confidence,
        "execution_time": exec_time
    }


# =========================================
# 2. Gaussian Mixture Model (GMM)
# =========================================
def run_gmm(X_train, X_test, y_train, y_test, n_components=2):
    start_time = time.time()

    model = GaussianMixture(n_components=n_components, random_state=42)
    model.fit(X_train)

    # Predict clusters
    train_clusters = model.predict(X_train)
    test_clusters = model.predict(X_test)

    # Correct mapping (using TRAIN only)
    mapping = {}

    for i in range(n_components):
        mask = (train_clusters == i)
        if np.sum(mask) > 0:
            mapping[i] = np.bincount(y_train[mask]).argmax()

    # Apply mapping to test
    y_pred_mapped = np.array([mapping.get(c, 0) for c in test_clusters])

    accuracy = accuracy_score(y_test, y_pred_mapped)

    # Confidence
    probs = model.predict_proba(X_test)
    confidence = np.mean(np.max(probs, axis=1))

    exec_time = time.time() - start_time

    return {
        "model_name": "GMM",
        "accuracy": accuracy,
        "confidence": confidence,
        "execution_time": exec_time
    }