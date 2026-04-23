import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================
# 1. Load Moons (Synthetic Data)
# =========================================
def load_moons_data(n_samples=300, noise=0.0, random_state=42):
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state
    )
    return X, y


# =========================================
# 2. Load MNIST (Kaggle Data)
# =========================================
def load_mnist_data(n_components=2, sample_size=1500, random_state=42):

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "mnist_train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "mnist_test.csv"))

    X_train = train_df.drop("label", axis=1).values
    y_train = train_df["label"].values

    X_test = test_df.drop("label", axis=1).values
    y_test = test_df["label"].values

    X_full = np.vstack((X_train, X_test))
    y_full = np.hstack((y_train, y_test))

    # 🔥 Sampling
    from sklearn.model_selection import train_test_split
    X_full, _, y_full, _ = train_test_split(
        X_full,
        y_full,
        train_size=sample_size,
        stratify=y_full,
        random_state=random_state
    )

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    # PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)

    # Split
    X_train_reduced, X_test_reduced, y_train_small, y_test_small = train_test_split(
        X_reduced,
        y_full,
        test_size=0.3,
        stratify=y_full,
        random_state=random_state
    )

    return (
        X_reduced, y_full,
        X_train_reduced, X_test_reduced,
        y_train_small, y_test_small
    )

def load_student_data():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv("data/iris.csv", sep=";")

    # 🎯 target
    y = df["Target"].values

    # تحويل النص إلى أرقام
    y = (y != "Graduate").astype(int)  # Graduate vs Others

    # 🔥 أهم خطوة: نختار 2 features فقط
    X = df[["Admission grade", "Age at enrollment"]].values

    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y