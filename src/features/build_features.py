import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_features(parquet_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Loads the raw training snapshot, engineers features, and returns
    scaled train/test splits along with the fitted scaler.

    Args:
        parquet_path: Path to the train_snapshot.parquet file.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # ------------------------------------------------------------------ #
    # 1. LOAD DATA
    # ------------------------------------------------------------------ #
    logging.info(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    logging.info(f"Loaded {len(df)} rows. Fraud ratio: {df['class'].mean():.4%}")

    # ------------------------------------------------------------------ #
    # 2. SEPARATE FEATURES AND TARGET
    #    We drop 'transaction_id' and 'timestamp' because they are
    #    identifiers — they carry no signal about fraud patterns.
    #    Keeping them would cause the model to "memorise" specific
    #    transactions rather than learn generalizable fraud patterns.
    # ------------------------------------------------------------------ #
    drop_cols = ['transaction_id', 'timestamp', 'class']
    X = df.drop(columns=drop_cols).values   # Shape: (n_samples, 29)
    y = df['class'].values                  # Shape: (n_samples,)
    feature_names = df.drop(columns=drop_cols).columns.tolist()
    logging.info(f"Feature matrix shape: {X.shape} | Target shape: {y.shape}")

    # ------------------------------------------------------------------ #
    # 3. STRATIFIED TRAIN/TEST SPLIT
    #    stratify=y ensures the 0.15% fraud ratio is preserved in BOTH
    #    the train and test sets. Without this, it is possible (by chance)
    #    that all fraud samples end up in one split, making evaluation
    #    meaningless.
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    logging.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    logging.info(f"Train fraud ratio: {y_train.mean():.4%} | Test fraud ratio: {y_test.mean():.4%}")

    # ------------------------------------------------------------------ #
    # 4. FEATURE SCALING — FIT ONLY ON TRAIN, TRANSFORM BOTH
    #    This is the most important rule in ML preprocessing:
    #    The scaler MUST only see training data when learning the mean
    #    and std. If we fit on all data (including test), test statistics
    #    "leak" into the scaler — the model indirectly "sees" the test
    #    set during training. This is called DATA LEAKAGE and inflates
    #    evaluation metrics, giving you false confidence.
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # Fit + Transform on train
    X_test = scaler.transform(X_test)         # Transform ONLY on test

    logging.info("Scaling complete. Features are now mean=0, std=1.")

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = build_features("data/raw/train_snapshot.parquet")
    print(f"\n--- Feature Engineering Summary ---")
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"y_train fraud count: {y_train.sum()} / {len(y_train)}")
    print(f"y_test  fraud count: {y_test.sum()} / {len(y_test)}")
    print(f"Scaler mean (first 3 features): {scaler.mean_[:3]}")
