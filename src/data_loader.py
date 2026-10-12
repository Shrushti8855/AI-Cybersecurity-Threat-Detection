
import pandas as pd

from src.config import TRAIN_FILE, TEST_FILE, TARGET_COLUMN


def load_dataset(path):
    """Load a CSV and normalize column names."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download the UNSW-NB15 train/test CSV files "
            "and place them in data/raw/."
        )

    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.str.strip()

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Missing target column '{TARGET_COLUMN}'. "
            f"Available columns: {data.columns.tolist()}"
        )

    # The target must contain numeric binary labels.
    data[TARGET_COLUMN] = pd.to_numeric(
        data[TARGET_COLUMN], errors="raise"
    ).astype(int)

    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("The target must contain only 0 and 1.")

    return data


def load_train_test():
    """Load the official training and testing splits."""
    train_data = load_dataset(TRAIN_FILE)
    test_data = load_dataset(TEST_FILE)

    if set(train_data.columns) != set(test_data.columns):
        raise ValueError(
            "Training and testing CSV files have different columns."
        )

    # Ensure both DataFrames use the same column order.
    test_data = test_data[train_data.columns]

    return train_data, test_data