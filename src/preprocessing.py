
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import TARGET_COLUMN, DROP_COLUMNS


def clean_frame(data):
    """Clean invalid numeric values and remove unused columns."""
    frame = data.copy()
    frame.columns = frame.columns.str.strip()

    # Convert infinity values to missing values.
    frame = frame.replace([np.inf, -np.inf], np.nan)

    # Remove the target and excluded identifier/category columns
    # from the predictor DataFrame.
    excluded = [
        col for col in DROP_COLUMNS
        if col in frame.columns
    ]

    return frame.drop(columns=excluded, errors="ignore")


def split_features_target(data):
    """Return predictor features X and target y."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing target: {TARGET_COLUMN}")

    y = data[TARGET_COLUMN].astype(int)
    X = clean_frame(data.drop(columns=[TARGET_COLUMN]))

    # All predictors must have compatible train/test types.
    for col in X.columns:
        if (
            pd.api.types.is_object_dtype(X[col])
            or isinstance(X[col].dtype, pd.CategoricalDtype)
        ):
            X[col] = X[col].astype("string").astype(object)

    return X, y


def build_preprocessor(X):
    """Build preprocessing fitted only during model training."""
    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric_columns = [
        col for col in X.columns
        if col not in categorical_columns
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])