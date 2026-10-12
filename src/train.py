
import json
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import (
    MODEL_DIR,
    MODEL_FILE,
    OUTPUT_DIR,
    RANDOM_STATE,
)
from src.data_loader import load_train_test
from src.preprocessing import split_features_target, build_preprocessor


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)

    train_data, test_data = load_train_test()

    X_train, y_train = split_features_target(train_data)
    X_test, y_test = split_features_target(test_data)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train/test feature schemas do not match.")

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Input features:", len(X_train.columns))
    print("\nTraining label distribution:")
    print(y_train.value_counts().sort_index())

    preprocessor = build_preprocessor(X_train)

    classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # One complete pipeline ensures inference uses the same
    # preprocessing that was learned during training.
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)

    metadata = {
        "model": "RandomForestClassifier",
        "target": "label",
        "classes": {
            "0": "Normal",
            "1": "Attack",
        },
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "input_features": list(X_train.columns),
        "random_state": RANDOM_STATE,
    }

    with open(
        OUTPUT_DIR / "reports" / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nModel saved to:", MODEL_FILE)
    print("Metadata saved.")


if __name__ == "__main__":
    main()