
import joblib
import pandas as pd

from src.config import MODEL_FILE, OUTPUT_DIR
from src.data_loader import load_train_test
from src.preprocessing import split_features_target


def main():
    _, test_data = load_train_test()
    X_test, _ = split_features_target(test_data)

    model = joblib.load(MODEL_FILE)

    # Select a few real feature rows as schema-compatible templates.
    demo = X_test.head(20).copy().reset_index(drop=True)

    # Make synthetic variations of numeric features.
    numeric = demo.select_dtypes(include="number").columns.tolist()

    if not numeric:
        raise RuntimeError("No numeric features available.")

    for column in numeric[:5]:
        demo.loc[0, column] = demo.loc[0, column] * 1.5

    for column in numeric[:5]:
        demo.loc[1, column] = demo.loc[1, column] * 2.0

    # All modified records are labeled synthetic for the demo.
    demo["simulation_type"] = "SYNTHETIC_FEATURE_VARIATION"

    # The model must receive only the same features used for training.
    input_features = demo[X_test.columns]

    demo["predicted_label"] = model.predict(input_features)
    demo["attack_probability"] = model.predict_proba(
        input_features
    )[:, 1]
    demo["prediction_name"] = demo["predicted_label"].map({
        0: "Normal",
        1: "Attack",
    })

    output_path = OUTPUT_DIR / "alerts" / "simulation_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(output_path, index=False)

    print("Synthetic demonstration completed.")
    print("Rows generated:", len(demo))
    print(demo[
        ["simulation_type", "prediction_name", "attack_probability"]
    ].to_string(index=False))
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()