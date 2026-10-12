
import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

from src.config import MODEL_FILE, OUTPUT_DIR
from src.data_loader import load_train_test
from src.preprocessing import split_features_target


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading trained model...")
    model = joblib.load(MODEL_FILE)

    print("Loading test dataset...")
    _, test_df = load_train_test()
    X_test, y_test = split_features_target(test_df)

    print("Generating predictions...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate model performance metrics.
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    print("\n=== Model Evaluation Results ===")
    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title()}: {value:.4%}")

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            zero_division=0,
        )
    )

    # Save metrics.
    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    # Save predictions and probabilities.
    prediction_df = pd.DataFrame({
        "actual_label": y_test.reset_index(drop=True),
        "predicted_label": predictions,
        "prediction_name": [
            "Attack" if value == 1 else "Normal"
            for value in predictions
        ],
        "attack_probability": probabilities,
    })

    prediction_df.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False,
    )

    # Generate a readable confusion matrix.
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Attack"],
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format=",d",
        colorbar=False,
    )

    ax.set_title("Network Threat Detection - Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\n=== Confusion Matrix ===")
    print("Rows = Actual class; Columns = Predicted class")
    print(cm)

    print("\nEvaluation files saved to:")
    print(OUTPUT_DIR.resolve())
    print("- metrics.json")
    print("- predictions.csv")
    print("- confusion_matrix.png")


if __name__ == "__main__":
    main()