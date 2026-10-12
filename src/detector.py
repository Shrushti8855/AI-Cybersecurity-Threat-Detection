
import joblib
import pandas as pd

from src.config import (
    MODEL_FILE,
    PREDICTIONS_FILE,
    ALERTS_FILE,
)
from src.alerts import save_alerts
from src.data_loader import load_train_test
from src.preprocessing import split_features_target


def main():
    model = joblib.load(MODEL_FILE)

    _, test_data = load_train_test()
    X_test, _ = split_features_target(test_data)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = pd.DataFrame({
        "predicted_label": predictions,
        "attack_probability": probabilities,
        "prediction_name": [
            "Attack" if value == 1 else "Normal"
            for value in predictions
        ],
    })

    results.to_csv(PREDICTIONS_FILE, index=False)

    alerts = save_alerts(results, ALERTS_FILE)

    print("\nDetection completed.")
    print("Total records:", len(results))
    print("Predicted attacks:", int((predictions == 1).sum()))
    print("Predicted normal:", int((predictions == 0).sum()))
    print("Alerts generated:", len(alerts))
    print("Alerts saved to:", ALERTS_FILE)


if __name__ == "__main__":
    main()