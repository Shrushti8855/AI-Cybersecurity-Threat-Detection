
import pandas as pd

from src.alerts import generate_alerts


def test_only_attack_predictions_generate_alerts():
    predictions = pd.DataFrame({
        "predicted_label": [0, 1, 1],
        "attack_probability": [0.10, 0.60, 0.90],
    })

    alerts = generate_alerts(predictions, threshold=0.70)

    assert len(alerts) == 2
    assert set(alerts["alert_status"]) == {"REQUIRES_REVIEW"}


def test_high_score_gets_high_priority():
    predictions = pd.DataFrame({
        "predicted_label": [1],
        "attack_probability": [0.95],
    })

    alerts = generate_alerts(predictions, threshold=0.70)

    assert alerts.iloc[0]["severity"] == "HIGH"
    