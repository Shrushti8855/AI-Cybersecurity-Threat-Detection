
import pandas as pd


def generate_alerts(predictions, threshold=0.70):
    """
    Generate review alerts from predicted network attacks.

    The threshold controls alert priority. It is a
    demonstration setting, not a calibrated risk score.
    """
    required = {"predicted_label", "attack_probability"}

    if not required.issubset(predictions.columns):
        raise ValueError(f"Required columns: {sorted(required)}")

    # Generate alerts only for records predicted as attacks.
    alerts = predictions[
        predictions["predicted_label"] == 1
    ].copy()

    def severity(probability):
        return "HIGH" if probability >= threshold else "MEDIUM"

    alerts["severity"] = alerts["attack_probability"].apply(severity)
    alerts["alert_status"] = "REQUIRES_REVIEW"
    alerts["alert_reason"] = (
        "Model classified this network flow as suspicious"
    )

    return alerts


def save_alerts(predictions, output_path):
    alerts = generate_alerts(predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(output_path, index=False)
    return alerts