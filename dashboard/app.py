
import json

import pandas as pd
import streamlit as st

from src.config import OUTPUT_DIR, PREDICTIONS_FILE, ALERTS_FILE, FIGURE_DIR


st.set_page_config(
    page_title="AI Cybersecurity Threat Detection",
    page_icon="🔒",
    layout="wide",
)

st.title("AI-Powered Cybersecurity Threat Detection")
st.caption(
    "Network intrusion detection prototype | UNSW-NB15 dataset | "
    "Offline evaluation"
)

# --------------------------------------------------
# Load evaluation results and predictions
# --------------------------------------------------

metrics_candidates = [
    OUTPUT_DIR / "metrics.json",
    OUTPUT_DIR / "reports" / "metrics.json",
]

predictions_candidates = [
    OUTPUT_DIR / "predictions.csv",
    PREDICTIONS_FILE,
]

metrics_path = next(
    (path for path in metrics_candidates if path.exists()), None
)
predictions_path = next(
    (path for path in predictions_candidates if path.exists()), None
)

if metrics_path is None or predictions_path is None:
    st.error("Evaluation files are missing.")
    st.code(
        "python -m src.train\n"
        "python -m src.evaluate\n"
        "python -m src.detector"
    )
    st.stop()

with open(metrics_path, "r", encoding="utf-8") as file:
    metrics = json.load(file)

predictions = pd.read_csv(predictions_path)

required = {"actual_label", "predicted_label", "attack_probability"}
if not required.issubset(predictions.columns):
    st.error(
        "The selected prediction file does not contain evaluation labels. "
        "Run `python -m src.evaluate` to regenerate outputs/predictions.csv."
    )
    st.stop()

predictions["actual_label"] = pd.to_numeric(
    predictions["actual_label"], errors="coerce"
)
predictions["predicted_label"] = pd.to_numeric(
    predictions["predicted_label"], errors="coerce"
)
predictions["attack_probability"] = pd.to_numeric(
    predictions["attack_probability"], errors="coerce"
)
predictions = predictions.dropna(
    subset=["actual_label", "predicted_label", "attack_probability"]
)

predictions["actual_label"] = predictions["actual_label"].astype(int)
predictions["predicted_label"] = predictions["predicted_label"].astype(int)

predictions["prediction_name"] = predictions["predicted_label"].map(
    {0: "Normal", 1: "Attack"}
)

predictions["actual_name"] = predictions["actual_label"].map(
    {0: "Normal", 1: "Attack"}
)

alerts = (
    pd.read_csv(ALERTS_FILE)
    if ALERTS_FILE.exists()
    else pd.DataFrame()
)

# --------------------------------------------------
# Security summary
# --------------------------------------------------

st.subheader("Security Summary")

total_records = len(predictions)
attack_count = int((predictions["predicted_label"] == 1).sum())
normal_count = int((predictions["predicted_label"] == 0).sum())

actual = predictions["actual_label"]
predicted = predictions["predicted_label"]

true_positives = int(((actual == 1) & (predicted == 1)).sum())
true_negatives = int(((actual == 0) & (predicted == 0)).sum())
false_positives = int(((actual == 0) & (predicted == 1)).sum())
false_negatives = int(((actual == 1) & (predicted == 0)).sum())

row1 = st.columns(4)
row1[0].metric("Records Evaluated", f"{total_records:,}")
row1[1].metric("Predicted Attacks", f"{attack_count:,}")
row1[2].metric("Predicted Normal", f"{normal_count:,}")
row1[3].metric(
    "Attack Recall",
    f"{metrics.get('recall', 0):.2%}",
)

row2 = st.columns(4)
row2[0].metric("Attacks Detected", f"{true_positives:,}")
row2[1].metric("False Alarms", f"{false_positives:,}")
row2[2].metric("Missed Attacks", f"{false_negatives:,}")
row2[3].metric("Correct Normal", f"{true_negatives:,}")

st.caption(
    "False alarms are normal records classified as attacks. "
    "Missed attacks are attack records classified as normal."
)

st.divider()

# --------------------------------------------------
# Model performance
# --------------------------------------------------

st.subheader("Model Performance")

metric_columns = st.columns(5)

metric_values = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1 Score", "f1_score"),
    ("ROC-AUC", "roc_auc"),
]

for column, (label, key) in zip(metric_columns, metric_values):
    value = metrics.get(key)
    column.metric(
        label,
        f"{value:.2%}" if isinstance(value, (int, float)) else "N/A",
    )

st.divider()

# --------------------------------------------------
# Charts
# --------------------------------------------------

st.subheader("Traffic Detection Overview")

left, right = st.columns(2)

with left:
    st.markdown("**Predicted Traffic**")
    distribution = pd.DataFrame({
        "Class": ["Attack", "Normal"],
        "Records": [attack_count, normal_count],
    })
    st.bar_chart(
        distribution,
        x="Class",
        y="Records",
        horizontal=True,
    )

with right:
    st.markdown("**Confusion Matrix Breakdown**")
    confusion_data = pd.DataFrame({
        "Outcome": [
            "Attacks detected",
            "False alarms",
            "Missed attacks",
            "Normal correctly identified",
        ],
        "Records": [
            true_positives,
            false_positives,
            false_negatives,
            true_negatives,
        ],
    })
    st.bar_chart(
        confusion_data,
        x="Outcome",
        y="Records",
        horizontal=True,
    )

st.divider()

# --------------------------------------------------
# Security alerts
# --------------------------------------------------

st.subheader("Security Alerts")

if alerts.empty:
    st.info(
        "No alert file found. Run `python -m src.detector` "
        "to generate alert records."
    )
else:
    st.write(f"Total generated alerts: **{len(alerts):,}**")

    visible_alerts = alerts.copy()

    if "severity" in visible_alerts.columns:
        severity_options = sorted(
            visible_alerts["severity"].dropna().unique().tolist()
        )
        selected_severities = st.multiselect(
            "Filter by severity",
            options=severity_options,
            default=severity_options,
        )
        visible_alerts = visible_alerts[
            visible_alerts["severity"].isin(selected_severities)
        ]

    if "attack_probability" in visible_alerts.columns:
        minimum_score = st.slider(
            "Minimum attack probability",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            key="alert_threshold",
        )
        visible_alerts = visible_alerts[
            visible_alerts["attack_probability"] >= minimum_score
        ]

    st.write(f"Alerts matching filters: **{len(visible_alerts):,}**")

    st.dataframe(
        visible_alerts.head(1000),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Filtered Alerts (CSV)",
        data=visible_alerts.to_csv(index=False).encode("utf-8"),
        file_name="filtered_security_alerts.csv",
        mime="text/csv",
    )

# --------------------------------------------------
# Predicted attack records
# --------------------------------------------------

st.subheader("Predicted Attack Records")

minimum_attack_score = st.slider(
    "Minimum score for attack records",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.05,
    key="attack_threshold",
)

attack_records = predictions[
    (predictions["predicted_label"] == 1)
    & (predictions["attack_probability"] >= minimum_attack_score)
]

st.write(f"Matching attack records: **{len(attack_records):,}**")

st.dataframe(
    attack_records.head(500),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download Attack Predictions (CSV)",
    data=attack_records.to_csv(index=False).encode("utf-8"),
    file_name="attack_predictions.csv",
    mime="text/csv",
)

# --------------------------------------------------
# Confusion matrix image
# --------------------------------------------------

figure_candidates = [
    (OUTPUT_DIR / "confusion_matrix.png", "Confusion Matrix"),
    (FIGURE_DIR / "confusion_matrix.png", "Confusion Matrix"),
]

for path, title in figure_candidates:
    if path.exists():
        st.divider()
        st.subheader(title)
        st.image(str(path), use_container_width=True)
        break
else:
    st.info("Confusion matrix image not found. Run `python -m src.evaluate`.")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
st.caption(
    "Educational prototype using offline UNSW-NB15 test data. "
    "Model predictions are not confirmed incidents. Human review is "
    "required. This application does not monitor live network traffic."
)