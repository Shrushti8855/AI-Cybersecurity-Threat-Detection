# AI-Powered Cybersecurity Threat Detection

A machine-learning-based network intrusion detection prototype that analyzes network traffic records and classifies them as normal or potentially malicious using a Random Forest classifier.

## Features

* Network traffic classification using machine learning
* UNSW-NB15 benchmark dataset
* Model evaluation with precision, recall, F1-score, and ROC-AUC
* Streamlit dashboard for prediction analysis
* Security alert filtering and CSV exports
* Automated tests with pytest

## Tech Stack

* Python
* pandas and NumPy
* scikit-learn
* Matplotlib
* Streamlit
* pytest

## Model Performance

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 90.75% |
| Precision | 87.68% |
| Recall    | 96.79% |
| F1-score  | 92.01% |
| ROC-AUC   | 98.44% |

*Results are from an offline evaluation on the benchmark test dataset. They do not establish production or live-network detection performance.*

## Project Structure

```text
AI-Cybersecurity-Threat-Detection/
├── dashboard/
│   └── app.py
├── data/
│   └── README.md
├── src/
│   ├── alerts.py
│   ├── config.py
│   ├── data_loader.py
│   ├── detector.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   ├── simulate.py
│   └── train.py
├── tests/
│   └── test_alerts.py
├── .gitignore
├── pytest.ini
└── requirements.txt
```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Shrushti8855/AI-Cybersecurity-Threat-Detection.git
   cd AI-Cybersecurity-Threat-Detection
   ```

2. Create and activate a virtual environment.

   Windows PowerShell:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place the required UNSW-NB15 dataset files in `data/raw/`. Follow the instructions in `data/README.md`.

5. Train the model:

   ```bash
   python -m src.train
   ```

6. Evaluate the model:

   ```bash
   python -m src.evaluate
   ```

7. Launch the dashboard:

   ```bash
   python -m streamlit run dashboard/app.py
   ```

8. Run tests:

   ```bash
   python -m pytest -q
   ```

## Responsible Use

This project is an educational prototype for offline network intrusion detection. Its predictions are potential indicators, not confirmed security incidents. It is not intended to replace production security monitoring or professional incident response.

## Author

**Shrushti Shewale**

GitHub: [Shrushti8855](https://github.com/Shrushti8855)
