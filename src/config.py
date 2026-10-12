
from pathlib import Path

# The project root is the parent directory of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"

MODEL_FILE = MODEL_DIR / "threat_detector.joblib"
METRICS_FILE = OUTPUT_DIR / "reports" / "metrics.json"
PREDICTIONS_FILE = OUTPUT_DIR / "reports" / "test_predictions.csv"
ALERTS_FILE = OUTPUT_DIR / "alerts" / "security_alerts.csv"

FIGURE_DIR = OUTPUT_DIR / "figures"

TARGET_COLUMN = "label"
RANDOM_STATE = 42

# These columns are not used as model predictors.
# Attack category is excluded because this is a binary classifier.
DROP_COLUMNS = [
    "id",
    "attack_cat",
    "srcip",
    "dstip",
    "stime",
    "ltime",
]