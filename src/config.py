"""Project-wide constants."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "raw" / "roman_urdu.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Roman Urdu Data Set (Sharf, 2017), UCI Machine Learning Repository, CC BY 4.0
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00458/Roman%20Urdu%20DataSet.csv"

LABELS = {
    "three": ["Negative", "Neutral", "Positive"],
    "binary": ["Negative", "Positive"],
}

DEFAULT_MODEL = "distilbert-base-multilingual-cased"
SEED = 42
MAX_LEN = 64  # reviews are short; 64 sub-word tokens covers the vast majority
