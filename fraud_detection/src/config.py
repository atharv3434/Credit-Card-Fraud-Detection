from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TX_CSV, USERS_CSV = str(DATA / "transactions.csv"), str(DATA / "users.csv")
OUT = DATA / "output"
VAL_PREDS, TEST_PREDS, IMPORTANCE = (str(OUT / n) for n in ("val_predictions.csv", "test_predictions.csv", "importance.csv"))

START = datetime(2025, 1, 1)
# Fraud labels arrive ~7 days late (chargebacks). So train/val/test are separated by a 7-day gap:
# at the time we'd train, labels from the most recent week are not yet known.
TRAIN_DAYS = (0, 56)
VAL_DAYS = (63, 75)
TEST_DAYS = (82, 90)
REVIEW_COST = 5.0        # dollars to manually review one flagged transaction
SEED = 42
