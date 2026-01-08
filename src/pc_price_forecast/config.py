from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    ROOT: Path = Path(__file__).resolve().parents[2]
    DATA_RAW: Path = ROOT / "data" / "raw"
    DATA_INTERIM: Path = ROOT / "data" / "interim"
    DATA_PROCESSED: Path = ROOT / "data" / "processed"
    MODELS: Path = ROOT / "models"
    REPORTS: Path = ROOT / "reports"

PATHS = Paths()

# ---- Raw data ----
RAW_CSV_FILENAME = "computer_prices_all.csv"
TARGET_COL = "price"

# Optional ID column (set None if not exist)
ID_COL = None

# If True: use all columns except target (recommended baseline)
USE_ALL_FEATURES = True

# If you want to explicitly remove high-cardinality columns for generalization tests
DROP_COLS = []  # e.g. ["model", "cpu_model", "gpu_model"]

# Split
TEST_SIZE = 0.2
VALID_SIZE = 0.2  # fraction of remaining train after test split
RANDOM_SEED = 42

# Model
CATBOOST_PARAMS = dict(
    loss_function="RMSE",
    iterations=3000,
    learning_rate=0.05,
    depth=8,
    random_seed=RANDOM_SEED,
    verbose=200,
)
EARLY_STOPPING_ROUNDS = 200
