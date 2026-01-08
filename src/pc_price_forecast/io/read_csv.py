import pandas as pd
from ..config import PATHS, RAW_CSV_FILENAME

def read_raw_csv() -> pd.DataFrame:
    path = PATHS.DATA_RAW / RAW_CSV_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")
    df = pd.read_csv(path)
    return df
