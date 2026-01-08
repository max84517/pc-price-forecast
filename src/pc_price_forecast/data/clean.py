import pandas as pd
from ..config import TARGET_COL
from ..utils.logger import get_logger

log = get_logger()

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strip column names
    df.columns = [c.strip() for c in df.columns]

    # Strip string cells
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        df[c] = df[c].astype("string").str.strip()

    # Ensure target is numeric
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    after = len(df)
    if after < before:
        log.info(f"Dropped rows with NA target: {before-after:,}")

    return df
