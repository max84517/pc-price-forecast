import pandas as pd
from ..config import TARGET_COL, ID_COL
from ..utils.logger import get_logger

log = get_logger()

def validate_raw_df(df: pd.DataFrame) -> None:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in columns: {list(df.columns)}")

    if ID_COL is not None and ID_COL not in df.columns:
        raise ValueError(f"ID column '{ID_COL}' not found in columns: {list(df.columns)}")

    if df[TARGET_COL].isna().mean() > 0:
        na_rate = df[TARGET_COL].isna().mean()
        raise ValueError(f"Target '{TARGET_COL}' has missing values. NA rate={na_rate:.3f}")

    if not pd.api.types.is_numeric_dtype(df[TARGET_COL]):
        raise ValueError(f"Target '{TARGET_COL}' must be numeric dtype, got {df[TARGET_COL].dtype}")

    log.info(f"Rows: {len(df):,}, Cols: {df.shape[1]}")
    log.info(f"Target '{TARGET_COL}' stats: min={df[TARGET_COL].min():.2f}, "
             f"mean={df[TARGET_COL].mean():.2f}, max={df[TARGET_COL].max():.2f}")
