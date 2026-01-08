import pandas as pd

from pc_price_forecast.io.read_csv import read_raw_csv
from pc_price_forecast.io.save_artifacts import save_parquet, save_json
from pc_price_forecast.data.validate import validate_raw_df
from pc_price_forecast.data.clean import basic_clean
from pc_price_forecast.data.split import split_train_valid_test
from pc_price_forecast.config import PATHS, TARGET_COL, ID_COL, USE_ALL_FEATURES, DROP_COLS
from pc_price_forecast.utils.logger import get_logger

log = get_logger()

def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = df.columns.tolist()

    if USE_ALL_FEATURES:
        feats = [c for c in cols if c != TARGET_COL]
        if ID_COL is not None and ID_COL in feats:
            feats.remove(ID_COL)
        for c in DROP_COLS:
            if c in feats:
                feats.remove(c)
        return feats

    # (If later you want manual selection, implement here)
    raise NotImplementedError("Set USE_ALL_FEATURES=True for baseline.")

def infer_cat_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    X = df[feature_cols]
    cat_cols = []
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]) or str(X[c].dtype) in ("object", "string", "category"):
            cat_cols.append(c)
    return cat_cols

def main():
    df = read_raw_csv()
    validate_raw_df(df)
    df = basic_clean(df)

    feature_cols = infer_feature_columns(df)
    cat_cols = infer_cat_columns(df, feature_cols)

    # Keep only needed columns (features + target + optional id)
    keep_cols = feature_cols + [TARGET_COL]
    if ID_COL is not None:
        keep_cols = [ID_COL] + keep_cols
    df = df[keep_cols].copy()

    train_df, valid_df, test_df = split_train_valid_test(df)

    save_parquet(train_df, PATHS.DATA_PROCESSED / "train.parquet")
    save_parquet(valid_df, PATHS.DATA_PROCESSED / "valid.parquet")
    save_parquet(test_df, PATHS.DATA_PROCESSED / "test.parquet")

    meta = {
        "target_col": TARGET_COL,
        "id_col": ID_COL,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "drop_cols": DROP_COLS,
        "rows": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
    }
    save_json(meta, PATHS.MODELS / "dataset_meta.json")

    log.info(f"Saved processed datasets to: {PATHS.DATA_PROCESSED}")
    log.info(f"Feature cols: {len(feature_cols)} | Cat cols: {len(cat_cols)}")
    log.info(f"Train/Valid/Test = {len(train_df):,} / {len(valid_df):,} / {len(test_df):,}")
    print("✅ Dataset build done.")

if __name__ == "__main__":
    main()
