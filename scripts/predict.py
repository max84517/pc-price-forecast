import pandas as pd
from catboost import CatBoostRegressor, Pool

from pc_price_forecast.config import PATHS, TARGET_COL
from pc_price_forecast.io.save_artifacts import load_parquet, load_json, save_csv
from pc_price_forecast.utils.logger import get_logger

log = get_logger()

def main():
    meta = load_json(PATHS.MODELS / "dataset_meta.json")
    feature_cols = meta["feature_cols"]
    cat_cols = meta["cat_cols"]

    model = CatBoostRegressor()
    model.load_model(str(PATHS.MODELS / "catboost_model.cbm"))

    test_df = load_parquet(PATHS.DATA_PROCESSED / "test.parquet").copy()
    X = test_df[feature_cols]
    preds = model.predict(Pool(X, cat_features=cat_cols))

    out = test_df.copy()
    out["pred_price"] = preds
    out["error"] = out["pred_price"] - out[TARGET_COL]
    out["abs_error"] = out["error"].abs()

    save_csv(out, PATHS.REPORTS / "test_predictions.csv")
    log.info(f"Saved predictions: {PATHS.REPORTS / 'test_predictions.csv'}")
    print("✅ Prediction export done.")

if __name__ == "__main__":
    main()
