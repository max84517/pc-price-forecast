import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pc_price_forecast.config import PATHS, TARGET_COL, CATBOOST_PARAMS, EARLY_STOPPING_ROUNDS
from pc_price_forecast.io.save_artifacts import load_parquet, load_json, save_json, save_csv
from pc_price_forecast.utils.logger import get_logger

log = get_logger()

def main():
    meta = load_json(PATHS.MODELS / "dataset_meta.json")
    feature_cols = meta["feature_cols"]
    cat_cols = meta["cat_cols"]

    train_df = load_parquet(PATHS.DATA_PROCESSED / "train.parquet")
    valid_df = load_parquet(PATHS.DATA_PROCESSED / "valid.parquet")
    test_df  = load_parquet(PATHS.DATA_PROCESSED / "test.parquet")

    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_valid, y_valid = valid_df[feature_cols], valid_df[TARGET_COL]
    X_test,  y_test  = test_df[feature_cols],  test_df[TARGET_COL]

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_cols)
    test_pool  = Pool(X_test,  y_test,  cat_features=cat_cols)

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    preds = model.predict(test_pool)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae  = float(mean_absolute_error(y_test, preds))
    r2   = float(r2_score(y_test, preds))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    save_json(metrics, PATHS.MODELS / "metrics.json")

    # Feature importance
    importances = model.get_feature_importance(train_pool)
    fi = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
    save_csv(fi, PATHS.REPORTS / "feature_importance.csv")

    # Save model
    model_path = PATHS.MODELS / "catboost_model.cbm"
    model.save_model(str(model_path))

    log.info(f"RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")
    log.info(f"Saved model: {model_path}")
    log.info(f"Saved feature importance: {PATHS.REPORTS / 'feature_importance.csv'}")
    print("✅ Training done.")

if __name__ == "__main__":
    main()
