"""Train the LightGBM weekly sales model and log to MLflow."""
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import pandas as pd
from pathlib import Path


PROCESSED = Path("data/processed/train_features.parquet")


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("walmart_weekly_sales")

    df = pd.read_parquet(PROCESSED)
    target = "Weekly_Sales"
    features = [c for c in df.columns if c not in {target, "Date"}]

    with mlflow.start_run():
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            random_state=42,
        )
        model.fit(df[features], df[target])
        mlflow.lightgbm.log_model(model, "model")
        mlflow.log_params({"n_estimators": 500, "learning_rate": 0.05})


if __name__ == "__main__":
    main()
