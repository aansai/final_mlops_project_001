from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import yaml
import numpy as np
import dagshub
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from MLOPS_Clothes_project.config import ConfigLoader, get_logger

logger = get_logger(__name__)
cfg = ConfigLoader()

PROCESSED_DIR  = Path(cfg.get("paths", "processed"))
MODELS_DIR     = Path(cfg.get("paths", "models"))
PROCESSOR_PATH = Path(cfg.get("paths", "processor"))
REPORTS_DIR    = Path(cfg.get("paths", "reports"))
PARAMS_PATH = cfg.get("paths", "params")  


def load_params():
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)["model"]
        logger.info(f"Params Loaded from params.yaml")
        return params
    except Exception as e:
        logger.exception(f"Error While Loading params.yaml: {e}")
        raise


def load_data():
    try:
        X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
        y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
        logger.info(f"Loaded Test Data — {X_test.shape}")
        return X_test, y_test
    except Exception as e:
        logger.exception(f"Error While Loading Test Data: {e}")
        raise


def load_model():
    try:
        model = joblib.load(MODELS_DIR / "model.pkl")
        logger.info("Model Loaded Successfully")
        return model
    except Exception as e:
        logger.exception(f"Error While Loading Model: {e}")
        raise


def evaluate(model, X_test, y_test):
    try:
        predictions = model.predict(X_test)
        r2   = r2_score(y_test, predictions)
        mae  = mean_absolute_error(y_test, predictions)
        mse  = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        metrics = {
            "r2_score": round(r2,   4),
            "mae":      round(mae,  4),
            "mse":      round(mse,  4),
            "rmse":     round(rmse, 4),
        }
        logger.info(f"R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | MSE: {mse:.4f}")
        return metrics, predictions
    except Exception as e:
        logger.exception(f"Error While Evaluating Model: {e}")
        raise


def save_metrics(metrics):
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics Saved to {path}")
    except Exception as e:
        logger.exception(f"Error While Saving Metrics: {e}")
        raise


def main():
    try:
        params         = load_params()        
        X_test, y_test = load_data()
        model          = load_model()
        metrics, _     = evaluate(model, X_test, y_test)

        mlflow.set_tracking_uri("https://dagshub.com/aansai/final_mlops_project_001.mlflow")
        dagshub.init(repo_owner='aansai', repo_name='final_mlops_project_001', mlflow=True)
       # mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("MLOPS_Clothes_Project")

        with mlflow.start_run(run_name="XGBRegressor_Best_Model") as run:

            mlflow.log_param("model_name",   "XGBRegressor")
            mlflow.log_param("test_size",    cfg.get("data", "test_size"))
            mlflow.log_param("random_state", cfg.get("data", "random_state"))

            for key, value in params.items():
                mlflow.log_param(key, value)

            mlflow.log_metric("r2_score", metrics["r2_score"])
            mlflow.log_metric("mae",      metrics["mae"])
            mlflow.log_metric("mse",      metrics["mse"])
            mlflow.log_metric("rmse",     metrics["rmse"])

            mlflow.xgboost.log_model(model, artifact_path="model")
            mlflow.log_artifact(str(PROCESSOR_PATH), artifact_path="processor")

            save_metrics(metrics)
            mlflow.log_artifact(str(REPORTS_DIR / "metrics.json"))

            mlflow.register_model(f"runs:/{run.info.run_id}/model",
            "XGBRegressor_Clothes"
            )

            logger.info(f"MLflow Run Completed — R2: {metrics['r2_score']}")

    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)
    except ValueError as e:
        logger.error(e)
        sys.exit(1)
    except Exception as e:
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()