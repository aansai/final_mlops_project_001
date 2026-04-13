from __future__ import annotations
import sys
from pathlib import Path
import joblib
import pandas as pd
import yaml
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from MLOPS_Clothes_project.config import ConfigLoader, get_logger

logger = get_logger(__name__)
cfg    = ConfigLoader()

PROCESSED_DIR = Path(cfg.get("paths", "processed"))
MODELS_DIR    = Path(cfg.get("paths", "models"))
PARAMS_PATH   = cfg.get("paths", "params")
THRESHOLD     = float(cfg.get("data", "r2_threshold"))

def load_params():
    try:
        with open(PARAMS_PATH, "r") as f:
            return yaml.safe_load(f)["model"]
    except Exception as e:
        logger.exception(f"Error Loading params.yaml: {e}")
        raise

def build_models(xgb_params):
    models = {
        "Linear Regression":     LinearRegression(),
        "Decision Tree":         DecisionTreeRegressor(),
        "Random Forest":         RandomForestRegressor(),
        "Gradient Boosting":     GradientBoostingRegressor(),
        "XGBRegressor":          XGBRegressor(**xgb_params),
        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
        "AdaBoost Regressor":    AdaBoostRegressor(),
    }
    params = {
        "Linear Regression":     {},
        "Decision Tree":         {"criterion": ["squared_error", "friedman_mse"]},
        "Random Forest":         {"n_estimators": [32, 64, 128], "max_depth": [4, 6, 8]},
        "Gradient Boosting":     {"learning_rate": [0.1, 0.05, 0.01], "n_estimators": [64, 128]},
        "XGBRegressor":          {"learning_rate": [0.1, 0.05, 0.01], "n_estimators": [32, 64, 128]},
        "CatBoosting Regressor": {"depth": [6, 8], "learning_rate": [0.05, 0.1], "iterations": [50, 100]},
        "AdaBoost Regressor":    {"learning_rate": [0.1, 0.05, 0.01], "n_estimators": [64, 128]},
    }
    return models, params

def load_data():
    try:
        X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
        X_test  = pd.read_csv(PROCESSED_DIR / "X_test.csv")
        y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
        y_test  = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
        logger.info(f"Loaded Splits — Train: {X_train.shape} | Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.exception(f"Error While Loading Splits: {e}")
        raise

def evaluate_models(X_train, X_test, y_train, y_test, models, params):
    try:
        report = {}
        for name, model in models.items():
            grid = params.get(name, {})
            if grid:
                gs   = GridSearchCV(model, grid, cv=3, scoring="r2", n_jobs=-1)
                gs.fit(X_train, y_train)
                best = gs.best_estimator_
            else:
                best = model.fit(X_train, y_train)
            score        = r2_score(y_test, best.predict(X_test))
            report[name] = {"model": best, "r2": score}
            logger.info(f"{name} — R2: {score:.4f}")
        return report
    except Exception as e:
        logger.exception(f"Error While Evaluating Models: {e}")
        raise

def save_model(model, name):
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / "model.pkl"
        joblib.dump(model, path)
        logger.info(f"Best Model '{name}' Saved to {path}")
    except Exception as e:
        logger.exception(f"Error While Saving Model: {e}")
        raise

def main():
    try:
        xgb_params                       = load_params()
        models, params                   = build_models(xgb_params)
        X_train, X_test, y_train, y_test = load_data()
        report                           = evaluate_models(X_train, X_test, y_train, y_test, models, params)
        best_name  = max(report, key=lambda k: report[k]["r2"])
        best_score = report[best_name]["r2"]
        best_model = report[best_name]["model"]
        logger.info(f"Best Model: {best_name} | R2: {best_score:.4f}")
        if best_score < THRESHOLD:
            raise ValueError(f"No Good Model Found — Best R2: {best_score:.4f}")
        save_model(best_model, best_name)
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
