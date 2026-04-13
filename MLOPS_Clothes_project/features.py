from __future__ import annotations

import sys
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from MLOPS_Clothes_project.config import ConfigLoader, get_logger

logger = get_logger(__name__)
cfg = ConfigLoader()

RAW_PATH       = Path("data/raw/dataset.csv")
Processed_path = cfg.get("paths", "processed")       
processor_path = cfg.get("paths", "processor")        
cat_columns    = cfg.get("features", "cat_columns")   
num_columns    = cfg.get("features", "num_columns")  
BULK_THRESHOLD = cfg.get("features", "bulk_threshold") 


def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed Data not Found at: {path}")
    logger.info(f"Loading Data from: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loading DataFrame is Empty")
    logger.info(f"Loaded {len(df):,} Rows x {df.shape[1]} Columns")
    return df


def is_return(df):
    try:
        df["is_return"] = (df["Units_Sold"] < 0).astype(int)
        logger.info("Is Return Feature Made by Unit_Sold")
        return df
    except Exception as e:
        logger.exception(f"Error While Making a Is_return Feature:{e}")
        raise


def is_bulk_order(df):
    try:
        df["is_bulk_order"] = (df["Units_Sold"] >= BULK_THRESHOLD).astype(int)
        logger.info(f"is_bulk_order feature made by Units_sold Column (threshold={BULK_THRESHOLD})")
        return df
    except Exception as e:
        logger.exception(f"Error While Making a is_bulk_order Feature:{e}")
        raise


def revenue_per_unit(df):
    try:
        df["revenue_per_unit"] = np.where(
            (df["Units_Sold"].notna()) & (df["Units_Sold"] > 0),
            df["Sales_Amount"] / df["Units_Sold"],
            np.nan,
        )
        logger.info("revenue_per_unit feature made by Unit_sold or Sales Amount")
        return df
    except Exception as e:
        logger.exception(f"Error While Making a revenue_per_unit feature:{e}")
        raise


def profit_margin_pct(df):
    try:
        df["profit_margin_pct"] = np.where(
            df["Sales_Amount"] != 0, df["Profit"] / df["Sales_Amount"], np.nan
        )
        logger.info("profit_margin_pct feature made by Sales Amount or Profit")
        return df
    except Exception as e:
        logger.exception(f"Error While Making a profit_margin_pct feature:{e}")
        raise


def effective_discount(df):
    try:
        gross = df["Units_Sold"] * df["Unit_Price"]
        df["effective_discount"] = np.where(
            gross > 0, 1 - (df["Sales_Amount"] / gross), np.nan
        )
        logger.info("effective_discount feature made by Unit_price or Sales_Amount")
        return df
    except Exception as e:
        logger.exception(f"Error While Making a effective_discount feature:{e}")
        raise


def Order_Date_parsed(df):
    try:
        df["Order_Date_parsed"] = pd.to_datetime(
            df["Order_Date"], dayfirst=True, errors="coerce"
        )
        df["order_month"] = df["Order_Date_parsed"].dt.month
        df["order_quarter"] = df["Order_Date_parsed"].dt.quarter
        df["order_dayofweek"] = df["Order_Date_parsed"].dt.dayofweek
        logger.info("Month, Quarter, Days of Week features made by Order_Date")
        return df
    except Exception as e:
        logger.exception(f"Error While making Month, Quarter, Days of Week:{e}")
        raise


def is_zero_sale(df):
    try:
        df["is_zero_sale"] = (df["Sales_Amount"] == 0).astype(int)
        logger.info("is_zero_sale feature created by Sales_Amount")
        return df
    except Exception as e:
        logger.exception(f"Error While Creating is_zero_sale:{e}")
        raise


def discount_applied(df):
    try:
        df["discount_applied"] = (
            df["Discount_%"].notna() & (df["Discount_%"] > 0)
        ).astype(int)
        logger.info("discount_applied Feature Created by Discount_%")
        return df
    except Exception as e:
        logger.exception(f"Error While Creating discount_applied:{e}")
        raise


def price_tier(df):
    try:
        df["price_tier"] = pd.qcut(
            df["Unit_Price"],
            q=4,
            labels=["budget", "mid", "upper-mid", "premium"],
            duplicates="drop",
        )
        logger.info("price_tier Feature Created by Unit_price")
        return df
    except Exception as e:
        logger.exception(f"Error While Creating price_tier:{e}")
        raise


def drop_cols(df):
    try:
        df.drop(
            columns=[
                "Order_Date", "Order_Date_parsed", "profit_margin_pct",
                "effective_discount", "revenue_per_unit",
            ],
            inplace=True,
        )
        logger.info("High Missing Values Features Dropped")
        return df
    except Exception as e:
        logger.exception(f"Error While Dropping Columns:{e}")
        raise


def split_data(df):
    try:
        X = df.drop(columns=["Profit"])
        y = df["Profit"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info(f"X_TRAIN: {len(X_train):,} | X_TEST: {len(X_test):,}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.exception(f"Error Occurred While Splitting Data:{e}")
        raise


def processor_build(cat_cols, num_cols):
    try:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohc", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ])
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        processor = ColumnTransformer(
            [("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)],
            remainder="drop",
        )
        logger.info("Processor Built Successfully")
        return processor
    except Exception as e:
        logger.exception(f"Error Occurred While Building Processor:{e}")
        raise


def save_data(X_train, X_test, y_train, y_test, processor):
    try:
        output_dir = Path(Processed_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        X_train_proc = processor.fit_transform(X_train)
        X_test_proc  = processor.transform(X_test)

        pd.DataFrame(X_train_proc).to_csv(output_dir / "X_train.csv", index=False)
        pd.DataFrame(X_test_proc).to_csv(output_dir  / "X_test.csv",  index=False)
        y_train.to_csv(output_dir / "y_train.csv", index=False)
        y_test.to_csv(output_dir  / "y_test.csv",  index=False)

        joblib.dump(processor, Path(processor_path)) 
        logger.info(f"Saved train/test splits and processor to {output_dir}")
    except Exception as e:
        logger.exception(f"Error While Saving Data:{e}")
        raise


def main():
    try:
        df = load_data(RAW_PATH)
        df = is_return(df)
        df = is_bulk_order(df)
        df = revenue_per_unit(df)
        df = profit_margin_pct(df)
        df = effective_discount(df)
        df = Order_Date_parsed(df)
        df = is_zero_sale(df)
        df = discount_applied(df)
        df = price_tier(df)
        df = drop_cols(df)

        X_train, X_test, y_train, y_test = split_data(df)
        processor = processor_build(cat_columns, num_columns)
        save_data(X_train, X_test, y_train, y_test, processor)

        logger.info(
            f"Features Pipeline Completed — {df.shape[1]} Features, {len(df):,} Rows"
        )
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