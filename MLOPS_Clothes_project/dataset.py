from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from MLOPS_Clothes_project.config import ConfigLoader, get_logger

logger = get_logger(__name__)
cfg = ConfigLoader()
Raw_path = cfg.get("paths","raw_data")


def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw Data not found at: {path}")
    logger.info(f"Loadind Data From {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loading DataFrame is Empty")
    logger.info(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    return df

def drop_cols(df):
    try:
        df.drop(columns=['Order_ID','Customer_Name'],inplace=True)
        logger.info("Dropped Columns Order_ID,Customer_Name")
        return df
    except Exception as e:
        logger.exception(f"Error while Dropping Columns:{e}") 

CITY_MAP = {
    "bengaluru":  "Bangalore",
    "bangalore":  "Bangalore",
    "hyd":        "Hyderabad",
    "hyderbad":   "Hyderabad",
    "hyderabad":  "Hyderabad",
    "delhi":      "Delhi",
    "mumbai":     "Mumbai",
    "ahmedabad":  "Ahmedabad",
    "pune":       "Pune",
}

def city(df,CITY_MAP):
    try:
        df["City"] = (df["City"].str.strip().str.lower().map(CITY_MAP).fillna(df["City"].str.strip()))
        logger.info('City Columns Values Cleaned')
        return df
    except Exception as e:
        logger.exception(f'Error While Cleaning City Columns: {e}')
        raise
  
def Order_Date(df):
    try:
        def parse_date(date_str):
            if pd.isna(date_str):
                return pd.NaT
            parts = date_str.replace('/', '-').split('-')
            if len(parts[0]) == 4:
                return pd.to_datetime(date_str.replace('/', '-'), format='%Y-%m-%d')
            else:
                return pd.to_datetime(date_str.replace('/', '-'), format='%d-%m-%Y')
        df['Order_Date'] = df['Order_Date'].apply(parse_date)
        logger.info("Irrelevent Date Format Cleaned")
        return df
    except Exception as e:
        logger.exception(f"Error While Cleaning Order Data Column:{e}")
        raise

def save_data(df):
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True,exist_ok=True)
    output_path = raw_dir / "dataset.csv"
    df.to_csv(output_path,index=False)
    logger.info(f"Saved {len(df):,} rows {output_path}")

def main():
    try:
        df = load_data(Raw_path)
        df = drop_cols(df)
        df = city(df,CITY_MAP)
        df = Order_Date(df)
        save_data(df)
        logger.info("Dataset Pipeline Complete")
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)
    except ValueError as e:
        logger.error(e)
        sys.exit(1)
    except Exception as e:
        logger.exception(e)
        sys.exit(1)

if __name__ == '__main__':
    main()
