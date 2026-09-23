import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_PATH, DATA_PROFILE_PATH, TARGET_COLUMN
from src.data.loader import load_raw_data
from src.data.cleaner import clean_housing_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def profile():
    logger.info("Executing automated data profiling...")
    df_raw = load_raw_data(RAW_DATA_PATH, normalize=True)
    
    # Target analysis
    target_series = df_raw[TARGET_COLUMN].dropna()
    q25, q75 = float(target_series.quantile(0.25)), float(target_series.quantile(0.75))
    iqr = q75 - q25
    lower_bound = max(0.0, q25 - 1.5 * iqr)
    upper_bound = q75 + 1.5 * iqr

    target_stats = {
        "count": int(target_series.count()),
        "mean": round(float(target_series.mean()), 2),
        "median": round(float(target_series.median()), 2),
        "std": round(float(target_series.std()), 2),
        "min": float(target_series.min()),
        "q25": q25,
        "q75": q75,
        "max": float(target_series.max()),
        "skewness": round(float(target_series.skew()), 4),
        "kurtosis": round(float(target_series.kurtosis()), 4),
        "log_skewness": round(float(np.log1p(target_series).skew()), 4),
        "iqr_outlier_count": int(((target_series < lower_bound) | (target_series > upper_bound)).sum()),
        "iqr_lower_bound": round(lower_bound, 2),
        "iqr_upper_bound": round(upper_bound, 2)
    }

    # Missing values
    missing_counts = df_raw.isnull().sum()
    missing_dict = {
        col: {
            "missing_count": int(count),
            "missing_pct": round(float((count / len(df_raw)) * 100), 2)
        }
        for col, count in missing_counts.items() if count > 0
    }

    # Feature breakdown
    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_raw.select_dtypes(include=["object"]).columns.tolist()
    date_cols = [c for c in df_raw.columns if any(k in c.lower() for k in ["year", "yr", "mo", "date"])]
    id_cols = [c for c in df_raw.columns if c.lower() in ["order", "pid", "id"]]

    profile_data = {
        "dataset_name": "Ames Housing Dataset (Dean De Cock, 2011)",
        "rows": int(len(df_raw)),
        "columns": int(len(df_raw.columns)),
        "duplicate_rows": int(df_raw.duplicated().sum()),
        "target_column": TARGET_COLUMN,
        "target_dtype": str(df_raw[TARGET_COLUMN].dtype),
        "numeric_features_count": len(num_cols),
        "categorical_features_count": len(cat_cols),
        "date_features": date_cols,
        "potential_id_columns": id_cols,
        "total_missing_cells": int(missing_counts.sum()),
        "missing_features_count": len(missing_dict),
        "top_missing_features": sorted(missing_dict.items(), key=lambda x: x[1]["missing_count"], reverse=True)[:10],
        "target_summary": target_stats,
        "de_cock_large_homes_count": int((df_raw["GrLivArea"] > 4000).sum()) if "GrLivArea" in df_raw.columns else 0
    }

    DATA_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)

    logger.info(f"Data profile report saved to: {DATA_PROFILE_PATH}")
    print(json.dumps(profile_data, indent=2))

if __name__ == "__main__":
    profile()
