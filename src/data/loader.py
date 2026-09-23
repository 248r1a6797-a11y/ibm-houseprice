"""Robust data loader for HouseWise AI."""
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import logging
from ..config import RAW_DATA_PATH

logger = logging.getLogger(__name__)

def normalize_columns(columns: pd.Index) -> list[str]:
    """Normalize column names by removing spaces, slashes, and special characters."""
    clean_cols = []
    for c in columns:
        clean = (
            str(c)
            .strip()
            .replace(" ", "")
            .replace("/", "")
            .replace("-", "")
            .replace(".", "")
        )
        clean_cols.append(clean)
    return clean_cols

def load_raw_data(
    file_path: Optional[Union[str, Path]] = None,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Load raw housing dataset from CSV or Parquet.
    Handles encodings, delimiter detection, and optional column normalization.
    """
    path = Path(file_path) if file_path else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Housing data file not found at: {path}")

    logger.info(f"Loading raw housing dataset from: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in [".csv", ".txt"]:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("UTF-8 decoding failed, trying latin1 encoding...")
            df = pd.read_csv(path, encoding="latin1")
    elif path.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if df.empty:
        raise ValueError(f"Dataset at {path} is empty!")

    logger.info(f"Loaded raw dataset with shape: {df.shape}")

    if normalize:
        orig_cols = list(df.columns)
        df.columns = normalize_columns(df.columns)
        # Store original mapping as an attribute
        df.attrs["orig_columns"] = orig_cols

    return df
