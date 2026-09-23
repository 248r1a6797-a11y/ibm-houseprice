"""Data splitting module with strict anti-leakage train/test isolation."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
from ..config import (
    RANDOM_STATE, TEST_SIZE, PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, TARGET_COLUMN
)

logger = logging.getLogger(__name__)

def split_and_save_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into 80% Train/CV pool and 20% untouched Holdout Test set.
    Saves clean splits to data/processed/.
    """
    logger.info(f"Splitting {len(df)} records into train and test (test_size={test_size}, random_state={random_state})...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    logger.info(f"Train set: {len(train_df)} rows | Test set: {len(test_df)} rows")

    if save:
        PROCESSED_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(PROCESSED_TRAIN_PATH, index=False)
        test_df.to_parquet(PROCESSED_TEST_PATH, index=False)
        logger.info(f"Saved processed train data to {PROCESSED_TRAIN_PATH}")
        logger.info(f"Saved processed test data to {PROCESSED_TEST_PATH}")

    return train_df, test_df
