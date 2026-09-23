"""CLI prediction script for HouseWise AI."""
import argparse
import json
import logging
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predictor import predict_single_property, predict_batch_dataframe
from src.data.loader import load_raw_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run():
    parser = argparse.ArgumentParser(description="HouseWise AI Valuation & Prediction Tool")
    parser.add_argument("--sample", action="store_true", help="Run prediction on a realistic sample property")
    parser.add_argument("--csv", type=str, help="Path to input CSV for batch scoring")
    parser.add_argument("--output", type=str, default="batch_predictions.csv", help="Output path for batch predictions")
    args = parser.parse_args()

    if args.sample:
        logger.info("Running single property prediction demo...")
        # Realistic sample home based on Ames median
        sample_property = {
            "GrLivArea": 1650,
            "LotArea": 9500,
            "OverallQual": 7,
            "OverallCond": 5,
            "YearBuilt": 2003,
            "YearRemodAdd": 2004,
            "TotalBsmtSF": 900,
            "1stFlrSF": 900,
            "2ndFlrSF": 750,
            "FullBath": 2,
            "HalfBath": 1,
            "BsmtFullBath": 1,
            "BsmtHalfBath": 0,
            "BedroomAbvGr": 3,
            "KitchenAbvGr": 1,
            "TotRmsAbvGrd": 7,
            "Fireplaces": 1,
            "GarageCars": 2,
            "GarageArea": 550,
            "WoodDeckSF": 150,
            "OpenPorchSF": 60,
            "EnclosedPorch": 0,
            "3SsnPorch": 0,
            "ScreenPorch": 0,
            "PoolArea": 0,
            "Neighborhood": "CollgCr",
            "BldgType": "1Fam",
            "HouseStyle": "2Story",
            "MSZoning": "RL",
            "ExterQual": "Gd",
            "KitchenQual": "Gd",
            "Foundation": "PConc",
            "CentralAir": "Y",
            "YrSold": 2009,
            "MoSold": 6
        }
        res = predict_single_property(sample_property, dataset_medians={"median_living_area": 1442.0})
        print("\n" + "="*50)
        print("  HOUSEWISE AI — VALUATION ESTIMATE")
        print("="*50)
        print(f"  Estimated Valuation:       {res['formatted_price']}")
        print(f"  Calibrated 80% Interval:   {res['formatted_interval']}")
        print(f"  Model Engine:              {res['model_name']} (v{res['model_version']})")
        print(f"  Holdout Test MAE:          ${res['test_mae']:,.0f}")
        print("\n  Top Positive Price Drivers:")
        for i, d in enumerate(res["positive_drivers"], 1):
            print(f"    {i}. {d['feature']} (+{d['contribution']:.3f} log impact)")
        print("\n  Top Negative Price Drivers:")
        for i, d in enumerate(res["negative_drivers"], 1):
            print(f"    {i}. {d['feature']} ({d['contribution']:.3f} log impact)")
        print("\n  Property Profile:")
        for c in res["comparisons"]:
            print(f"    • {c}")
        print("="*50 + "\n")

    elif args.csv:
        logger.info(f"Scoring batch CSV: {args.csv}")
        df = pd.read_csv(args.csv)
        scored_df = predict_batch_dataframe(df)
        out_path = Path(args.output)
        scored_df.to_csv(out_path, index=False)
        logger.info(f"Successfully scored {len(scored_df)} properties! Saved to {out_path}")

    else:
        parser.print_help()

if __name__ == "__main__":
    run()
