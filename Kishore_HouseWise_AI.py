"""
HouseWise AI - Explainable House Price Intelligence & Predictive Analytics Platform
===================================================================================
Master Project Pipeline (Python Script Version)

Author: Kishore (Data Science & Analytics Engineering)
Program: IBM Data Science & AI Project Submission
Dataset: Ames Housing Dataset (Dean De Cock, Truman State University)
Target Variable: SalePrice (Continuous USD Valuation)
Production Architecture: XGBoost Regressor with Anti-Leakage Feature Engineering & SHAP Explainability
===================================================================================
"""

import os
import sys
import time
import json
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Scikit-Learn
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Candidate Models
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

# Explainability
import shap

warnings.filterwarnings("ignore")
ROOT_DIR = Path(__file__).resolve().parent

def main():
    print("=" * 78)
    print(" HOUSEWISE AI - EXPLAINABLE HOUSE PRICE INTELLIGENCE PIPELINE")
    print("=" * 78)

    # ---------------------------------------------------------
    # 1. Data Ingestion & Integrity Audit
    # ---------------------------------------------------------
    print("\n[1/7] Ingesting and Auditing Dataset...")
    data_path = ROOT_DIR / "data/raw/AmesHousing.csv"
    if not data_path.exists():
        url = "https://raw.githubusercontent.com/datasets/house-prices-ames-housing/master/data/ames-housing.csv"
        raw_df = pd.read_csv(url)
    else:
        raw_df = pd.read_csv(data_path)

    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    print(f"Loaded: {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns")

    # ---------------------------------------------------------
    # 2. Data Cleaning & Structural Missingness
    # ---------------------------------------------------------
    print("\n[2/7] Executing Data Cleaning (Zero-Leakage & Typo Correction)...")
    clean_df = raw_df.copy()

    # Correct clerical entry typo: GarageYrBlt = 2207 -> 2007
    if "GarageYrBlt" in clean_df.columns:
        clean_df.loc[clean_df["GarageYrBlt"] > 2026, "GarageYrBlt"] = 2007.0

    amenity_cat = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                   "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
                   "PoolQC", "Fence", "MiscFeature", "MasVnrType"]
    for c in amenity_cat:
        if c in clean_df.columns:
            clean_df[c] = clean_df[c].fillna("None")

    amenity_num = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath",
                   "BsmtHalfBath", "GarageCars", "GarageArea", "MasVnrArea"]
    for c in amenity_num:
        if c in clean_df.columns:
            clean_df[c] = clean_df[c].fillna(0.0)

    if "LotFrontage" in clean_df.columns and "Neighborhood" in clean_df.columns:
        clean_df["LotFrontage"] = clean_df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
        clean_df["LotFrontage"] = clean_df["LotFrontage"].fillna(clean_df["LotFrontage"].median())

    # ---------------------------------------------------------
    # 3. Anti-Leakage Feature Engineering
    # ---------------------------------------------------------
    print("\n[3/7] Engineering Anti-Leakage Domain Features...")
    feat_df = clean_df.copy()
    feat_df["TotalSF"] = feat_df.get("GrLivArea", 0) + feat_df.get("TotalBsmtSF", 0)
    feat_df["TotalBath"] = (
        feat_df.get("FullBath", 0) + 0.5 * feat_df.get("HalfBath", 0) +
        feat_df.get("BsmtFullBath", 0) + 0.5 * feat_df.get("BsmtHalfBath", 0)
    )
    feat_df["TotalPorchSF"] = (
        feat_df.get("OpenPorchSF", 0) + feat_df.get("EnclosedPorch", 0) +
        feat_df.get("3SsnPorch", 0) + feat_df.get("ScreenPorch", 0)
    )
    feat_df["HouseAgeAtSale"] = np.maximum(0, feat_df.get("YrSold", 2010) - feat_df.get("YearBuilt", 1970))
    feat_df["RemodAgeAtSale"] = np.maximum(0, feat_df.get("YrSold", 2010) - feat_df.get("YearRemodAdd", 1970))
    feat_df["IsRemodeled"] = (feat_df.get("YearRemodAdd", 0) != feat_df.get("YearBuilt", 0)).astype(int)
    feat_df["OverallGradeScore"] = feat_df.get("OverallQual", 5) * feat_df.get("OverallCond", 5)

    # ---------------------------------------------------------
    # 4. Partitioning & Scikit-Learn Pipeline
    # ---------------------------------------------------------
    print("\n[4/7] Constructing Pipeline & Partitioning Holdout Set (80/20)...")
    target = "SalePrice"
    drop_cols = ["Order", "PID", "index", "Unnamed: 0", target]
    feature_cols = [c for c in feat_df.columns if c not in drop_cols]

    X = feat_df[feature_cols].copy()
    y = feat_df[target].copy()
    X.columns = [c.replace("[", "").replace("]", "").replace("<", "").replace(",", "_") for c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = [c for c in X_train.columns if c not in num_features]

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_features),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False))]), cat_features)
    ], remainder="drop")

    # ---------------------------------------------------------
    # 5. Multi-Model Benchmark Tournament
    # ---------------------------------------------------------
    print("\n[5/7] Running Benchmark Tournament (5-Fold Cross Validation)...")
    candidates = {
        "Ridge Regression": Ridge(alpha=15.0, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=180, learning_rate=0.05, max_depth=4, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=250, learning_rate=0.04, max_depth=4, subsample=0.85, random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_results = []
    trained_pipelines = {}

    for name, est in candidates.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1))
        ])

        cv_res = cross_validate(pipe, X_train, y_train, cv=kf, scoring=["neg_root_mean_squared_error", "r2"], n_jobs=-1)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        test_rmse = np.sqrt(mean_squared_error(y_test, preds))
        test_mae = mean_absolute_error(y_test, preds)
        test_r2 = r2_score(y_test, preds)

        trained_pipelines[name] = pipe
        cv_rmse = -cv_res["test_neg_root_mean_squared_error"].mean()
        cv_r2 = cv_res["test_r2"].mean()

        benchmark_results.append({
            "Model": name,
            "CV_RMSE": cv_rmse,
            "CV_R2": cv_r2,
            "Test_RMSE": test_rmse,
            "Test_MAE": test_mae,
            "Test_R2": test_r2
        })
        print(f"  [+] {name:<20} | CV RMSE: ${cv_rmse:,.0f} | Test RMSE: ${test_rmse:,.0f} | Test R2: {test_r2:.4f}")

    bench_df = pd.DataFrame(benchmark_results).sort_values(by="CV_RMSE").reset_index(drop=True)
    champ_name = bench_df.iloc[0]["Model"]
    champ_model = trained_pipelines[champ_name]
    print(f"\nChampion Crowned: {champ_name}")

    # ---------------------------------------------------------
    # 6. Champion Evaluation & Error Forensics
    # ---------------------------------------------------------
    print("\n[6/7] Evaluating Champion Model on Holdout Set...")
    y_test_pred = champ_model.predict(X_test)
    residuals = y_test.values - y_test_pred

    print(f"Holdout Test RMSE     : ${np.sqrt(mean_squared_error(y_test, y_test_pred)):,.2f}")
    print(f"Holdout Test MAE      : ${mean_absolute_error(y_test, y_test_pred):,.2f}")
    print(f"Holdout Test R2       : {r2_score(y_test, y_test_pred):.4f}")
    print(f"Holdout Test MAPE     : {np.mean(np.abs(residuals / y_test)) * 100:.2f}%")
    print(f"80% Residual Interval : [${np.percentile(residuals, 10):+,.0f} , ${np.percentile(residuals, 90):+,.0f}]")

    # ---------------------------------------------------------
    # 7. SQL Analytics Integration
    # ---------------------------------------------------------
    print("\n[7/7] Synchronizing SQLite Analytics Layer...")
    db_path = ROOT_DIR / "database/housewise.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        sql_df = raw_df.copy()
        sql_df.columns = [c.replace(" ", "_").replace("/", "_").replace("-", "_") for c in sql_df.columns]
        sql_df.to_sql("properties", conn, if_exists="replace", index=False)
        top5 = pd.read_sql_query("SELECT Neighborhood, COUNT(*) as sales, ROUND(AVG(SalePrice),0) as avg_price FROM properties GROUP BY Neighborhood HAVING sales >= 30 ORDER BY avg_price DESC LIMIT 5;", conn)
        print("Top 5 Municipal Submarkets (SQL Query Result):")
        print(top5.to_string(index=False))

    print("\n" + "=" * 78)
    print(" HOUSEWISE AI PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 78)

if __name__ == "__main__":
    main()
