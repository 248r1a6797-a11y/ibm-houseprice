"""Descriptive analytics and market KPI computation for HouseWise AI."""
import pandas as pd
import numpy as np
from typing import Dict, Any

def compute_market_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate executive market-level KPIs."""
    target_series = df["SalePrice"].dropna()
    liv_area = df["GrLivArea"].dropna() if "GrLivArea" in df.columns else pd.Series([])

    # Compute derived price per square foot if not already present
    if "PricePerSqFt" in df.columns:
        ppsf_series = df["PricePerSqFt"].dropna()
    elif "GrLivArea" in df.columns:
        valid = (df["GrLivArea"] > 0) & (df["SalePrice"] > 0)
        ppsf_series = (df.loc[valid, "SalePrice"] / df.loc[valid, "GrLivArea"]).dropna()
    else:
        ppsf_series = pd.Series([])

    kpis = {
        "total_properties": int(len(df)),
        "median_sale_price": float(target_series.median()),
        "average_sale_price": float(target_series.mean()),
        "min_sale_price": float(target_series.min()),
        "max_sale_price": float(target_series.max()),
        "std_sale_price": float(target_series.std()),
        "median_living_area": float(liv_area.median()) if not liv_area.empty else 0.0,
        "average_living_area": float(liv_area.mean()) if not liv_area.empty else 0.0,
        "median_price_per_sqft": float(ppsf_series.median()) if not ppsf_series.empty else 0.0,
        "average_price_per_sqft": float(ppsf_series.mean()) if not ppsf_series.empty else 0.0,
        "total_neighborhoods": int(df["Neighborhood"].nunique()) if "Neighborhood" in df.columns else 0
    }
    return kpis

def compute_price_distribution(df: pd.DataFrame, bins: int = 30) -> Dict[str, Any]:
    """Compute distribution statistics for raw and log-transformed price."""
    prices = df["SalePrice"].dropna()
    log_prices = np.log1p(prices)

    counts, bin_edges = np.histogram(prices, bins=bins)
    log_counts, log_bin_edges = np.histogram(log_prices, bins=bins)

    return {
        "raw_counts": counts.tolist(),
        "raw_bin_edges": bin_edges.tolist(),
        "log_counts": log_counts.tolist(),
        "log_bin_edges": log_bin_edges.tolist(),
        "raw_skewness": float(prices.skew()),
        "log_skewness": float(log_prices.skew())
    }
