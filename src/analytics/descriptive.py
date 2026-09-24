"""Descriptive analytics and market KPI computation for HouseWise AI."""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from ..data.schema import infer_target_column

def compute_market_kpis(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    area_col: Optional[str] = None,
    loc_col: Optional[str] = None
) -> Dict[str, Any]:
    """Calculate executive market-level KPIs with dynamic column adaptation."""
    if df.empty:
        return {
            "total_properties": 0, "median_sale_price": 0.0, "average_sale_price": 0.0,
            "min_sale_price": 0.0, "max_sale_price": 0.0, "std_sale_price": 0.0,
            "median_living_area": 0.0, "average_living_area": 0.0,
            "median_price_per_sqft": 0.0, "average_price_per_sqft": 0.0,
            "total_neighborhoods": 0
        }

    # 1. Resolve Target Column
    target = target_col or ("SalePrice" if "SalePrice" in df.columns else infer_target_column(df))
    target_series = df[target].dropna() if target and target in df.columns else pd.Series([], dtype=float)

    # 2. Resolve Area Column
    area = area_col
    if not area:
        for c in ["GrLivArea", "LivingArea", "sqft", "area", "TotalSF"]:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                area = c
                break

    liv_area = df[area].dropna() if area and area in df.columns else pd.Series([], dtype=float)

    # 3. Resolve Location Column
    loc = loc_col
    if not loc:
        for c in ["Neighborhood", "Location", "suburb", "city", "district", "zipcode"]:
            if c in df.columns:
                loc = c
                break

    # 4. Compute Price per Sq Ft
    if "PricePerSqFt" in df.columns:
        ppsf_series = df["PricePerSqFt"].dropna()
    elif area and target and area in df.columns and target in df.columns:
        valid = (df[area] > 0) & (df[target] > 0)
        ppsf_series = (df.loc[valid, target] / df.loc[valid, area]).dropna()
    else:
        ppsf_series = pd.Series([], dtype=float)

    kpis = {
        "total_properties": int(len(df)),
        "median_sale_price": float(target_series.median()) if not target_series.empty else 0.0,
        "average_sale_price": float(target_series.mean()) if not target_series.empty else 0.0,
        "min_sale_price": float(target_series.min()) if not target_series.empty else 0.0,
        "max_sale_price": float(target_series.max()) if not target_series.empty else 0.0,
        "std_sale_price": float(target_series.std()) if len(target_series) > 1 else 0.0,
        "median_living_area": float(liv_area.median()) if not liv_area.empty else 0.0,
        "average_living_area": float(liv_area.mean()) if not liv_area.empty else 0.0,
        "median_price_per_sqft": float(ppsf_series.median()) if not ppsf_series.empty else 0.0,
        "average_price_per_sqft": float(ppsf_series.mean()) if not ppsf_series.empty else 0.0,
        "total_neighborhoods": int(df[loc].nunique()) if loc and loc in df.columns else 0
    }
    return kpis

def compute_price_distribution(
    df: pd.DataFrame, 
    bins: int = 30,
    target_col: Optional[str] = None
) -> Dict[str, Any]:
    """Compute distribution statistics for raw and log-transformed price."""
    target = target_col or ("SalePrice" if "SalePrice" in df.columns else infer_target_column(df))
    if not target or target not in df.columns:
        return {"raw_counts": [], "raw_bin_edges": [], "log_counts": [], "log_bin_edges": [], "raw_skewness": 0.0, "log_skewness": 0.0}

    prices = df[target].dropna()
    prices = prices[prices > 0]
    if prices.empty:
        return {"raw_counts": [], "raw_bin_edges": [], "log_counts": [], "log_bin_edges": [], "raw_skewness": 0.0, "log_skewness": 0.0}

    log_prices = np.log1p(prices)

    counts, bin_edges = np.histogram(prices, bins=bins)
    log_counts, log_bin_edges = np.histogram(log_prices, bins=bins)

    return {
        "raw_counts": counts.tolist(),
        "raw_bin_edges": bin_edges.tolist(),
        "log_counts": log_counts.tolist(),
        "log_bin_edges": log_bin_edges.tolist(),
        "raw_skewness": float(prices.skew()) if len(prices) > 2 else 0.0,
        "log_skewness": float(log_prices.skew()) if len(log_prices) > 2 else 0.0
    }
