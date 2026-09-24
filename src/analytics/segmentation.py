"""Property and market segmentation analytics."""
import pandas as pd
import numpy as np
from typing import Optional
from ..data.schema import infer_target_column

def segment_by_neighborhood(
    df: pd.DataFrame,
    loc_col: Optional[str] = None,
    target_col: Optional[str] = None,
    area_col: Optional[str] = None
) -> pd.DataFrame:
    """Group statistics by neighborhood or submarket location."""
    if df.empty:
        return pd.DataFrame()

    # Resolve location column
    loc = loc_col
    if not loc:
        for c in ["Neighborhood", "Location", "suburb", "city", "district", "zipcode"]:
            if c in df.columns:
                loc = c
                break

    # Resolve target column
    target = target_col or ("SalePrice" if "SalePrice" in df.columns else infer_target_column(df))
    
    if not loc or not target or loc not in df.columns or target not in df.columns:
        return pd.DataFrame()

    # Resolve area column
    area = area_col
    if not area:
        for c in ["GrLivArea", "LivingArea", "sqft", "area", "TotalSF"]:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                area = c
                break

    df_calc = df.copy()
    if "PricePerSqFt" not in df_calc.columns and area and area in df_calc.columns:
        valid = (df_calc[area] > 0) & (df_calc[target] > 0)
        df_calc["PricePerSqFt"] = np.where(valid, df_calc[target] / df_calc[area], np.nan)

    agg_dict = {
        target: ["count", "median", "mean", "min", "max", "std"]
    }
    if area and area in df_calc.columns:
        agg_dict[area] = ["median"]
    if "PricePerSqFt" in df_calc.columns:
        agg_dict["PricePerSqFt"] = ["median"]

    grouped = df_calc.groupby(loc).agg(agg_dict)
    grouped.columns = ["_".join(c).strip() for c in grouped.columns]

    rename_map = {
        f"{target}_count": "transaction_count",
        f"{target}_median": "median_price",
        f"{target}_mean": "mean_price",
        f"{target}_min": "min_price",
        f"{target}_max": "max_price",
        f"{target}_std": "std_price"
    }
    if area:
        rename_map[f"{area}_median"] = "median_living_area"
    if "PricePerSqFt" in df_calc.columns:
        rename_map["PricePerSqFt_median"] = "median_ppsf"

    grouped = grouped.rename(columns=rename_map).sort_values(by="median_price", ascending=False).reset_index()
    if loc != "Neighborhood":
        grouped["Neighborhood"] = grouped[loc]

    return grouped

def segment_by_price_tier(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    area_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Segment properties into Budget, Mid-Range, Premium, and Luxury tiers.
    Uses dynamic empirical quartiles to support any currency, scale, or region.
    """
    if df.empty:
        return pd.DataFrame()

    target = target_col or ("SalePrice" if "SalePrice" in df.columns else infer_target_column(df))
    if not target or target not in df.columns:
        return pd.DataFrame()

    # Resolve area column
    area = area_col
    if not area:
        for c in ["GrLivArea", "LivingArea", "sqft", "area", "TotalSF"]:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                area = c
                break

    valid_prices = df[target].dropna()
    valid_prices = valid_prices[valid_prices > 0]
    if valid_prices.empty:
        return pd.DataFrame()

    # Dynamic empirical quartile thresholds
    q25 = float(valid_prices.quantile(0.25))
    q50 = float(valid_prices.quantile(0.50))
    q75 = float(valid_prices.quantile(0.75))

    def assign_tier(price):
        if pd.isna(price) or price <= 0:
            return "Unknown"
        if price <= q25:
            return f"1. Budget (≤ ${q25:,.0f})"
        elif price <= q50:
            return f"2. Mid-Range (${q25:,.0f}–${q50:,.0f})"
        elif price <= q75:
            return f"3. Upper-Mid (${q50:,.0f}–${q75:,.0f})"
        else:
            return f"4. Luxury (> ${q75:,.0f})"

    df_tier = df.copy()
    df_tier["PriceTier"] = df_tier[target].apply(assign_tier)

    agg_dict = {
        target: ["count", "median", "mean"]
    }
    if area and area in df_tier.columns:
        agg_dict[area] = ["median"]

    summary = df_tier.groupby("PriceTier").agg(agg_dict)
    summary.columns = ["_".join(c).strip() for c in summary.columns]

    rename_map = {
        f"{target}_count": "property_count",
        f"{target}_median": "median_price",
        f"{target}_mean": "mean_price"
    }
    if area:
        rename_map[f"{area}_median"] = "median_living_area"

    summary = summary.rename(columns=rename_map).reset_index()
    summary = summary.sort_values(by="median_price").reset_index(drop=True)
    return summary

def segment_by_size(
    df: pd.DataFrame,
    area_col: Optional[str] = None,
    target_col: Optional[str] = None
) -> pd.DataFrame:
    """Segment properties by size buckets (Compact, Mid-Size, Spacious, Estate)."""
    if df.empty:
        return pd.DataFrame()

    target = target_col or ("SalePrice" if "SalePrice" in df.columns else infer_target_column(df))
    area = area_col
    if not area:
        for c in ["GrLivArea", "LivingArea", "sqft", "area", "TotalSF"]:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                area = c
                break

    if not area or area not in df.columns:
        return pd.DataFrame()

    valid_area = df[area].dropna()
    valid_area = valid_area[valid_area > 0]
    if valid_area.empty:
        return pd.DataFrame()

    q25 = float(valid_area.quantile(0.25))
    q50 = float(valid_area.quantile(0.50))
    q75 = float(valid_area.quantile(0.75))

    def assign_size_bucket(val):
        if pd.isna(val) or val <= 0:
            return "Unknown"
        if val <= q25:
            return f"1. Compact (≤ {q25:,.0f} sqft)"
        elif val <= q50:
            return f"2. Mid-Size ({q25:,.0f}–{q50:,.0f} sqft)"
        elif val <= q75:
            return f"3. Spacious ({q50:,.0f}–{q75:,.0f} sqft)"
        else:
            return f"4. Estate (> {q75:,.0f} sqft)"

    df_size = df.copy()
    df_size["SizeTier"] = df_size[area].apply(assign_size_bucket)

    agg_dict = {
        area: ["count", "median", "mean"]
    }
    if target and target in df_size.columns:
        agg_dict[target] = ["median", "mean"]

    summary = df_size.groupby("SizeTier").agg(agg_dict)
    summary.columns = ["_".join(c).strip() for c in summary.columns]
    return summary.reset_index()
