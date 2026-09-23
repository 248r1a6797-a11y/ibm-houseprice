"""Property and market segmentation analytics."""
import pandas as pd
import numpy as np

def segment_by_neighborhood(df: pd.DataFrame) -> pd.DataFrame:
    """Group statistics by neighborhood."""
    if "Neighborhood" not in df.columns or "SalePrice" not in df.columns:
        return pd.DataFrame()

    df_calc = df.copy()
    if "PricePerSqFt" not in df_calc.columns and "GrLivArea" in df_calc.columns:
        valid = (df_calc["GrLivArea"] > 0) & (df_calc["SalePrice"] > 0)
        df_calc["PricePerSqFt"] = np.where(valid, df_calc["SalePrice"] / df_calc["GrLivArea"], np.nan)

    agg_dict = {
        "SalePrice": ["count", "median", "mean", "min", "max", "std"],
        "GrLivArea": ["median"]
    }
    if "PricePerSqFt" in df_calc.columns:
        agg_dict["PricePerSqFt"] = ["median"]

    grouped = df_calc.groupby("Neighborhood").agg(agg_dict)
    grouped.columns = ["_".join(c).strip() for c in grouped.columns]
    grouped = grouped.rename(columns={
        "SalePrice_count": "transaction_count",
        "SalePrice_median": "median_price",
        "SalePrice_mean": "mean_price",
        "SalePrice_min": "min_price",
        "SalePrice_max": "max_price",
        "SalePrice_std": "std_price",
        "GrLivArea_median": "median_living_area",
        "PricePerSqFt_median": "median_ppsf"
    }).sort_values(by="median_price", ascending=False).reset_index()

    return grouped

def segment_by_price_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Segment properties into Budget, Mid-Range, Premium, and Luxury tiers based on distribution."""
    if "SalePrice" not in df.columns:
        return pd.DataFrame()

    # Domain percentile thresholds: 25th ($129,500), 75th ($213,500), 95th ($326,000)
    def assign_tier(price):
        if price < 130000:
            return "1. Budget (<$130k)"
        elif price < 215000:
            return "2. Mid-Range ($130k-$215k)"
        elif price < 320000:
            return "3. Premium ($215k-$320k)"
        else:
            return "4. Luxury (>$320k)"

    df_tier = df.copy()
    df_tier["PriceTier"] = df_tier["SalePrice"].apply(assign_tier)

    summary = df_tier.groupby("PriceTier").agg(
        property_count=("SalePrice", "count"),
        median_price=("SalePrice", "median"),
        mean_price=("SalePrice", "mean"),
        median_living_area=("GrLivArea", "median") if "GrLivArea" in df.columns else ("SalePrice", "count"),
        median_year_built=("YearBuilt", "median") if "YearBuilt" in df.columns else ("SalePrice", "count")
    ).reset_index()

    summary["market_share_pct"] = (summary["property_count"] / len(df)) * 100
    return summary

def segment_by_size(df: pd.DataFrame) -> pd.DataFrame:
    """Segment properties by above-grade square footage."""
    if "GrLivArea" not in df.columns or "SalePrice" not in df.columns:
        return pd.DataFrame()

    def assign_size_bin(area):
        if area < 1200:
            return "1. Compact (<1,200 sq ft)"
        elif area < 1800:
            return "2. Medium (1,200-1,800 sq ft)"
        elif area < 2500:
            return "3. Large (1,800-2,500 sq ft)"
        else:
            return "4. Estate (>2,500 sq ft)"

    df_size = df.copy()
    df_size["SizeCategory"] = df_size["GrLivArea"].apply(assign_size_bin)

    summary = df_size.groupby("SizeCategory").agg(
        property_count=("SalePrice", "count"),
        median_price=("SalePrice", "median"),
        mean_price=("SalePrice", "mean"),
        median_ppsf=("PricePerSqFt", "median") if "PricePerSqFt" in df.columns else ("SalePrice", "median")
    ).reset_index()

    summary["pct_of_homes"] = (summary["property_count"] / len(df)) * 100
    return summary
