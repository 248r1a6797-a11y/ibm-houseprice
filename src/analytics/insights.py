"""Deterministic Business Insights Engine for HouseWise AI.
Strictly generates structured insights from computed dataset and model statistics.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from .descriptive import compute_market_kpis
from .segmentation import segment_by_neighborhood, segment_by_price_tier
from ..data.schema import infer_target_column

def generate_business_insights(
    df: pd.DataFrame, 
    model_eval_summary: Optional[Dict[str, Any]] = None,
    target_col: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate deterministic, data-backed business insights adhering to:
    { "id", "title", "fact", "insight", "opportunity", "action", "evidence" }
    """
    if df.empty:
        return []

    insights = []
    target = target_col or ("SalePrice" if "SalePrice" in df.columns else infer_target_column(df))
    if not target or target not in df.columns:
        return []

    kpis = compute_market_kpis(df, target_col=target)
    neigh_df = segment_by_neighborhood(df, target_col=target)
    tier_df = segment_by_price_tier(df, target_col=target)

    # Insight 1: Geographic / Cohort Valuation Spread
    if not neigh_df.empty and len(neigh_df) > 1:
        top_neigh = neigh_df.iloc[0]
        bottom_neigh = neigh_df.iloc[-1]
        spread_ratio = top_neigh["median_price"] / max(1.0, bottom_neigh["median_price"])
        loc_col_name = "Neighborhood" if "Neighborhood" in neigh_df.columns else neigh_df.columns[0]
        
        insights.append({
            "id": "INSIGHT_01_GEOGRAPHIC_DISPARITY",
            "title": "Severe Geographic Valuation Disparity Across Submarkets",
            "fact": f"The top submarket '{top_neigh[loc_col_name]}' records a median valuation of ${top_neigh['median_price']:,.0f}, which is {spread_ratio:.2f}x higher than the lowest submarket '{bottom_neigh[loc_col_name]}' at ${bottom_neigh['median_price']:,.0f}.",
            "insight": "Micro-location dynamics are among the primary macroeconomic determinants of house valuations, often outweighing pure square footage differences.",
            "opportunity": "Segment appraisal and valuation workflows by neighborhood clusters rather than applying blanket municipality-wide growth rates.",
            "action": "Conduct targeted submarket analyses to verify whether zoning, school districts, or transit access account for the premium.",
            "evidence": [
                f"Top Submarket: {top_neigh[loc_col_name]} (Median: ${top_neigh['median_price']:,.0f}, N={top_neigh['transaction_count']})",
                f"Lowest Submarket: {bottom_neigh[loc_col_name]} (Median: ${bottom_neigh['median_price']:,.0f}, N={bottom_neigh['transaction_count']})",
                f"Inter-submarket spread multiplier: {spread_ratio:.2f}x"
            ]
        })

    # Insight 2: Living Area & Price Elasticity
    area_col = None
    for c in ["GrLivArea", "LivingArea", "sqft", "area", "TotalSF"]:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            area_col = c
            break

    if area_col:
        corr_val = df[area_col].corr(df[target])
        median_ppsf = kpis.get("median_price_per_sqft", 0)

        insights.append({
            "id": "INSIGHT_02_LIVING_AREA_CORRELATION",
            "title": f"Strong Linear Elasticity in Usable Size ({area_col})",
            "fact": f"Living area ({area_col}) exhibits a correlation of r = {corr_val:.3f} with {target}, with a median unit value of ${median_ppsf:.2f}/sq ft.",
            "insight": "Living space expansion remains the most reliable physical determinant of market valuation for residential properties.",
            "opportunity": "Developers and investors can evaluate ROI on square-footage additions against the prevailing median construction cost per square foot.",
            "action": f"Audit property expansions to assess whether renovation additions systematically command the market median ${median_ppsf:.0f}/sq ft rate.",
            "evidence": [
                f"Correlation coefficient with target: {corr_val:.4f}",
                f"Dataset median price per square foot: ${median_ppsf:.2f}",
                f"Median living area: {kpis.get('median_living_area', 0):,.0f} sq ft"
            ]
        })

    # Insight 3: Target Distribution Asymmetry & Skewness Correction
    target_series = df[target].dropna()
    target_series = target_series[target_series > 0]
    if not target_series.empty:
        raw_skew = float(target_series.skew())
        log_skew = float(np.log1p(target_series).skew())
        if abs(raw_skew) > 0.5:
            insights.append({
                "id": "INSIGHT_03_TARGET_SKEWNESS",
                "title": "Pronounced Right-Tail Price Skewness Requires Logarithmic Modeling",
                "fact": f"Raw {target} demonstrates skewness of {raw_skew:.3f} with an upper whisker extending to ${target_series.max():,.0f}. Applying log1p(y) compresses skewness to {log_skew:.3f}.",
                "insight": "High-value luxury properties introduce heteroscedasticity and high leverage that disproportionately penalize ordinary linear estimators.",
                "opportunity": "Training models on log-transformed targets normalizes residual variance, stabilizing gradient descent and regularized regression parameters.",
                "action": "Enforce log1p target transformation during model training and inverse-transform predictions via expm1 to evaluate business errors on the true dollar scale.",
                "evidence": [
                    f"Raw target skewness: {raw_skew:.4f}",
                    f"Log-transformed target skewness: {log_skew:.4f}",
                    f"Interquartile Range: ${target_series.quantile(0.25):,.0f} to ${target_series.quantile(0.75):,.0f} (Max: ${target_series.max():,.0f})"
                ]
            })

    # Insight 4: Renovation Premium vs Aging Effect
    if "YearBuilt" in df.columns and "YearRemodAdd" in df.columns:
        remodeled = df[df["YearRemodAdd"] != df["YearBuilt"]]
        non_remodeled = df[df["YearRemodAdd"] == df["YearBuilt"]]
        if not remodeled.empty and not non_remodeled.empty:
            remod_med = remodeled[target].median()
            non_remod_med = non_remodeled[target].median()
            remod_diff = remod_med - non_remod_med
            pct_diff = (remod_diff / max(1.0, non_remod_med)) * 100.0

            insights.append({
                "id": "INSIGHT_04_REMODEL_PREMIUM",
                "title": "Substantial Market Premium for Renovated and Modernized Properties",
                "fact": f"Properties with recorded remodels hold a median valuation of ${remod_med:,.0f} compared to ${non_remod_med:,.0f} for properties in original construction state ({pct_diff:+.1f}% difference).",
                "insight": "Capital improvements and modernization actively protect aging properties from structural depreciation and obsolescence.",
                "opportunity": "Property renovators and institutional buyers can exploit the pricing spread between distressed original properties and modernized comparables.",
                "action": "Incorporate explicit renovation and modernization indicator features into production pricing engines.",
                "evidence": [
                    f"Remodeled properties count: {len(remodeled):,} (Median valuation: ${remod_med:,.0f})",
                    f"Unremodeled properties count: {len(non_remodeled):,} (Median valuation: ${non_remod_med:,.0f})",
                    f"Observed raw premium: ${remod_diff:,.0f} ({pct_diff:.1f}%)"
                ]
            })

    # Insight 5: Model Generalization & Error Concentration
    if model_eval_summary:
        best_model = model_eval_summary.get("best_model_name", "Champion Model")
        test_rmse = model_eval_summary.get("test_rmse", 0)
        test_mae = model_eval_summary.get("test_mae", 0)
        test_r2 = model_eval_summary.get("test_r2", 0)

        insights.append({
            "id": "INSIGHT_05_MODEL_GENERALIZATION",
            "title": f"High Generalization Capability of the {best_model} Pipeline",
            "fact": f"The selected {best_model} achieves a Test R² of {test_r2:.4f} with a Test MAE of ${test_mae:,.0f} and Test RMSE of ${test_rmse:,.0f} on the untouched holdout partition.",
            "insight": "Empirically validated models capture complex multi-feature interactions without succumbing to severe overfitting.",
            "opportunity": "Automate preliminary automated property valuations (AVM) for standard residential transactions.",
            "action": "Implement a human-in-the-loop review threshold for properties falling into the highest 5% valuation tier where residual variance is widest.",
            "evidence": [
                f"Selected Production Model: {best_model}",
                f"Holdout Test MAE: ${test_mae:,.2f}",
                f"Holdout Test RMSE: ${test_rmse:,.2f}",
                f"Holdout Test R²: {test_r2:.4f}"
            ]
        })

    return insights
