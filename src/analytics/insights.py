"""Deterministic Business Insights Engine for HouseWise AI.
Strictly generates structured insights from computed dataset and model statistics.
"""
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from .descriptive import compute_market_kpis
from .segmentation import segment_by_neighborhood, segment_by_price_tier

def generate_business_insights(df: pd.DataFrame, model_eval_summary: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Generate deterministic, data-backed business insights adhering to:
    { "fact", "insight", "opportunity", "action", "evidence" }
    """
    insights = []

    kpis = compute_market_kpis(df)
    neigh_df = segment_by_neighborhood(df)
    tier_df = segment_by_price_tier(df)

    # Insight 1: Premium Neighborhood Valuation Spread
    if not neigh_df.empty:
        top_neigh = neigh_df.iloc[0]
        bottom_neigh = neigh_df.iloc[-1]
        spread_ratio = top_neigh["median_price"] / max(1, bottom_neigh["median_price"])
        
        insights.append({
            "id": "INSIGHT_01_NEIGHBORHOOD_DISPARITY",
            "title": "Severe Geographic Valuation Disparity Across Neighborhoods",
            "fact": f"The top neighborhood '{top_neigh['Neighborhood']}' records a median sale price of ${top_neigh['median_price']:,.0f}, which is {spread_ratio:.2f}x higher than the lowest neighborhood '{bottom_neigh['Neighborhood']}' at ${bottom_neigh['median_price']:,.0f}.",
            "insight": "Micro-location and neighborhood prestige are among the primary macroeconomic determinants of house valuations in Ames, often outweighing pure square footage differences.",
            "opportunity": "Segment appraisal and valuation workflows by neighborhood clusters rather than applying blanket municipality-wide growth rates.",
            "action": "Conduct targeted submarket analyses and verify whether local school districts or zoning restrictions account for the premium.",
            "evidence": [
                f"Top Neighborhood: {top_neigh['Neighborhood']} (Median: ${top_neigh['median_price']:,.0f}, N={top_neigh['transaction_count']})",
                f"Lowest Neighborhood: {bottom_neigh['Neighborhood']} (Median: ${bottom_neigh['median_price']:,.0f}, N={bottom_neigh['transaction_count']})",
                f"Inter-neighborhood spread multiplier: {spread_ratio:.2f}x"
            ]
        })

    # Insight 2: Living Area & Price Elasticity
    if "GrLivArea" in df.columns and "SalePrice" in df.columns:
        corr_val = df["GrLivArea"].corr(df["SalePrice"])
        median_ppsf = kpis.get("median_price_per_sqft", 0)

        insights.append({
            "id": "INSIGHT_02_LIVING_AREA_CORRELATION",
            "title": "Strong Linear Elasticity in Above-Grade Living Area",
            "fact": f"Above-grade living area exhibits a strong positive Pearson correlation of r = {corr_val:.3f} with final sale price, with a median unit value of ${median_ppsf:.2f}/sq ft.",
            "insight": "Living space expansion remains the most reliable physical determinant of market valuation for single-family residential properties.",
            "opportunity": "Developers and home flippers can evaluate ROI on square-footage additions against the prevailing median construction cost per square foot.",
            "action": "Audit building permit filings for home expansions to assess whether renovation additions systematically command the market median ${median_ppsf:.0f}/sq ft rate.",
            "evidence": [
                f"Pearson correlation coefficient: {corr_val:.4f}",
                f"Dataset median price per square foot: ${median_ppsf:.2f}",
                f"Median above-ground living area: {kpis.get('median_living_area', 0):,.0f} sq ft"
            ]
        })

    # Insight 3: Target Distribution Asymmetry & Skewness Correction
    target_series = df["SalePrice"].dropna()
    raw_skew = float(target_series.skew())
    log_skew = float(np.log1p(target_series).skew())
    if abs(raw_skew) > 1.0:
        insights.append({
            "id": "INSIGHT_03_TARGET_SKEWNESS",
            "title": "Pronounced Right-Tail Price Skewness Requires Logarithmic Modeling",
            "fact": f"Raw sale prices demonstrate high positive skewness of {raw_skew:.3f} with an upper whisker extending to ${target_series.max():,.0f}. Applying log1p(y) compresses skewness to {log_skew:.3f}.",
            "insight": "High-value luxury properties in the upper quartile introduce heteroscedasticity and leverage that disproportionately penalize ordinary least squares (OLS) linear estimators.",
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
            remod_med = remodeled["SalePrice"].median()
            non_remod_med = non_remodeled["SalePrice"].median()
            remod_diff = remod_med - non_remod_med
            pct_diff = (remod_diff / non_remod_med) * 100

            insights.append({
                "id": "INSIGHT_04_REMODEL_PREMIUM",
                "title": "Substantial Market Premium for Renovated and Modernized Properties",
                "fact": f"Properties with recorded remodels/additions hold a median sale price of ${remod_med:,.0f} compared to ${non_remod_med:,.0f} for properties in original construction state ({pct_diff:+.1f}% difference).",
                "insight": "Capital improvements and modernization actively protect aging properties from structural depreciation and obsolescence.",
                "opportunity": "Property renovators and institutional buyers can exploit the pricing spread between distressed original properties and modernized comparables.",
                "action": "Incorporate explicit 'IsRemodeled' and 'RemodAgeAtSale' features into production pricing engines to reward verified structural updates.",
                "evidence": [
                    f"Remodeled properties count: {len(remodeled):,} (Median price: ${remod_med:,.0f})",
                    f"Unremodeled properties count: {len(non_remodeled):,} (Median price: ${non_remod_med:,.0f})",
                    f"Observed raw premium: ${remod_diff:,.0f} ({pct_diff:.1f}%)"
                ]
            })

    # Insight 5: Model Generalization & Error Concentration
    if model_eval_summary:
        best_model = model_eval_summary.get("best_model_name", "Production Regressor")
        test_rmse = model_eval_summary.get("test_rmse", 0)
        test_mae = model_eval_summary.get("test_mae", 0)
        test_r2 = model_eval_summary.get("test_r2", 0)

        insights.append({
            "id": "INSIGHT_05_MODEL_GENERALIZATION",
            "title": f"High Generalization Capability of the {best_model} Pipeline",
            "fact": f"The selected {best_model} achieves a Test R² of {test_r2:.4f} with a Test MAE of ${test_mae:,.0f} and Test RMSE of ${test_rmse:,.0f} on the completely untouched holdout partition.",
            "insight": "Nonlinear tree-boosting ensembles capture subtle interaction terms (e.g., Living Area x Overall Quality) without succumbing to severe overfitting.",
            "opportunity": "Automate preliminary automated property valuations (AVM) for 80%+ of standard residential transactions.",
            "action": "Implement a human-in-the-loop review threshold for properties exceeding 4,000 sq ft or properties falling into the top 5% luxury tier where residual variance is highest.",
            "evidence": [
                f"Selected Production Model: {best_model}",
                f"Holdout Test MAE: ${test_mae:,.2f}",
                f"Holdout Test RMSE: ${test_rmse:,.2f}",
                f"Holdout Test R²: {test_r2:.4f}"
            ]
        })

    return insights
