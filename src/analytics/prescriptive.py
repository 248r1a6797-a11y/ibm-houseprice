"""
HouseWise AI — Prescriptive Analytics & Resource-Constrained Prioritization Engine
===================================================================================
Provides institutional-grade property prioritization and portfolio capital allocation:
1. Multi-Criteria Property Prioritization Index (PPI)
2. Resource-Constrained Capital Allocation Optimizer (Knapsack / Greedy under budget)
3. Valuation Divergence Action Tiers with operational playbooks
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "test.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "models", "production", "production_model.joblib")

class PrescriptiveEngine:
    """Multi-criteria property ranking and capital-constrained allocation engine."""

    def __init__(
        self,
        test_path: str = TEST_PATH,
        model_path: str = MODEL_PATH,
        df_test: Optional[pd.DataFrame] = None,
        model: Optional[Any] = None,
        target_col: str = "SalePrice"
    ):
        self.target_col = target_col
        if df_test is not None:
            self.df_test = df_test.copy()
            self.model = model
            self.test_path = test_path
            self.model_path = model_path
        else:
            self.test_path = test_path
            self.model_path = model_path
            self.df_test = pd.read_parquet(self.test_path) if os.path.exists(self.test_path) else pd.DataFrame()
            self.model = joblib.load(self.model_path) if os.path.exists(self.model_path) else None

        self.df_scored = pd.DataFrame()
        if not self.df_test.empty and self.model is not None:
            self._score_baseline()

    def _score_baseline(self):
        """Generates baseline model predictions on the holdout partition."""
        t_col = self.target_col if self.target_col in self.df_test.columns else ("SalePrice" if "SalePrice" in self.df_test.columns else self.df_test.columns[-1])
        X_test = self.df_test.drop(columns=[t_col]) if t_col in self.df_test.columns else self.df_test.copy()
        y_test = pd.to_numeric(self.df_test[t_col], errors="coerce").fillna(0.0).values if t_col in self.df_test.columns else np.zeros(len(self.df_test))
        
        try:
            preds = self.model.predict(X_test)
        except Exception:
            preds = y_test.copy()
        
        self.df_scored = self.df_test.copy()
        self.df_scored["Predicted_Price"] = np.round(preds, 2)
        spread_usd = preds - y_test
        denom = np.where(y_test == 0, 1.0, y_test)
        self.df_scored["Valuation_Spread_USD"] = np.round(spread_usd, 2)
        self.df_scored["Valuation_Spread_Pct"] = np.round((spread_usd / denom) * 100.0, 2)

    def calculate_prioritization_index(
        self,
        weight_upside: float = 0.40,
        weight_remodel: float = 0.25,
        weight_neighborhood: float = 0.20,
        weight_liquidity: float = 0.15
    ) -> pd.DataFrame:
        """
        Computes the composite Property Prioritization Index (PPI) [0 - 100].
        Weights are normalized to sum to 1.0.
        """
        w_total = weight_upside + weight_remodel + weight_neighborhood + weight_liquidity
        if w_total == 0:
            w_total = 1.0
        w1 = weight_upside / w_total
        w2 = weight_remodel / w_total
        w3 = weight_neighborhood / w_total
        w4 = weight_liquidity / w_total

        df = self.df_scored.copy()
        if df.empty:
            return df

        # 1. Valuation Upside Score: percentile rank of Valuation_Spread_Pct
        df["Score_Upside"] = df["Valuation_Spread_Pct"].rank(pct=True) * 100.0

        # 2. Remodel Potential Score: higher for sound structure with older remodel/condition
        if "OverallQual" in df.columns and "YearRemodAdd" in df.columns:
            cond = df["OverallCond"] if "OverallCond" in df.columns else 5
            remodel_signal = (
                (df["OverallQual"] / 10.0) * 0.4 +
                (1.0 - (df["YearRemodAdd"] - 1950).clip(lower=0) / 60.0) * 0.4 +
                ((10.0 - cond) / 10.0) * 0.2
            )
        else:
            remodel_signal = pd.Series(50.0, index=df.index)
        df["Score_Remodel"] = remodel_signal.rank(pct=True) * 100.0

        # 3. Neighborhood Momentum Score
        loc_col = next((c for c in ["Neighborhood", "location", "submarket", "suburb", "city", "district", "zipcode"] if c in df.columns), None)
        size_col = next((c for c in ["GrLivArea", "LivingArea", "sqft", "area", "TotalSF"] if c in df.columns), None)
        t_col = self.target_col if self.target_col in df.columns else ("SalePrice" if "SalePrice" in df.columns else "Predicted_Price")
        
        target_series = pd.to_numeric(df[t_col], errors="coerce").fillna(0.0)
        if size_col and (df[size_col] > 0).any():
            ppsf = target_series / pd.to_numeric(df[size_col], errors="coerce").replace(0, 1.0)
        else:
            ppsf = target_series

        if loc_col and df[loc_col].nunique() > 1:
            neigh_medians = df.groupby(loc_col)[t_col].transform("median")
            df["Score_Neighborhood"] = neigh_medians.rank(pct=True) * 100.0
        else:
            df["Score_Neighborhood"] = ppsf.rank(pct=True) * 100.0

        # 4. Liquidity & Size Efficiency Score
        if size_col:
            size_s = pd.to_numeric(df[size_col], errors="coerce").fillna(0.0)
            size_diff = (size_s - size_s.median()).abs()
            s_max = size_diff.max()
            liquidity_signal = 1.0 - (size_diff / (s_max if s_max > 0 else 1.0))
        else:
            liquidity_signal = pd.Series(50.0, index=df.index)
        df["Score_Liquidity"] = liquidity_signal.rank(pct=True) * 100.0

        # Composite PPI
        df["PPI_Score"] = np.round(
            w1 * df["Score_Upside"] +
            w2 * df["Score_Remodel"] +
            w3 * df["Score_Neighborhood"] +
            w4 * df["Score_Liquidity"],
            2
        )

        # Valuation Divergence Action Tier
        def assign_tier(spread_pct):
            if spread_pct >= 10.0:
                return "Tier 1: Deep Value Opportunity"
            elif spread_pct >= -5.0 and spread_pct < 10.0:
                return "Tier 2: Fair Market Value"
            elif spread_pct >= -15.0 and spread_pct < -5.0:
                return "Tier 3: Overvaluation Risk"
            else:
                return "Tier 4: Speculative / High Volatility"

        df["Action_Tier"] = df["Valuation_Spread_Pct"].apply(assign_tier)
        return df.sort_values(by="PPI_Score", ascending=False)

    def allocate_capital_budget(
        self,
        capital_budget_usd: float = 3000000.0,
        min_tier_filter: Optional[List[str]] = None,
        weight_upside: float = 0.40,
        weight_remodel: float = 0.25,
        weight_neighborhood: float = 0.20,
        weight_liquidity: float = 0.15
    ) -> Dict[str, Any]:
        """
        Executes a resource-constrained capital allocation:
        Greedily selects highest-PPI properties until the capital budget is exhausted.
        """
        ranked_df = self.calculate_prioritization_index(
            weight_upside, weight_remodel, weight_neighborhood, weight_liquidity
        )

        if min_tier_filter:
            ranked_df = ranked_df[ranked_df["Action_Tier"].isin(min_tier_filter)]

        selected = []
        cumulative_spend = 0.0

        for _, row in ranked_df.iterrows():
            cost = row["SalePrice"]
            if cumulative_spend + cost <= capital_budget_usd:
                selected.append(row)
                cumulative_spend += cost

        portfolio_df = pd.DataFrame(selected)

        if len(portfolio_df) == 0:
            return {
                "budget_usd": capital_budget_usd,
                "capital_spent_usd": 0.0,
                "unspent_capital_usd": capital_budget_usd,
                "properties_acquired": 0,
                "portfolio": pd.DataFrame(),
                "metrics": {}
            }

        total_spent = float(portfolio_df["SalePrice"].sum())
        total_market_pred = float(portfolio_df["Predicted_Price"].sum())
        net_spread_usd = total_market_pred - total_spent
        net_spread_pct = (net_spread_usd / total_spent) * 100.0 if total_spent > 0 else 0.0

        metrics = {
            "properties_acquired": len(portfolio_df),
            "capital_spent_usd": total_spent,
            "unspent_capital_usd": capital_budget_usd - total_spent,
            "capital_utilization_pct": np.round((total_spent / capital_budget_usd) * 100.0, 2),
            "estimated_portfolio_value_usd": total_market_pred,
            "net_equity_spread_usd": np.round(net_spread_usd, 2),
            "expected_portfolio_roi_pct": np.round(net_spread_pct, 2),
            "avg_property_price_usd": np.round(portfolio_df["SalePrice"].mean(), 2),
            "avg_ppi_score": np.round(portfolio_df["PPI_Score"].mean(), 2),
            "top_neighborhoods": portfolio_df["Neighborhood"].value_counts().head(3).to_dict()
        }

        return {
            "budget_usd": capital_budget_usd,
            "metrics": metrics,
            "portfolio": portfolio_df
        }
