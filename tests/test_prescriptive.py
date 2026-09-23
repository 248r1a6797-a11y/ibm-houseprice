"""Unit tests for Prescriptive Analytics and Grounded Advisor engines."""
import pytest
import pandas as pd
from src.analytics.prescriptive import PrescriptiveEngine
from src.analytics.advisor import GroundedAdvisor

def test_prescriptive_engine_scoring():
    engine = PrescriptiveEngine()
    df_ranked = engine.calculate_prioritization_index()
    assert isinstance(df_ranked, pd.DataFrame)
    assert len(df_ranked) > 0
    assert "PPI_Score" in df_ranked.columns
    assert "Action_Tier" in df_ranked.columns
    assert "Valuation_Spread_Pct" in df_ranked.columns
    
    # Check PPI range
    assert df_ranked["PPI_Score"].min() >= 0.0
    assert df_ranked["PPI_Score"].max() <= 100.0

def test_prescriptive_budget_allocation():
    engine = PrescriptiveEngine()
    budget = 2500000.0
    result = engine.allocate_capital_budget(capital_budget_usd=budget)
    
    assert "metrics" in result
    assert "portfolio" in result
    metrics = result["metrics"]
    portfolio = result["portfolio"]
    
    assert metrics["capital_spent_usd"] <= budget
    assert metrics["properties_acquired"] == len(portfolio)
    assert metrics["capital_spent_usd"] > 0
    assert metrics["capital_utilization_pct"] > 90.0  # High capital deployment efficiency

def test_grounded_advisor_queries():
    advisor = GroundedAdvisor()
    
    # Accuracy query
    ans_acc = advisor.ask("What is the model accuracy and RMSE?")
    assert "FACT:" in ans_acc["fact"]
    assert "0.9342" in ans_acc["fact"]
    assert "INTERPRETATION & ACTION:" in ans_acc["interpretation"]
    
    # Drivers query
    ans_drv = advisor.ask("What are the top features and drivers?")
    assert "FACT:" in ans_drv["fact"]
    assert "SHAP" in ans_drv["fact"]
    
    # Neighborhoods query
    ans_neigh = advisor.ask("Which neighborhood is most expensive per sqft?")
    assert "FACT:" in ans_neigh["fact"]
    assert "sqft" in ans_neigh["fact"]
