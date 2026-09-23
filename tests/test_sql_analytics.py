"""Unit tests for SQLite Analytics Engine."""
import os
import pytest
import pandas as pd
from src.analytics.sql_engine import SQLEngine, DB_PATH, SQL_DIR

def test_database_exists():
    assert os.path.exists(DB_PATH), f"Database file missing at {DB_PATH}"

def test_sql_engine_connection():
    engine = SQLEngine()
    with engine.get_connection() as conn:
        df = pd.read_sql_query("SELECT COUNT(*) AS total FROM properties;", conn)
        assert len(df) == 1
        assert df["total"].iloc[0] == 2930

def test_sql_schema_columns():
    engine = SQLEngine()
    schema_df = engine.get_table_schema("properties")
    assert len(schema_df) >= 80
    assert "SalePrice" in schema_df["name"].values
    assert "Neighborhood" in schema_df["name"].values

def test_all_sql_scripts_execution():
    engine = SQLEngine()
    queries = engine.list_available_queries()
    assert len(queries) >= 11
    
    for q in queries:
        res = engine.execute_script_query(q)
        assert isinstance(res, pd.DataFrame)
        assert len(res) > 0, f"Query {q} returned 0 rows"

def test_custom_query_execution_and_security():
    engine = SQLEngine()
    
    # Valid SELECT
    df, err = engine.execute_custom_query("SELECT Neighborhood, AVG(SalePrice) AS p FROM properties GROUP BY Neighborhood LIMIT 5;")
    assert err is None
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5

    # Blocked dangerous mutation
    df_bad, err_bad = engine.execute_custom_query("DROP TABLE properties;")
    assert df_bad is None
    assert "Security Error" in err_bad
