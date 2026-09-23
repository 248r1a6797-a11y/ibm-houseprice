"""
HouseWise AI — SQL Analytics Query Engine Module
================================================
Provides programmatic access to the SQLite relational database at data/db/housewise.db.
Supports running pre-built SQL scripts from src/sql/ and executing read-only ad-hoc queries.
"""

import os
import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "db", "housewise.db")
SQL_DIR = os.path.join(BASE_DIR, "src", "sql")

class SQLEngine:
    """Interface to SQLite database for property analytics queries."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at {self.db_path}. Run scripts/setup_sql_analytics.py first.")

    def get_connection(self) -> sqlite3.Connection:
        """Returns a read-only or standard sqlite3 connection."""
        return sqlite3.connect(self.db_path)

    def list_available_queries(self) -> List[str]:
        """Returns list of available pre-built SQL query scripts."""
        if not os.path.exists(SQL_DIR):
            return []
        return sorted([f for f in os.listdir(SQL_DIR) if f.endswith(".sql") and not f.startswith("01_schema")])

    def get_query_sql(self, script_name: str) -> str:
        """Reads SQL string from a script file."""
        filepath = os.path.join(SQL_DIR, script_name)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Script {script_name} not found in {SQL_DIR}")
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()

    def execute_script_query(self, script_name: str) -> pd.DataFrame:
        """Executes a pre-built SQL script file and returns a pandas DataFrame."""
        sql = self.get_query_sql(script_name)
        with self.get_connection() as conn:
            return pd.read_sql_query(sql, conn)

    def execute_custom_query(self, query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Executes a custom user-supplied SQL query safely.
        Restricted to SELECT queries only to prevent accidental table drop/mutation.
        """
        stripped = query.strip().upper()
        if not (stripped.startswith("SELECT") or stripped.startswith("WITH")):
            return None, "Security Error: Only read-only SELECT and WITH statements are permitted."
        
        try:
            with self.get_connection() as conn:
                df = pd.read_sql_query(query, conn)
                return df, None
        except Exception as e:
            return None, str(e)

    def get_table_schema(self, table_name: str = "properties") -> pd.DataFrame:
        """Inspects table schema and columns."""
        with self.get_connection() as conn:
            return pd.read_sql_query(f"PRAGMA table_info({table_name});", conn)
