"""System health check and diagnostic tool for HouseWise AI."""
import sys
from pathlib import Path
import logging

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHECKLIST = [
    ("Config File", Path("config/config.yaml")),
    ("Feature Config", Path("config/feature_config.yaml")),
    ("Raw Dataset", Path("data/raw/AmesHousing.csv")),
    ("Data Documentation", Path("data/raw/DataDocumentation.txt")),
    ("Data Profile", Path("reports/data_profile.json")),
    ("Production Model", Path("models/production/production_model.joblib")),
    ("Model Metadata", Path("models/production/model_metadata.json")),
    ("Model Comparison Table", Path("reports/tables/model_comparison.csv")),
    ("Worst Underpredictions", Path("reports/tables/worst_underpredictions.csv")),
    ("Worst Overpredictions", Path("reports/tables/worst_overpredictions.csv")),
    ("SHAP Feature Importance", Path("reports/tables/shap_feature_importance.csv")),
    ("Streamlit Master App", Path("app/app.py")),
    ("Data Dictionary", Path("docs/data_dictionary.md")),
    ("PRD Document", Path("docs/PRD.md")),
    ("Architecture Document", Path("docs/architecture.md")),
    ("Data Sources Document", Path("docs/data_sources.md")),
    ("Master Documentation", Path("DOCUMENTATION.md")),
    ("SQLite Database", Path("data/db/housewise.db")),
    ("SQL Analytics Summary", Path("reports/sql_analytics/sql_analytics_summary.md"))
]

def run_health_check() -> bool:
    logger.info("============================================================")
    logger.info("HOUSEWISE AI — SYSTEM HEALTH & ARTIFACT AUDIT")
    logger.info("============================================================")

    all_passed = True
    for name, path in CHECKLIST:
        exists = path.exists()
        status = "PASS" if exists else "FAIL"
        if not exists:
            all_passed = False
            logger.error(f"[{status}] {name} missing at: {path}")
        else:
            logger.info(f"[{status}] {name} verified.")

    # Check Model Loading & Inference
    try:
        from src.inference.predictor import load_production_model, predict_single_property
        model, meta = load_production_model()
        logger.info(f"[PASS] Model Engine: {meta.get('model_name')} loads properly.")
        
        sample_prop = {
            "GrLivArea": 1600, "OverallQual": 7, "YearBuilt": 2005,
            "TotalBsmtSF": 800, "FullBath": 2, "Neighborhood": "CollgCr"
        }
        res = predict_single_property(sample_prop)
        logger.info(f"[PASS] Sample Inference test succeeded: {res['formatted_price']} ({res['formatted_interval']})")
    except Exception as e:
        all_passed = False
        logger.error(f"[FAIL] Inference verification failed: {e}")

    logger.info("============================================================")
    if all_passed:
        logger.info("PROJECT HEALTH STATUS: PASS (All critical systems verified)")
        return True
    else:
        logger.error("PROJECT HEALTH STATUS: FAIL (Resolve missing items above)")
        return False

if __name__ == "__main__":
    success = run_health_check()
    sys.exit(0 if success else 1)
