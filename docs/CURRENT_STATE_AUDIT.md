# HouseWise AI — Exhaustive Current State Architecture & Forensic Audit
**Date**: September 24, 2026 | **Author**: Antigravity Data Science & ML Engineering Team
**Status**: Architecture Audit Complete | Implementation Code Modifications Paused Awaiting User Sign-Off

---

## 1. Executive Summary

An exhaustive forensic audit of the entire `ibm` repository was conducted to diagnose why the application operates primarily as a static, Ames Housing demo report rather than a truly schema-adaptive machine learning and analytics workbench.

### Primary Audit Conclusion
While the codebase contains high-quality modular Python components and visually impressive UI layouts, **it was fundamentally hardwired to the Ames Housing dataset schema (2,930 rows, 82 columns)** and depends heavily on pre-computed static files on disk (`model_metadata.json`, `model_comparison.csv`, `shap_feature_importance.csv`, `test.parquet`). 

When a user attempts to upload real-world housing data with arbitrary schemas (such as a 3,986-record, 8-feature dataset with columns like `price, area, bedrooms, bathrooms, location, age, parking`), the application either:
1. Silently ignores the uploaded data and renders pre-cached Ames metrics and charts,
2. Injects medians or zeros for missing Ames columns, rendering user inputs ineffective,
3. Crashes in preprocessing/validation due to hardcoded column references (`SalePrice`, `GrLivArea`, `OverallQual`, `YearBuilt`), or
4. Misleads the user with pre-calculated performance numbers (`Test RMSE: $22,972`, `MAE: $13,835`, `R²: 0.9342`) that bear no mathematical relation to the uploaded data.

---

## 2. Forensic Trace of Hardcoded Values & Demo Artifacts

The user explicitly requested the forensic origin of specific figures, forms, charts, and pipeline indicators. Below is the exact code-level trace:

### 2.1 The "3,986 Records" & "8 Features" Paradox
- **Origin & Phenomenon**: In real-world user scenarios, users frequently upload standard real estate tables (e.g. 3,986 transactions across 8 core attributes: `price`, `area`, `bedrooms`, `bathrooms`, `location`, `age`, `parking`, `lot_size`). 
- **Code Failure Point**: In `app/state.py` lines 27–43, the application silently loads `data/raw/AmesHousing.csv` (2,930 rows × 82 columns) upon session launch. The UI displays Ames row counts (2,930) and 82 features everywhere. When a user uploads their 3,986-record file in `01_data_workspace.py`, the downstream pages (`11_property_predictor.py`, `12_what_if_simulator.py`, `15_neighborhood_intelligence.py`, `16_risk_analysis.py`, `18_executive_overview.py`) still attempt to access Ames column names (`GrLivArea`, `OverallQual`, `TotalBsmtSF`, `Neighborhood`), failing to reflect the user's 8 features.

### 2.2 Model Metrics: RMSE $22,972 | MAE $13,835 | R² 0.9342
These numbers are not computed dynamically on application load; they originate from offline training of an XGBoost regressor on the Ames dataset:
- `models/production/model_metadata.json` (Line 103): `"RMSE": 22972.84`, `"MAE": 13834.55`, `"R2": 0.9342`.
- `reports/tables/model_comparison.csv` (Line 2): `XGBoost,24949.12,6334.44,14416.64,0.889,22972.84,13834.55,0.9342...`
- `reports/tables/evaluation_summary.json` (Line 4): `"Holdout_Test_RMSE": 22972.84`.
- **Hardcoded Fallbacks in UI**:
  - `app/pages/11_property_predictor.py` (Line 101): `metadata.get("metrics", {}).get("holdout_test", {}).get("RMSE", 22972)`
  - `app/pages/18_executive_overview.py` (Lines 64–69): Defaults `r2_val = 0.9342`, `rmse_val = 22972`, `mae_val = 13835`.
  - `app/pages/17_insights_actions.py` (Lines 69–71): Hardcodes fallback `test_rmse: 22972`, `test_mae: 13835`, `test_r2: 0.9342`.
  - `src/analytics/advisor.py` (Line 60): Hardcoded dictionary `test_rmse: 22972.84`.

### 2.3 The Quality Score
- Computed in `app/pages/02_data_profiling.py` (Lines 21–32):
  $$\text{Quality Score} = \text{clip}\left(0.6 \times \text{Completeness} + 0.4 \times \text{Uniqueness} - \text{Validity Penalty}, 0, 100\right)$$
  For the auto-loaded Ames dataset, 13,995 missing cells (predominantly structural nulls in `PoolQC`, `MiscFeature`, `Alley`, `Fence`) produce a completeness score of 94.2%, uniqueness of 100%, yielding a quality score of ~96/100 (or 92.4% in static pre-calculated report cards). If a user uploads arbitrary data, this formula runs, but the UI displays hardcoded Ames narrative text ("Verified assessor records", "De Cock 2011").

### 2.4 Prediction Form (`app/pages/11_property_predictor.py`)
- **Lines 30–51**: Form widgets are hardcoded with explicit Ames variables:
  - Above-Grade Living Area (`GrLivArea`, min 300, max 8000, default 1650)
  - Lot Size (`LotArea`, default 9500)
  - Bedrooms Above Grade (`BedroomAbvGr`, default 3)
  - Overall Quality (`OverallQual`, 1–10 slider, default 7)
  - Overall Condition (`OverallCond`, 1–9 slider, default 5)
  - Year Built (`YearBuilt`, default 2003)
  - Year Remodeled (`YearRemodAdd`, default 2004)
  - Basement Area (`TotalBsmtSF`, default 900)
  - Full Bathrooms (`FullBath`, default 2)
  - Garage Capacity (`GarageCars`, default 2)
  - Garage Area (`GarageArea`, default 500)
  - Neighborhood (`Neighborhood` selectbox with Ames submarkets: `CollgCr`, `NAmes`, etc.)
- **Lines 57–83**: Constructs `prop_record` assuming these exact 12 Ames columns. If a model was trained on arbitrary features (`area`, `bedrooms`, `bathrooms`, `age`), the user inputs are completely ignored, and lines 80–82 substitute training-set medians!

### 2.5 What-If Simulator (`app/pages/12_what_if_simulator.py`)
- **Lines 30–68**: Sliders are hardcoded to `b_liv` (`GrLivArea`), `b_qual` (`OverallQual`), `b_cond` (`OverallCond`), `b_cars` (`GarageCars`), `b_remod` (`YearRemodAdd`).
- Synthetic row builder `make_row()` creates a dictionary with 17 hardcoded Ames columns, failing on arbitrary datasets.

### 2.6 SHAP Values & Attributions
- `app/pages/10_model_explainability.py` (Line 41): Reads `reports/tables/shap_feature_importance.csv`. It never calculates live SHAP explanations on newly trained models.
- `app/pages/18_executive_overview.py` (Line 146): Also reads `shap_feature_importance.csv` directly from disk.

### 2.7 Pipeline Lifecycle Status
- `app/state.py` (Lines 113–124): Upon initial session load, `pipeline_status` is marked as:
  ```python
  "DATA": "✓ Loaded",
  "QUALITY": "✓ Profiled",
  "CLEANING": "✓ Applied",
  "FEATURES": "✓ 81 Selected",
  "EDA": "✓ Ready",
  "TRAINING": "✓ Completed",
  "EVALUATION": "✓ Evaluated",
  "EXPLAINABILITY": "✓ Available",
  "PRODUCTION": "✓ Champion Ready"
  ```
  This creates the false illusion that an entire machine learning lifecycle has successfully run on user data before the user has touched a single control.

### 2.8 The Production Model Artifact
- `models/production/production_model.joblib` (6.2 MB): A frozen Scikit-Learn `TransformedTargetRegressor(Pipeline)` trained offline strictly on 89 one-hot encoded Ames features.
- `src/inference/predictor.py` (Line 64): Populates missing features from `metadata["numerical_features"]` + `metadata["categorical_features"]` (Ames columns).
- `src/data/validator.py` (Lines 60–100): `validate_input_features` explicitly rejects any payload missing `GrLivArea`, `OverallQual`, or `YearBuilt` with `ValueError: Living area (GrLivArea) is required`.

---

## 3. Detailed Audit by Repository Subsystem

### 3.1 Data Ingestion & State Management (`app/state.py`, `src/data/loader.py`)
| Component | Existing Implementation | Architectural Limitation |
| :--- | :--- | :--- |
| `app/state.py` | Auto-loads Ames dataset at session start | Prevents clean onboarding state; assumes Ames is the only universe |
| `src/data/loader.py` | `load_raw_data(path, normalize=True)` | Strips column spaces, but does not infer semantic column types (ID, Target, Numerical, Categorical, Datetime) |
| Target Selection | Defaults to `SalePrice` | Fails if target column is `price`, `target`, `value`, `cost`, or user-specified |

### 3.2 Data Cleaning & Transformation (`src/data/cleaner.py`, `src/data/validator.py`)
| Component | Existing Implementation | Architectural Limitation |
| :--- | :--- | :--- |
| `src/data/cleaner.py` | Hardcodes De Cock outlier removal (`GrLivArea > 4000`), `GarageYrBlt == 2207` fix, and `LotFrontage` imputation | Completely inapplicable to arbitrary real estate data |
| `src/data/validator.py` | Requires `SalePrice`, `GrLivArea`, `LotArea`, `YearBuilt` | Hard crash (`ValidationError`) when validating arbitrary datasets |

### 3.3 Feature Engineering & Preprocessing (`src/features/`)
| Component | Existing Implementation | Architectural Limitation |
| :--- | :--- | :--- |
| `src/features/engineering.py` | Calculates `TotalSF = GrLivArea + TotalBsmtSF`, `TotalBath`, `HouseAgeAtSale` | Returns constant zeroes if columns are named `area`, `bathrooms`, `age` |
| `src/features/preprocessing.py` | `EXCLUDE_COLUMNS = ["Order", "PID", "SalePrice", "PricePerSqFt", "IsExtremeOutlier"]` | Hardcoded column exclusion causes target leakage if target is named `price` |

### 3.4 Model Training & Registry (`src/models/`, `app/pages/07_model_laboratory.py`)
| Component | Existing Implementation | Architectural Limitation |
| :--- | :--- | :--- |
| `07_model_laboratory.py` | Dynamic K-fold tournament exists, but requires manual transition from Page 6 | User must navigate through 6 pages before being able to train; no 1-click fallback |
| `08_hyperparameter_tuning.py` | Pre-defined grids for 6 algorithms | Grid execution runs, but results don't automatically update production predictor unless manually assigned |
| Model Persistence | Writes to `models/production/` | In multi-user or new-dataset sessions, previous dataset's `.joblib` might bleed into active session |

### 3.5 Analytics & Decision Intelligence (`src/analytics/`, `app/pages/14-19`)
| Component | Existing Implementation | Architectural Limitation |
| :--- | :--- | :--- |
| `src/analytics/descriptive.py` | `compute_market_kpis` requires `SalePrice` | Crashes with `KeyError: 'SalePrice'` on arbitrary targets |
| `src/analytics/segmentation.py` | Requires `Neighborhood`, `SalePrice`, fixed $130k/$215k price tiers | Empty DataFrames on custom datasets; fixed dollar tiers fail for non-USD or different price scales |
| `src/analytics/advisor.py` | Returns hardcoded 2,930 properties and $529.73M volume | Violates zero-hallucination principle on custom datasets |
| `app/pages/19_sql_studio.py` | Executes queries on static `database/housewise.db` | Does not sync uploaded dataset into an in-memory SQLite table |

---

## 4. Verification of Audit Claims
- Every finding documented above has been verified directly against source code lines in `app/` and `src/`.
- Unit tests (`pytest`) currently pass (24/24) only because tests specifically test Ames-derived sample data (`tests/test_cleaner.py`, `tests/test_features.py`). Tests for arbitrary schema adaptability are currently missing.

---

## 5. Next Steps
Proceed to `docs/REAL_DATA_FAILURE_ANALYSIS.md` to trace the exact failure mechanics of arbitrary schemas, followed by `docs/REBUILD_PLAN.md`.
