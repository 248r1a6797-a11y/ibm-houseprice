# HouseWise AI — Real-Data Failure Mode & Schema Adaptability Analysis
**Date**: September 24, 2026 | **Author**: Antigravity Data Science & ML Engineering Team
**Status**: Forensic Failure Analysis Complete | Implementation Code Modifications Paused Awaiting User Sign-Off

---

## 1. Executive Summary

This document provides a rigorous, step-by-step failure analysis demonstrating why the current implementation of **HouseWise AI** fails when provided with arbitrary real-world housing datasets.

A truly production-grade real estate analytics and machine learning platform must be **schema-adaptive**. It must gracefully ingest any tabular dataset (e.g., a real-world file with 3,986 records and 8 columns: `price`, `area`, `bedrooms`, `bathrooms`, `location`, `age`, `parking`, `lot_size`), infer column roles, dynamically build preprocessing pipelines, train and benchmark models, and generate predictions, explanations, and business intelligence without hardcoded dependencies on the Ames Housing dataset.

Below is an exhaustive post-mortem of how the existing codebase breaks across all functional modules.

---

## 2. Walk-Through of a User Scenario (e.g. 3,986 Records, 8 Features)

Consider a user uploading a clean residential transactions dataset:
- **Rows**: 3,986
- **Columns**: `price` (Target), `area`, `bedrooms`, `bathrooms`, `location`, `age`, `parking`, `lot_size`

```
   price     area  bedrooms  bathrooms    location  age  parking  lot_size
0  245000   1450.0         3        2.0    Downtown   12        1    4500.0
1  389000   2100.0         4        2.5     Suburbs    5        2    7200.0
...
```

Here is the exact lifecycle failure path when this dataset is fed into the existing codebase:

```
[User Uploads CSV] 
       │
       ▼
[01_data_workspace.py] ──> Overwrites raw_df, sets target_col = 'lot_size' (last col!)
       │
       ▼
[03_data_cleaner.py]   ──> Checks for 'GarageYrBlt', 'LotFrontage', 'GrLivArea' (None found!)
       │                   De Cock extreme outlier filter looks for 'GrLivArea' -> Does nothing.
       │
       ▼
[04_engineering.py]    ──> Calculates TotalSF = GrLivArea + TotalBsmtSF (0 + 0 = 0.0 CONSTANT!)
       │                   Calculates TotalBath = FullBath + 0.5*HalfBath (0.0 CONSTANT!)
       │                   Injects 7 useless constant columns into the dataset.
       │
       ▼
[preprocessing.py]     ──> EXCLUDE_COLUMNS = ['Order', 'PID', 'SalePrice', 'PricePerSqFt']
       │                   CRITICAL FLAW: 'price' is NOT in EXCLUDE_COLUMNS!
       │                   'price' is added to candidate_features!
       │                   TARGET LEAKAGE: Model achieves 100% R² / $0 RMSE by memorizing target!
       │
       ▼
[11_predictor.py]      ──> Calls validate_input_features(payload)
       │                   Throws: ValueError("Living area (GrLivArea) is required.")
       │                   Form renders 12 Ames widgets, ignoring the user's 8 features!
       │
       ▼
[10_explainability.py] ──> Reads reports/tables/shap_feature_importance.csv from disk!
       │                   Displays Ames features (OverallQual, GrLivArea) for the user's data!
       │
       ▼
[18_overview.py]       ──> Displays: "Ames Housing Assessor Dataset (De Cock, 2011)"
                           Displays: $22,972 RMSE, $180,796 Mean Price (Ames numbers!)
```

---

## 3. Systematic Module-by-Module Failure Analysis

### 3.1 Module 1: Ingestion & Session State (`app/state.py`)
- **Failure 1.1: Silent Ames Initialization**: `init_session_state` immediately loads `AmesHousing.csv` at startup. If the user never uploads a file, they see a pre-baked Ames demo. If they do upload a file, downstream states are only partially reset.
- **Failure 1.2: Naive Target Variable Guessing**: The code assigns `target_col = df.columns[-1]`. If the target column is `price` (first column) or `Sale_Price` in the middle, the application picks the wrong target column (e.g., `lot_size` or `parking`), converting regression into predicting parking stalls.
- **Failure 1.3: Premature Pipeline Completion Indicators**: `pipeline_status` is initialized with all checkmarks `✓` at launch, creating false user confidence.

### 3.2 Module 2: Data Cleaning & Imputation (`src/data/cleaner.py`)
- **Failure 2.1: Domain-Specific Clerical Corrections**:
  ```python
  # cleaner.py line 51-58
  if "GarageYrBlt" in df_clean.columns:
      df_clean.loc[typo_mask & (df_clean["GarageYrBlt"] == 2207), "GarageYrBlt"] = 2007.0
  ```
  Hardcoded clerical fixes for a 1927 garage in Ames, Iowa have zero generalizability to external datasets.
- **Failure 2.2: Hardcoded LotFrontage Neighborhood Median Imputation**:
  ```python
  # cleaner.py line 61-64
  df_clean["LotFrontage"] = df_clean.groupby("Neighborhood")["LotFrontage"].transform(...)
  ```
  If either `LotFrontage` or `Neighborhood` is absent, generic numerical missing values elsewhere in the dataset are ignored.
- **Failure 2.3: Rigid Outlier Removal**:
  Assumes `GrLivArea > 4000 & SalePrice < 300000` (the De Cock outliers). On user datasets, this filter does nothing or fails silently.

### 3.3 Module 3: Schema Validation (`src/data/validator.py`)
- **Failure 3.1: Hardcoded Mandatory Columns**:
  ```python
  # validator.py line 26
  target = TARGET_COLUMN # Defined as "SalePrice" in config.py
  if target not in df.columns:
      errors.append(f"Required target column '{target}' is missing.")
  ```
  `validate_housing_schema` hard-fails on any dataset where the target is not named `SalePrice`.
- **Failure 3.2: Rigid Inference Validation**:
  ```python
  # validator.py line 78
  living_area = features.get("GrLivArea")
  if living_area is None:
      errors.append("Living area (GrLivArea) is required.")
  ```
  Any single-property prediction with custom feature names fails before reaching the model.

### 3.4 Module 4: Feature Engineering (`src/features/engineering.py`)
- **Failure 4.1: Constant Column Injection**:
  `engineer_housing_features` creates:
  - `TotalSF` = 0.0 (constant variance = 0)
  - `TotalBath` = 0.0 (constant variance = 0)
  - `TotalPorchSF` = 0.0 (constant variance = 0)
  - `HasPool` = 0, `HasGarage` = 0, `HasFireplace` = 0
  Injecting zero-variance constants degrades tree splits and produces singular matrices in linear models (`LinAlgError`).
- **Failure 4.2: Temporal Feature Assumptions**:
  Assumes `YrSold` and `YearBuilt` exist. If a dataset provides property age directly as `age = 12`, the code ignores `age` and fails to derive age features.

### 3.5 Module 5: Preprocessing & Severe Target Leakage (`src/features/preprocessing.py`)
- **Failure 5.1: Catastrophic Target Leakage**:
  ```python
  # preprocessing.py line 13
  EXCLUDE_COLUMNS = ["Order", "PID", "SalePrice", "PricePerSqFt", "IsExtremeOutlier"]
  ```
  If the uploaded dataset has a target named `price`, `target`, `value`, or `MEDV`, it is **not excluded from candidate features**.
  The preprocessing pipeline includes the target in the feature matrix $X$.
  Any ML model trained on this pipeline trivially achieves $R^2 = 1.000$ and $RMSE \approx 0$ by learning the identity function $f(y) = y$, completely corrupting the model benchmark tournament!
- **Failure 5.2: Unhandled High-Cardinality Categoricals**:
  Uses `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. If an uploaded dataset has a column like `address` or `street_name` with 3,000 unique text values, One-Hot Encoding explodes memory ($3,000 \times 3,986 \times 8 \text{ bytes} \approx 95 \text{ MB}$) and creates severe curse of dimensionality.

### 3.6 Module 6: Model Training & Laboratory (`app/pages/07_model_laboratory.py`)
- **Failure 6.1: Disconnected Experiment Setup**:
  If the user skips Page 6 (Experiment Setup) and goes straight to Page 7, `X_train` is `None`, and training cannot be run.
- **Failure 6.2: Unhandled Column Name Formatting**:
  XGBoost and LightGBM reject feature names containing `[`, `]`, `<`, or `,`. If raw user columns contain special characters, training crashes with:
  `ValueError: feature_names may not contain [, ] or <`.

### 3.7 Module 7: Inference & Single-Property Predictor (`app/pages/11_property_predictor.py`)
- **Failure 7.1: Static Input UI Form**:
  The form is physically coded with 12 Ames widgets (`GrLivArea`, `OverallQual`, `YearBuilt`, `TotalBsmtSF`, etc.). 
  If the active model was trained on the user's 8 features (`price, area, bedrooms, bathrooms, location, age, parking, lot_size`), the form does not display inputs for `bedrooms`, `location`, `age`, or `parking`.
- **Failure 7.2: Feature Replacement by Imputed Medians**:
  Lines 80–82 loop through `active_model.feature_names_in_` and fill missing values with training medians. Since the form widgets didn't match the model features, the model evaluates training-set medians, and the predicted price never changes regardless of user input!

### 3.8 Module 8: What-If Simulator (`app/pages/12_what_if_simulator.py`)
- **Failure 8.1: Hardcoded Base vs Scenario Controls**:
  Sliders are hardwired to `GrLivArea`, `OverallQual`, `OverallCond`, `GarageCars`, and `YearRemodAdd`.
  On an arbitrary dataset, tweaking these sliders produces 0 marginal impact because none of these columns exist in the user's model.

### 3.9 Module 9: Explainability & Diagnostics (`app/pages/10_model_explainability.py`)
- **Failure 9.1: Disconnected Static SHAP Artifact**:
  Tab 1 loads `reports/tables/shap_feature_importance.csv`. It displays Ames feature attributions (`OverallQual`, `GrLivArea`) even when the user has trained a custom model on their own data.
- **Failure 9.2: Crash on Local Waterfall**:
  In Tab 2, `explain_single_prediction` expects the production pipeline preprocessor structure. If the active pipeline structure differs or feature counts do not match `model_metadata.json`, it throws an exception.

### 3.10 Module 10: Market Analytics & Neighborhood Intelligence (`app/pages/14`, `15`)
- **Failure 10.1: Hardcoded Column Checking in Spatial Analysis**:
  ```python
  # 15_neighborhood_intelligence.py line 23
  loc_col = "Neighborhood" if "Neighborhood" in df.columns else None
  area_col = "GrLivArea" if "GrLivArea" in df.columns else None
  ```
  If columns are named `location`, `district`, `city`, `suburb` or `area`, `sqft`, the page displays:
  `"Neighborhood intelligence requires 'Neighborhood' and 'GrLivArea' columns."` and renders nothing.
- **Failure 10.2: Static Descriptive Analytics**:
  `src/analytics/descriptive.py` directly references `df["SalePrice"]`. If target is `price`, it throws `KeyError: 'SalePrice'`.

### 3.11 Module 11: Risk & Error Slicing (`app/pages/16_risk_analysis.py`)
- **Failure 11.1: Static Test Parquet Fallback**:
  If the user hasn't completed in-session training, the page reads `data/processed/test.parquet` (the Ames test split), presenting error metrics from Ames rather than user data.
- **Failure 11.2: Hardcoded Dollar Price Tiers**:
  ```python
  bins = [0, 130000, 215000, 350000, 10000000]
  labels = ["Budget (<$130k)", "Mid-Range ($130k-$215k)", "Upper-Mid ($215k-$350k)", "Luxury (>$350k)"]
  ```
  If the user's prices are in millions (e.g. London / California) or thousands, or in another currency (e.g., INR, EUR), these bins fail, grouping 100% of rows into a single tier or producing empty buckets.

### 3.12 Module 12: Business Intelligence & Grounded Advisor (`app/pages/17_insights_actions.py`)
- **Failure 12.1: Hardcoded Statistical Hallucination**:
  `GroundedAdvisor` in `src/analytics/advisor.py` returns hardcoded numbers:
  ```python
  "total_properties": 2930,
  "market_volume_millions": 529.73,
  "mean_sale_price": 180796.06,
  "champion_model": "XGBoost Regressor",
  "test_rmse": 22972.84
  ```
  This is a direct violation of the Zero-Hallucination mandate when operating on user-provided data.

### 3.13 Module 13: SQL Studio (`app/pages/19_sql_studio.py`)
- **Failure 13.1: Static Database Disconnect**:
  Queries execute against `database/housewise.db`. When a user loads new data in the session, `database/housewise.db` is not updated. Queries looking for `OverallQual` or `Neighborhood` crash on the new table.

---

## 4. Architectural Root Cause Summary

| Root Cause ID | Underlying Flaw | Architectural Solution |
| :--- | :--- | :--- |
| **RC-1** | **Implicit Ames Coupling** across data, cleaner, validator, features, and predictors | Decouple pipeline from column names; introduce dynamic `DatasetContext` |
| **RC-2** | **Static File Fallbacks** masquerading as live session results | Remove all static file fallbacks in user workflows; enforce strict `EMPTY` onboarding state |
| **RC-3** | **Hardcoded Preprocessing Exclusion List** creating catastrophic target leakage | Dynamic feature partitioning: explicitly exclude the user-selected target and identified ID columns |
| **RC-4** | **Static UI Form Layouts** unable to adapt to arbitrary feature schemas | Dynamic form generator inspecting active model's `feature_names_in_` and training distributions |
| **RC-5** | **Hardcoded Quantile & Binning Thresholds** breaking on non-Ames price scales | Dynamic distribution-based binning (quartiles, quantiles, Freedman-Diaconis rule) |
| **RC-6** | **Static SHAP & Metric Caching** | Compute live SHAP values and permutation importances directly on active model & test fold |

---

## 5. Next Steps
Proceed to `docs/REBUILD_PLAN.md` for the comprehensive 25-phase schema-adaptive architecture specification.
