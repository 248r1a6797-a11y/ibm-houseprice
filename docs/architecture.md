# Architecture & System Design Document
## HouseWise AI Platform

### 1. High-Level Architectural Flow

```
[ Raw Ames Housing Data ]
           │
           ▼
[ Data Ingestion & Schema Validator ] (src/data/loader.py, validator.py)
           │
           ▼
[ Data Cleaning & Quality Audit ] (src/data/cleaner.py)
  ├── Structural Null Conversion (NA -> 'None', 0)
  └── Outlier Auditing (>4000 sq ft flag)
           │
           ▼
[ Leakage-Free Train/Test Split ] (src/data/splitter.py)
  ├── 80% Training / Cross-Validation Pool (5-Fold KFold)
  └── 20% Untouched Holdout Test Set
           │
           ▼
[ Feature Engineering Layer ] (src/features/engineering.py)
  ├── TotalSF (GrLivArea + TotalBsmtSF)
  ├── TotalBath (Full + 0.5*Half + BsmtFull + 0.5*BsmtHalf)
  ├── HouseAgeAtSale (YrSold - YearBuilt)
  └── RemodAgeAtSale (YrSold - YearRemodAdd)
           │
           ▼
[ Reusable Preprocessing Pipeline ] (src/features/preprocessing.py)
  ├── Numerical: Median Imputer + Robust/Standard Scaler
  └── Categorical: Constant Imputer + OneHotEncoder(handle_unknown='ignore')
           │
           ▼
[ Model Registry & Benchmarking ] (src/models/)
  ├── Baselines (Median, Mean, Neighborhood Median)
  ├── Linear & Regularized (OLS, Ridge, Lasso, ElasticNet)
  ├── Tree Ensembles (Random Forest, Extra Trees)
  ├── Gradient Boosters (HistGB, LightGBM, XGBoost)
  └── Meta-Learner (StackingRegressor)
           │
           ▼
[ Model Selection Gate & Optimization ] (src/models/tuning.py, registry.py)
  ├── Hyperparameter Search (RandomizedSearchCV)
  ├── Target Transformation Experiment (Raw vs log1p(y))
  └── Selection Gate (RMSE, MAE, Stability, Latency)
           │
           ▼
[ Persistence & Metadata Serialization ] (models/production/)
  ├── production_model.joblib
  └── model_metadata.json
           │
           ▼
[ Explainability & Error Diagnostics ] (src/explainability/, src/evaluation/)
  ├── SHAP TreeExplainer (Local waterfall & Global Beeswarm)
  ├── Permutation Importance
  ├── Residual Diagnostics & Heteroscedasticity Analysis
  └── Calibrated Uncertainty Intervals (Quantile Bands)
           │
           ▼
[ User Interface & Analytics Layer ] (app/app.py)
  ├── 12 Interactive Streamlit Pages
  ├── Single Property Estimation & Interactive Sliders
  ├── Batch CSV Processing & Export
  └── Deterministic Business Intelligence Engine
```

### 2. Anti-Leakage Safeguards
1. **Fit-Transform Isolation**: All scalers, imputers, and encoders are fit strictly on training folds within `Pipeline` or `cross_validate` closures; never on the full dataset or holdout set.
2. **Temporal Correctness**: `HouseAge` and `RemodAge` are computed using `YrSold` (the historical transaction year) rather than the runtime calendar year.
3. **Identifier Sanitization**: Tax assessor tracking columns (`Order`, `PID`) are eliminated prior to modeling.
4. **Target Integrity**: Any target scaling or transformation ($\log(1+y)$) is reversed ($\exp(y) - 1$) before calculating dollar-denominated business error metrics (RMSE, MAE).
