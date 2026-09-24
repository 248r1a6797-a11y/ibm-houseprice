# HouseWise AI — Comprehensive Production Rebuild & Schema-Adaptive Architecture Specification
**Date**: September 24, 2026 | **Author**: Antigravity Data Science & ML Engineering Team
**Status**: Architecture Rebuild Plan Complete | Implementation Code Modifications Paused Awaiting User Sign-Off

---

## 1. Architectural Vision & Core Principles

The rebuilt **HouseWise AI** platform transforms from a static Ames Housing dashboard into a **truly schema-adaptive, reactive machine-learning workbench and decision intelligence system**.

### Non-Negotiable Core Architectural Invariants:
1. **Schema Adaptability**: Operates flawlessly on any valid tabular real estate dataset (e.g. 3,986 records across 8 columns: `price, area, bedrooms, bathrooms, location, age, parking, lot_size`, or the 82-column Ames benchmark, or California housing, or custom Excel sheets).
2. **Zero Hardcoded Schema Dependencies**: No column name (`SalePrice`, `GrLivArea`, `OverallQual`, `YearBuilt`, `Neighborhood`) is ever presumed mandatory. All column references are dynamically bound through `DatasetContext`.
3. **Zero Static Offline Fallbacks in Live Workflows**: No precomputed CSV or `.joblib` file on disk may masquerade as live results. If an experiment has not been executed in the active session, downstream pages display clean, informative "Awaiting Execution" states with clear navigation calls to action.
4. **Zero Target Leakage**: Preprocessing pipelines strictly isolate features $X$ from target $y$. Preprocessor transformers (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) are fitted strictly on training folds inside Scikit-Learn `Pipeline` objects.
5. **Dynamic Form & UI Synthesis**: Prediction forms, what-if sliders, correlation heatmaps, and spatial aggregations dynamically inspect the active dataset schema and model estimators at runtime.
6. **Zero Hallucination in Advisory Layer**: All strategic insights, metrics, and advisory answers are mathematically computed from the active dataset and active trained champion.

---

## 2. Core Architecture: The `DatasetContext` & Dynamic Pipeline

```
                               ┌────────────────────────────────────────┐
                               │             USER RAW DATA              │
                               │  (Upload CSV/Excel or Select Ames Benchmark)│
                               └──────────────────┬─────────────────────┘
                                                  │
                                                  ▼
                               ┌────────────────────────────────────────┐
                               │             DatasetContext             │
                               │  • raw_df, clean_df, feature_df        │
                               │  • target_col, problem_type            │
                               │  • num_cols, cat_cols, id_cols, date_cols│
                               │  • location_col, size_col, age_col     │
                               │  • feature_ranges, categorical_levels  │
                               └──────────────────┬─────────────────────┘
                                                  │
                                                  ▼
                 ┌────────────────────────────────┴────────────────────────────────┐
                 │                                                                 │
                 ▼                                                                 ▼
┌─────────────────────────────────┐                             ┌─────────────────────────────────┐
│       DATA SCIENCE WORKSPACE    │                             │       DYNAMIC INFERENCE & BI    │
│  • Page 1: Data Workspace       │                             │  • Page 11: Dynamic Predictor   │
│  • Page 2: Quality Profiler     │                             │  • Page 12: Dynamic What-If Sim │
│  • Page 3: Cleaning Studio      │                             │  • Page 13: Error Diagnostics   │
│  • Page 4: Feature Studio       │                             │  • Page 14: Market Analytics    │
│  • Page 5: EDA Lab              │                             │  • Page 15: Spatial Intel       │
│  • Page 6: Experiment Setup     │                             │  • Page 16: Risk Slicing        │
│  • Page 7: Model Tournament     │                             │  • Page 17: Insights & Advisor  │
│  • Page 8: Tuning Studio        │                             │  • Page 18: Executive Overview  │
│  • Page 9: Experiment History   │                             │  • Page 19: Dynamic SQL Studio  │
│  • Page 10: Live SHAP Lab       │                             │  • Pages 20-21: Docs & Method   │
└─────────────────────────────────┘                             └─────────────────────────────────┘
```

---

## 3. The 25-Phase Comprehensive Rebuild Plan

### Phase 1: Dynamic Dataset Context & Semantic Schema Inference (`src/data/schema.py`)
- Implement `DatasetContext` dataclass:
  - Stores `raw_df`, `clean_df`, `feature_df`.
  - Automatically identifies candidate target variables (e.g. numeric columns with names containing `price`, `val`, `target`, `cost`, `medv`).
  - Automatically identifies potential ID columns (e.g. `id`, `pid`, `order`, index-like columns with 100% uniqueness).
  - Automatically partitions feature columns into:
    - Continuous numerical
    - Discrete numerical / counts
    - Categorical / ordinal
    - Datetime / temporal
    - Spatial / location candidates (`neighborhood`, `location`, `city`, `zipcode`, `district`, `suburb`)
    - Size / area candidates (`area`, `grlivarea`, `sqft`, `size`, `surface`)
    - Age / year candidates (`yearbuilt`, `year`, `age`, `built`)
  - Computes and caches training distributions (min, max, median, mean, standard deviation, unique categories) for dynamic UI generation.

### Phase 2: Reactive Session State Machine & Clean Onboarding (`app/state.py`)
- Refactor `init_session_state(st)`:
  - **Clean Onboarding State**: When a user first opens the application, do NOT silently load Ames data as if it were the user's data. Present a clear choice:
    1. Upload Custom Real Estate Data (CSV / Excel), or
    2. Load Curated Benchmark Dataset (Ames Housing Assessor Dataset).
  - Initialize pipeline status to `⏳ Awaiting Data` until data is uploaded or chosen.
  - Implement strict downstream invalidation: mutating cleaning invalidates features and models; mutating features invalidates splits and models; mutating splits invalidates models.

### Phase 3: Schema-Adaptive Ingestion & File Parsing (`app/pages/01_data_workspace.py`)
- Multi-format file uploader (`.csv`, `.xlsx`, `.xls`, `.parquet`).
- Automatic encoding detection (UTF-8, ISO-8859-1, cp1252).
- Delimiter autodetection (comma, semicolon, tab).
- Target variable selector with smart auto-recommendation.
- Problem type selector (Fixed to Regression with verification that target is continuous numeric).
- Ingestion preview displaying row count, column count, memory footprint, and column data types.

### Phase 4: Dynamic Data Profiling & Quality Scoring (`app/pages/02_data_profiling.py`)
- Calculate Data Quality Score (0–100) dynamically derived from active dataset:
  - Completeness metric (% missing values across cells).
  - Uniqueness metric (% duplicate rows).
  - Validity metric (type conformity, non-negative bounds for prices/areas).
  - Constant column penalty.
- Interactive column-by-column profiling directory:
  - Column name, detected type, missing count, missing %, unique values, min, median, max, skewness.
  - Missingness heatmap and correlation indicator.

### Phase 5: Interactive & Schema-Adaptive Data Cleaning Studio (`app/pages/03_data_cleaning_studio.py`, `src/data/cleaner.py`)
- Universal cleaning operations:
  - Missing numerical imputation: Median, Mean, Zero, or Drop.
  - Missing categorical imputation: Mode, Constant ("Missing" / "None"), or Drop.
  - Deduplication toggle with live duplicate row counter.
  - Outlier handling: IQR (1.5x, 3.0x), Z-Score (>3.0 std), with actions: Cap/Winsorize, Remove, or Flag without modifying.
- Real-time **Before vs. After Cleaning Audit Card**:
  - Displays original rows vs. clean rows, null cells before vs. after, duplicate rows eliminated, outlier rows adjusted.

### Phase 6: Dynamic Feature Engineering & Selection Studio (`app/pages/04_feature_engineering_studio.py`, `src/features/engineering.py`)
- Dynamic Derived Feature Generator:
  - User can combine any two numerical columns via basic arithmetic ($+, -, \times, \div$).
  - Built-in smart templates that activate only when relevant columns are detected:
    - *Unit Price* ($\text{Price} / \text{Area}$) — flagged strictly for analysis, excluded from model training.
    - *Total Area* (Summing detected area columns, e.g. basement + living area).
    - *Total Bathrooms* (Summing full + half baths).
    - *Property Age* ($\text{Current Year or Year Sold} - \text{Year Built}$).
- Feature Selection Matrix:
  - Interactive multiselect with "Select All", "Deselect All", and filter by type.
  - Variance thresholding (automatically flags and deselects zero-variance columns).
  - Mutual Information and Pearson correlation ranking against target.

### Phase 7: Dynamic Preprocessing Pipeline with Zero Leakage (`src/features/preprocessing.py`)
- Rebuild `build_preprocessing_pipeline`:
  - Takes dynamic lists of `num_cols` and `cat_cols` from `DatasetContext`.
  - Numerical Pipeline: `SimpleImputer(strategy="median")` + `StandardScaler()`.
  - Categorical Pipeline: `SimpleImputer(strategy="constant", fill_value="Missing")` + `OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01)`.
  - Enforces strict exclusion of the target variable and designated ID columns.
  - Wraps everything in Scikit-Learn `ColumnTransformer(remainder="drop")`.

### Phase 8: Dynamic Exploratory Data Analysis Laboratory (`app/pages/05_eda_laboratory.py`)
- Dynamic Multi-Variable Chart Builder:
  - Scatter plots, box plots, violin plots, histograms, density contours.
  - X, Y, Color (Hue), and Size selectors populated dynamically from active dataset columns.
- Target Variable Distribution Studio:
  - Visualizes raw target vs $\log(1 + y)$ transformed distribution.
  - Computes skewness and kurtosis dynamically.
- Interactive Bivariate Hypothesis Testing:
  - Numerical vs Numerical correlation scatter with OLS trendline.
  - Categorical vs Target boxplot with ANOVA / Kruskal-Wallis test p-values.

### Phase 9: Experiment Setup & Holdout Partitioning (`app/pages/06_experiment_setup.py`)
- Train/Test holdout split slider (10% to 40%, default 20%).
- Cross-validation fold selector ($K = 3$ to $10$, default 5).
- Stratification strategy: Standard Shuffle K-Fold or Target Quantile-Stratified K-Fold.
- Target transformation toggle: `log1p(y)` wrapping via `TransformedTargetRegressor`.
- Random seed selector for deterministic reproducibility.
- Partition inspection card: Displays training sample count, test sample count, and feature count.

### Phase 10: Interactive Model Laboratory & Benchmark Tournament (`app/pages/07_model_laboratory.py`, `src/models/`)
- Candidate Algorithm Selection:
  - Linear Family: OLS, Ridge, Lasso, ElasticNet.
  - Tree Family: Decision Tree, Random Forest, Extra Trees.
  - Boosting Family: Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost (if available).
- Live Tournament Execution:
  - Cross-validation scoring across all folds ($R^2$, RMSE, MAE).
  - Holdout test evaluation.
  - Real-time progress bar with per-model execution timer.
- Dynamic Champion Selection Gate:
  - Automatically identifies and crowns the Champion Model based on lowest CV RMSE / highest holdout $R^2$.
  - Saves the active fitted pipeline in `st.session_state.champion_model`.
  - Dynamic Model Comparison Table & Bar Chart.

### Phase 11: Hyperparameter Tuning Studio (`app/pages/08_hyperparameter_tuning.py`, `src/models/tuning.py`)
- Interactive tuning for Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM.
- Tuning method: RandomizedSearchCV or GridSearchCV.
- Configurable hyperparameter sliders (learning rate, max depth, n_estimators, alpha/regularization).
- Side-by-side **Baseline vs. Tuned Performance Comparison**:
  - Displays $\Delta \text{RMSE}$, $\Delta R^2$, and runtime.
- "Promote Tuned Model to Champion" button to immediately update the active production pipeline.

### Phase 12: Dynamic Experiment History Tracker (`app/pages/09_experiment_history.py`)
- Maintains an in-memory chronological ledger of all training and tuning runs during the session.
- Columns: Run ID, Timestamp, Model Name, Features Count, CV RMSE, CV $R^2$, Test RMSE, Test $R^2$, Tuning Status.
- Run-to-run delta diffs and ability to restore a prior run as the active champion.

### Phase 13: Live SHAP & Game-Theoretic Explainability (`app/pages/10_model_explainability.py`, `src/explainability/`)
- Global Feature Attribution:
  - Computes TreeSHAP (for tree models) or Permutation Importance / Linear SHAP directly on the active champion and holdout test set.
  - Interactive horizontal bar chart of mean absolute attribution values.
- Single-Property Waterfall Breakdown:
  - Select any observation from the holdout dataset or input a custom property.
  - Computes real-time positive value drivers (green) and downward price penalties (red).

### Phase 14: Dynamic Property Predictor (`app/pages/11_property_predictor.py`, `src/inference/predictor.py`)
- **Completely Dynamic Form Synthesis**:
  - Automatically generates form input widgets from `active_model.named_steps["preprocessor"]` or `active_model.regressor_.named_steps["preprocessor"]`.
  - For numerical features: Generates `st.number_input` or slider bounded by $[\min, \max]$ with default set to training median.
  - For categorical features: Generates `st.selectbox` populated with unique categories observed in training data.
- Single Prediction Output:
  - Point valuation estimate.
  - Empirical 90% and 95% prediction intervals computed from holdout residual quantiles.
  - Real-time financial waterfall drivers for this specific property.
- Batch Prediction Mode:
  - Upload a CSV of unseen properties, generate batch predictions, and download scored CSV.

### Phase 15: Dynamic What-If Scenario Simulator (`app/pages/12_what_if_simulator.py`)
- Dynamic Dual-Column Interface (Base Property vs. Scenario Property).
- Dynamically selects top 4 most influential numerical features for side-by-side slider tuning.
- Real-time delta valuation ($\Delta \text{Price}$) and marginal return per unit change.

### Phase 16: Prediction Diagnostics & Residual Forensics (`app/pages/13_prediction_diagnostics.py`)
- Evaluates active champion against holdout test set:
  - Predicted vs. Actual scatter plot with $45^\circ$ reference line.
  - Residual distribution histogram with normal curve fit.
  - Residual vs. Fitted plot (heteroscedasticity diagnostic).
  - QQ plot for normality of residuals.
- Worst Predictions Forensics:
  - Top 5 largest underpredictions (Actual $\gg$ Predicted).
  - Top 5 largest overpredictions (Actual $\ll$ Predicted).
  - Detailed feature breakdown to identify boundary failure causes.

### Phase 17: Schema-Adaptive Market Analytics (`app/pages/14_market_analytics.py`, `src/analytics/`)
- Dynamic Feature Correlation Matrix:
  - Correlates all numerical columns in active dataset using Pearson or Spearman.
- Target Feature Ranking:
  - Bar chart of highest positive and negative correlations with the target variable.
- Non-linear relationships:
  - Scatter plots with LOWESS smoothed trendlines.

### Phase 18: Spatial & Cohort Intelligence (`app/pages/15_neighborhood_intelligence.py`)
- Dynamic Spatial Selector:
  - User can select which categorical column represents location/cohort (e.g. `neighborhood`, `location`, `zipcode`, `city`, `district`).
  - User can select which numerical column represents property area (e.g. `area`, `sqft`, `living_area`).
- Submarket Directory:
  - Aggregates median price, mean price, median price per square foot, and transaction volume.
  - Interactive horizontal ranking bar chart and boxplots.

### Phase 19: Sliced Risk & Uncertainty Analysis (`app/pages/16_risk_analysis.py`)
- Dynamic Price Tier Slicing:
  - Computes 4 price tiers based on empirical quartiles (Q1, Q2, Q3, Q4) of the target variable.
  - Computes sliced MAE, RMSE, and MAPE across each quartile.
- Dynamic Cohort Slicing:
  - User selects any categorical feature or binned numerical feature to inspect slice performance.

### Phase 20: Evidence-Based Insights & Grounded Advisor (`app/pages/17_insights_actions.py`, `src/analytics/advisor.py`)
- Structured Strategic Action Cards:
  - Synthesizes insights strictly from active dataset statistics and model metrics.
  - Strictly follows the BI Hierarchy: DATA $\rightarrow$ INFORMATION $\rightarrow$ INSIGHT $\rightarrow$ DECISION $\rightarrow$ ACTION.
- Grounded Real Estate Advisor (Zero Hallucination):
  - Answers user queries strictly grounded in the active dataset's verified facts.
  - Rejects answering questions about unverified data.

### Phase 21: Schema-Adaptive Executive Overview (`app/pages/18_executive_overview.py`)
- Top-level C-Suite KPI Dashboard:
  - Active Dataset Name & Row Count.
  - Median & Mean Target Valuation.
  - Active Champion Model & Holdout Generalization Metrics ($R^2$, RMSE, MAE).
  - Top 5 Empirical Price Drivers from live SHAP.
  - Model Health & Calibration Status.

### Phase 22: Live Dynamic SQL Studio (`app/pages/19_sql_studio.py`)
- In-Memory SQLite Database Engine:
  - Upon loading or cleaning data, writes the active dataset into an in-memory SQLite table (`properties`).
  - Schema inspection tab displays live table DDL and column types.
  - Interactive SQL Query Runner: User can write and execute arbitrary `SELECT` queries with real-time tabular output and CSV export.
  - Pre-built adaptive analytical queries (e.g. Top cohorts, quantile distributions, missing value counts).

### Phase 23: Methodology & Theoretical Rigor (`app/pages/20_methodology.py`)
- Mathematical formulation of regression metrics:
  $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}, \quad R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
- Documentation of log-target transformation mechanics:
  $$\hat{y} = \exp\left(\hat{z}\right) - 1, \quad \text{where } z = \ln(y + 1)$$
- Shapley Value game-theoretic formulation and TreeSHAP polynomial complexity guarantees.

### Phase 24: Enterprise Documentation & In-App Manual (`app/pages/21_documentation.py`)
- Comprehensive architecture guide and workflow manual.
- Data dictionary generator inspecting the active dataset.
- Troubleshooting guide for data ingestion, model convergence, and prediction bounds.

### Phase 25: Automated Testing, Edge-Case Verification & QA (`tests/`)
- Test dynamic ingestion of generic datasets (e.g. 3,986 records × 8 features).
- Test zero target leakage in preprocessing.
- Test dynamic predictor form synthesis and prediction interval calculation.
- Test graceful handling of missing columns, high cardinality, and negative values.
- Verify 100% test pass rate with `pytest`.

---

## 4. Verification and Rollout Sequence

1. **Obtain User Approval**: Present this rebuild plan and the forensic audits.
2. **Execute Phase by Phase**: Rebuild core modules (`src/data/`, `src/features/`, `src/models/`, `src/inference/`), then rebuild the reactive state and Streamlit UI pages (`app/pages/`).
3. **Verify with Multiple Real Datasets**:
   - Test with Ames Housing (benchmark).
   - Test with generic 3,986-record real estate dataset (`price, area, bedrooms, bathrooms, location, age, parking, lot_size`).
   - Test with single-variable or minimal-column datasets.
4. **Final Regression & UI Testing**: Ensure all 21 pages render cleanly without warnings, errors, or hardcoded fallbacks.
