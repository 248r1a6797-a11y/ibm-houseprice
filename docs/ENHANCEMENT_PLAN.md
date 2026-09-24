# HouseWise AI — Platform Transformation & Enhancement Plan
### Transforming Static Reporting into an Interactive End-to-End ML Workbench
**Date**: September 24, 2026 | **Author**: Antigravity Data Science & ML Engineering Team

---

## 1. Architectural Vision & Core Principles

The fundamental transformation of HouseWise AI follows the enterprise product principle:

$$\textbf{USER ACTION} \longrightarrow \textbf{COMPUTATION} \longrightarrow \textbf{RESULT} \longrightarrow \textbf{VISUALIZATION} \longrightarrow \textbf{NEXT ACTION}$$

Rather than presenting pre-calculated figures from disk, the platform will maintain an active, reactive **Session State Pipeline** in memory. When default data is loaded, the pipeline is fully initialized. When a user uploads a new dataset, modifies cleaning strategies, engineers new features, or tunes models, the platform recomputes statistics dynamically, invalidates downstream dependencies safely, and provides an end-to-end data science lifecycle.

---

## 2. Restructured Sidebar & Page Hierarchy

The sidebar will be organized into five professional operational sections (21 comprehensive modules):

```
HOUSEWISE AI
Explainable Property Intelligence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WORKSPACE
  1. Data Workspace (Upload, Target, Problem Type, Sample)
  2. Data Profiling (Dynamic Quality Score, Schema, Missingness)
  3. Data Cleaning Studio (Imputation, Duplicates, Outlier Capping)
  4. Feature Engineering (Selection, Derived Formulas, Templates)
  5. EDA Laboratory (Interactive Chart Builder, Custom Aggregations)

MACHINE LEARNING
  6. Experiment Setup (Split, Folds, Random State, Log Target)
  7. Model Laboratory (Live Multi-Model Benchmark, Champion Selection)
  8. Hyperparameter Tuning (Interactive Search, Param Sliders, Delta)
  9. Experiment History (Run Tracker, Compare Experiments)
  10. Model Explainability (Live TreeSHAP Summary, Beeswarm, Waterfall)

PREDICTION
  11. Property Predictor (Dynamic Feature Form, Price & Interval)
  12. What-If Simulator (Base vs Scenario, Marginal Delta Analysis)
  13. Prediction Diagnostics (Actual vs Pred, Residuals, Worst Errors)

INTELLIGENCE
  14. Market Analytics & Relationships (Pearson/Spearman, Corrs)
  15. Neighborhood Intelligence (Submarket Tables, Price/SqFt)
  16. Error & Risk Analysis (Slices by Price, Quality, Vintage)
  17. Insights & Actions (Dynamically Generated Fact-Action Cards)

REPORTING
  18. Executive Overview (Dynamic KPI Dashboard from Active Session)
  19. Enterprise SQL Studio (Live Relational SQLite Queries)
  20. Methodology & Governance (Mathematical Rigor, Model Card)
  21. Project Documentation (Exhaustive 22-Section Reference)
```

---

## 3. Session State Architecture & Reactive Invalidation Engine

### 3.1 State Schema (`app/state.py`)
A centralized state manager (`init_session_state()`) will manage:
- `raw_df`: The currently loaded raw DataFrame (default: Ames Housing).
- `target_col`: Active target variable (default: `SalePrice`).
- `problem_type`: `regression` or `classification` (auto-detected).
- `clean_df`: Cleaned DataFrame after cleaning studio transformations.
- `cleaning_params`: Active missing value, duplicate, and outlier strategies.
- `cleaning_audit`: Before/After statistics and affected row counts.
- `feature_df`: DataFrame after feature engineering and selection.
- `selected_features`: List of active feature columns.
- `engineered_features`: Dict of user-created derived feature formulas.
- `split_config`: `test_size`, `cv_folds`, `random_state`, `use_log_target`.
- `X_train`, `X_test`, `y_train`, `y_test`: Active partitions.
- `benchmark_results`: DataFrame of live or baseline model comparison metrics.
- `champion_model`: Active trained production estimator pipeline.
- `champion_metadata`: Metrics, feature lists, and error bounds for active champion.
- `experiment_history`: List of completed training experiment runs.
- `pipeline_status`: Dict tracking status of each stage (`DATA`, `QUALITY`, `CLEANING`, `FEATURES`, `EDA`, `TRAINING`, `EVALUATION`, `EXPLAINABILITY`, `DEPLOYMENT`).

### 3.2 Invalidation Rules
To prevent stale results:
- If `raw_df` or `target_col` changes $\implies$ Invalidate `clean_df`, `feature_df`, `benchmark_results`, `champion_model`, `experiment_history`.
- If `cleaning_params` changes $\implies$ Invalidate `feature_df`, `benchmark_results`, `champion_model`.
- If `selected_features` or `engineered_features` changes $\implies$ Invalidate `benchmark_results`, `champion_model`.
- A prominent alert banner will inform the user: *"Pipeline input modified. Downstream models require re-running."*

---

## 4. Detailed Implementation Phases

### Phase 1: State Management & Pipeline Infrastructure
- Build `app/state.py` to initialize and expose the reactive session state.
- Build `app/components/pipeline_banner.py` rendering the 8-stage visual pipeline tracker at the top of the interface.
- Audit and correct technical terminology ("Ames Housing Assessor Dataset (De Cock, 2011)", "Explained Variance ($R^2$): 93.4%").

### Phase 2: Data Workspace & Profiling (Pages 1 & 2)
- Build **Data Workspace**: File uploader (CSV / Excel), load default Ames dataset, target variable selector dropdown, problem type detector, dataset summary metrics, and download processed dataset button.
- Build **Data Profiling**: Interactive quality score (0–100 calculated from completeness, uniqueness, and validity), detailed column-level profiling table (Name, Type, Missing, Missing %, Unique, Mean, Median, Std, Min, Max), and column filters (Numerical, Categorical, Missing, High Cardinality, Potential ID/Leakage).

### Phase 3: Data Cleaning Studio & Feature Engineering (Pages 3 & 4)
- Build **Data Cleaning Studio**:
  - Missing value strategy selector (Numerical: Mean, Median; Categorical: Most Frequent, "Unknown").
  - Duplicate detection and one-click removal with row count impact.
  - Outlier detection (IQR multiplier, Z-score threshold) and treatment (Cap/Winsorize, Remove, Keep).
  - Prominent **Before vs After** comparison card with an **"Apply Cleaning Transformations"** button.
- Build **Feature Engineering Studio**:
  - Feature selection table with checkboxes, Select All, Deselect All, Search, and Numeric/Categorical filters.
  - Pre-built housing derived feature templates (`TotalSF`, `TotalBathrooms`, `TotalPorchArea`, `HouseAge`, `RemodelAge`, `QualityPerSqFt`).
  - Custom Derived Feature Creator: Name, Formula (e.g. `GrLivArea * OverallQual`), and source column selector.
  - Feature selection algorithms: Mutual Information, Correlation with Target, and SelectKBest.

### Phase 4: Interactive EDA Laboratory & Correlation Analysis (Pages 5 & 14)
- Build **EDA Laboratory**:
  - Fully dynamic Chart Builder: User selects X Variable, Y Variable, Group-By (Hue), Aggregation (Mean, Median, Sum, None), and Chart Type (Histogram, Box Plot, Scatter Plot, Violin Plot, Bar Chart, Heatmap, KDE Distribution).
  - Dynamic descriptive text generator beneath charts summarizing the observed relationship.
- Build **Correlation & Relationships**:
  - Pearson vs Spearman toggle.
  - Interactive correlation heatmap for user-selected numerical features.
  - Ranked Top Positive and Top Negative relationships with the target variable.
  - Advisory callout explicitly distinguishing statistical correlation from causal impact.

### Phase 5: Experiment Setup, Model Laboratory & Tuning (Pages 6, 7, 8 & 9)
- Build **Experiment Setup**:
  - Train/Test split slider (e.g. 10% to 40%, default 20%).
  - Cross-validation folds slider (3 to 10 folds, default 5).
  - Random seed input (default 42).
  - Target transformation toggle (`log1p(y)` with `expm1` inverse vs `None`).
  - Interactive summary of resulting sample sizes ($N_{\text{train}}$, $N_{\text{test}}$).
- Build **Interactive Model Laboratory**:
  - Candidate algorithm selector with checkboxes (Linear Regression, Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, XGBoost, CatBoost, LightGBM).
  - **"RUN MODEL BENCHMARK"** button: Executes live 5-fold cross-validation on the active session partitions, displaying a real-time progress bar.
  - Interactive Leaderboard Table displaying CV RMSE, CV MAE, CV $R^2$, Train Time, and Holdout Test metrics.
  - Dynamic **Champion Model Crown**: Identifies the winning model strictly from measured validation evidence.
- Build **Hyperparameter Tuning Studio**:
  - Interactive sliders for hyperparameters of the selected model (e.g. for XGBoost: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`).
  - Search strategy selector (RandomizedSearchCV vs GridSearchCV).
  - **"START TUNING"** button: Tunes the pipeline live and displays the Before vs After metric improvement delta.
- Build **Experiment History Tracker**:
  - Saves every completed training/tuning run with Experiment ID, Timestamp, Model, Hyperparameters, and Metrics.
  - Side-by-side run comparison table and visual improvement chart.

### Phase 6: Live Explainability, Predictor & What-If Simulator (Pages 10, 11, 12 & 13)
- Build **Live Model Explainability**:
  - Global TreeSHAP feature importance bar and summary beeswarm plot computed from the active champion model.
  - Local Single-Property Waterfall Plot: User selects any row from the test set or inputs custom values, displaying positive and negative dollar attributions.
- Build **Interactive Property Predictor**:
  - Dynamic input form generated from the active model's required feature set (Numeric sliders, Categorical dropdowns).
  - Displays predicted price, active model name, evaluation metrics, and calibrated 80% confidence interval.
- Build **What-If Simulator**:
  - Side-by-side comparison: **Base Property** vs **Scenario Property**.
  - Sliders to modify features (Living Area, Quality, Year Built, Garage, Baths).
  - Real-time calculation of Estimated Value Difference ($\Delta\$$ and $\Delta\%$) and feature contribution breakdown.
- Build **Prediction Diagnostics & Error Analysis**:
  - Actual vs Predicted interactive scatter plot with identity diagonal line.
  - Residual distribution histogram and Residual vs Predicted plot.
  - Error breakdown across price segments (<$100k, $100k-$200k, $200k-$300k, >$300k).
  - Worst Over- and Under-Prediction forensic tables with root-cause analysis.

### Phase 7: Intelligence, Prescriptive Engine & Reporting (Pages 15, 16, 17, 18 & 19)
- Build **Neighborhood Intelligence & Price/SqFt**:
  - Tabular breakdown of median price, living area, price per sqft, and model error across neighborhoods.
  - Price per sqft analysis grouped by quality, vintage, and building type.
- Build **Dynamic Insights & Actions**:
  - Insight generator that computes real statistics from the active dataset and renders FACT $\to$ EVIDENCE $\to$ INTERPRETATION $\to$ ACTION cards.
  - Integrated Prescriptive Portfolio Allocator with capital budget sliders and multi-criteria scoring.
- Update **Executive Overview**:
  - Connect all KPI cards, histograms, and trend lines directly to the active session state (`st.session_state`), ensuring that if a user uploads a new dataset, the entire executive dashboard updates instantly.

---

## 5. Verification & Testing Strategy

1. **Unit Testing**:
   - Write tests for data workspace loaders (CSV and Excel parsing).
   - Write tests for interactive cleaning transformations (Winsorization, imputation, duplicate drops).
   - Write tests for derived feature engineering formulas.
   - Write tests for session state invalidation logic.
   - Run `python -m pytest -v` ensuring 100% pass across all existing and new test modules.
2. **End-to-End Workflow Verification**:
   - Step through the complete 28-step user journey from Upload to Insights on `http://localhost:8501`.
   - Verify that modifying a parameter properly invalidates downstream models.
   - Verify that live model training completes and crowns the champion from measured evidence.
