# HouseWise AI — Feature Gap Analysis
**Date**: September 24, 2026 | **Author**: Antigravity Data Science & ML Engineering Team

---

## 1. Overview of Gaps

The transformation from a static reporting dashboard to an interactive end-to-end Machine Learning Workbench requires bridging significant architectural and functional gaps. Below is the itemized gap analysis covering all 25 specific requirements.

---

## 2. Itemized Gap Matrix

| # | Requested Functional Area | Current Implementation Status | Gap Description & Action Required |
| :-: | :--- | :--- | :--- |
| **1** | **Data Workspace** | ❌ Missing | No file uploader (CSV/Excel). No target variable selector. No problem type auto-detection. Need a dedicated page with upload, inspect, target selection, and processed download. |
| **2** | **Data Profiling** | ⚠️ Partial (Static JSON) | `reports/data_profile.json` is rendered statically. Need dynamic profiling table with Type, Missing Count/%, Unique, Mean, Median, Std, Min, Max, and Data Quality Score (0-100). |
| **3** | **Data Cleaning Studio** | ❌ Missing | Cleaning is an offline script (`src/data/cleaner.py`). Need interactive controls: Missing strategy (Mean, Median, Frequent, Unknown), Duplicates removal, Outlier handling (IQR, Z-Score) with Winsorize/Remove/Keep, showing Before vs After. |
| **4** | **Feature Engineering Controls** | ⚠️ Partial (Offline only) | Pre-computed 7 features offline. Need interactive feature selector (checkboxes, search, numeric/categorical filters) and derived feature creator (custom formula builder + templates `TotalSF`, `TotalBath`, `HouseAge`, `QualityPerSqFt`). |
| **5** | **Interactive EDA Laboratory** | ⚠️ Partial (Fixed charts) | Market Analytics has fixed charts. Need dynamic Chart Builder: User selects X, Y, Hue, Aggregation, Chart Type (Histogram, Box, Scatter, Violin, Bar, Heatmap, KDE). |
| **6** | **Correlation & Relationships** | ⚠️ Partial (Fixed Pearson) | Correlation heatmap exists but is static. Need interactive toggle between Pearson and Spearman, top positive & negative relationship rankings with target, and correlation vs causation advisory. |
| **7** | **Experiment Configuration** | ❌ Missing | Train/test split was run offline. Need interactive configuration: Train/Test split slider (e.g. 80/20), CV folds slider (3–10), random seed, target transformation toggle (`log1p` vs `none`). |
| **8** | **Interactive Model Laboratory** | ⚠️ Partial (Static table) | `04_model_lab.py` simply displays `model_comparison.csv`. Need interactive candidate model checkboxes and a **"RUN MODEL BENCHMARK"** button that executes CV live, shows progress, and crowns the champion from measured evidence. |
| **9** | **Hyperparameter Tuning Studio** | ⚠️ Partial (Offline script) | `src/models/tuning.py` exists but is not accessible from the UI. Need interactive hyperparameter sliders for top models, search strategy selector (RandomizedSearchCV vs GridSearchCV), and before/after score delta display. |
| **10**| **Model Experiment History** | ❌ Missing | No experiment tracking database in session state. Need an Experiment Tracker storing Experiment IDs, timestamps, features, hyperparameters, CV scores, and run comparisons. |
| **11**| **Live Model Explainability** | ⚠️ Partial (Static figures) | SHAP figures are pre-rendered. Need interactive TreeSHAP summary, beeswarm, feature importance, and live single-property waterfall attribution for any selected property. |
| **12**| **Interactive Property Predictor** | ⚠️ Partial (Fixed fields) | Predictor uses hard-coded preset fields. Need dynamic form generated from the active model's feature set with real-time prediction and uncertainty intervals. |
| **13**| **What-If Simulator** | ❌ Missing | Completely absent. Need a side-by-side Base Property vs Scenario Property comparator with difference in $, difference in %, and feature attribution delta. |
| **14**| **Prediction Diagnostics** | ⚠️ Partial (Static CSVs) | Pre-computed worst error CSVs displayed. Need interactive Actual vs Predicted scatter, residual histogram, error by price range, and under vs overprediction forensics. |
| **15**| **Segment Performance Analysis** | ❌ Missing | No sliced performance breakdown. Need model evaluation sliced by Neighborhood, Price Segment (<$100k, $100k-$200k, $200k-$300k, >$300k), Age Cohort, and Quality Tier. |
| **16**| **Neighborhood Intelligence** | ⚠️ Partial (Static bar) | Bar chart of neighborhood medians exists. Need comprehensive submarket table: Property count, median price, median living area, price per sqft, average quality, and model error by neighborhood. |
| **17**| **Price per Sq Ft Analysis** | ⚠️ Partial (Single metric) | Card shows $120/sqft. Need deep dive: Overall median/min/max, breakdown by neighborhood, quality, vintage, and clear distinction between observed vs predicted. |
| **18**| **Error & Risk Analysis** | ⚠️ Partial (Fixed charts) | Conformal intervals exist. Need interactive risk analysis: high-residual flags, sensitivity to volatile features, and documented mathematical limitations. |
| **19**| **Dynamic Insight Generator** | ⚠️ Partial (Rule cards) | Cards exist but have static numbers. Need dynamically synthesized FACT $\to$ EVIDENCE $\to$ INTERPRETATION $\to$ ACTION cards computed directly from active data. |
| **20**| **Dynamic Executive Overview** | ⚠️ Partial (Hard-coded) | Currently displays static $180k, 2,930 rows, and hard-coded champion XGBoost. Must dynamically compute all KPIs from the current session state. |
| **21**| **Dataset Change Invalidation** | ❌ Missing | Critical flaw: changing data does not invalidate models or predictions. Need reactive state invalidation: if data/target/features change, invalidate downstream models and mark them pending. |
| **22**| **Global Pipeline Status Banner**| ❌ Missing | No visual progress tracker. Need an 8-stage progress tracker at the top of the UI showing completed, active, and pending steps. |
| **23**| **Technical Claims Audit** | ⚠️ Minor corrections | Wording like "Verified assessor records" and "93.4% Generalization accuracy" must be audited and corrected to "Ames Housing Assessor Dataset (De Cock, 2011)" and "Explained Variance ($R^2$): 93.4%". |
| **24**| **Clean Modular UI Design** | ✅ Good foundation | CSS theme and card structures are clean. Must preserve card/tab design while embedding interactive workflow controls without visual clutter. |
| **25**| **End-to-End User Journey** | ⚠️ Disconnected | Currently, user cannot run: Upload $\to$ Clean $\to$ Feature Engineer $\to$ Train $\to$ Tune $\to$ Predict $\to$ What-If $\to$ Insights. Need a unified state-driven session journey. |

---

## 3. High-Priority Architecture Deficits

1. **Session State Engine (`st.session_state`)**:
   The application requires a centralized state manager in `src/config.py` or `app/state.py` that initializes and coordinates the entire pipeline state across all pages.
2. **Dynamic Invalidation Hook**:
   Any mutation of upstream artifacts (e.g. uploading a new file or modifying cleaning parameters) must instantly set `st.session_state.models_valid = False` and notify the user with a warning banner.
3. **Execution Runtime**:
   Model training and tuning must execute smoothly in-browser without freezing the UI, using progress bars, execution time tracking, and caching of intermediate results.
