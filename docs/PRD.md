# Product Requirements Document (PRD)
## HouseWise AI: Explainable House Price Intelligence & Prediction Platform

### 1. Executive Summary & Purpose
HouseWise AI is an enterprise-grade housing intelligence platform designed to transform raw property sales records into explainable valuation estimates, transparent feature contribution attributions, and actionable business insights. Rather than treating machine learning as an opaque "black-box" regressor, HouseWise AI emphasizes scientific rigor, anti-leakage safeguards, cross-validated model selection, and multi-tier stakeholder storytelling.

### 2. Stakeholders & Personas
1. **Business Leadership / Executives**: Require top-line market KPIs (median prices, price distributions, transaction volume), high-level valuation accuracy metrics, macro risk factors, and recommended business actions.
2. **Real Estate / Data Analysts**: Require bivariate relationships, correlation heatmaps, neighborhood-level pricing benchmarks, price per square foot metrics, and granular error segmentations.
3. **Property Evaluators & Consumers**: Require an intuitive input form to test hypothetical property configurations, obtain an estimated market valuation with confidence/uncertainty intervals, and receive an itemized breakdown of positive and negative value drivers.
4. **Technical Reviewers / Internship Evaluators**: Require strict code modularity, automated test coverage (`pytest`), non-leaking preprocessors, baseline benchmarks, hyperparameter optimization logs, and comprehensive documentation.

### 3. Business Intelligence Hierarchy
The system strictly implements the five-level decision hierarchy:
- **Level 1 — KPIs (What is happening?)**: Volume, median and mean sale prices, living area benchmarks, baseline and model RMSE.
- **Level 2 — Trends (How is it changing?)**: Historical price dynamics across sale years (2006–2010), month-of-year seasonality, neighborhood appreciation.
- **Level 3 — Drivers (Why is it happening?)**: SHAP global and local contributions, permutation importance, living area and quality coefficients.
- **Level 4 — Risk (What could go wrong?)**: Residual skewness, heteroscedasticity, outlier vulnerability (>4000 sq ft custom estates), high-error neighborhoods.
- **Level 5 — Action (What should be done?)**: Targeted data acquisition in underrepresented neighborhoods, conservative valuation bands for high-end properties, seasonal listing strategies.

### 4. Technical Requirements
- **Dataset**: Ames Housing Dataset (2,930 observations, 82 variables, Dean De Cock).
- **Target**: `SalePrice` in USD ($).
- **Core Models**: Baselines (Mean, Median, Group), Ridge, Lasso, ElasticNet, DecisionTree, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, XGBoost, LightGBM, Stacking Ensemble.
- **Validation**: 5-Fold Cross-Validation, fixed seed 42, 80/20 train/test holdout.
- **Explainability**: SHAP (TreeExplainer) with waterfall plots, beeswarm plots, and permutation importance.
- **Inference**: Sub-100ms single property scoring and bulk CSV batch processing.
- **Dashboard**: 12-page Streamlit application with custom corporate styling.
- **Testing**: Automated `pytest` suite testing data loaders, feature engineering, model training, and inference.
