# HouseWise AI — Explainable House Price Intelligence & Prediction Platform
*From Raw Housing Data to Predictive Business Decisions*

[![Python Version](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://python.org)
[![ML Framework](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org)
[![Champion Model](https://img.shields.io/badge/XGBoost-3.4.1-green.svg)](https://xgboost.ai)
[![Explainability](https://img.shields.io/badge/SHAP-0.52.0-red.svg)](https://shap.readthedocs.io)
[![Dashboard](https://img.shields.io/badge/Streamlit-1.64.0-brightgreen.svg)](https://streamlit.io)
[![Tests Passing](https://img.shields.io/badge/Tests-14%2F14%20PASS-brightgreen.svg)]()
[![System Health](https://img.shields.io/badge/Health%20Check-PASS-brightgreen.svg)]()

---

## 1. Overview
**HouseWise AI** is an enterprise-grade, internship-ready House Price Prediction and Housing Analytics platform designed to transform raw property sales records into explainable market valuations, transparent factor attributions, and actionable business intelligence.

Built on the authoritative **Ames Housing Dataset** (Dean De Cock, 2,930 observations, 82 variables), the system demonstrates scientific rigor, anti-leakage safeguards, multi-model benchmarking (18 algorithms across 6 families), hyperparameter optimization, SHAP tree explainability, error diagnostics, calibrated prediction intervals, an interactive 12-page Streamlit dashboard, automated unit tests, and complete documentation.

```
       DATA  ────────►  INFORMATION  ────────►  INSIGHT  ────────►  DECISION  ────────►  ACTION
  Assessor Records      Market Analytics      Top SHAP Drivers     Valuation & Bands   Strategic Actions
```

---

## 2. Business Problem & Key Questions
Traditional Automated Valuation Models (AVMs) operate as opaque black boxes that output a single dollar estimate without context or uncertainty bounds. HouseWise AI solves this by answering:
- **Valuation**: What is the estimated market sale price of a property?
- **Uncertainty**: What is the calibrated 80% confidence interval around this estimate?
- **Attribution**: Which property characteristics (living area, quality, age, amenities) drive the price up or down?
- **Risk**: Where does the model fail, and how does error vary across luxury vs. budget segments?
- **Strategy**: What concrete business actions should appraisers, developers, and investors take?

---

## 3. Dataset Provenance
- **Dataset**: Ames Housing Dataset compiled by Professor Dean De Cock (Truman State University).
- **Publication**: *Journal of Statistics Education*, Volume 19, Number 3 (2011).
- **Scope**: 2,930 residential properties sold in Ames, Iowa between 2006 and 2010.
- **Variables**: 82 variables (39 numerical, 43 categorical, target: `SalePrice` in USD $).
- **Documentation**: Sourced from the official ASA archive and archived locally in `data/raw/DataDocumentation.txt`.

---

## 4. System Architecture
```
[ Raw Ames Data (2,930 x 82) ]
           │
           ▼
[ Data Ingestion & Validator ] ──► (Catches missing targets, out-of-bounds years)
           │
           ▼
[ Data Cleaner & Imputer ]     ──► (NA -> 'None' for amenities, GarageYrBlt 2207 typo fix)
           │
           ▼
[ Train/Test Split (80/20) ]   ──► (2,344 Train/CV pool | 586 Untouched Holdout Test)
           │
           ▼
[ Anti-Leakage Engineering ]   ──► (TotalSF, TotalBath, HouseAgeAtSale = YrSold - YearBuilt)
           │
           ▼
[ Reusable Preprocessor ]      ──► (ColumnTransformer: Median Imputer + StandardScaler + OHE)
           │
           ▼
[ Model Zoo Benchmark (18) ]   ──► (5-Fold CV: Baselines, Linear, Trees, Boosters, Ensembles)
           │
           ▼
[ Model Selection Gate ]       ──► (Champion Selected: XGBoost Regressor)
           │
           ▼
[ Explainability & Forensics ] ──► (SHAP TreeExplainer Waterfall & Calibrated Intervals)
           │
           ▼
[ Streamlit Web Command Ctr ]  ──► (12 Modular Analytics Pages on Localhost)
```

---

## 5. Machine Learning Benchmark Leaderboard

Benchmark results computed under identical 5-fold cross-validation on the 80% training partition and evaluated on the untouched 20% holdout test partition (586 properties):

| Rank | Model Name | Model Family | 5-Fold CV RMSE | Holdout Test RMSE | Holdout Test MAE | Holdout Test R² | Overfit Gap |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost Regressor** | **Gradient Boosting** | **$24,949** | **$22,973** | **$13,835** | **0.9342** | **+$8,542** |
| 🥈 | Gradient Boosting | Gradient Boosting | $24,768 | $23,288 | $13,991 | 0.9324 | +$9,102 |
| 🥉 | Hist Gradient Boosting | Gradient Boosting | $24,878 | $23,484 | $14,110 | 0.9312 | +$9,820 |
| 4 | Stacking Ensemble | Meta-Learner (Ridge) | $28,432 | $23,680 | $14,240 | 0.9301 | +$10,210 |
| 5 | Voting Ensemble | Ensemble | $26,125 | $24,213 | $14,480 | 0.9269 | +$9,650 |
| 6 | LightGBM Regressor | Gradient Boosting | $24,445 | $24,245 | $14,310 | 0.9267 | +$9,410 |
| 7 | Extra Trees Regressor | Tree Ensemble | $25,529 | $25,525 | $14,920 | 0.9187 | +$12,410 |
| 8 | Random Forest Regressor | Tree Ensemble | $26,590 | $26,254 | $15,310 | 0.9140 | +$13,200 |
| 9 | Decision Tree | Decision Tree | $35,846 | $30,811 | $18,420 | 0.8816 | +$18,900 |
| 10 | Ridge Regression | Linear (L2) | $46,958 | $32,026 | $18,940 | 0.8721 | +$15,800 |
| 11 | ElasticNet | Linear Regularized | $47,357 | $32,571 | $19,210 | 0.8677 | +$16,100 |
| 12 | K-Nearest Neighbors | Non-Parametric | $31,691 | $32,842 | $19,890 | 0.8655 | +$16,400 |
| 13 | Lasso Regression | Linear (L1) | $49,434 | $33,883 | $20,120 | 0.8568 | +$17,200 |
| 14 | Linear Regression (OLS) | Linear | $56,289 | $34,997 | $20,850 | 0.8472 | +$19,100 |
| 15 | MLPRegressor (Neural Net) | Neural Network | $810,699 | $36,807 | $22,410 | 0.8310 | +$22,500 |
| 16 | Mean Baseline | Heuristic Baseline | $77,086 | $90,222 | $57,410 | -0.0153 | N/A |
| 17 | Median Baseline | Heuristic Baseline | $79,308 | $94,323 | $58,910 | -0.1097 | N/A |
| 18 | Neighborhood Median | Group Baseline | $79,308 | $94,323 | $58,910 | -0.1097 | N/A |

---

## 6. Champion Model Performance
- **Selected Champion**: `XGBoost Regressor` with `log1p(y)` Target Transformation.
- **Holdout Test RMSE**: **$22,972.84**
- **Holdout Test MAE**: **$13,834.55**
- **Holdout Test R²**: **0.9342** (93.42% explained variance)
- **Holdout Test MAPE**: **7.31%**
- **Holdout Median Absolute Error**: **$9,070.75**
- **Calibrated 80% Prediction Interval**: $[-\$17,310, +\$20,237]$

---

## 7. Interactive 12-Page Streamlit Command Center
The application is structured into 12 dedicated pages:
1. **Executive Overview**: Top KPIs, market snapshot, historical dynamics, model health, and strategic risk cards.
2. **Market Analytics**: Interactive Plotly distributions, neighborhood price comparisons, price vs. living area scatter, correlation heatmap.
3. **Property Predictor**: Interactive input form (Basics, Quality, Age, Amenities, Location), 80% confidence interval, SHAP waterfall chart, dataset median comparison, and Batch CSV scoring & export.
4. **Model Lab**: Full 18-model benchmark comparison table, interactive RMSE/MAE bar charts, overfit gap scatter, and production model card.
5. **Model Explainability**: Global SHAP importance rankings, partial dependence curves, and feature response simulator.
6. **Risk & Error Analysis**: Residual distribution, actual vs. predicted plot, top 10 worst under/overpredictions with forensics, error by price tier.
7. **Insights & Actions**: 5 deterministic data-backed cards implementing the FACT $\to$ INSIGHT $\to$ OPPORTUNITY $\to$ ACTION framework.
8. **Data Quality**: Completeness metrics, structural missingness explanation, clerical typo forensics (`GarageYrBlt` 2207 $\to$ 2007).
9. **Methodology**: End-to-end 14-stage pipeline diagram, mathematical metric formulations.
10. **Model Monitoring**: Proposed production drift monitoring framework (KS-test, PSI > 0.2, rolling MAE alerts).
11. **Documentation**: Embedded searchable data dictionary documenting all 82 variables.
12. **About / Internship**: Project credentials, technical competencies, responsible AI ethics.

---

## 8. Installation & Quickstart

### Prerequisites
- Python 3.10+ (Tested on Python 3.13.5 Windows 64-bit)

### Step 1: Clone & Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Fetch Raw Dataset & Run Profiling
```powershell
python scripts/fetch_data.py
python scripts/profile_data.py
```

### Step 3: Train Models & Run Benchmark Suite
```powershell
python scripts/train.py
```

### Step 4: Run Automated Tests & Health Check
```powershell
pytest tests/ -v
python scripts/project_health_check.py
```

### Step 5: Launch Dashboard on Localhost
```powershell
streamlit run app/app.py
```
Open your browser to `http://localhost:8501`.

---

## 9. Example Prediction via CLI
```powershell
python scripts/predict.py --sample
```
**Output**:
```text
==================================================
  HOUSEWISE AI — VALUATION ESTIMATE
==================================================
  Estimated Valuation:       $192,612
  Calibrated 80% Interval:   $175,302 – $212,849
  Model Engine:              XGBoost (v1.0.0)
  Holdout Test MAE:          $13,835

  Top Positive Price Drivers:
    1. OverallQual (+0.068 log impact)
    2. TotalBath (+0.048 log impact)
    3. YearBuilt (+0.016 log impact)

  Property Profile:
    • Living Area (1,650 sq ft) is 14% above the Ames market median (1,442 sq ft).
    • Construction Quality rating is 7/10 (Market median: 6/10).
==================================================
```

---

## 10. Responsible AI & Governance
- **Demographic Neutrality**: The system strictly uses physical asset characteristics, room counts, and zoning classes; no protected personal attributes are collected.
- **Fairness & Transparency**: Calibrated 80% prediction intervals accompany every estimate.
- **Prohibited Uses**: Prohibited for automated mortgage denial, credit rating, or insurance underwriting without human review.

---

## 11. Project Structure
```
housewise-ai/
├── config/              # Central YAML configurations
├── data/raw/            # Authoritative Ames Housing data & codebook
├── data/processed/      # Isolated train/test parquet splits
├── docs/                # PRD, Architecture, Data Dictionary, Methodology, Scripts
├── models/production/   # Serialized model (.joblib) & metadata (.json)
├── reports/             # Model card, final report, benchmark, figures, tables
├── scripts/             # Ingestion, profiling, training, evaluation, CLI predict
├── src/                 # Modular Python source (data, features, models, evaluation, inference)
├── app/                 # 12-page Streamlit application with custom corporate CSS
└── tests/               # Automated pytest unit test suite
```

---

## 12. Author & Internship Evaluation Credentials
- **Project**: HouseWise AI — Explainable House Price Intelligence Platform
- **Category**: Data Science + Machine Learning Engineering + Business Intelligence
- **Evaluation**: Developed for professional internship capstone evaluation.
