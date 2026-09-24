# HouseWise AI — Explainable House Price Intelligence & Valuation Platform
### *Enterprise Predictive Modeling, Relational SQL Analytics & Decision-Support Workbench*

[![Python Version](https://img.shields.io/badge/Python-3.13.5-blue.svg)](https://python.org)
[![Dataset](https://img.shields.io/badge/Dataset-Ames_Housing_Assessor-informational.svg)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
[![Champion Model](https://img.shields.io/badge/Champion-XGBoost_Regressor-green.svg)](https://xgboost.ai)
[![R2 Score](https://img.shields.io/badge/Holdout_R²-0.9342-success.svg)]()
[![RMSE](https://img.shields.io/badge/Holdout_RMSE-$22,973-success.svg)]()
[![Explainability](https://img.shields.io/badge/XAI-TreeSHAP-purple.svg)](https://shap.readthedocs.io)
[![Relational Engine](https://img.shields.io/badge/Database-SQLite_3-blue.svg)]()
[![UI Framework](https://img.shields.io/badge/Dashboard-Streamlit_1.41+-red.svg)](https://streamlit.io)
[![Test Suite](https://img.shields.io/badge/Tests-100%25_PASS-brightgreen.svg)]()

---

## 1. Project Overview & Business Value
**HouseWise AI** is a production-grade residential real estate intelligence and valuation platform built on municipal assessor records. Moving beyond simplistic machine-learning toys, HouseWise AI bridges the gap between raw data engineering, statistical machine learning, and executive capital allocation following the five-stage Business Intelligence hierarchy:

$$\text{DATA} \longrightarrow \text{INFORMATION} \longrightarrow \text{INSIGHT} \longrightarrow \text{DECISION} \longrightarrow \text{ACTION}$$

The platform integrates:
- **Clean Ingestion & Anti-Leakage Preprocessing**: Zero-leakage data cleaning, clerical error resolution (e.g. `GarageYrBlt = 2207` correction), and 7 domain-engineered structural features.
- **Relational SQL Analytics Engine**: An embedded SQLite 3 database (`data/db/housewise.db`) executing 12 production analytical scripts with CTEs and window functions, cross-validated against Pandas.
- **18-Model Benchmark Tournament**: Systematic 5-fold cross-validation across 6 model families (Linear, Regularized, Tree Ensembles, Gradient Boosting, Stacking, Neural Networks).
- **Champion XGBoost Regressor**: Achieving **$R^2 = 0.9342$**, **RMSE = \$22,972.84**, **MAE = \$13,834.55**, and **MAPE = 7.31%** on an untouched 20% holdout test partition (586 properties).
- **Calibrated Prediction Intervals**: Conformal empirical 80% error offsets ([-$17,310, +$20,237]) to protect mortgage collateral underwriting.
- **Game-Theoretic Explainability (TreeSHAP)**: Global feature importance rankings and single-property dollar-denominated financial waterfall plots.
- **Prescriptive Capital Allocator**: Multi-criteria Property Prioritization Index (PPI) and bounded knapsack optimization delivering +13.33% net equity gains on capital deployment.
- **12-Page Streamlit Command Center**: Complete interactive dashboard with live SQL console, scenario simulators, and error forensics active on `http://localhost:8501`.

---

## 2. Dataset Links & Provenance

The system is trained and evaluated on the official **Ames Housing Assessor Dataset** compiled by Professor Dean De Cock (Truman State University):

| Resource | Direct Link & Access | Description |
| :--- | :--- | :--- |
| **Kaggle Housing Dataset** | [Kaggle Competition: House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) | Primary Kaggle dataset mirror and benchmark test platform. |
| **Kaggle Ames Open Dataset** | [Kaggle: Ames Housing Dataset (Full 2,930)](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) | Complete 2,930-record dataset with all 82 original assessor features. |
| **Authoritative JSE Paper** | [Journal of Statistics Education (De Cock, 2011)](http://jse.amstat.org/v19n3/decock.pdf) | Peer-reviewed academic paper detailing data collection methodology. |
| **Raw Assessor Data (ASA)**| [American Statistical Association Archive](http://jse.amstat.org/v19n3/decock/AmesHousing.txt) | Original uncompressed tab-delimited municipal assessor records. |
| **Local Raw Storage** | `data/raw/AmesHousing.csv` | Local authoritative CSV archive stored within this repository. |

### Dataset Key Characteristics
- **Total Transactions**: 2,930 residential property sales in Ames, Iowa (2006 – 2010).
- **Partitioning**: 80% Training Pool (2,344 records) / 20% Holdout Test Partition (586 records).
- **Total Variables**: 82 columns (38 continuous/discrete numeric, 44 nominal/ordinal categorical).
- **Target Variable**: `SalePrice` in true US Dollars (Range: \$12,789 to \$755,000; Median: \$160,000; Mean: \$180,796).
- **Target Normalization**: Log-transformation $y_{\text{log}} = \log(1 + y)$ reduces skewness from +1.74 to -0.01.

---

## 3. Technologies Used

HouseWise AI is built on a modern, reproducible Python 3.13+ data science and engineering stack:

| Layer | Technologies / Packages | Purpose & Utility |
| :--- | :--- | :--- |
| **Core Environment** | Python 3.13.5 (64-bit), Virtualenv | Base execution runtime and dependency isolation |
| **Data Processing** | Pandas 2.2+, NumPy 2.1+, PyArrow | Vectorized tabular operations and Parquet storage |
| **Relational Database** | SQLite 3 (Embedded SQL Engine) | Relational storage, indexing, and 12 analytical SQL scripts |
| **Machine Learning** | Scikit-Learn 1.6+, XGBoost 2.1+, LightGBM 4.5+ | Preprocessing pipelines, cross-validation, and estimators |
| **Explainable AI (XAI)**| SHAP (Shapley Additive exPlanations) 0.46+ | Exact TreeSHAP game-theoretic feature attribution |
| **Interactive UI** | Streamlit 1.41+, Plotly Express | 12-page web command center with reactive visualization |
| **Documentation & BI** | python-docx 1.2+, Matplotlib, Seaborn | Programmatic Word report generation and static figures |
| **Quality Assurance** | Pytest 8.3+, Sys/Subprocess Testing | 100% automated test coverage across all pipelines |

---

## 4. Benchmark Tournament Results (18 Models)

All 18 candidate algorithms were evaluated using identical 5-fold cross-validation on the 2,344-property training set and scored against the 586-property holdout test set:

| Rank | Model Name | Algorithmic Family | 5-Fold CV RMSE | Holdout Test RMSE | Holdout Test MAE | Holdout Test R² |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost Regressor** | **Gradient Boosting** | **$24,949.12** | **$22,972.84** | **$13,834.55** | **0.9342** |
| 🥈 | Gradient Boosting | Gradient Boosting | $24,767.55 | $23,288.31 | $14,000.49 | 0.9324 |
| 🥉 | Hist Gradient Boosting | Gradient Boosting | $24,878.30 | $23,483.50 | $14,228.72 | 0.9312 |
| 4 | Stacking Ensemble | Meta-Learner (Ridge) | $28,432.42 | $23,680.22 | $13,597.88 | 0.9301 |
| 5 | Voting Ensemble | Ensemble Averaging | $26,124.89 | $24,213.12 | $13,694.67 | 0.9269 |
| 6 | LightGBM Regressor | Gradient Boosting | $24,839.90 | $24,334.09 | $14,563.64 | 0.9261 |
| 7 | Extra Trees Regressor | Tree Ensemble | $25,528.91 | $25,525.33 | $15,264.45 | 0.9187 |
| 8 | Random Forest Regressor| Tree Ensemble | $26,590.29 | $26,253.55 | $15,599.51 | 0.9140 |
| 9 | Decision Tree | Decision Tree | $35,845.88 | $30,810.50 | $20,529.65 | 0.8816 |
| 10 | Ridge Regression | Regularized L2 Linear| $46,957.51 | $32,026.03 | $15,495.34 | 0.8721 |
| 11 | ElasticNet | Regularized L1/L2 | $47,356.53 | $32,571.36 | $15,379.84 | 0.8677 |
| 12 | K-Nearest Neighbors | Non-Parametric Distance| $31,691.28 | $32,841.99 | $19,773.02 | 0.8655 |
| 13 | Lasso Regression | Regularized L1 Linear| $49,434.30 | $33,883.18 | $15,821.10 | 0.8568 |
| 14 | Linear Regression (OLS)| Ordinary Least Squares| $56,289.21 | $34,997.30 | $14,569.89 | 0.8472 |
| 15 | MLPRegressor (Neural Net)| Deep Feedforward Net | $810,699.09 | $36,807.03 | $17,948.59 | 0.8310 |
| 16 | Mean Baseline | Heuristic Baseline | $77,086.27 | $90,222.37 | $64,611.41 | -0.0153 |
| 17 | Median Baseline | Heuristic Baseline | $79,307.86 | $94,322.62 | $63,838.10 | -0.1097 |
| 18 | Neighborhood Median | Micro-Market Baseline | $79,307.86 | $94,322.62 | $63,838.10 | -0.1097 |

### Champion Model Specification
- **Algorithm**: `XGBoost Regressor` with Target Transformation (`log1p` $\to$ `expm1`)
- **Holdout Test $R^2$**: **0.9342** (Explains 93.42% of property price variance)
- **Holdout Test RMSE**: **\$22,972.84**
- **Holdout Test MAE**: **\$13,834.55**
- **Holdout Test MAPE**: **7.31%** (Commercial appraisal industry threshold is $< 10\%$)
- **Calibrated 80% Prediction Interval**: $[-\$17,310.02, +\$20,236.56]$

---

## 5. Relational SQL Analytics Layer

HouseWise AI includes an embedded SQLite 3 database (`data/db/housewise.db`) populated with 2,930 cleaned property records and indexed for high-concurrency analytical queries:

- `01_schema.sql`: Table DDL, primary keys, and composite indexes.
- `02_market_overview.sql`: Macro market aggregates (\$529.73M total capitalization, \$180.8k mean price).
- `03_neighborhood_price_per_sqft.sql`: Submarket rankings by price per sqft using `DENSE_RANK() OVER()`.
- `04_quality_condition_matrix.sql`: Quality-condition cross-tabulations.
- `05_property_age_depreciation.sql`: Vintage cohort depreciation (Pre-1950 vs Modern cohorts).
- `06_remodel_roi_impact.sql`: Renovation ROI and modernization price premiums.
- `07_basement_living_area_value.sql`: Finished basement space value contribution.
- `08_garage_capacity_premiums.sql`: Marginal dollar premium per additional car stall using `LAG() OVER()`.
- `09_bath_ratio_analysis.sql`: Value scaling by bathroom-to-bedroom ratios.
- `10_neighborhood_volatility_tiers.sql`: Submarket price dispersion and quartile tiers via `NTILE(4) OVER()`.
- `11_outlier_divergence_candidates.sql`: Micro-market valuation divergence identification.
- `12_temporal_sales_velocity.sql`: Calendar month transaction velocity and seasonality patterns.

---

## 6. Setup & Installation Instructions

### Prerequisites
- Python 3.10+ (Tested extensively on Python 3.13.5 Windows 64-bit).
- Git installed.

### Step 1: Clone Repository & Create Virtual Environment
```bash
# Clone repository
git clone https://github.com/Kishore/HouseWise-AI.git
cd ibm

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Initialize Database & Run SQL Analytics
```bash
python scripts/setup_sql_analytics.py
```

### Step 4: Run Automated Verification Tests
```bash
pytest -v
```

### Step 5: Launch the 12-Page Streamlit Web Platform
```bash
streamlit run app/app.py --server.port 8501
```
Open **`http://localhost:8501`** in any web browser to access the full platform.

### Step 6: CLI Property Valuation (Optional)
```bash
python scripts/predict.py --sample
```

---

## 7. Submission Deliverables Manifest

This repository contains all four required capstone submission deliverables in the project root:

| Deliverable Name | File Path in Root | Format & Specification |
| :--- | :--- | :--- |
| **1. Code File (Notebook)** | [`Kishore_HouseWise_AI.ipynb`](file:///c:/Users/Kishore/Documents/ibm/Kishore_HouseWise_AI.ipynb)<br>[`YourName_ProjectName.ipynb`](file:///c:/Users/Kishore/Documents/ibm/YourName_ProjectName.ipynb) | Complete, fully executed end-to-end Jupyter Notebook with all 14 stages, markdown commentary, EDA plots, 18-model CV tournament, SHAP waterfalls, SQL queries, and knapsack optimization. |
| **Executable Script** | [`Kishore_HouseWise_AI.py`](file:///c:/Users/Kishore/Documents/ibm/Kishore_HouseWise_AI.py) | Standalone Python pipeline reproducing the complete data, SQL, ML, SHAP, and prescriptive workflow. |
| **2. Requirements File** | [`requirements.txt`](file:///c:/Users/Kishore/Documents/ibm/requirements.txt) | Complete list of all Python dependencies with pinned minimum versions for exact reproducibility. |
| **3. Project Report** | [`Kishore_ProjectReport.docx`](file:///c:/Users/Kishore/Documents/ibm/Kishore_ProjectReport.docx)<br>[`YourName_ProjectReport.docx`](file:///c:/Users/Kishore/Documents/ibm/YourName_ProjectReport.docx) | Comprehensive 22-section Microsoft Word document featuring executive styling, embedded figures, model tables, SHAP importance charts, and prescriptive business frameworks. |
| **4. README File** | [`README.md`](file:///c:/Users/Kishore/Documents/ibm/README.md)<br>[`README`](file:///c:/Users/Kishore/Documents/ibm/README) | Complete overview, dataset provenance links, architecture diagrams, installation instructions, and tournament benchmarks. |

---

## 8. Prescriptive Capital Allocation & Operational Tiers

HouseWise AI formulates a **Property Prioritization Index (PPI)**:

$$\text{PPI} = 0.40 \cdot \text{Upside} + 0.30 \cdot \text{RemodelPotential} + 0.20 \cdot \text{NeighborhoodQuality} + 0.10 \cdot \text{Liquidity}$$

### 4-Tier Operational Investment Playbook
- **Tier 1: Deep Value Opportunity ($\Delta\% \ge +10.0\%$)**: Immediate cash acquisition signal; substantial equity cushion.
- **Tier 2: Fair Market Value ($-5.0\% \le \Delta\% < +10.0\%$)**: Standard transaction range; normal mortgage underwriting.
- **Tier 3: Overvaluation Risk ($-15.0\% \le \Delta\% < -5.0\%$)**: Price renegotiation required; demand seller concessions or appraisal contingency.
- **Tier 4: Speculative / High Volatility ($\Delta\% < -15.0\%$ or $> +30.0\%$)**: Forensic inspection hold; investigate unpermitted additions or distressed sale conditions.

---

## 9. Author & Project Metadata

- **Author / Lead Engineer**: Kishore
- **Project Title**: HouseWise AI — Interactive Explainable House Price Analytics & ML Platform
- **Track**: IBM SkillsBuild / Enterprise Data Science & Machine Learning Capstone
- **Local Application URL**: `http://localhost:8501`
- **Verification Status**: **100% Production Ready** (All 14 tests passing, 0 errors)
