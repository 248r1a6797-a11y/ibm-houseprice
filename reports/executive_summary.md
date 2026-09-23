# Executive Summary: HouseWise AI Valuation & Intelligence Platform

## What Was Analyzed?
An authoritative municipal dataset of **2,930 residential property sales** in Ames, Iowa, spanning 2006 to 2010 with **82 explanatory variables**, compiled by Professor Dean De Cock (Truman State University). Variables capture physical dimensions, material grades, zoning classifications, room counts, and neighborhood micro-locations.

## What Was Predicted?
The **estimated market sale price** (in USD $\$$) of residential single-family properties at transaction time. The system deliberately characterizes outputs as *estimated market valuations based on historical assessor records*, avoiding misleading claims of "true intrinsic value".

## Which Model Was Selected?
An **Extreme Gradient Boosting (XGBoost) Regressor** trained within an atomic, leakage-free pipeline with a logarithmic target transformation ($\log(1+y)$) and inverse exponential mapping ($\exp(\hat{z})-1$). It was selected via the automated Model Selection Gate over 17 competing regressors (including Random Forest, HistGradientBoosting, LightGBM, Stacking Ensembles, and regularized linear models).

## How Well Did It Perform?
On the completely untouched 20% holdout test partition (586 properties):
- **Holdout Test RMSE**: **$22,972.84** (lowest among all 18 evaluated models)
- **Holdout Test MAE**: **$13,834.55** (typical absolute deviation)
- **Coefficient of Determination ($R^2$)**: **0.9342** (explaining 93.42% of market variance)
- **Mean Absolute Percentage Error (MAPE)**: **7.31%**
- **Median Absolute Error**: **$9,070.75**
- **Calibrated 80% Uncertainty Band**: $[-\$17,310, +\$20,237]$

## What Were the Main Drivers?
Using TreeExplainer Shapley Additive Explanations (SHAP), the dominant value determinants in order of impact are:
1. **Overall Material & Finish Quality (`OverallQual`)**
2. **Above-Ground Living Area (`GrLivArea`)**
3. **Total Combined Square Footage (`TotalSF`)**
4. **Combined Fractional Bathrooms (`TotalBath`)**
5. **Property Age at Time of Sale (`HouseAgeAtSale`)**
6. **Neighborhood Location** (e.g. `StoneBr`, `NridgHt` commanding significant premiums; `MeadowV`, `Edwards` exhibiting lower baseline medians)

## Where Does the Model Perform Poorly?
1. **Upper-Tail Luxury Properties (SalePrice > $350,000)**: Heteroscedastic error expansion increases median absolute error to ~$28,000 on custom-built estates.
2. **Unusual Non-Normal Sales**: Distressed foreclosures, partial sales, or family property transfers where sale price was disconnected from physical asset characteristics.
3. **Sparse Neighborhoods**: Areas with fewer than 15 historical records.

## What Limitations Exist?
- **Temporal Constraint**: Dataset spans 2006–2010 during the U.S. subprime financial crisis; does not reflect modern inflation or post-2020 interest rates.
- **Geographic Specificity**: Reflects Ames, Iowa (a college town hosting Iowa State University); cannot be generalized to metropolitan financial hubs without local retraining.
- **Non-Causal Interpretation**: SHAP values denote statistical associations within historical records, not guaranteed causal price increases.

## What Stakeholders Should Understand
Machine learning models excel at capturing complex, non-linear pricing relationships for 80%+ of standard residential inventory. However, automated valuations must serve as decision-support intelligence for appraisers and portfolio managers, coupled with human appraisal reviews on properties exceeding $350,000.
