# Model Card: HouseWise AI Production Regressor

## 1. Model Details
- **Model Name**: Extreme Gradient Boosting Regressor (XGBoost)
- **Model Version**: 1.0.0
- **Model Architecture**: `TransformedTargetRegressor` with `np.log1p` target transformation wrapping a scikit-learn `Pipeline` (ColumnTransformer preprocessor + `XGBRegressor`).
- **Base Algorithms**: HistGradientBoosting, GradientBoosting, LightGBM, Random Forest, Ridge benchmarks.
- **Training Timestamp**: September 23, 2026
- **License / Governance**: Open Academic & Educational Use

## 2. Intended Use
- **Primary Use Case**: Automated residential valuation decision support, comparative market analysis (CMA), and feature-level price driver explanations for properties in Ames, Iowa.
- **Out-of-Scope / Prohibited Uses**: Automated denial of residential mortgage applications, credit underwriting, insurance rate setting, or tenant background screening without certified human appraisal and legal compliance.

## 3. Training & Validation Strategy
- **Training Data**: 80% partition (2,344 properties) of the authoritative Ames Housing dataset (Dean De Cock, 2011).
- **Validation**: 5-Fold Cross-Validation with stratified random state (seed: 42).
- **Holdout Test Set**: 20% untouched partition (586 properties).

## 4. Performance Metrics (Original USD $ Scale)
- **Holdout Test RMSE**: $22,972.84
- **Holdout Test MAE**: $13,834.55
- **Holdout Test R²**: 0.9342 (93.42% of target variance explained)
- **Holdout Test RMSLE**: 0.1059
- **Holdout Test MAPE**: 7.31%
- **Holdout Median Absolute Error**: $9,070.75
- **Calibrated 80% Prediction Interval**: [-$17,310, +$20,237]

## 5. Explainability & Factors
- **Methodology**: Shapley Additive Explanations (SHAP TreeExplainer).
- **Primary Value Drivers**: `OverallQual`, `GrLivArea`, `TotalSF`, `TotalBath`, `HouseAgeAtSale`, `Neighborhood`.
- **Limitation**: SHAP attributions reflect historical predictive associations, not causal relationships.
