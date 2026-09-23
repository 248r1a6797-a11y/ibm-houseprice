# Machine Learning Experiment Tracking Log

**Date**: 2026-09-23  
**Target**: `SalePrice` (USD $)  
**Validation Strategy**: 5-Fold Cross-Validation on 80% training partition with 20% holdout test.

## Experiment Benchmark Results

| Rank | Model Name | 5-Fold CV RMSE ($) | Holdout Test RMSE ($) | Holdout Test MAE ($) | Holdout Test R² | Overfit Gap ($) | Training Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **XGBoost** | $24,949 | $22,973 | $13,835 | 0.9342 | $10,181 | 1.5s |
| 2 | **Gradient Boosting** | $24,768 | $23,288 | $14,000 | 0.9324 | $11,163 | 9.6s |
| 3 | **Hist Gradient Boosting** | $24,878 | $23,484 | $14,229 | 0.9312 | $14,426 | 13.5s |
| 4 | **Stacking Ensemble** | $28,432 | $23,680 | $13,598 | 0.9301 | $10,950 | 33.3s |
| 5 | **Voting Ensemble** | $26,125 | $24,213 | $13,695 | 0.9269 | $11,240 | 20.0s |
| 6 | **LightGBM** | $24,840 | $24,334 | $14,564 | 0.9261 | $15,352 | 1.4s |
| 7 | **Extra Trees** | $25,529 | $25,525 | $15,264 | 0.9187 | $19,952 | 2.9s |
| 8 | **Random Forest** | $26,590 | $26,254 | $15,600 | 0.9140 | $13,993 | 2.8s |
| 9 | **Decision Tree** | $35,846 | $30,810 | $20,530 | 0.8816 | $10,061 | 0.4s |
| 10 | **Ridge Regression** | $46,958 | $32,026 | $15,495 | 0.8721 | $9,566 | 0.3s |
| 11 | **ElasticNet** | $47,357 | $32,571 | $15,380 | 0.8677 | $8,095 | 1.2s |
| 12 | **K-Nearest Neighbors** | $31,691 | $32,842 | $19,773 | 0.8655 | $32,842 | 0.7s |
| 13 | **Lasso Regression** | $49,434 | $33,883 | $15,821 | 0.8568 | $4,222 | 0.7s |
| 14 | **Linear Regression** | $56,289 | $34,997 | $14,570 | 0.8472 | $18,319 | 2.4s |
| 15 | **Artificial Neural Network (MLP)** | $810,699 | $36,807 | $17,949 | 0.8310 | $15,478 | 7.7s |
| 16 | **Mean Baseline** | $77,086 | $90,222 | $64,611 | -0.0153 | $13,114 | 0.3s |
| 17 | **Neighborhood Median Baseline** | $79,308 | $94,323 | $63,838 | -0.1097 | $15,007 | 0.3s |
| 18 | **Median Baseline** | $79,308 | $94,323 | $63,838 | -0.1097 | $15,007 | 0.3s |