# Model Selection Report & Gate Documentation

## 1. Executive Summary
The HouseWise AI Model Selection Gate benchmarked 18 distinct regression algorithms across 6 model families using 5-fold cross-validation and a 20% holdout test partition (586 records).

## 2. Benchmark Leaderboard Summary

| Rank | Model Family | Algorithm | 5-Fold CV RMSE | Holdout Test RMSE | Holdout Test MAE | Holdout Test R² | Overfit Gap | Inference Latency |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Gradient Boosting** | **XGBoost Regressor** | **$24,949** | **$22,973** | **$13,835** | **0.9342** | **+$8,542** | **<35ms** |
| 🥈 | **Gradient Boosting** | Gradient Boosting Regressor | $24,768 | $23,288 | $13,991 | 0.9324 | +$9,102 | <40ms |
| 🥉 | **Gradient Boosting** | Hist Gradient Boosting | $24,878 | $23,484 | $14,110 | 0.9312 | +$9,820 | <30ms |
| 4 | **Ensemble / Meta** | Stacking Ensemble (Ridge Meta) | $28,432 | $23,680 | $14,240 | 0.9301 | +$10,210 | <95ms |
| 5 | **Ensemble / Meta** | Voting Ensemble | $26,125 | $24,213 | $14,480 | 0.9269 | +$9,650 | <80ms |
| 6 | **Gradient Boosting** | LightGBM Regressor (Tuned) | $24,445 | $24,245 | $14,310 | 0.9267 | +$9,410 | <30ms |
| 7 | **Tree Ensemble** | Extra Trees Regressor | $25,529 | $25,525 | $14,920 | 0.9187 | +$12,410 | <50ms |
| 8 | **Tree Ensemble** | Random Forest Regressor | $26,590 | $26,254 | $15,310 | 0.9140 | +$13,200 | <60ms |
| 9 | **Decision Tree** | Decision Tree Regressor | $35,846 | $30,811 | $18,420 | 0.8816 | +$18,900 | <15ms |
| 10 | **Linear L2** | Ridge Regression | $46,958 | $32,026 | $18,940 | 0.8721 | +$15,800 | <10ms |
| 11 | **Linear Regularized**| ElasticNet | $47,357 | $32,571 | $19,210 | 0.8677 | +$16,100 | <10ms |
| 12 | **Non-Parametric** | K-Nearest Neighbors | $31,691 | $32,842 | $19,890 | 0.8655 | +$16,400 | <25ms |
| 13 | **Linear L1** | Lasso Regression | $49,434 | $33,883 | $20,120 | 0.8568 | +$17,200 | <10ms |
| 14 | **Linear OLS** | Ordinary Least Squares | $56,289 | $34,997 | $20,850 | 0.8472 | +$19,100 | <10ms |
| 15 | **Neural Network** | MLPRegressor | $810,699 | $36,807 | $22,410 | 0.8310 | +$22,500 | <45ms |
| 16 | **Baseline 2** | Mean Baseline | $77,086 | $90,222 | $57,410 | -0.0153 | N/A | <5ms |
| 17 | **Baseline 1** | Median Baseline | $79,308 | $94,323 | $58,910 | -0.1097 | N/A | <5ms |
| 18 | **Baseline 3** | Neighborhood Median Baseline | $79,308 | $94,323 | $58,910 | -0.1097 | N/A | <5ms |

## 3. Decision Rationale
1. **Accuracy Superiority**: XGBoost demonstrated the lowest holdout test error (RMSE: $22,972.84, MAE: $13,834.55) and highest explanatory power (93.42% variance explained).
2. **Stability & Generalization**: Maintained close alignment between cross-validation RMSE ($24,949) and test RMSE ($22,973), without severe overfit spread.
3. **Interpretability**: Enables exact Shapley attribution through `TreeExplainer`, providing single-property waterfall explanations and global feature rankings.
4. **Production Efficiency**: Model artifact size is <3MB and scoring latency is <35 milliseconds, perfectly suited for real-time web inference.
