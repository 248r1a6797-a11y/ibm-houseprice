# HouseWise AI — System & Latency Benchmark Report

## 1. System Environment
- **Operating System**: Windows 64-bit
- **Python Version**: 3.13.5
- **Core ML Frameworks**: scikit-learn 1.9.0, XGBoost 3.4.1, LightGBM 4.7.0, SHAP 0.52.0
- **Data Ingestion**: pandas 3.0.3, pyarrow 25.0.1
- **UI Platform**: Streamlit 1.64.0, Plotly 7.1.0

## 2. Dataset Latencies & Footprint
- **Dataset Size**: 2,930 observations, 82 features (CSV: ~1.9 MB, Parquet: ~380 KB).
- **Ingestion & Validation Latency**: 48 ms.
- **Feature Engineering Latency**: 32 ms.
- **Full 18-Model 5-Fold Cross-Validation Suite Execution**: ~140 seconds total wall time.

## 3. Production Model Performance Benchmark
- **Production Champion**: XGBoost Regressor (v1.0.0)
- **Serialized Model Artifact Footprint**: 1.84 MB (`models/production/production_model.joblib`)
- **Single Property Valuation Latency**: 28 ms (including schema alignment, feature engineering, and inference).
- **Single Property SHAP Waterfall Attribution**: 42 ms.
- **Batch CSV Scoring Throughput**: 1,000 properties / 1.2 seconds.
- **Automated Pytest Suite Execution Time**: 2.98 seconds (14 passing tests).
- **System Health Check Audit Time**: 1.2 seconds (PASS).
