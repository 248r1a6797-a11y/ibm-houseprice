"""Page 7 — Interactive Machine Learning Model Laboratory & Benchmark Tournament."""
import time
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.components.header import render_header
from src.features.preprocessing import build_preprocessing_pipeline
from src.models.registry import build_full_pipeline, get_all_candidate_models
from src.models.linear_models import get_linear_models
from src.models.tree_models import get_tree_models
from src.models.boosting_models import get_boosting_models

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def render_page():
    render_header(
        title="Interactive Model Laboratory & Tournament",
        subtitle="Benchmark multiple candidate architectures across K-Fold cross validation and select the production champion from empirical evidence.",
        badge="BENCHMARK LAYER"
    )

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    split_config = st.session_state.split_config

    # Auto-initialize experiment partitions if not yet committed
    if X_train is None or y_train is None:
        df_src = st.session_state.feature_df if not st.session_state.feature_df.empty else (
            st.session_state.clean_df if not st.session_state.clean_df.empty else st.session_state.raw_df
        )
        target_col = st.session_state.target_col
        sel_features = st.session_state.get("selected_features", [])

        if df_src.empty or not target_col or target_col not in df_src.columns:
            st.warning("No dataset or target available. Please visit 1. Data Workspace first.")
            return

        if not sel_features:
            sel_features = [c for c in df_src.columns if c != target_col]
            st.session_state.selected_features = sel_features

        y = pd.to_numeric(df_src[target_col], errors="coerce")
        valid_idx = y.dropna().index
        if len(valid_idx) < 10:
            st.warning(f"Target column `{target_col}` does not contain sufficient numeric data for training.")
            return

        df_valid = df_src.loc[valid_idx]
        y_valid = y.loc[valid_idx]

        X = df_valid[[c for c in sel_features if c in df_valid.columns]].copy()
        clean_cols = [c.replace("[", "").replace("]", "").replace("<", "").replace(",", "_") for c in X.columns]
        X.columns = clean_cols

        X_tr, X_te, y_tr, y_te = train_test_split(X, y_valid, test_size=0.20, random_state=42)
        st.session_state.X_train = X_tr
        st.session_state.X_test = X_te
        st.session_state.y_train = y_tr
        st.session_state.y_test = y_te
        X_train, X_test, y_train, y_test = X_tr, X_te, y_tr, y_te

    st.markdown("<div class='card-section'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🎯 Candidate Model Selection</div>", unsafe_allow_html=True)
    st.caption("Select candidate algorithms to evaluate in the benchmark tournament:")

    rnd = split_config.get("random_state", 42)
    all_available = {
        "Ridge Regression": get_linear_models(rnd).get("Ridge Regression"),
        "Lasso Regression": get_linear_models(rnd).get("Lasso Regression"),
        "Decision Tree": get_tree_models(rnd).get("Decision Tree"),
        "Random Forest": get_tree_models(rnd).get("Random Forest"),
        "Extra Trees": get_tree_models(rnd).get("Extra Trees"),
        "Gradient Boosting": get_boosting_models(rnd).get("Gradient Boosting"),
        "HistGradientBoosting": get_boosting_models(rnd).get("Hist Gradient Boosting")
    }

    boosting = get_boosting_models(rnd)
    if "XGBoost" in boosting:
        all_available["XGBoost"] = boosting["XGBoost"]
    if "LightGBM" in boosting:
        all_available["LightGBM"] = boosting["LightGBM"]

    col1, col2, col3, col4 = st.columns(4)
    selected_models = []
    
    model_keys = list(all_available.keys())
    for idx, key in enumerate(model_keys):
        col = [col1, col2, col3, col4][idx % 4]
        with col:
            # Default check top performers
            default_check = key in ["XGBoost", "Random Forest", "Gradient Boosting", "Ridge Regression", "HistGradientBoosting"]
            if st.checkbox(key, value=default_check, key=f"sel_m_{key}"):
                selected_models.append(key)

    st.markdown("---")
    
    if st.button("🚀 RUN MODEL BENCHMARK TOURNAMENT", type="primary", disabled=len(selected_models) == 0):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Sanitize column names for gradient boosting algorithms
        X_train_clean = X_train.copy()
        X_test_clean = X_test.copy()
        clean_cols = [c.replace("[", "").replace("]", "").replace("<", "").replace(",", "_") for c in X_train.columns]
        X_train_clean.columns = clean_cols
        X_test_clean.columns = clean_cols

        num_cols = X_train_clean.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X_train_clean.columns if c not in num_cols]
        preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
        use_log = split_config.get("use_log_target", True)

        results = []
        trained_pipelines = {}

        cv_folds = split_config.get("cv_folds", 5)
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=rnd)

        for idx, m_name in enumerate(selected_models):
            status_text.text(f"Evaluating {m_name} ({idx+1}/{len(selected_models)})...")
            base_estimator = all_available[m_name]
            full_pipe = build_full_pipeline(base_estimator, preprocessor, use_log_target=use_log)

            t0 = time.perf_counter()
            cv_scores = cross_validate(
                full_pipe,
                X_train_clean,
                y_train,
                cv=kf,
                scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
                n_jobs=-1
            )
            train_time = time.perf_counter() - t0

            # Fit on full training set and evaluate on holdout test partition
            full_pipe.fit(X_train_clean, y_train)
            test_preds = full_pipe.predict(X_test_clean)
            test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
            test_mae = float(mean_absolute_error(y_test, test_preds))
            test_r2 = float(r2_score(y_test, test_preds))

            trained_pipelines[m_name] = full_pipe

            results.append({
                "Model": m_name,
                "CV_RMSE": float(-cv_scores["test_neg_root_mean_squared_error"].mean()),
                "CV_RMSE_Std": float(cv_scores["test_neg_root_mean_squared_error"].std()),
                "CV_MAE": float(-cv_scores["test_neg_mean_absolute_error"].mean()),
                "CV_R2": float(cv_scores["test_r2"].mean()),
                "Test_RMSE": test_rmse,
                "Test_MAE": test_mae,
                "Test_R2": test_r2,
                "Train_Time_Sec": round(train_time, 2)
            })

            progress_bar.progress((idx + 1) / len(selected_models))

        status_text.text("Benchmark complete! Crowned champion model from empirical evidence.")
        
        bench_df = pd.DataFrame(results).sort_values(by="CV_RMSE", ascending=True).reset_index(drop=True)
        st.session_state.benchmark_results = bench_df
        
        # Champion is the model with lowest CV RMSE
        champ_row = bench_df.iloc[0]
        champ_pipe = trained_pipelines[champ_row["Model"]]
        st.session_state.champion_name = champ_row["Model"]
        st.session_state.champion_model = champ_pipe
        
        # Calculate residual quantiles for prediction intervals
        holdout_residuals = y_test.values - champ_pipe.predict(X_test_clean)
        lower_residual = float(np.percentile(holdout_residuals, 10))
        upper_residual = float(np.percentile(holdout_residuals, 90))

        champ_metadata = {
            "model_name": champ_row["Model"],
            "model_version": "1.0.0",
            "metrics": {
                "holdout_test": {
                    "RMSE": champ_row["Test_RMSE"],
                    "MAE": champ_row["Test_MAE"],
                    "R2": champ_row["Test_R2"]
                },
                "cv_rmse": champ_row["CV_RMSE"],
                "cv_r2": champ_row["CV_R2"]
            },
            "prediction_intervals": {
                "lower_residual_offset": lower_residual,
                "upper_residual_offset": upper_residual,
                "coverage": 0.80
            },
            "features_count": len(X_train.columns),
            "numerical_features": num_cols,
            "categorical_features": cat_cols,
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
        st.session_state.champion_metadata = champ_metadata

        # Save to disk as active production model
        try:
            prod_dir = ROOT_DIR / "models/production"
            prod_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(champ_pipe, prod_dir / "production_model.joblib")
            with open(prod_dir / "model_metadata.json", "w", encoding="utf-8") as f:
                json.dump(champ_metadata, f, indent=2)
        except Exception as e:
            st.warning(f"Could not persist model to disk: {e}")

        # Append to experiment tracker
        st.session_state.experiment_history.append({
            "experiment_id": f"EXP-{len(st.session_state.experiment_history)+1:03d}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
            "model": champ_row["Model"],
            "features_count": len(X_train.columns),
            "cv_rmse": float(champ_row["CV_RMSE"]),
            "cv_r2": float(champ_row["CV_R2"]),
            "test_rmse": float(champ_row["Test_RMSE"]),
            "test_r2": float(champ_row["Test_R2"]),
            "status": "Tournament Champion"
        })

        st.session_state.pipeline_status["TRAINING"] = "✓ Completed"
        st.session_state.pipeline_status["EVALUATION"] = "✓ Evaluated"
        st.session_state.pipeline_status["EXPLAINABILITY"] = "✓ Available"
        st.session_state.pipeline_status["PRODUCTION"] = f"✓ {champ_row['Model']} Ready"
        st.session_state.needs_retraining = False

        st.success(f"🏆 Champion Crowned: **{champ_row['Model']}** (Test RMSE: ${champ_row['Test_RMSE']:,.0f} | Test R²: {champ_row['Test_R2']:.4f})")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Render Benchmark Leaderboard
    bench_df = st.session_state.get("benchmark_results")
    if bench_df is not None and not bench_df.empty:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🏆 Model Tournament Leaderboard</div>", unsafe_allow_html=True)

        st.dataframe(
            bench_df.style.format({
                "CV_RMSE": "${:,.0f}",
                "CV_RMSE_Std": "±${:,.0f}",
                "CV_MAE": "${:,.0f}",
                "CV_R2": "{:.3f}",
                "Test_RMSE": "${:,.0f}",
                "Test_MAE": "${:,.0f}",
                "Test_R2": "{:.3f}",
                "Train_Time_Sec": "{:.2f}s"
            }),
            use_container_width=True
        )

        fig_comp = px.bar(
            bench_df,
            x="Model",
            y="Test_RMSE",
            color="Test_RMSE",
            color_continuous_scale="Blues_r",
            labels={"Test_RMSE": "Holdout Test RMSE ($) [Lower is Better]"},
            title="Algorithm Comparison by Generalization Error"
        )
        fig_comp.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
