"""Page 8 — Interactive Hyperparameter Optimization & Fine-Tuning."""
import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

from app.components.header import render_header
from src.features.preprocessing import build_preprocessing_pipeline
from src.models.registry import build_full_pipeline
from src.models.tree_models import get_tree_models
from src.models.boosting_models import get_boosting_models
from src.models.linear_models import get_linear_models

def render_page():
    render_header(
        title="Interactive Hyperparameter Optimization Studio",
        subtitle="Fine-tune architecture hyperparameters via cross-validated random search or grid search to maximize generalization.",
        badge="TUNING LAYER"
    )

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    split_config = st.session_state.split_config

    if X_train is None or y_train is None:
        from sklearn.model_selection import train_test_split
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
            st.warning(f"Target column `{target_col}` does not contain sufficient numeric data for tuning.")
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
    st.markdown("<div class='card-title'>⚙️ Tuning Configuration & Search Space</div>", unsafe_allow_html=True)

    tune_model_name = st.selectbox(
        "Select Model Architecture to Optimize:",
        ["XGBoost", "Random Forest", "Gradient Boosting", "Ridge Regression"]
    )

    col1, col2 = st.columns(2)
    with col1:
        search_method = st.radio("Search Strategy:", ["RandomizedSearchCV", "GridSearchCV"], horizontal=True)
    with col2:
        n_iters = st.slider("Search Iterations (for RandomizedSearch):", 4, 20, 8) if search_method == "RandomizedSearchCV" else 0

    st.markdown("##### Parameter Search Grid Bounds")
    param_grid = {}

    if tune_model_name == "XGBoost":
        c1, c2, c3 = st.columns(3)
        with c1:
            n_est = st.multiselect("n_estimators:", [100, 200, 300, 500], default=[100, 200])
            param_grid["regressor__regressor__n_estimators"] = n_est
        with c2:
            lr = st.multiselect("learning_rate:", [0.03, 0.05, 0.08, 0.1, 0.15], default=[0.05, 0.08])
            param_grid["regressor__regressor__learning_rate"] = lr
        with c3:
            depth = st.multiselect("max_depth:", [3, 4, 5, 6, 8], default=[3, 4, 6])
            param_grid["regressor__regressor__max_depth"] = depth

    elif tune_model_name == "Random Forest":
        c1, c2, c3 = st.columns(3)
        with c1:
            n_est = st.multiselect("n_estimators:", [100, 150, 200], default=[100, 200])
            param_grid["regressor__regressor__n_estimators"] = n_est
        with c2:
            depth = st.multiselect("max_depth:", [10, 15, 20, None], default=[10, 15])
            param_grid["regressor__regressor__max_depth"] = depth
        with c3:
            min_leaf = st.multiselect("min_samples_leaf:", [1, 2, 4], default=[1, 2])
            param_grid["regressor__regressor__min_samples_leaf"] = min_leaf

    elif tune_model_name == "Gradient Boosting":
        c1, c2 = st.columns(2)
        with c1:
            n_est = st.multiselect("n_estimators:", [100, 150, 200], default=[100, 150])
            param_grid["regressor__regressor__n_estimators"] = n_est
        with c2:
            lr = st.multiselect("learning_rate:", [0.03, 0.05, 0.1], default=[0.05, 0.1])
            param_grid["regressor__regressor__learning_rate"] = lr

    elif tune_model_name == "Ridge Regression":
        alphas = st.multiselect("alpha (L2 Regularization):", [0.1, 1.0, 5.0, 10.0, 20.0, 50.0], default=[1.0, 5.0, 10.0, 20.0])
        param_grid["regressor__regressor__alpha"] = alphas

    st.markdown("---")

    if st.button("🚀 START HYPERPARAMETER TUNING", type="primary"):
        with st.spinner(f"Running cross-validated search for {tune_model_name}..."):
            t0 = time.perf_counter()

            # Build base pipeline
            num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = [c for c in X_train.columns if c not in num_cols]
            preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
            use_log = split_config.get("use_log_target", True)

            if tune_model_name == "XGBoost":
                base_est = get_boosting_models(split_config["random_state"])["XGBoost"]
            elif tune_model_name == "Random Forest":
                base_est = get_tree_models(split_config["random_state"])["Random Forest"]
            elif tune_model_name == "Gradient Boosting":
                base_est = get_boosting_models(split_config["random_state"])["Gradient Boosting"]
            else:
                base_est = get_linear_models(split_config["random_state"])["Ridge Regression"]

            pipe = build_full_pipeline(base_est, preprocessor, use_log_target=use_log)

            cv_folds = split_config.get("cv_folds", 5)
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=split_config.get("random_state", 42))

            if search_method == "RandomizedSearchCV":
                search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=param_grid,
                    n_iter=min(n_iters, np.prod([len(v) for v in param_grid.values()])),
                    scoring="neg_root_mean_squared_error",
                    cv=kf,
                    random_state=split_config.get("random_state", 42),
                    n_jobs=-1
                )
            else:
                search = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,
                    scoring="neg_root_mean_squared_error",
                    cv=kf,
                    n_jobs=-1
                )

            search.fit(X_train, y_train)
            duration = time.perf_counter() - t0

            best_estimator = search.best_estimator_
            best_cv_rmse = float(-search.best_score_)
            best_params = {k.replace("regressor__regressor__", ""): v for k, v in search.best_params_.items()}

            # Test evaluation
            test_preds = best_estimator.predict(X_test)
            test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
            test_r2 = float(r2_score(y_test, test_preds))

            # Benchmark comparison
            prev_bench = st.session_state.benchmark_results
            prev_rmse = float(prev_bench[prev_bench["Model"] == tune_model_name]["CV_RMSE"].iloc[0]) if (not prev_bench.empty and tune_model_name in prev_bench["Model"].values) else best_cv_rmse
            improvement_delta = prev_rmse - best_cv_rmse

            st.session_state.tuned_result = {
                "model": tune_model_name,
                "best_params": best_params,
                "best_cv_rmse": best_cv_rmse,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "duration": round(duration, 2),
                "improvement": round(improvement_delta, 2)
            }

            # Update champion if better
            if best_cv_rmse <= prev_rmse:
                st.session_state.champion_model = best_estimator
                st.session_state.champion_name = f"{tune_model_name} (Tuned)"
                st.session_state.champion_metadata["model_name"] = f"{tune_model_name} (Tuned)"
                st.session_state.champion_metadata["metrics"]["holdout_test"]["RMSE"] = test_rmse
                st.session_state.champion_metadata["metrics"]["holdout_test"]["R2"] = test_r2

            # Append to experiment tracker
            st.session_state.experiment_history.append({
                "experiment_id": f"EXP-TUNE-{len(st.session_state.experiment_history)+1:03d}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": f"{tune_model_name} (Tuned)",
                "features_count": len(X_train.columns),
                "cv_rmse": best_cv_rmse,
                "cv_r2": None,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "status": "Tuning Complete"
            })

            st.success(f"Tuning finished in {duration:.2f} seconds!")

    st.markdown("</div>", unsafe_allow_html=True)

    # Render Tuning Results
    res = st.session_state.get("tuned_result")
    if res:
        st.markdown("<div class='card-section'>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-title'>🏆 Optimal Hyperparameters for {res['model']}</div>", unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Optimized CV RMSE", f"${res['best_cv_rmse']:,.0f}", delta=f"-${res['improvement']:,.0f}" if res['improvement'] > 0 else "Baseline")
        k2.metric("Holdout Test RMSE", f"${res['test_rmse']:,.0f}")
        k3.metric("Explained Variance (R²)", f"{res['test_r2']:.3f}")
        k4.metric("Search Duration", f"{res['duration']}s")

        st.markdown("##### Best Parameters Identified:")
        st.json(res["best_params"])
        st.markdown("</div>", unsafe_allow_html=True)
