"""
Модуль для тестирования стратегий прогнозирования и горизонтов (Этап 3).
Реализует прямую, рекурсивную и гибридную стратегии прогнозирования,
оценивает их эффективность на разных горизонтах (h=1, 7, 30).
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

# Подавляем FutureWarning от sklearn (не критично, но засоряет вывод)
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*force_all_finite.*')

import numpy as np
import pandas as pd
import streamlit as st

from utils import parse_int_list


def prepare_direct_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    if df.empty or horizon <= 0:
        return pd.DataFrame(columns=feature_cols), pd.Series(dtype=float)

    working = df.copy()
    working["target_shift"] = working[target_col].shift(-horizon)
    working.dropna(subset=["target_shift"], inplace=True)

    if working.empty:
        return pd.DataFrame(columns=feature_cols), pd.Series(dtype=float)

    X = working[feature_cols]
    y = working["target_shift"]
    
    # Удаляем строки с NaN в признаках (X), так как модели не могут обработать NaN
    # Сохраняем индексы для синхронизации с y
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    if X.empty or y.empty:
        return pd.DataFrame(columns=feature_cols), pd.Series(dtype=float)
    
    return X, y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan}

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def recursive_forecast(
    model,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lag_cols: List[str],
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    working = df.reset_index(drop=True).copy()
    max_steps = min(horizon, len(working))

    preds: List[float] = []
    actual: List[float] = []
    
    for step in range(max_steps):
        # Проверяем наличие NaN в признаках перед прогнозированием
        # Используем pandas методы для проверки, так как могут быть нечисловые типы
        x_row = working.loc[step, feature_cols]
        if x_row.isna().any():
            # Если есть NaN, пропускаем этот шаг
            continue
        
        x = x_row.to_numpy().reshape(1, -1)
        y_hat = model.predict(x)[0]
        preds.append(float(y_hat))
        actual.append(float(working.loc[step, target_col]))

        working.loc[step, target_col] = y_hat
        for lag_col in lag_cols:
            try:
                lag_value = int(lag_col.split("_")[1])
            except (IndexError, ValueError):
                continue

            target_idx = step + lag_value
            if target_idx < len(working) and lag_col in working.columns:
                working.loc[target_idx, lag_col] = y_hat

    return np.array(preds), np.array(actual)


def evaluate_direct_strategy(
    model_factory,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    horizon: int,
) -> Dict[str, float]:
    X_train, y_train = prepare_direct_dataset(train_df, feature_cols, target_col, horizon)
    if y_train.empty:
        return {
            "status": "insufficient_data",
            "train_time": np.nan,
            "predict_time": np.nan,
            "val_mae": np.nan,
            "val_rmse": np.nan,
            "val_mape": np.nan,
            "test_mae": np.nan,
            "test_rmse": np.nan,
            "test_mape": np.nan,
        }

    model = model_factory()

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    results = {
        "train_time": train_time,
        "predict_time": 0.0,
        "status": "ok",
    }

    for split_name, (X_split, y_split) in {
        "val": prepare_direct_dataset(val_df, feature_cols, target_col, horizon),
        "test": prepare_direct_dataset(test_df, feature_cols, target_col, horizon),
    }.items():
        if y_split.empty:
            results[f"{split_name}_mae"] = np.nan
            results[f"{split_name}_rmse"] = np.nan
            results[f"{split_name}_mape"] = np.nan
            continue

        start = time.perf_counter()
        y_pred = model.predict(X_split)
        predict_time = time.perf_counter() - start
        results["predict_time"] += predict_time

        metrics = compute_metrics(y_split.to_numpy(), y_pred)
        results[f"{split_name}_mae"] = metrics["mae"]
        results[f"{split_name}_rmse"] = metrics["rmse"]
        results[f"{split_name}_mape"] = metrics["mape"]

    return results


def evaluate_recursive_strategy(
    model_factory,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lag_cols: List[str],
    horizon: int,
) -> Dict[str, float]:
    if train_df.empty:
        return {
            "status": "insufficient_data",
            "train_time": np.nan,
            "predict_time": np.nan,
            "val_mae": np.nan,
            "val_rmse": np.nan,
            "val_mape": np.nan,
            "test_mae": np.nan,
            "test_rmse": np.nan,
            "test_mape": np.nan,
        }

    # Подготовка данных для обучения: удаляем строки с NaN в признаках
    train_X = train_df[feature_cols].copy()
    train_y = train_df[target_col].copy()
    
    # Удаляем строки с NaN в признаках
    valid_mask = train_X.notna().all(axis=1)
    train_X = train_X[valid_mask]
    train_y = train_y[valid_mask]
    
    if train_X.empty or train_y.empty:
        return {
            "status": "insufficient_data",
            "train_time": np.nan,
            "predict_time": np.nan,
            "val_mae": np.nan,
            "val_rmse": np.nan,
            "val_mape": np.nan,
            "test_mae": np.nan,
            "test_rmse": np.nan,
            "test_mape": np.nan,
        }
    
    model = model_factory()
    start = time.perf_counter()
    model.fit(train_X, train_y)
    train_time = time.perf_counter() - start

    results = {
        "train_time": train_time,
        "predict_time": 0.0,
        "status": "ok",
    }

    for split_name, df_split in {"val": val_df, "test": test_df}.items():
        if df_split.empty:
            results[f"{split_name}_mae"] = np.nan
            results[f"{split_name}_rmse"] = np.nan
            results[f"{split_name}_mape"] = np.nan
            continue

        # Удаляем строки с NaN в признаках перед прогнозированием
        df_split_clean = df_split.copy()
        valid_mask = df_split_clean[feature_cols].notna().all(axis=1)
        df_split_clean = df_split_clean[valid_mask]
        
        if df_split_clean.empty:
            results[f"{split_name}_mae"] = np.nan
            results[f"{split_name}_rmse"] = np.nan
            results[f"{split_name}_mape"] = np.nan
            continue

        start = time.perf_counter()
        preds, actual = recursive_forecast(
            model, df_split_clean, feature_cols, target_col, lag_cols, horizon
        )
        predict_time = time.perf_counter() - start
        results["predict_time"] += predict_time

        metrics = compute_metrics(actual, preds)
        results[f"{split_name}_mae"] = metrics["mae"]
        results[f"{split_name}_rmse"] = metrics["rmse"]
        results[f"{split_name}_mape"] = metrics["mape"]

    return results


def aggregate_hybrid_results(
    horizon: int,
    recursive_cache: Dict[int, Dict[str, float]],
    direct_cache: Dict[int, Dict[str, float]],
) -> Dict[str, float]:
    """
    Гибридная стратегия: рекурсивная для h ≤ 3, прямая для h > 3.
    """
    if horizon <= 3:
        # Для горизонтов ≤ 3 используем рекурсивную стратегию
        result = recursive_cache.get(horizon)
        if result:
            return result
    else:
        # Для горизонтов > 3 используем прямую стратегию
        result = direct_cache.get(horizon)
        if result:
            return result
    
    # Fallback: если результаты не найдены, возвращаем доступные
    if horizon <= 3:
        result = direct_cache.get(horizon)
        if result:
            return result
    else:
        result = recursive_cache.get(horizon)
        if result:
            return result
    
    # Если нет результатов ни для одной стратегии, возвращаем пустые метрики
    return {
        "status": "insufficient_data",
        "train_time": np.nan,
        "predict_time": np.nan,
        "val_mae": np.nan,
        "val_rmse": np.nan,
        "val_mape": np.nan,
        "test_mae": np.nan,
        "test_rmse": np.nan,
        "test_mape": np.nan,
    }


def stage3(
    analysis_data: Optional[Dict[str, Any]],
    lab_state: Dict[str, bool],
    model_factories: Dict[str, Callable[[], object]],
    default_horizons: List[int],
    default_split: Tuple[int, int, int],
) -> Dict[str, Any]:
    if analysis_data is None:
        analysis_data = {}


    if not lab_state.get("stage2_completed"):
        st.info("Сначала завершите этап 2, чтобы перейти к стратегиям прогнозирования.")
        return analysis_data

    features_df = analysis_data.get("features_df")
    feature_cols = analysis_data.get("feature_cols", [])
    target_feature_name = analysis_data.get("target_feature_name")

    train_df = analysis_data.get("train_df")
    val_df = analysis_data.get("val_df")
    test_df = analysis_data.get("test_df")

    if (
        features_df is None
        or features_df.empty
        or not feature_cols
        or target_feature_name is None
        or train_df is None
        or val_df is None
        or test_df is None
    ):
        st.warning("Недостаточно признаков или выборок для обучения стратегий. Перегенерируйте признаки на этапе 2.")
        return analysis_data

    horizons_defaults = analysis_data.get("selected_horizons", default_horizons)
    selected_models_default = analysis_data.get("selected_models", list(model_factories.keys()))

    lag_cols = [col for col in feature_cols if col.startswith("lag_")]
    split_defaults = analysis_data.get("split_percentages", default_split)

    with st.form("forecasting_strategies_form"):
        col_left, col_right = st.columns(2)
        with col_left:
            horizons_input = st.text_input(
                "Горизонты прогноза (через запятую)",
                value=", ".join(str(h) for h in horizons_defaults),
                help="Пример: 1, 7, 14, 30",
            )
        with col_right:
            st.markdown("**Текущие пропорции разбиения:**")
            split_col1, split_col2, split_col3 = st.columns(3)
            split_col1.metric("Train", f"{split_defaults[0]}%", help="Доля обучающей выборки")
            split_col2.metric("Validation", f"{split_defaults[1]}%", help="Доля валидационной выборки")
            split_col3.metric("Test", f"{split_defaults[2]}%", help="Доля тестовой выборки")

        st.markdown("**Выбор моделей**")
        model_options = list(model_factories.keys())
        selected_models = st.multiselect(
            "Модели для оценки",
            model_options,
            default=selected_models_default,
            help="Выберите модели, которые будут обучаться на выбранных стратегиях.",
        )

        run_stage3 = st.form_submit_button("Запустить оценку стратегий")

    if run_stage3:
        try:
            horizons = parse_int_list(horizons_input)
            if not horizons:
                raise ValueError("Не указаны горизонты.")
        except ValueError as exc:
            st.error(f"Ошибка в списке горизонтов: {exc}")
            horizons = horizons_defaults

        if not selected_models:
            st.error("Выберите хотя бы одну модель.")
            selected_models = selected_models_default

        stage3_results: List[Dict[str, Any]] = []
        recursive_cache: Dict[int, Dict[str, float]] = {}
        direct_cache: Dict[int, Dict[str, float]] = {}

        progress = st.progress(0)
        total_jobs = len(selected_models) * len(horizons)
        current_job = 0

        for model_name in selected_models:
            factory = model_factories[model_name]
            for horizon in horizons:
                status_placeholder = st.empty()
                status_placeholder.write(f"Обучение {model_name} на горизонте {horizon}")

                direct_metrics = evaluate_direct_strategy(
                    factory, train_df, val_df, test_df, feature_cols, target_feature_name, horizon
                )
                direct_cache[horizon] = direct_metrics

                recursive_metrics = evaluate_recursive_strategy(
                    factory, train_df, val_df, test_df, feature_cols, target_feature_name, lag_cols, horizon
                )
                recursive_cache[horizon] = recursive_metrics

                hybrid_metrics = aggregate_hybrid_results(horizon, recursive_cache, direct_cache)

                stage3_results.extend(
                    [
                        {
                            "model": model_name,
                            "strategy": "direct",
                            "horizon": horizon,
                            **direct_metrics,
                        },
                        {
                            "model": model_name,
                            "strategy": "recursive",
                            "horizon": horizon,
                            **recursive_metrics,
                        },
                        {
                            "model": model_name,
                            "strategy": "hybrid",
                            "horizon": horizon,
                            **hybrid_metrics,
                        },
                    ]
                )

                current_job += 1
                progress.progress(min(current_job / total_jobs, 1.0))
                status_placeholder.write(f"Завершено: {model_name} на горизонте {horizon}")

        progress.empty()
        st.success("Оценка стратегий завершена.")

        results_df = pd.DataFrame(stage3_results)
        analysis_data.update(
            {
                "stage3_results": results_df,
                "recursive_cache": recursive_cache,
                "direct_cache": direct_cache,
                "selected_horizons": horizons,
                "selected_models": selected_models,
            }
        )
        lab_state["stage3_completed"] = True
        lab_state["stage4_completed"] = False
        lab_state["stage5_completed"] = False
    else:
        results_df = analysis_data.get("stage3_results")

    stage3_results_df = analysis_data.get("stage3_results")
    if stage3_results_df is not None and not stage3_results_df.empty:
        st.markdown("#### 📊 Сводка по стратегиям")
        st.dataframe(stage3_results_df)

        st.markdown("#### 🏆 Лучшие комбинации по метрикам")
        metric_cols = ["val_mae", "val_rmse", "val_mape", "test_mae", "test_rmse", "test_mape"]
        best_results = []
        for metric in metric_cols:
            metric_df = stage3_results_df.dropna(subset=[metric])
            if metric_df.empty:
                continue
            best_row = metric_df.loc[metric_df[metric].idxmin()]
            best_results.append(
                {
                    "metric": metric,
                    "model": best_row["model"],
                    "strategy": best_row["strategy"],
                    "horizon": best_row["horizon"],
                    "value": best_row[metric],
                }
            )
        if best_results:
            st.table(pd.DataFrame(best_results))
        
        # Анализ накопления ошибки по горизонтам
        st.markdown("#### 📈 Анализ накопления ошибки по горизонтам")
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Фильтруем данные по стратегиям
            for strategy in ["direct", "recursive", "hybrid"]:
                strategy_df = stage3_results_df[stage3_results_df["strategy"] == strategy].copy()
                if strategy_df.empty:
                    continue
                
                # Группируем по горизонтам и моделям
                horizons = sorted(strategy_df["horizon"].unique())
                models = strategy_df["model"].unique()
                
                if len(horizons) > 0:
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=("MAE по горизонтам", "RMSE по горизонтам", 
                                      "MAPE по горизонтам", "Время обучения и прогноза"),
                        vertical_spacing=0.12,
                        horizontal_spacing=0.1
                    )
                    
                    # Цвета для каждой модели (консистентные)
                    model_colors = {}
                    
                    # Предопределенные цвета для известных моделей
                    predefined_colors = {
                        "LinearRegression": {"main": "blue", "time": "pink"},
                        "RandomForestRegressor": {"main": "lightgreen", "time": "purple"},
                    }
                    
                    color_palette = [
                        "orange",         # Другие модели
                        "red",
                        "green",
                        "cyan",
                        "magenta",
                        "yellow",
                        "brown",
                        "gray",
                    ]
                    
                    # Назначаем цвета для каждой модели
                    color_idx = 0
                    for model in models:
                        if model in predefined_colors:
                            model_colors[model] = predefined_colors[model]
                        else:
                            # Для других моделей используем палитру
                            model_colors[model] = {
                                "main": color_palette[color_idx % len(color_palette)],
                                "time": color_palette[(color_idx + 1) % len(color_palette)],
                            }
                            color_idx += 2
                    
                    for model in models:
                        model_df = strategy_df[strategy_df["model"] == model].sort_values("horizon")
                        if model_df.empty:
                            continue
                        
                        valid_horizons = model_df["horizon"].dropna().unique()
                        valid_horizons = sorted([h for h in valid_horizons if pd.notna(h)])
                        
                        model_color = model_colors.get(model, {}).get("main", "blue")
                        model_time_color = model_colors.get(model, {}).get("time", "pink")
                        
                        # MAE
                        if "val_mae" in model_df.columns:
                            mae_values = [model_df[model_df["horizon"] == h]["val_mae"].values[0] 
                                         if len(model_df[model_df["horizon"] == h]) > 0 else np.nan 
                                         for h in valid_horizons]
                            fig.add_trace(
                                go.Scatter(
                                    x=valid_horizons, 
                                    y=mae_values, 
                                    name=f"{model} (MAE)",
                                    mode="lines+markers", 
                                    legendgroup=model,
                                    line=dict(color=model_color, width=2),
                                    marker=dict(color=model_color, size=6)
                                ),
                                row=1, col=1
                            )
                        
                        # RMSE
                        if "val_rmse" in model_df.columns:
                            rmse_values = [model_df[model_df["horizon"] == h]["val_rmse"].values[0] 
                                          if len(model_df[model_df["horizon"] == h]) > 0 else np.nan 
                                          for h in valid_horizons]
                            fig.add_trace(
                                go.Scatter(
                                    x=valid_horizons, 
                                    y=rmse_values, 
                                    name=f"{model} (RMSE)",
                                    mode="lines+markers", 
                                    legendgroup=model, 
                                    showlegend=False,
                                    line=dict(color=model_color, width=2),
                                    marker=dict(color=model_color, size=6)
                                ),
                                row=1, col=2
                            )
                        
                        # MAPE
                        if "val_mape" in model_df.columns:
                            mape_values = [model_df[model_df["horizon"] == h]["val_mape"].values[0] 
                                          if len(model_df[model_df["horizon"] == h]) > 0 else np.nan 
                                          for h in valid_horizons]
                            fig.add_trace(
                                go.Scatter(
                                    x=valid_horizons, 
                                    y=mape_values, 
                                    name=f"{model} (MAPE)",
                                    mode="lines+markers", 
                                    legendgroup=model, 
                                    showlegend=False,
                                    line=dict(color=model_color, width=2),
                                    marker=dict(color=model_color, size=6)
                                ),
                                row=2, col=1
                            )
                        
                        # Время
                        if "train_time" in model_df.columns and "predict_time" in model_df.columns:
                            train_times = [model_df[model_df["horizon"] == h]["train_time"].values[0] 
                                          if len(model_df[model_df["horizon"] == h]) > 0 else np.nan 
                                          for h in valid_horizons]
                            predict_times = [model_df[model_df["horizon"] == h]["predict_time"].values[0] 
                                            if len(model_df[model_df["horizon"] == h]) > 0 else np.nan 
                                            for h in valid_horizons]
                            fig.add_trace(
                                go.Scatter(
                                    x=valid_horizons, 
                                    y=train_times, 
                                    name=f"{model} (Train)",
                                    mode="lines+markers", 
                                    legendgroup=f"{model}_time",
                                    line=dict(color=model_time_color, width=2),
                                    marker=dict(color=model_time_color, size=6)
                                ),
                                row=2, col=2
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=valid_horizons, 
                                    y=predict_times, 
                                    name=f"{model} (Predict)",
                                    mode="lines+markers", 
                                    legendgroup=f"{model}_time", 
                                    showlegend=True,
                                    line=dict(color=model_time_color, width=2, dash="dash"),
                                    marker=dict(color=model_time_color, size=6)
                                ),
                                row=2, col=2
                            )
                    
                    fig.update_xaxes(title_text="Горизонт прогноза (h)", row=2, col=1)
                    fig.update_xaxes(title_text="Горизонт прогноза (h)", row=2, col=2)
                    fig.update_yaxes(title_text="MAE", row=1, col=1)
                    fig.update_yaxes(title_text="RMSE", row=1, col=2)
                    fig.update_yaxes(title_text="MAPE (%)", row=2, col=1)
                    fig.update_yaxes(title_text="Время (сек)", row=2, col=2)
                    fig.update_layout(
                        height=800,
                        title_text=f"Анализ стратегии: {strategy.upper()}",
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Для визуализации требуется plotly. Установите: pip install plotly")
        except Exception as e:
            st.warning(f"Ошибка при создании графиков: {e}")
        
        # Анализ и выбор лучшей стратегии
        st.markdown("#### 🎯 Анализ и рекомендация по выбору стратегии")
        
        try:
            # Группируем по стратегиям и считаем средние метрики
            strategy_comparison = []
            
            for strategy in ["direct", "recursive", "hybrid"]:
                strategy_df = stage3_results_df[stage3_results_df["strategy"] == strategy].copy()
                if strategy_df.empty:
                    continue
                
                # Вычисляем средние метрики по всем моделям и горизонтам
                avg_metrics = {
                    "strategy": strategy.upper(),
                    "val_mae": strategy_df["val_mae"].mean() if "val_mae" in strategy_df.columns else np.nan,
                    "val_rmse": strategy_df["val_rmse"].mean() if "val_rmse" in strategy_df.columns else np.nan,
                    "val_mape": strategy_df["val_mape"].mean() if "val_mape" in strategy_df.columns else np.nan,
                    "test_mae": strategy_df["test_mae"].mean() if "test_mae" in strategy_df.columns else np.nan,
                    "test_rmse": strategy_df["test_rmse"].mean() if "test_rmse" in strategy_df.columns else np.nan,
                    "test_mape": strategy_df["test_mape"].mean() if "test_mape" in strategy_df.columns else np.nan,
                    "avg_train_time": strategy_df["train_time"].mean() if "train_time" in strategy_df.columns else np.nan,
                    "avg_predict_time": strategy_df["predict_time"].mean() if "predict_time" in strategy_df.columns else np.nan,
                }
                
                # Вычисляем среднее время (обучение + прогноз)
                if not np.isnan(avg_metrics["avg_train_time"]) and not np.isnan(avg_metrics["avg_predict_time"]):
                    avg_metrics["avg_total_time"] = avg_metrics["avg_train_time"] + avg_metrics["avg_predict_time"]
                else:
                    avg_metrics["avg_total_time"] = np.nan
                
                strategy_comparison.append(avg_metrics)
            
            if strategy_comparison:
                comparison_df = pd.DataFrame(strategy_comparison)
                
                # Отображаем таблицу сравнения
                st.markdown("**📊 Сравнение стратегий (средние метрики):**")
                
                display_cols = ["strategy", "val_mae", "val_rmse", "val_mape", "test_mae", "test_rmse", "test_mape", "avg_total_time"]
                available_cols = [col for col in display_cols if col in comparison_df.columns]
                
                # Форматируем значения для отображения
                display_df = comparison_df[available_cols].copy()
                for col in ["val_mae", "val_rmse", "test_mae", "test_rmse", "avg_total_time"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
                
                for col in ["val_mape", "test_mape"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}%" if pd.notna(x) else "N/A")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Определяем лучшую стратегию по метрикам на тестовой выборке
                st.markdown("**🏆 Рекомендация по выбору стратегии:**")
                
                # Считаем взвешенную оценку (чем меньше метрики, тем лучше)
                best_strategies = {}
                
                # По тестовой выборке (основной критерий)
                for metric in ["test_mae", "test_rmse", "test_mape"]:
                    if metric in comparison_df.columns:
                        metric_df = comparison_df.dropna(subset=[metric])
                        if not metric_df.empty:
                            best_idx = metric_df[metric].idxmin()
                            best_strategy = metric_df.loc[best_idx, "strategy"]
                            best_strategies[metric] = {
                                "strategy": best_strategy,
                                "value": metric_df.loc[best_idx, metric]
                            }
                
                # Считаем голоса за каждую стратегию
                strategy_votes = {}
                for metric, result in best_strategies.items():
                    strategy = result["strategy"]
                    if strategy not in strategy_votes:
                        strategy_votes[strategy] = 0
                    strategy_votes[strategy] += 1
                
                # Определяем лучшую стратегию (по большинству голосов)
                if strategy_votes:
                    best_strategy = max(strategy_votes, key=strategy_votes.get)
                    vote_count = strategy_votes[best_strategy]
                    
                    # Отображаем результаты
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.success(f"**✅ Рекомендуемая стратегия: {best_strategy}**")
                        st.markdown(f"*Выбрана по {vote_count} из {len(best_strategies)} метрик*")
                        
                        # Обоснование
                        st.markdown("**📈 Обоснование выбора:**")
                        
                        for metric, result in best_strategies.items():
                            metric_name = metric.replace("test_", "").upper()
                            strategy = result["strategy"]
                            value = result["value"]
                            
                            if metric == "test_mae":
                                value_str = f"{value:.6f}"
                            elif metric == "test_rmse":
                                value_str = f"{value:.6f}"
                            elif metric == "test_mape":
                                value_str = f"{value:.4f}%"
                            else:
                                value_str = f"{value:.6f}"
                            
                            if strategy == best_strategy:
                                st.markdown(f"✅ **{metric_name}**: {strategy} — {value_str} (лучший результат)")
                            else:
                                st.markdown(f"   {metric_name}: {strategy} — {value_str}")
                    
                    with col2:
                        # Сравнение по времени выполнения
                        if "avg_total_time" in comparison_df.columns:
                            time_df = comparison_df.dropna(subset=["avg_total_time"])
                            if not time_df.empty:
                                fastest_idx = time_df["avg_total_time"].idxmin()
                                fastest_strategy = time_df.loc[fastest_idx, "strategy"]
                                fastest_time = time_df.loc[fastest_idx, "avg_total_time"]
                                
                                st.markdown("**⏱️ Время выполнения:**")
                                for _, row in comparison_df.iterrows():
                                    strategy_name = row["strategy"]
                                    time_val = row.get("avg_total_time", np.nan)
                                    if pd.notna(time_val):
                                        if strategy_name == fastest_strategy:
                                            st.success(f"{strategy_name}: {time_val:.4f} сек ⚡")
                                        else:
                                            st.info(f"{strategy_name}: {time_val:.4f} сек")
                
                # Дополнительные рекомендации
                st.markdown("---")
                st.markdown("**💡 Дополнительные рекомендации:**")
                
                recommendations = []
                
                # Проверяем стабильность стратегий по валидационной выборке
                if "val_mae" in comparison_df.columns:
                    val_mae_df = comparison_df.dropna(subset=["val_mae"])
                    if not val_mae_df.empty and len(val_mae_df) > 1:
                        val_mae_std = val_mae_df["val_mae"].std()
                        val_mae_mean = val_mae_df["val_mae"].mean()
                        # Коэффициент вариации для оценки стабильности
                        cv = val_mae_std / val_mae_mean if val_mae_mean > 0 else np.inf
                        if cv < 0.1:  # Порог стабильности (менее 10% вариации)
                            recommendations.append("✅ Все стратегии показывают стабильные результаты на валидационной выборке")
                        else:
                            recommendations.append("⚠️ Есть различия в стабильности стратегий на валидационной выборке")
                
                # Рекомендации в зависимости от стратегии
                if best_strategy == "DIRECT":
                    recommendations.append("📌 **Прямая стратегия** оптимальна для:")
                    recommendations.append("   - Длинных горизонтов прогноза (h > 7)")
                    recommendations.append("   - Критичных приложений, где важна максимальная точность")
                    recommendations.append("   - Сценариев, где допустимо обучение нескольких моделей")
                elif best_strategy == "RECURSIVE":
                    recommendations.append("📌 **Рекурсивная стратегия** оптимальна для:")
                    recommendations.append("   - Коротких горизонтов прогноза (h ≤ 3)")
                    recommendations.append("   - Быстрого прогнозирования с одной моделью")
                    recommendations.append("   - Сценариев с ограниченными вычислительными ресурсами")
                elif best_strategy == "HYBRID":
                    recommendations.append("📌 **Гибридная стратегия** оптимальна для:")
                    recommendations.append("   - Средних горизонтов прогноза (3 < h ≤ 7)")
                    recommendations.append("   - Баланса между точностью и вычислительной сложностью")
                    recommendations.append("   - Сценариев, требующих компромисса между качеством и скоростью")
                
                for rec in recommendations:
                    st.markdown(rec)
                
                # Сохраняем результат анализа
                analysis_data["best_strategy"] = best_strategy
                analysis_data["strategy_comparison"] = comparison_df.to_dict('records')
                
        except Exception as e:
            st.warning(f"Ошибка при анализе стратегий: {e}")
            import traceback
            with st.expander("Подробности ошибки"):
                st.code(traceback.format_exc(), language="python")

    return analysis_data


__all__ = [
    "stage3",
    "compute_metrics",
    "prepare_direct_dataset",
    "recursive_forecast",
    "evaluate_direct_strategy",
    "evaluate_recursive_strategy",
    "aggregate_hybrid_results",
]

