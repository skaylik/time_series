"""
Модуль для кросс-валидации временных рядов (Этап 4).
Реализует расширяющееся окно, скользящее окно и TimeSeriesSplit схемы,
оценивает стабильность качества моделей на разных временных интервалах.
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
from sklearn.model_selection import TimeSeriesSplit

from forecasting_strategies_horizons import compute_metrics


def expanding_window_indices(
    n_samples: int,
    min_train_size: int,
    test_size: int,
    step: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    if (
        n_samples <= 0
        or min_train_size < 1
        or test_size < 1
        or step < 1
        or min_train_size + test_size > n_samples
    ):
        return splits

    train_end = min_train_size
    while train_end + test_size <= n_samples:
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, train_end + test_size)
        splits.append((train_idx, test_idx))
        train_end += step
    return splits


def sliding_window_indices(
    n_samples: int,
    window_size: int,
    test_size: int,
    step: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    if (
        n_samples <= 0
        or window_size < 1
        or test_size < 1
        or step < 1
        or window_size + test_size > n_samples
    ):
        return splits

    split_start = window_size
    while split_start + test_size <= n_samples:
        train_idx = np.arange(split_start - window_size, split_start)
        test_idx = np.arange(split_start, split_start + test_size)
        splits.append((train_idx, test_idx))
        split_start += step
    return splits


def timeseries_split_indices(
    df_length: int,
    n_splits: int,
    max_train_size: int,
    test_size: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if (
        df_length <= 0
        or n_splits < 2
        or max_train_size < 1
        or test_size < 1
        or max_train_size + test_size > df_length
    ):
        return []

    tss = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=max_train_size,
        test_size=test_size,
    )
    return [(train_idx, test_idx) for train_idx, test_idx in tss.split(np.zeros(df_length))]


def evaluate_cv_splits(
    df: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    model_factory: Callable[[], object],
    feature_cols: List[str],
    target_col: str,
) -> pd.DataFrame:
    if not splits:
        return pd.DataFrame()

    records: List[Dict[str, object]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        # Удаляем строки с NaN в признаках перед обучением
        train_valid_mask = X_train.notna().all(axis=1) & y_train.notna()
        X_train = X_train[train_valid_mask]
        y_train = y_train[train_valid_mask]
        
        test_valid_mask = X_test.notna().all(axis=1) & y_test.notna()
        X_test = X_test[test_valid_mask]
        y_test = y_test[test_valid_mask]

        if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
            continue

        model = model_factory()
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        start = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_time = time.perf_counter() - start

        metrics = compute_metrics(y_test.to_numpy(), y_pred)
        records.append(
            {
                "fold": fold_idx,
                "train_start": train_df["datetime"].iloc[0],
                "train_end": train_df["datetime"].iloc[-1],
                "test_start": test_df["datetime"].iloc[0],
                "test_end": test_df["datetime"].iloc[-1],
                "train_size": len(train_df),
                "test_size": len(test_df),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "train_time": train_time,
                "predict_time": predict_time,
            }
        )

    return pd.DataFrame.from_records(records) if records else pd.DataFrame()


def summarize_cv_results(folds_df: pd.DataFrame) -> Dict[str, float | int]:
    if folds_df.empty:
        return {
            "mean_mae": np.nan,
            "std_mae": np.nan,
            "mean_rmse": np.nan,
            "std_rmse": np.nan,
            "mean_mape": np.nan,
            "std_mape": np.nan,
            "mean_train_time": np.nan,
            "mean_predict_time": np.nan,
            "n_folds": 0,
        }

    summary = {
        "mean_mae": float(folds_df["mae"].mean()),
        "std_mae": float(folds_df["mae"].std(ddof=1)) if len(folds_df) > 1 else 0.0,
        "mean_rmse": float(folds_df["rmse"].mean()),
        "std_rmse": float(folds_df["rmse"].std(ddof=1)) if len(folds_df) > 1 else 0.0,
        "mean_mape": float(folds_df["mape"].mean()),
        "std_mape": float(folds_df["mape"].std(ddof=1)) if len(folds_df) > 1 else 0.0,
        "mean_train_time": float(folds_df["train_time"].mean()),
        "mean_predict_time": float(folds_df["predict_time"].mean()),
        "n_folds": int(len(folds_df)),
    }
    return summary


def stage4(
    analysis_data: Optional[Dict[str, Any]],
    lab_state: Dict[str, bool],
    model_factories: Dict[str, Callable[[], object]],
) -> Dict[str, Any]:
    if analysis_data is None:
        analysis_data = {}


    if not lab_state.get("stage3_completed"):
        st.info("Сначала завершите этап 3, чтобы перейти к кросс-валидации.")
        return analysis_data

    features_df = analysis_data.get("features_df")
    feature_cols = analysis_data.get("feature_cols", [])
    target_feature_name = analysis_data.get("target_feature_name")

    if features_df is None or features_df.empty or not feature_cols or target_feature_name is None:
        st.warning("Недостаточно признаков для кросс-валидации. Пересоздайте признаки и повторите попытку.")
        return analysis_data

    features_sorted = features_df.sort_values("datetime").reset_index(drop=True)
    total_points = len(features_sorted)

    cv_defaults = analysis_data.get(
        "cv_config",
        {
            "expanding": {"min_train": max(total_points // 5, 20), "test": max(total_points // 10, 10), "step": 1},
            "sliding": {"window": max(total_points // 4, 30), "test": max(total_points // 10, 10), "step": 1},
            "tss": {
                "n_splits": 5,
                "max_train": min(365, total_points - 10) if total_points > 365 else max(total_points // 3, 30),  # По умолчанию 365, если данных достаточно
                "test": max(total_points // 10, 10),
            },
        },
    )

    with st.form("cross_validation_form"):
        st.markdown("#### ⚙️ Настройки расширяющегося окна")
        col1, col2, col3 = st.columns(3)
        min_train = col1.number_input(
            "Мин. train размер",
            min_value=10,
            max_value=max(total_points - 10, 10),
            value=int(cv_defaults["expanding"]["min_train"]),
            key="expanding_min_train",
        )
        expanding_test = col2.number_input(
            "Test размер",
            min_value=5,
            max_value=max(total_points - 10, 5),
            value=int(cv_defaults["expanding"]["test"]),
            key="expanding_test",
        )
        expanding_step = col3.number_input(
            "Шаг",
            min_value=1,
            max_value=50,
            value=int(cv_defaults["expanding"]["step"]),
            key="expanding_step",
        )

        st.markdown("#### 🪟 Настройки скользящего окна")
        col1, col2, col3 = st.columns(3)
        sliding_window = col1.number_input(
            "Размер окна",
            min_value=10,
            max_value=max(total_points - 10, 10),
            value=int(cv_defaults["sliding"]["window"]),
            key="sliding_window",
        )
        sliding_test = col2.number_input(
            "Test размер",
            min_value=5,
            max_value=max(total_points - 10, 5),
            value=int(cv_defaults["sliding"]["test"]),
            key="sliding_test",
        )
        sliding_step = col3.number_input(
            "Шаг",
            min_value=1,
            max_value=50,
            value=int(cv_defaults["sliding"]["step"]),
            key="sliding_step",
        )

        st.markdown("#### 🔁 TimeSeriesSplit")
        col1, col2, col3 = st.columns(3)
        tss_splits = col1.number_input(
            "Число сплитов",
            min_value=2,
            max_value=10,
            value=int(cv_defaults["tss"]["n_splits"]),
            key="tss_splits",
        )
        tss_max_train = col2.number_input(
            "Макс. размер train",
            min_value=10,
            max_value=max(total_points - 10, 10),
            value=int(cv_defaults["tss"]["max_train"]),
            key="tss_max_train",
        )
        tss_test = col3.number_input(
            "Test размер",
            min_value=5,
            max_value=max(total_points - 10, 5),
            value=int(cv_defaults["tss"]["test"]),
            key="tss_test",
        )

        st.markdown("#### 🧠 Выбор модели для валидации")
        model_options = list(model_factories.keys())
        default_model = analysis_data.get("cv_model_name", model_options[0] if model_options else "")
        selected_model = st.selectbox(
            "Модель для оценки",
            model_options,
            index=model_options.index(default_model) if default_model in model_options else 0,
            key="cv_model_select",
        )

        run_cv = st.form_submit_button("Запустить кросс-валидацию")

    if run_cv:
        model_factory = model_factories[selected_model]

        expanding_splits = expanding_window_indices(total_points, min_train, expanding_test, expanding_step)
        sliding_splits = sliding_window_indices(total_points, sliding_window, sliding_test, sliding_step)
        tss_splits_indices = timeseries_split_indices(total_points, tss_splits, tss_max_train, tss_test)

        results = {}
        progress = st.progress(0)
        split_types = {
            "expanding": expanding_splits,
            "sliding": sliding_splits,
            "timeseries_split": tss_splits_indices,
        }

        for idx, (split_name, splits) in enumerate(split_types.items(), start=1):
            progress.progress(idx / len(split_types))
            folds_df = evaluate_cv_splits(features_sorted, splits, model_factory, feature_cols, target_feature_name)

            if not folds_df.empty:
                summary = summarize_cv_results(folds_df)
                st.markdown(f"#### 📈 {split_name.capitalize()} ({len(folds_df)} фолдов)")
                st.dataframe(folds_df)
                
                # Красивое отображение сводки метрик
                st.markdown("**📊 Сводка метрик:**")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("MAE (среднее)", f"{summary.get('mean_mae', np.nan):.4f}", 
                             delta=f"±{summary.get('std_mae', 0):.4f}", help="Среднее абсолютное отклонение")
                    st.metric("RMSE (среднее)", f"{summary.get('mean_rmse', np.nan):.4f}", 
                             delta=f"±{summary.get('std_rmse', 0):.4f}", help="Среднеквадратичное отклонение")
                
                with metric_col2:
                    st.metric("MAPE (среднее)", f"{summary.get('mean_mape', np.nan):.2f}%", 
                             delta=f"±{summary.get('std_mape', 0):.2f}%", help="Средняя абсолютная процентная ошибка")
                    st.metric("Число фолдов", f"{summary.get('n_folds', 0)}", help="Количество фолдов")
                
                with metric_col3:
                    st.metric("Время обучения (среднее)", f"{summary.get('mean_train_time', np.nan):.4f} сек", 
                             help="Среднее время обучения модели")
                    st.metric("Время прогноза (среднее)", f"{summary.get('mean_predict_time', np.nan):.4f} сек", 
                             help="Среднее время прогнозирования")
            else:
                summary = {}
                st.warning(f"Не удалось провести кросс-валидацию для {split_name}.")

            results[split_name] = {
                "folds": folds_df,
                "summary": summary,
            }

        progress.empty()
        st.success("Кросс-валидация завершена.")

        analysis_data.update(
            {
                "cv_results": results,
                "cv_config": {
                    "expanding": {"min_train": min_train, "test": expanding_test, "step": expanding_step},
                    "sliding": {"window": sliding_window, "test": sliding_test, "step": sliding_step},
                    "tss": {"n_splits": tss_splits, "max_train": tss_max_train, "test": tss_test},
                },
                "cv_model_name": selected_model,
            }
        )
        lab_state["stage4_completed"] = True
        lab_state["stage5_completed"] = False

    if lab_state.get("stage4_completed"):
        cv_results = analysis_data.get("cv_results", {})
        
        # Визуализация динамики ошибки во времени
        st.markdown("#### 📈 Визуализация динамики ошибки во времени")
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            for name, result in cv_results.items():
                folds = result.get("folds", pd.DataFrame())
                if folds.empty:
                    continue
                
                # Проверяем наличие необходимых колонок
                if "test_start" not in folds.columns or "mae" not in folds.columns:
                    continue
                
                # Сортируем по дате начала теста
                folds_sorted = folds.sort_values("test_start").copy()
                
                # Создаем график с несколькими метриками
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("MAE по времени", "RMSE по времени", 
                                  "MAPE по времени", "Размеры выборок"),
                    vertical_spacing=0.12,
                    horizontal_spacing=0.1
                )
                
                # MAE
                if "mae" in folds_sorted.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=folds_sorted["test_start"],
                            y=folds_sorted["mae"],
                            mode="lines+markers",
                            name="MAE",
                            marker=dict(color="blue", size=8),
                            line=dict(color="blue", width=2)
                        ),
                        row=1, col=1
                    )
                
                # RMSE
                if "rmse" in folds_sorted.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=folds_sorted["test_start"],
                            y=folds_sorted["rmse"],
                            mode="lines+markers",
                            name="RMSE",
                            marker=dict(color="red", size=8),
                            line=dict(color="red", width=2)
                        ),
                        row=1, col=2
                    )
                
                # MAPE
                if "mape" in folds_sorted.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=folds_sorted["test_start"],
                            y=folds_sorted["mape"],
                            mode="lines+markers",
                            name="MAPE",
                            marker=dict(color="green", size=8),
                            line=dict(color="green", width=2)
                        ),
                        row=2, col=1
                    )
                
                # Размеры выборок
                if "train_size" in folds_sorted.columns and "test_size" in folds_sorted.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=folds_sorted["test_start"],
                            y=folds_sorted["train_size"],
                            mode="lines+markers",
                            name="Train размер",
                            marker=dict(color="orange", size=8),
                            line=dict(color="orange", width=2)
                        ),
                        row=2, col=2
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=folds_sorted["test_start"],
                            y=folds_sorted["test_size"],
                            mode="lines+markers",
                            name="Test размер",
                            marker=dict(color="purple", size=8),
                            line=dict(color="purple", width=2)
                        ),
                        row=2, col=2
                    )
                
                # Добавляем среднюю линию для метрик
                if "mae" in folds_sorted.columns and not folds_sorted["mae"].isna().all():
                    mean_mae = folds_sorted["mae"].mean()
                    fig.add_hline(
                        y=mean_mae,
                        line_dash="dash",
                        line_color="blue",
                        annotation_text=f"Среднее MAE: {mean_mae:.2f}",
                        row=1, col=1
                    )
                
                if "rmse" in folds_sorted.columns and not folds_sorted["rmse"].isna().all():
                    mean_rmse = folds_sorted["rmse"].mean()
                    fig.add_hline(
                        y=mean_rmse,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Среднее RMSE: {mean_rmse:.2f}",
                        row=1, col=2
                    )
                
                if "mape" in folds_sorted.columns and not folds_sorted["mape"].isna().all():
                    mean_mape = folds_sorted["mape"].mean()
                    fig.add_hline(
                        y=mean_mape,
                        line_dash="dash",
                        line_color="green",
                        annotation_text=f"Среднее MAPE: {mean_mape:.2f}",
                        row=2, col=1
                    )
                
                # Обновляем подписи осей
                fig.update_xaxes(title_text="Дата начала теста", row=2, col=1)
                fig.update_xaxes(title_text="Дата начала теста", row=2, col=2)
                fig.update_yaxes(title_text="MAE", row=1, col=1)
                fig.update_yaxes(title_text="RMSE", row=1, col=2)
                fig.update_yaxes(title_text="MAPE (%)", row=2, col=1)
                fig.update_yaxes(title_text="Размер выборки", row=2, col=2)
                
                # Обновляем layout
                fig.update_layout(
                    height=800,
                    title_text=f"Динамика ошибки во времени: {name.capitalize()}",
                    showlegend=True,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except ImportError:
            st.info("Для визуализации требуется plotly. Установите: pip install plotly")
        except Exception as e:
            st.warning(f"Ошибка при создании графиков: {e}")

    return analysis_data


__all__ = [
    "stage4",
    "expanding_window_indices",
    "sliding_window_indices",
    "timeseries_split_indices",
    "evaluate_cv_splits",
    "summarize_cv_results",
]

