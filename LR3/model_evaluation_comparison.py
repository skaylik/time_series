"""
Модуль для оценки качества моделей и статистического сравнения (Этап 7).
Вычисляет метрики качества (MAE, RMSE, MAPE, MASE, SMAPE, R², RMSLE),
проводит тест Diebold–Mariano для сравнения моделей,
создает сравнительные таблицы и ранжирует модели по взвешенной оценке.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
except ImportError:
    acorr_ljungbox = None


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Средняя абсолютная ошибка (MAE)"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    return float(mean_absolute_error(y_true, y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Среднеквадратичная ошибка (RMSE)"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Средняя абсолютная процентная ошибка (MAPE)"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    denominator = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Симметричная средняя абсолютная процентная ошибка (SMAPE)"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.maximum(denominator, 1e-8)
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def compute_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """
    Средняя абсолютная масштабированная ошибка (MASE)
    MASE = MAE / MAE_naive, где MAE_naive - MAE наивного прогноза на обучающей выборке
    """
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_train) == 0:
        return np.nan
    
    mae_forecast = mean_absolute_error(y_true, y_pred)
    
    # Вычисляем MAE наивного прогноза на обучающей выборке
    if len(y_train) < seasonality + 1:
        # Если данных недостаточно для сезонного наивного прогноза, используем обычный наивный
        mae_naive = mean_absolute_error(y_train[1:], y_train[:-1])
    else:
        # Сезонный наивный прогноз
        mae_naive = mean_absolute_error(y_train[seasonality:], y_train[:-seasonality])
    
    if mae_naive == 0 or np.isnan(mae_naive):
        return np.nan
    
    return float(mae_forecast / mae_naive)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Коэффициент детерминации (R²)"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan


def compute_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Среднеквадратичная логарифмическая ошибка (RMSLE)
    Для лог-рядов
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    
    # Проверяем, что значения положительные
    y_true_positive = np.maximum(y_true, 1e-8)
    y_pred_positive = np.maximum(y_pred, 1e-8)
    
    try:
        log_true = np.log1p(y_true_positive)
        log_pred = np.log1p(y_pred_positive)
        return float(np.sqrt(mean_squared_error(log_true, log_pred)))
    except Exception:
        return np.nan


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    seasonality: int = 1,
) -> Dict[str, float]:
    """
    Вычисляет все метрики качества прогнозирования.
    """
    metrics = {
        "mae": compute_mae(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "mape": compute_mape(y_true, y_pred),
        "smape": compute_smape(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
        "rmsle": compute_rmsle(y_true, y_pred),
    }
    
    if y_train is not None:
        metrics["mase"] = compute_mase(y_true, y_pred, y_train, seasonality)
    else:
        metrics["mase"] = np.nan
    
    return metrics


def diebold_mariano_test(
    forecast1: np.ndarray,
    forecast2: np.ndarray,
    actual: np.ndarray,
    test: str = "two_sided",
) -> Dict[str, Any]:
    """
    Тест Diebold–Mariano для сравнения двух прогнозов.
    
    Args:
        forecast1: Прогноз первой модели
        forecast2: Прогноз второй модели
        actual: Фактические значения
        test: Тип теста ('two_sided', 'greater', 'smaller')
    
    Returns:
        Словарь с результатами теста
    """
    if len(forecast1) != len(forecast2) or len(forecast1) != len(actual):
        return {
            "statistic": np.nan,
            "pvalue": np.nan,
            "error": "Длины прогнозов и фактических значений не совпадают",
        }
    
    if len(forecast1) < 2:
        return {
            "statistic": np.nan,
            "pvalue": np.nan,
            "error": "Недостаточно данных для теста (минимум 2 наблюдения)",
        }
    
    try:
        # Вычисляем ошибки прогнозов
        error1 = actual - forecast1
        error2 = actual - forecast2
        
        # Вычисляем разность квадратов ошибок (loss function)
        loss_diff = error1 ** 2 - error2 ** 2
        
        # Реализация теста Diebold–Mariano
        # Используем формулу из литературы (Diebold & Mariano, 1995)
        mean_loss_diff = np.mean(loss_diff)
        n = len(loss_diff)
        
        # Вычисляем дисперсию разности потерь с учетом автокорреляции (Newey–West корректировка)
        # Используем формулу для долгосрочной дисперсии
        autocov = []
        max_lag = min(max(1, int(np.sqrt(n))), n - 1)  # Выбираем количество лагов
        
        # Вычисляем автоковариацию разности потерь
        for lag in range(max_lag + 1):
            if lag == 0:
                cov = np.var(loss_diff, ddof=1)
            else:
                # Автоковариация с лагом lag
                diff_centered = loss_diff - mean_loss_diff
                if len(diff_centered) > lag:
                    cov = np.mean(diff_centered[lag:] * diff_centered[:-lag])
                else:
                    cov = 0.0
            autocov.append(cov)
        
        # Вычисляем дисперсию с учетом автокорреляции (Newey–West HAC estimator)
        # Используем Bartlett kernel для весов
        variance = autocov[0]
        for lag in range(1, len(autocov)):
            # Bartlett kernel: w(l) = 1 - l / (h + 1), где h = max_lag
            weight = 1.0 - (lag / (max_lag + 1))
            variance += 2 * weight * autocov[lag]
        
        # Избегаем деления на ноль
        variance = max(abs(variance), 1e-10)
        
        # Вычисляем t-статистику
        if variance > 0:
            t_stat = mean_loss_diff / np.sqrt(variance / n)
        else:
            t_stat = 0.0
        
        # Вычисляем p-value в зависимости от типа теста
        if test == "two_sided":
            pvalue = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        elif test == "greater":
            # H0: модель 1 не лучше модели 2
            # H1: модель 1 лучше модели 2 (меньше ошибок)
            pvalue = 1 - stats.norm.cdf(t_stat)
        else:  # smaller
            # H0: модель 1 не хуже модели 2
            # H1: модель 1 хуже модели 2 (больше ошибок)
            pvalue = stats.norm.cdf(t_stat)
        
        return {
            "statistic": float(t_stat),
            "pvalue": float(pvalue),
            "test_type": test,
            "mean_loss_diff": float(mean_loss_diff),
            "variance": float(variance),
        }
    except Exception as e:
        return {
            "statistic": np.nan,
            "pvalue": np.nan,
            "error": str(e),
        }


def compute_weighted_score(
    mase: float,
    ljung_box_pvalue: float,
    weight_mase: float = 0.7,
    weight_lb: float = 0.3,
) -> float:
    """
    Вычисляет взвешенную оценку на основе MASE и p-value теста Льюнга-Бокса.
    
    Args:
        mase: MASE метрика (меньше = лучше)
        ljung_box_pvalue: p-value теста Льюнга-Бокса (больше = лучше, белый шум)
        weight_mase: Вес для MASE (по умолчанию 0.7)
        weight_lb: Вес для p-value Льюнга-Бокса (по умолчанию 0.3)
    
    Returns:
        Взвешенная оценка (меньше = лучше)
    """
    if np.isnan(mase) or np.isnan(ljung_box_pvalue):
        return np.nan
    
    # Нормализуем MASE (меньше = лучше, поэтому используем как есть)
    # Нормализуем p-value Льюнга-Бокса (больше = лучше, поэтому инвертируем: 1 - pvalue)
    # Но если p-value близко к 1, это хорошо, поэтому используем (1 - pvalue) как штраф
    mase_score = mase if not np.isnan(mase) else 1.0
    lb_penalty = (1 - ljung_box_pvalue) if not np.isnan(ljung_box_pvalue) else 1.0
    
    # Взвешенная сумма: чем меньше, тем лучше
    weighted_score = weight_mase * mase_score + weight_lb * lb_penalty
    
    return float(weighted_score)


def stage7(
    analysis_data: Optional[Dict[str, Any]],
    lab_state: Dict[str, bool],
) -> Dict[str, Any]:
    """
    Этап 7. Оценка качества и статистическое сравнение
    
    Метрики: MAE, RMSE, MAPE, MASE, SMAPE, R², RMSLE
    Тест Diebold–Mariano для сравнения моделей
    Сравнительная таблица с параметрами, метриками, p-value, временем обучения, статусом стационарности
    Ранжирование по взвешенной оценке (по MASE и p(LB))
    """
    if analysis_data is None:
        analysis_data = {}


    if not lab_state.get("stage5_completed"):
        st.info("Завершите этап 5, чтобы перейти к оценке качества моделей.")
        return analysis_data

    # Получаем результаты прогнозирования из этапа 5
    forecast_results: List[Any] = analysis_data.get("forecast_results", [])
    if not forecast_results:
        st.warning("Не найдены результаты прогнозирования. Завершите этап 5.")
        return analysis_data

    # Получаем обучающие данные для вычисления MASE
    train_series = analysis_data.get("stage5_train_series")
    if train_series is None:
        st.warning("Не найдены обучающие данные для вычисления MASE. Завершите этап 5.")
        return analysis_data

    # Получаем результаты диагностики из этапа 6 (если доступны)
    diagnostics_results = analysis_data.get("diagnostics_results", [])
    diagnostics_dict = {diag.get("model_name"): diag for diag in diagnostics_results if diag.get("model_name")}

    # Получаем информацию о стационарности из этапа 1
    # Проверяем различные источники информации о стационарности
    is_stationary = False
    stationarity_status = "⚠️ Неизвестно"
    
    # Пробуем получить из residual_analysis
    residual_analysis = analysis_data.get("residual_analysis", {})
    if residual_analysis:
        stationarity = residual_analysis.get("stationarity", {})
        if stationarity:
            adf = stationarity.get("adf", {})
            if adf and adf.get("is_stationary", False):
                is_stationary = True
                stationarity_status = "✅ Стационарен"
            else:
                stationarity_status = "⚠️ Нестационарен"

    # Подготовка данных для сравнения
    st.markdown("#### 📊 Сравнительная таблица моделей")
    
    # Определяем горизонт прогнозирования из результатов
    if forecast_results:
        current_horizon = len(forecast_results[0].forecast)
        st.info(f"Текущий горизонт прогнозирования: {current_horizon}. Для сравнения на разных горизонтах (h=1, 7, 30) запустите этап 5 с разными горизонтами.")
    else:
        current_horizon = 1
    
    # Собираем данные для сравнительной таблицы
    comparison_data = []
    
    for result in forecast_results:
        model_name = result.name
        model_group = result.group
        actual = result.actual
        forecast = result.forecast
        details = result.details or {}
        metrics = result.metrics or {}
        
        # Получаем диагностику для этой модели
        diagnostic = diagnostics_dict.get(model_name, {})
        ljung_box = diagnostic.get("ljung_box", {})
        shapiro_wilk = diagnostic.get("shapiro_wilk", {})
        
        # Горизонт прогнозирования
        horizon = len(forecast)
        
        # Вычисляем все метрики
        y_true = actual.values.flatten() if actual.values.ndim > 1 else actual.values
        y_pred = forecast.values.flatten() if forecast.values.ndim > 1 else forecast.values
        y_train = train_series.values.flatten() if train_series is not None and train_series.values.ndim > 1 else (train_series.values if train_series is not None else None)
        
        # Определяем сезонность на основе данных
        seasonality = 7 if train_series is not None and len(train_series) >= 14 else 1
        
        all_metrics = compute_all_metrics(y_true, y_pred, y_train, seasonality)
        
        # Получаем параметры модели
        model_params = {}
        if details:
            if isinstance(details, dict):
                model_params = details.copy()
            else:
                model_params = {"details": str(details)}
        
        # Получаем время обучения (если доступно)
        train_time = model_params.get("train_time", np.nan)
        predict_time = model_params.get("predict_time", np.nan)
        
        # Формируем запись для таблицы
        row_data = {
            "Модель": model_name,
            "Группа": model_group,
            "Горизонт": horizon,
            "MAE": all_metrics.get("mae", np.nan),
            "RMSE": all_metrics.get("rmse", np.nan),
            "MAPE": all_metrics.get("mape", np.nan),
            "MASE": all_metrics.get("mase", np.nan),
            "SMAPE": all_metrics.get("smape", np.nan),
            "R²": all_metrics.get("r2", np.nan),
            "RMSLE": all_metrics.get("rmsle", np.nan),
            "p-value (Ljung-Box)": ljung_box.get("pvalue", np.nan),
            "p-value (Shapiro-Wilk)": shapiro_wilk.get("pvalue", np.nan),
            "Время обучения (сек)": train_time if not np.isnan(train_time) else np.nan,
            "Время прогноза (сек)": predict_time if not np.isnan(predict_time) else np.nan,
            "Статус белый шум": "✅" if ljung_box.get("is_white_noise", False) else "⚠️",
            "Статус нормальность": "✅" if shapiro_wilk.get("is_normal", False) else "⚠️",
            "Статус стационарности": stationarity_status,
        }
        
        # Добавляем параметры модели
        if "order" in model_params:
            row_data["Порядок (order)"] = str(model_params.get("order"))
        if "seasonal_order" in model_params:
            row_data["Сезонный порядок"] = str(model_params.get("seasonal_order"))
        if "aic" in model_params:
            row_data["AIC"] = model_params.get("aic")
        if "bic" in model_params:
            row_data["BIC"] = model_params.get("bic")
        if "lambda" in model_params:
            row_data["λ (Box-Cox)"] = model_params.get("lambda")
        
        comparison_data.append(row_data)
    
    # Создаем DataFrame для сравнения
    if not comparison_data:
        st.warning("Не удалось собрать данные для сравнения.")
        return analysis_data
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if comparison_df.empty:
        st.warning("Не удалось собрать данные для сравнения.")
        return analysis_data
    
    # Вычисляем взвешенную оценку
    if "MASE" in comparison_df.columns and "p-value (Ljung-Box)" in comparison_df.columns:
        comparison_df["Взвешенная оценка"] = comparison_df.apply(
            lambda row: compute_weighted_score(
                row.get("MASE", np.nan),
                row.get("p-value (Ljung-Box)", np.nan),
            ),
            axis=1
        )
        
        # Ранжируем по взвешенной оценке (только для валидных значений)
        valid_weighted = comparison_df["Взвешенная оценка"].notna()
        if valid_weighted.any():
            comparison_df["Ранг"] = comparison_df["Взвешенная оценка"].rank(ascending=True, method="min", na_option="bottom")
        else:
            # Если все значения nan, ранжируем по RMSE
            if "RMSE" in comparison_df.columns:
                comparison_df["Ранг"] = comparison_df["RMSE"].rank(ascending=True, method="min")
            else:
                comparison_df["Ранг"] = range(1, len(comparison_df) + 1)
    
    # Сортируем по взвешенной оценке (если доступна и есть валидные значения)
    if "Взвешенная оценка" in comparison_df.columns:
        valid_weighted = comparison_df["Взвешенная оценка"].notna()
        if valid_weighted.any():
            # Сортируем по взвешенной оценке (nan в конце)
            comparison_df = comparison_df.sort_values("Взвешенная оценка", na_position="last")
        else:
            # Если все значения nan, сортируем по RMSE
            if "RMSE" in comparison_df.columns:
                comparison_df = comparison_df.sort_values("RMSE", na_position="last")
    else:
        # Сортируем по RMSE
        if "RMSE" in comparison_df.columns:
            comparison_df = comparison_df.sort_values("RMSE", na_position="last")
    
    # Отображаем сравнительную таблицу
    st.markdown("**📋 Сравнительная таблица всех моделей:**")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Визуализация метрик
    st.markdown("#### 📈 Визуализация метрик")
    
    metric_options = ["MAE", "RMSE", "MAPE", "MASE", "SMAPE", "R²"]
    available_metrics = [m for m in metric_options if m in comparison_df.columns]
    
    if available_metrics:
        selected_metric = st.selectbox(
            "Выберите метрику для визуализации",
            available_metrics,
            key="stage7_metric_viz"
        )
        
        if selected_metric in comparison_df.columns:
            # Сортируем по выбранной метрике
            sorted_df = comparison_df.sort_values(selected_metric)
            
            # Создаем график
            try:
                fig = go.Figure()
                
                # Группируем по группам моделей
                for group in sorted_df["Группа"].unique():
                    group_df = sorted_df[sorted_df["Группа"] == group]
                    fig.add_trace(
                        go.Bar(
                            x=group_df["Модель"],
                            y=group_df[selected_metric],
                            name=group,
                            text=group_df[selected_metric].round(4),
                            textposition="outside",
                        )
                    )
                
                fig.update_layout(
                    title=f"Сравнение моделей по метрике {selected_metric}",
                    xaxis_title="Модель",
                    yaxis_title=selected_metric,
                    barmode="group",
                    height=500,
                    xaxis=dict(tickangle=-45),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Ошибка при создании графика: {e}")
    
    # Тест Diebold–Mariano для сравнения моделей
    st.markdown("#### 🔬 Статистическое сравнение моделей (тест Diebold–Mariano)")
    
    # Выбираем модели для сравнения
    model_names = comparison_df["Модель"].unique().tolist()
    if len(model_names) < 2:
        st.info("Недостаточно моделей для сравнения (нужно минимум 2).")
    else:
        # Выбираем базовую модель для сравнения (лучшая по взвешенной оценке или RMSE)
        if "Взвешенная оценка" in comparison_df.columns:
            # Проверяем, есть ли валидные значения взвешенной оценки
            valid_weighted = comparison_df["Взвешенная оценка"].notna()
            if valid_weighted.any():
                # Используем взвешенную оценку
                best_idx = comparison_df.loc[valid_weighted, "Взвешенная оценка"].idxmin()
                if pd.notna(best_idx):
                    best_model_row = comparison_df.loc[best_idx]
                    baseline_model = best_model_row["Модель"]
                else:
                    # Если idxmin вернул nan, используем RMSE
                    best_idx = comparison_df["RMSE"].idxmin()
                    if pd.notna(best_idx):
                        best_model_row = comparison_df.loc[best_idx]
                        baseline_model = best_model_row["Модель"]
                    else:
                        # Используем первую модель как базовую
                        baseline_model = comparison_df.iloc[0]["Модель"]
            else:
                # Все значения взвешенной оценки - nan, используем RMSE
                best_idx = comparison_df["RMSE"].idxmin()
                if pd.notna(best_idx):
                    best_model_row = comparison_df.loc[best_idx]
                    baseline_model = best_model_row["Модель"]
                else:
                    # Используем первую модель как базовую
                    baseline_model = comparison_df.iloc[0]["Модель"]
        else:
            # Используем модель с наименьшим RMSE
            best_idx = comparison_df["RMSE"].idxmin()
            if pd.notna(best_idx):
                best_model_row = comparison_df.loc[best_idx]
                baseline_model = best_model_row["Модель"]
            else:
                # Используем первую модель как базовую
                baseline_model = comparison_df.iloc[0]["Модель"]
        
        st.markdown(f"**Базовая модель для сравнения:** {baseline_model} (лучшая модель)")
        
        # Получаем прогноз базовой модели
        baseline_result = next((r for r in forecast_results if r.name == baseline_model), None)
        if baseline_result is None:
            st.warning("Не найдена базовая модель для сравнения.")
        else:
            baseline_forecast = baseline_result.forecast.values.flatten() if baseline_result.forecast.values.ndim > 1 else baseline_result.forecast.values
            baseline_actual = baseline_result.actual.values.flatten() if baseline_result.actual.values.ndim > 1 else baseline_result.actual.values
            
            # Сравниваем каждую модель с базовой
            dm_results = []
            
            for model_name in model_names:
                if model_name == baseline_model:
                    continue
                
                # Находим результат модели
                model_result = next((r for r in forecast_results if r.name == model_name), None)
                if model_result is None:
                    continue
                
                model_forecast = model_result.forecast.values.flatten() if model_result.forecast.values.ndim > 1 else model_result.forecast.values
                model_actual = model_result.actual.values.flatten() if model_result.actual.values.ndim > 1 else model_result.actual.values
                
                # Выравниваем длину прогнозов
                min_len = min(len(baseline_forecast), len(model_forecast), len(baseline_actual))
                if min_len < 2:
                    continue
                
                baseline_forecast_aligned = baseline_forecast[:min_len]
                model_forecast_aligned = model_forecast[:min_len]
                actual_aligned = baseline_actual[:min_len]
                
                # Выполняем тест Diebold–Mariano
                dm_result = diebold_mariano_test(
                    baseline_forecast_aligned,
                    model_forecast_aligned,
                    actual_aligned,
                    test="two_sided",
                )
                
                if "error" in dm_result:
                    significance = f"❌ Ошибка: {dm_result.get('error', 'Неизвестная ошибка')}"
                else:
                    pvalue = dm_result.get("pvalue", 1.0)
                    significance = "✅ Значимо" if pvalue < 0.05 else "❌ Не значимо"
                
                dm_results.append({
                    "Модель": model_name,
                    "Базовая модель": baseline_model,
                    "Статистика DM": dm_result.get("statistic", np.nan),
                    "p-value DM": dm_result.get("pvalue", np.nan),
                    "Значимость": significance,
                })
            
            if dm_results:
                dm_df = pd.DataFrame(dm_results)
                st.dataframe(dm_df, use_container_width=True, hide_index=True)
                
                # Интерпретация результатов
                st.markdown("**📝 Интерпретация теста Diebold–Mariano:**")
                st.markdown("""
                - **p-value < 0.05**: Разница в точности прогнозов статистически значима (одна модель лучше другой)
                - **p-value ≥ 0.05**: Разница в точности прогнозов не значима (модели статистически эквивалентны)
                - Тест сравнивает квадраты ошибок двух моделей
                - Тест учитывает автокорреляцию ошибок (Newey–West корректировка)
                """)
    
    # Ранжирование моделей
    st.markdown("#### 🏆 Ранжирование моделей")
    
    if "Взвешенная оценка" in comparison_df.columns:
        valid_weighted = comparison_df["Взвешенная оценка"].notna()
        if valid_weighted.any():
            st.markdown("**Ранжирование по взвешенной оценке (MASE × 0.7 + (1 - p(LB)) × 0.3):**")
            st.markdown("*Меньше = лучше (низкий MASE и высокий p-value Льюнга-Бокса)*")
            
            # Сортируем по взвешенной оценке и добавляем ранг
            ranking_df = comparison_df.copy()
            ranking_df = ranking_df.sort_values("Взвешенная оценка", na_position="last")
            ranking_df["Ранг"] = range(1, len(ranking_df) + 1)
            
            # Отображаем таблицу рангов
            rank_display_cols = ["Ранг", "Модель", "Группа", "Взвешенная оценка", "MASE", "p-value (Ljung-Box)", "RMSE", "MAE"]
            available_rank_cols = [col for col in rank_display_cols if col in ranking_df.columns]
            st.dataframe(ranking_df[available_rank_cols], use_container_width=True, hide_index=True)
            
            # Визуализация взвешенной оценки (только для валидных значений)
            try:
                fig = go.Figure()
                
                # Фильтруем только валидные значения для графика
                valid_data = ranking_df[ranking_df["Взвешенная оценка"].notna()].sort_values("Взвешенная оценка")
                
                if not valid_data.empty:
                    fig.add_trace(
                        go.Bar(
                            x=valid_data["Модель"],
                            y=valid_data["Взвешенная оценка"],
                            text=valid_data["Взвешенная оценка"].round(4),
                            textposition="outside",
                            marker=dict(
                                color=valid_data["Взвешенная оценка"],
                                colorscale="RdYlGn_r",  # Обратная шкала: зеленый = лучше
                                showscale=True,
                            ),
                        )
                    )
                    
                    fig.update_layout(
                        title="Ранжирование моделей по взвешенной оценке",
                        xaxis_title="Модель",
                        yaxis_title="Взвешенная оценка (меньше = лучше)",
                        height=500,
                        xaxis=dict(tickangle=-45),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Нет валидных данных для визуализации взвешенной оценки.")
            except Exception as e:
                st.warning(f"Ошибка при создании графика ранжирования: {e}")
        else:
            # Все значения взвешенной оценки - nan, используем RMSE
            st.info("Взвешенная оценка недоступна (все значения nan). Ранжирование выполняется по RMSE.")
            ranking_df = comparison_df.copy()
            if "RMSE" in ranking_df.columns:
                ranking_df = ranking_df.sort_values("RMSE", na_position="last")
                ranking_df["Ранг"] = range(1, len(ranking_df) + 1)
                rank_display_cols = ["Ранг", "Модель", "Группа", "RMSE", "MAE", "MAPE", "MASE"]
                available_rank_cols = [col for col in rank_display_cols if col in ranking_df.columns]
                st.dataframe(ranking_df[available_rank_cols], use_container_width=True, hide_index=True)
            else:
                st.warning("Недостаточно данных для ранжирования.")
    else:
        st.info("Взвешенная оценка недоступна. Ранжирование выполняется по RMSE.")
        ranking_df = comparison_df.copy()
        if "RMSE" in ranking_df.columns:
            ranking_df = ranking_df.sort_values("RMSE", na_position="last")
            ranking_df["Ранг"] = range(1, len(ranking_df) + 1)
            rank_display_cols = ["Ранг", "Модель", "Группа", "RMSE", "MAE", "MAPE", "MASE"]
            available_rank_cols = [col for col in rank_display_cols if col in ranking_df.columns]
            st.dataframe(ranking_df[available_rank_cols], use_container_width=True, hide_index=True)
        else:
            st.warning("Недостаточно данных для ранжирования.")
    
    # Экспорт сравнительной таблицы
    st.markdown("#### 💾 Экспорт результатов сравнения")
    
    if not comparison_df.empty:
        # Экспорт сравнительной таблицы
        comparison_csv = comparison_df.to_csv(index=False, encoding='utf-8-sig')
        comparison_filename = f"model_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        st.download_button(
            label="📥 Скачать сравнительную таблицу в CSV",
            data=comparison_csv,
            file_name=comparison_filename,
            mime="text/csv",
            help="Экспортировать сравнительную таблицу всех моделей с метриками и статистическими тестами",
            key="stage7_download_comparison_csv"
        )
        
        # Экспорт таблицы теста Diebold–Mariano (если есть)
        if len(model_names) >= 2:
            try:
                # Повторно собираем результаты DM теста для экспорта
                if "Взвешенная оценка" in comparison_df.columns:
                    valid_weighted = comparison_df["Взвешенная оценка"].notna()
                    if valid_weighted.any():
                        best_idx = comparison_df.loc[valid_weighted, "Взвешенная оценка"].idxmin()
                        if pd.notna(best_idx):
                            baseline_model = comparison_df.loc[best_idx, "Модель"]
                        else:
                            baseline_model = comparison_df.loc[comparison_df["RMSE"].idxmin(), "Модель"]
                    else:
                        baseline_model = comparison_df.loc[comparison_df["RMSE"].idxmin(), "Модель"]
                else:
                    baseline_model = comparison_df.loc[comparison_df["RMSE"].idxmin(), "Модель"]
                
                baseline_result = next((r for r in forecast_results if r.name == baseline_model), None)
                if baseline_result:
                    baseline_forecast = baseline_result.forecast.values.flatten() if baseline_result.forecast.values.ndim > 1 else baseline_result.forecast.values
                    baseline_actual = baseline_result.actual.values.flatten() if baseline_result.actual.values.ndim > 1 else baseline_result.actual.values
                    
                    dm_export_data = []
                    for model_name in model_names:
                        if model_name == baseline_model:
                            continue
                        
                        model_result = next((r for r in forecast_results if r.name == model_name), None)
                        if model_result is None:
                            continue
                        
                        model_forecast = model_result.forecast.values.flatten() if model_result.forecast.values.ndim > 1 else model_result.forecast.values
                        model_actual = model_result.actual.values.flatten() if model_result.actual.values.ndim > 1 else model_result.actual.values
                        
                        min_len = min(len(baseline_forecast), len(model_forecast), len(baseline_actual))
                        if min_len < 2:
                            continue
                        
                        baseline_forecast_aligned = baseline_forecast[:min_len]
                        model_forecast_aligned = model_forecast[:min_len]
                        actual_aligned = baseline_actual[:min_len]
                        
                        dm_result = diebold_mariano_test(
                            baseline_forecast_aligned,
                            model_forecast_aligned,
                            actual_aligned,
                            test="two_sided",
                        )
                        
                        dm_export_data.append({
                            "Модель": model_name,
                            "Базовая_модель": baseline_model,
                            "Статистика_DM": dm_result.get("statistic", np.nan),
                            "p_value_DM": dm_result.get("pvalue", np.nan),
                            "Значимость": "Значимо" if dm_result.get("pvalue", 1) < 0.05 else "Не значимо",
                        })
                    
                    if dm_export_data:
                        dm_export_df = pd.DataFrame(dm_export_data)
                        dm_csv = dm_export_df.to_csv(index=False, encoding='utf-8-sig')
                        dm_filename = f"diebold_mariano_test_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        
                        st.download_button(
                            label="📥 Скачать результаты теста Diebold–Mariano в CSV",
                            data=dm_csv,
                            file_name=dm_filename,
                            mime="text/csv",
                            help="Экспортировать результаты статистического сравнения моделей",
                            key="stage7_download_dm_csv"
                        )
            except Exception as e:
                # Если не удалось создать экспорт DM теста, просто пропускаем
                pass
    
    # Сохраняем результаты
    analysis_data["comparison_table"] = comparison_df
    analysis_data["stage7_completed"] = True
    lab_state["stage7_completed"] = True
    
    st.success("Оценка качества и статистическое сравнение завершены.")
    
    return analysis_data


__all__ = [
    "stage7",
    "compute_all_metrics",
    "compute_mae",
    "compute_rmse",
    "compute_mape",
    "compute_smape",
    "compute_mase",
    "compute_r2",
    "compute_rmsle",
    "diebold_mariano_test",
    "compute_weighted_score",
]

