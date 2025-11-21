"""
Модуль для диагностики моделей временных рядов (Этап 6).
Выполняет статистические тесты: Льюнга-Бокса (автокорреляция остатков),
Бройша-Пагана (гомоскедастичность), Шапиро-Уилка (нормальность),
а также визуализацию ACF/PACF остатков и Q-Q графиков.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.stattools import acf, pacf


def ljung_box_test(
    residuals: np.ndarray,
    lags: Optional[int] = None,
    return_df: bool = True,
) -> Dict[str, Any]:
    """
    Тест Льюнга-Бокса для проверки автокорреляции остатков.
    H0: остатки не имеют автокорреляции (белый шум)
    p > 0.05 → остатки — белый шум
    """
    if len(residuals) < 10:
        return {
            "statistic": np.nan,
            "pvalue": np.nan,
            "is_white_noise": False,
            "error": "Недостаточно данных для теста (минимум 10 наблюдений)",
        }

    if lags is None:
        # Автоматический выбор числа лагов
        lags = min(10, len(residuals) // 4)

    try:
        # Используем statsmodels для теста Льюнга-Бокса
        # acorr_ljungbox возвращает DataFrame с колонками ['lb_stat', 'lb_pvalue']
        lb_result = acorr_ljungbox(residuals, lags=lags, return_df=True, boxpierce=False)
        
        if isinstance(lb_result, pd.DataFrame) and not lb_result.empty:
            # Берем последний результат (для максимального числа лагов)
            lb_stat = lb_result["lb_stat"].iloc[-1]
            lb_pvalue = lb_result["lb_pvalue"].iloc[-1]
        else:
            # Если результат в другом формате, пытаемся извлечь значения
            if isinstance(lb_result, tuple):
                if len(lb_result) >= 2:
                    lb_stat = lb_result[0]
                    lb_pvalue = lb_result[1]
                else:
                    lb_stat = lb_result[0] if len(lb_result) > 0 else np.nan
                    lb_pvalue = np.nan
            else:
                lb_stat = np.nan
                lb_pvalue = np.nan

        # Преобразуем в float, проверяя на NaN
        lb_stat = float(lb_stat) if not (isinstance(lb_stat, (float, int)) and np.isnan(lb_stat)) else np.nan
        lb_pvalue = float(lb_pvalue) if not (isinstance(lb_pvalue, (float, int)) and np.isnan(lb_pvalue)) else np.nan

        is_white_noise = lb_pvalue > 0.05 if not np.isnan(lb_pvalue) else False

        return {
            "statistic": lb_stat,
            "pvalue": lb_pvalue,
            "is_white_noise": is_white_noise,
            "lags": lags,
        }
    except Exception as e:
        return {
            "statistic": np.nan,
            "pvalue": np.nan,
            "is_white_noise": False,
            "error": str(e),
        }


def breusch_pagan_test(
    residuals: np.ndarray,
    fitted_values: np.ndarray,
) -> Dict[str, Any]:
    """
    Тест Бройша-Пагана для проверки гомоскедастичности остатков.
    H0: остатки гомоскедастичны (постоянная дисперсия)
    p > 0.05 → гомоскедастичность
    """
    if len(residuals) < 10 or len(fitted_values) < 10:
        return {
            "statistic": np.nan,
            "pvalue": np.nan,
            "is_homoscedastic": False,
            "error": "Недостаточно данных для теста (минимум 10 наблюдений)",
        }

    try:
        # Подготовка данных для теста
        # Тест Бройша-Пагана требует регрессию квадратов остатков на fitted values
        residuals_squared = residuals ** 2
        X = np.column_stack([np.ones(len(fitted_values)), fitted_values])
        
        # Используем statsmodels для теста Бройша-Пагана
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals_squared, X)
        
        is_homoscedastic = bp_pvalue > 0.05 if not np.isnan(bp_pvalue) else False

        return {
            "statistic": float(bp_stat) if not np.isnan(bp_stat) else np.nan,
            "pvalue": float(bp_pvalue) if not np.isnan(bp_pvalue) else np.nan,
            "is_homoscedastic": is_homoscedastic,
        }
    except Exception as e:
        return {
            "statistic": np.nan,
            "pvalue": np.nan,
            "is_homoscedastic": False,
            "error": str(e),
        }


def shapiro_wilk_test(residuals: np.ndarray) -> Dict[str, Any]:
    """
    Тест Шапиро-Уилка для проверки нормальности остатков.
    H0: остатки нормально распределены
    p > 0.05 → нормальность
    """
    if len(residuals) < 3:
        return {
            "statistic": np.nan,
            "pvalue": np.nan,
            "is_normal": False,
            "error": "Недостаточно данных для теста (минимум 3 наблюдения)",
        }

    # Тест Шапиро-Уилка работает только для выборок до 5000 наблюдений
    if len(residuals) > 5000:
        # Для больших выборок используем случайную подвыборку
        residuals_sample = np.random.choice(residuals, size=5000, replace=False)
    else:
        residuals_sample = residuals

    try:
        statistic, pvalue = stats.shapiro(residuals_sample)
        is_normal = pvalue > 0.05 if not np.isnan(pvalue) else False

        return {
            "statistic": float(statistic) if not np.isnan(statistic) else np.nan,
            "pvalue": float(pvalue) if not np.isnan(pvalue) else np.nan,
            "is_normal": is_normal,
            "sample_size": len(residuals_sample),
        }
    except Exception as e:
        return {
            "statistic": np.nan,
            "pvalue": np.nan,
            "is_normal": False,
            "error": str(e),
        }


def compute_acf_pacf(
    residuals: np.ndarray,
    nlags: Optional[int] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Вычисляет ACF и PACF остатков с доверительными интервалами.
    """
    if len(residuals) < 10:
        return {
            "acf": None,
            "pacf": None,
            "acf_lags": None,
            "pacf_lags": None,
            "acf_conf_int": None,
            "pacf_conf_int": None,
            "error": "Недостаточно данных для ACF/PACF (минимум 10 наблюдений)",
        }

    if nlags is None:
        nlags = min(40, len(residuals) // 4)

    try:
        # Вычисляем ACF
        acf_result = acf(residuals, nlags=nlags, alpha=alpha, fft=True)
        pacf_result = pacf(residuals, nlags=nlags, alpha=alpha)

        # Обрабатываем результаты ACF
        if isinstance(acf_result, tuple):
            acf_values = acf_result[0]
            acf_conf_int = acf_result[1] if len(acf_result) > 1 else None
        else:
            acf_values = acf_result
            acf_conf_int = None

        # Обрабатываем результаты PACF
        if isinstance(pacf_result, tuple):
            pacf_values = pacf_result[0]
            pacf_conf_int = pacf_result[1] if len(pacf_result) > 1 else None
        else:
            pacf_values = pacf_result
            pacf_conf_int = None

        # Убираем первый элемент (lag 0), который всегда равен 1
        if len(acf_values) > 1:
            acf_values = acf_values[1:]
            if acf_conf_int is not None and isinstance(acf_conf_int, np.ndarray):
                if acf_conf_int.ndim == 2 and len(acf_conf_int) > 1:
                    acf_conf_int = acf_conf_int[1:]
                elif acf_conf_int.ndim == 1 and len(acf_conf_int) > 1:
                    acf_conf_int = acf_conf_int[1:]
        
        if len(pacf_values) > 1:
            pacf_values = pacf_values[1:]
            if pacf_conf_int is not None and isinstance(pacf_conf_int, np.ndarray):
                if pacf_conf_int.ndim == 2 and len(pacf_conf_int) > 1:
                    pacf_conf_int = pacf_conf_int[1:]
                elif pacf_conf_int.ndim == 1 and len(pacf_conf_int) > 1:
                    pacf_conf_int = pacf_conf_int[1:]

        lags = np.arange(1, len(acf_values) + 1)

        return {
            "acf": acf_values,
            "pacf": pacf_values,
            "acf_lags": lags,
            "pacf_lags": lags,
            "acf_conf_int": acf_conf_int,
            "pacf_conf_int": pacf_conf_int,
        }
    except Exception as e:
        return {
            "acf": None,
            "pacf": None,
            "acf_lags": None,
            "pacf_lags": None,
            "acf_conf_int": None,
            "pacf_conf_int": None,
            "error": str(e),
        }


def diagnose_model(
    actual: pd.Series,
    forecast: pd.Series,
    residuals: Optional[np.ndarray] = None,
    lower: Optional[pd.Series] = None,
    upper: Optional[pd.Series] = None,
    model_name: str = "Model",
    horizon: int = 1,
) -> Dict[str, Any]:
    """
    Выполняет полную диагностику модели:
    - Тест Льюнга-Бокса
    - Тест Бройша-Пагана
    - Тест Шапиро-Уилка
    - ACF/PACF остатков
    """
    if residuals is None:
        # Вычисляем остатки
        residuals = (actual.values - forecast.values).flatten()
    else:
        residuals = residuals.flatten()

    # Удаляем NaN значения
    valid_mask = ~(np.isnan(residuals) | np.isnan(actual.values) | np.isnan(forecast.values))
    residuals_clean = residuals[valid_mask]
    actual_clean = actual.values[valid_mask]
    forecast_clean = forecast.values[valid_mask]

    if len(residuals_clean) < 3:
        return {
            "error": f"Недостаточно данных для диагностики (доступно {len(residuals_clean)} остатков, минимум 3)",
            "residuals": residuals_clean,
            "residual_count": len(residuals_clean),
        }

    # Тест Льюнга-Бокса (требует минимум 10 наблюдений)
    if len(residuals_clean) >= 10:
        ljung_box = ljung_box_test(residuals_clean)
    else:
        ljung_box = {
            "statistic": np.nan,
            "pvalue": np.nan,
            "is_white_noise": False,
            "error": f"Недостаточно данных для теста (доступно {len(residuals_clean)} остатков, минимум 10). Для диагностики остатков рекомендуется использовать горизонт прогнозирования не менее 10 точек или остатки из обучающей выборки.",
            "residual_count": len(residuals_clean),
        }

    # Тест Бройша-Пагана (требует минимум 10 наблюдений)
    if len(residuals_clean) >= 10:
        breusch_pagan = breusch_pagan_test(residuals_clean, forecast_clean)
    else:
        breusch_pagan = {
            "statistic": np.nan,
            "pvalue": np.nan,
            "is_homoscedastic": False,
            "error": f"Недостаточно данных для теста (доступно {len(residuals_clean)} остатков, минимум 10). Для диагностики остатков рекомендуется использовать горизонт прогнозирования не менее 10 точек или остатки из обучающей выборки.",
            "residual_count": len(residuals_clean),
        }

    # Тест Шапиро-Уилка
    shapiro_wilk = shapiro_wilk_test(residuals_clean)

    # ACF/PACF
    acf_pacf = compute_acf_pacf(residuals_clean)

    # Описательная статистика остатков
    residual_stats = {
        "mean": float(np.mean(residuals_clean)),
        "std": float(np.std(residuals_clean)),
        "min": float(np.min(residuals_clean)),
        "max": float(np.max(residuals_clean)),
        "median": float(np.median(residuals_clean)),
        "skewness": float(stats.skew(residuals_clean)),
        "kurtosis": float(stats.kurtosis(residuals_clean)),
    }

    return {
        "model_name": model_name,
        "horizon": horizon,
        "residuals": residuals_clean,
        "actual": actual_clean,
        "forecast": forecast_clean,
        "lower": lower.values[valid_mask] if lower is not None else None,
        "upper": upper.values[valid_mask] if upper is not None else None,
        "ljung_box": ljung_box,
        "breusch_pagan": breusch_pagan,
        "shapiro_wilk": shapiro_wilk,
        "acf_pacf": acf_pacf,
        "residual_stats": residual_stats,
    }


def plot_residuals_diagnostics(
    diagnostics: Dict[str, Any],
    index: Optional[pd.Index] = None,
) -> go.Figure:
    """
    Создает комплексную визуализацию диагностики остатков.
    """
    model_name = diagnostics.get("model_name", "Model")
    residuals = diagnostics.get("residuals")
    actual = diagnostics.get("actual")
    forecast = diagnostics.get("forecast")
    lower = diagnostics.get("lower")
    upper = diagnostics.get("upper")
    acf_pacf = diagnostics.get("acf_pacf", {})

    if residuals is None or len(residuals) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Недостаточно данных для визуализации", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    # Создаем subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Ряд и прогноз",
            "Остатки",
            "ACF остатков",
            "PACF остатков",
            "Q-Q plot остатков",
            "Остатки vs Прогноз (гетероскедастичность)",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Подготовка данных для визуализации
    # Создаем маску для валидных данных
    valid_mask = ~(np.isnan(residuals) | np.isnan(actual) | np.isnan(forecast))
    
    if not valid_mask.any():
        # Если все данные NaN, используем все данные
        valid_mask = np.ones(len(residuals), dtype=bool)
    
    residuals_clean = residuals[valid_mask]
    actual_clean = actual[valid_mask]
    forecast_clean = forecast[valid_mask]
    lower_clean = lower[valid_mask] if lower is not None and len(lower) == len(residuals) else None
    upper_clean = upper[valid_mask] if upper is not None and len(upper) == len(residuals) else None
    
    # Индексы для графиков
    if index is not None and len(index) == len(residuals):
        index_clean = index[valid_mask]
    else:
        index_clean = np.arange(len(residuals_clean))

    # 1. Ряд и прогноз (row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=index_clean,
            y=actual_clean,
            mode="lines+markers",
            name="Факт",
            line=dict(color="blue", width=2),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=index_clean,
            y=forecast_clean,
            mode="lines+markers",
            name="Прогноз",
            line=dict(color="red", width=2),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )
    if lower_clean is not None and upper_clean is not None:
        fig.add_trace(
            go.Scatter(
                x=index_clean,
                y=upper_clean,
                mode="lines",
                name="Верхняя граница",
                line=dict(width=0),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=index_clean,
                y=lower_clean,
                mode="lines",
                name="Доверительный интервал",
                fill="tonexty",
                fillcolor="rgba(255,0,0,0.2)",
                line=dict(width=0),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # 2. Остатки (row=1, col=2)
    fig.add_trace(
        go.Scatter(
            x=index_clean,
            y=residuals_clean,
            mode="lines+markers",
            name="Остатки",
            line=dict(color="green", width=1.5),
            marker=dict(size=4),
        ),
        row=1,
        col=2,
    )
    # Добавляем горизонтальную линию на нуле
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)

    # 3. ACF остатков (row=2, col=1)
    if acf_pacf.get("acf") is not None:
        acf_values = acf_pacf["acf"]
        acf_lags = acf_pacf.get("acf_lags", np.arange(1, len(acf_values) + 1))
        acf_conf_int = acf_pacf.get("acf_conf_int")

        fig.add_trace(
            go.Bar(
                x=acf_lags,
                y=acf_values,
                name="ACF",
                marker=dict(color="blue"),
            ),
            row=2,
            col=1,
        )

        # Добавляем доверительные интервалы
        if acf_conf_int is not None:
            try:
                acf_conf_int_array = np.asarray(acf_conf_int)
                if acf_conf_int_array.ndim == 2:
                    # Если conf_int - это массив [lower, upper] для каждого lag
                    if acf_conf_int_array.shape[0] == len(acf_values):
                        acf_lower = acf_conf_int_array[:, 0]
                        acf_upper = acf_conf_int_array[:, 1]
                    else:
                        # Если размерности не совпадают, используем значения как есть
                        acf_lower = acf_conf_int_array[0] if acf_conf_int_array.shape[0] > 0 else None
                        acf_upper = acf_conf_int_array[1] if acf_conf_int_array.shape[0] > 1 else None
                elif acf_conf_int_array.ndim == 1:
                    # Если conf_int - это одномерный массив, это может быть половина ширины интервала
                    acf_lower = -acf_conf_int_array
                    acf_upper = acf_conf_int_array
                else:
                    acf_lower = None
                    acf_upper = None
                
                if acf_lower is not None and acf_upper is not None:
                    # Ограничиваем длину до длины acf_values
                    if len(acf_lower) > len(acf_values):
                        acf_lower = acf_lower[:len(acf_values)]
                    if len(acf_upper) > len(acf_values):
                        acf_upper = acf_upper[:len(acf_values)]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=acf_lags[:len(acf_lower)],
                            y=acf_lower,
                            mode="lines",
                            name="Нижняя граница",
                            line=dict(color="red", dash="dash", width=1),
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=acf_lags[:len(acf_upper)],
                            y=acf_upper,
                            mode="lines",
                            name="Верхняя граница",
                            line=dict(color="red", dash="dash", width=1),
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )
            except Exception:
                # Если не удалось обработать доверительные интервалы, пропускаем их
                pass

        # Добавляем горизонтальную линию на нуле
        fig.add_hline(y=0, line_dash="solid", line_color="black", row=2, col=1)

    # 4. PACF остатков (row=2, col=2)
    if acf_pacf.get("pacf") is not None:
        pacf_values = acf_pacf["pacf"]
        pacf_lags = acf_pacf.get("pacf_lags", np.arange(1, len(pacf_values) + 1))
        pacf_conf_int = acf_pacf.get("pacf_conf_int")

        fig.add_trace(
            go.Bar(
                x=pacf_lags,
                y=pacf_values,
                name="PACF",
                marker=dict(color="green"),
            ),
            row=2,
            col=2,
        )

        # Добавляем доверительные интервалы
        if pacf_conf_int is not None:
            try:
                pacf_conf_int_array = np.asarray(pacf_conf_int)
                if pacf_conf_int_array.ndim == 2:
                    # Если conf_int - это массив [lower, upper] для каждого lag
                    if pacf_conf_int_array.shape[0] == len(pacf_values):
                        pacf_lower = pacf_conf_int_array[:, 0]
                        pacf_upper = pacf_conf_int_array[:, 1]
                    else:
                        # Если размерности не совпадают, используем значения как есть
                        pacf_lower = pacf_conf_int_array[0] if pacf_conf_int_array.shape[0] > 0 else None
                        pacf_upper = pacf_conf_int_array[1] if pacf_conf_int_array.shape[0] > 1 else None
                elif pacf_conf_int_array.ndim == 1:
                    # Если conf_int - это одномерный массив, это может быть половина ширины интервала
                    pacf_lower = -pacf_conf_int_array
                    pacf_upper = pacf_conf_int_array
                else:
                    pacf_lower = None
                    pacf_upper = None
                
                if pacf_lower is not None and pacf_upper is not None:
                    # Ограничиваем длину до длины pacf_values
                    if len(pacf_lower) > len(pacf_values):
                        pacf_lower = pacf_lower[:len(pacf_values)]
                    if len(pacf_upper) > len(pacf_values):
                        pacf_upper = pacf_upper[:len(pacf_values)]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=pacf_lags[:len(pacf_lower)],
                            y=pacf_lower,
                            mode="lines",
                            name="Нижняя граница",
                            line=dict(color="red", dash="dash", width=1),
                            showlegend=False,
                        ),
                        row=2,
                        col=2,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=pacf_lags[:len(pacf_upper)],
                            y=pacf_upper,
                            mode="lines",
                            name="Верхняя граница",
                            line=dict(color="red", dash="dash", width=1),
                            showlegend=False,
                        ),
                        row=2,
                        col=2,
                    )
            except Exception:
                # Если не удалось обработать доверительные интервалы, пропускаем их
                pass

        # Добавляем горизонтальную линию на нуле
        fig.add_hline(y=0, line_dash="solid", line_color="black", row=2, col=2)

    # 5. Q-Q plot остатков (row=3, col=1)
    try:
        # Теоретические квантили нормального распределения
        n_points = len(residuals_clean)
        if n_points > 0:
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n_points))
            sample_quantiles = np.sort(residuals_clean)

            if len(theoretical_quantiles) > 0 and len(sample_quantiles) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=theoretical_quantiles,
                        y=sample_quantiles,
                        mode="markers",
                        name="Q-Q plot",
                        marker=dict(color="blue", size=6),
                    ),
                    row=3,
                    col=1,
                )

                # Добавляем диагональную линию (идеальная нормальность)
                min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
                max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        name="Теоретическая линия",
                        line=dict(color="red", dash="dash", width=2),
                    ),
                    row=3,
                    col=1,
                )
    except Exception:
        pass

    # 6. Остатки vs Прогноз (проверка гетероскедастичности) (row=3, col=2)
    fig.add_trace(
        go.Scatter(
            x=forecast_clean,
            y=residuals_clean,
            mode="markers",
            name="Остатки vs Прогноз",
            marker=dict(color="purple", size=6),
        ),
        row=3,
        col=2,
    )
    # Добавляем горизонтальную линию на нуле
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=2)

    # Обновляем layout
    fig.update_xaxes(title_text="Время", row=1, col=1)
    fig.update_xaxes(title_text="Время", row=1, col=2)
    fig.update_xaxes(title_text="Lag", row=2, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=2)
    fig.update_xaxes(title_text="Теоретические квантили", row=3, col=1)
    fig.update_xaxes(title_text="Прогноз", row=3, col=2)

    fig.update_yaxes(title_text="Значение", row=1, col=1)
    fig.update_yaxes(title_text="Остатки", row=1, col=2)
    fig.update_yaxes(title_text="ACF", row=2, col=1)
    fig.update_yaxes(title_text="PACF", row=2, col=2)
    fig.update_yaxes(title_text="Выборочные квантили", row=3, col=1)
    fig.update_yaxes(title_text="Остатки", row=3, col=2)

    fig.update_layout(
        height=1200,
        title_text=f"Диагностика модели: {model_name}",
        showlegend=True,
        hovermode="closest",
    )

    return fig


def stage6(
    analysis_data: Optional[Dict[str, Any]],
    lab_state: Dict[str, bool],
) -> Dict[str, Any]:
    """
    Этап 6. Диагностика моделей
    
    Для каждой модели и каждого горизонта:
    - Тест Льюнга–Бокса (p > 0.05 → остатки — белый шум)
    - Гомоскедастичность: визуальный анализ + (опционально) Breusch–Pagan
    - Нормальность: Q-Q plot + тест Шапиро–Уилка
    - ACF/PACF остатков
    - Визуализация: ряд, прогноз, остатки, CI
    
    Диагностика обязательна для топ-3 моделей по метрикам.
    """
    if analysis_data is None:
        analysis_data = {}


    if not lab_state.get("stage5_completed"):
        st.info("Завершите этап 5, чтобы перейти к диагностике моделей.")
        return analysis_data

    # Получаем результаты прогнозирования из этапа 5
    forecast_results: List[Any] = analysis_data.get("forecast_results", [])
    if not forecast_results:
        st.warning("Не найдены результаты прогнозирования. Завершите этап 5.")
        return analysis_data

    # Диагностика всех моделей
    st.markdown("#### 🏆 Выбор моделей для диагностики")
    
    # Создаем DataFrame с метриками для сортировки
    model_metrics = []
    for result in forecast_results:
        model_metrics.append({
            "model": result.name,
            "group": result.group,
            "rmse": result.metrics.get("rmse", np.nan),
            "mae": result.metrics.get("mae", np.nan),
            "mape": result.metrics.get("mape", np.nan),
        })
    
    if not model_metrics:
        st.warning("Не найдены метрики для моделей.")
        return analysis_data
    
    metrics_df = pd.DataFrame(model_metrics)
    metrics_df = metrics_df.dropna(subset=["rmse"])
    metrics_df = metrics_df.sort_values("rmse")
    
    # Показываем таблицу метрик
    st.markdown("**Метрики всех моделей:**")
    st.dataframe(metrics_df, use_container_width=True)
    
    # Выбираем топ-3 модели для информации
    top_models_df = metrics_df.head(3)
    top_model_names = top_models_df["model"].tolist()
    
    st.markdown(f"**Топ-3 модели (по RMSE):**")
    for idx, (_, row) in enumerate(top_models_df.iterrows(), start=1):
        st.markdown(f"{idx}. **{row['model']}** ({row['group']}) - RMSE: {row['rmse']:.4f}")
    
    # Позволяем пользователю выбрать модели для диагностики (по умолчанию все модели)
    all_model_names = [result.name for result in forecast_results]
    selected_models = st.multiselect(
        "Выберите модели для диагностики (по умолчанию выбраны все модели)",
        all_model_names,
        default=all_model_names,  # По умолчанию выбираем все модели
        help="По умолчанию диагностируются все модели. Вы можете выбрать конкретные модели для ускорения.",
        key="stage6_model_selection"
    )
    
    # Кнопка для запуска диагностики
    if st.button("🔍 Запустить диагностику моделей", key="stage6_run_diagnostics"):
        if not selected_models:
            st.warning("Выберите хотя бы одну модель для диагностики.")
            return analysis_data
        
        # Фильтруем результаты для выбранных моделей
        selected_results = [result for result in forecast_results if result.name in selected_models]
        
        if not selected_results:
            st.warning("Не найдены результаты для выбранных моделей.")
            return analysis_data
        
        # Выполняем диагностику для каждой выбранной модели
        st.markdown("#### 📊 Результаты диагностики")
        
        diagnostics_results = []
        
        for result in selected_results:
            st.markdown(f"---")
            st.markdown(f"### 🔍 {result.name} ({result.group})")
            
            # Показываем метрики модели
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            mae = result.metrics.get('mae', np.nan)
            rmse = result.metrics.get('rmse', np.nan)
            mape = result.metrics.get('mape', np.nan)
            metrics_col1.metric("MAE", f"{mae:.4f}")
            metrics_col2.metric("RMSE", f"{rmse:.4f}")
            metrics_col3.metric("MAPE", f"{mape:.2f}%")
            
            # Проверяем, не равны ли метрики нулю (подозрительно для реальных данных)
            if not np.isnan(mae) and not np.isnan(rmse) and mae == 0.0 and rmse == 0.0:
                st.error("⚠️ **Внимание:** Метрики MAE и RMSE равны нулю. Это подозрительно и может указывать на: "
                        "1) Идеальное совпадение прогноза с фактическими значениями (маловероятно для реальных данных), "
                        "2) Проблему с данными (все значения одинаковые), "
                        "3) Ошибку в вычислении метрик или прогноза. "
                        "Рекомендуется проверить данные и процесс прогнозирования.")
            
            # Вычисляем остатки
            residuals = (result.actual.values - result.forecast.values).flatten()
            residual_count = len(residuals[~np.isnan(residuals)])
            
            # Проверяем, все ли остатки равны нулю
            if residual_count > 0:
                non_zero_residuals = residuals[~np.isnan(residuals)]
                if len(non_zero_residuals) > 0 and np.allclose(non_zero_residuals, 0, atol=1e-10):
                    st.warning("⚠️ **Внимание:** Все остатки равны нулю (или очень близки к нулю). "
                              "Это означает, что прогноз идеально совпадает с фактическими значениями, "
                              "что крайне маловероятно для реальных временных рядов. "
                              "Возможна проблема с данными или процессом прогнозирования.")
            
            # Показываем информацию о количестве остатков
            if residual_count < 10:
                st.warning(f"⚠️ **Внимание:** Для диагностики доступно только {residual_count} остатков (горизонт прогнозирования: {len(result.forecast)}). "
                          f"Тесты Льюнга-Бокса и Бройша-Пагана требуют минимум 10 наблюдений. "
                          f"Для полной диагностики рекомендуется использовать горизонт прогнозирования не менее 10 точек.")
            else:
                st.info(f"📊 Количество остатков для диагностики: {residual_count}")
            
            # Выполняем диагностику
            diagnostic = diagnose_model(
                actual=result.actual,
                forecast=result.forecast,
                residuals=residuals,
                lower=result.lower,
                upper=result.upper,
                model_name=result.name,
                horizon=len(result.forecast),
            )
            
            diagnostics_results.append(diagnostic)
            
            # Краткая сводка результатов тестов
            st.markdown("**📋 Краткая сводка диагностики:**")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            # Тест Льюнга-Бокса
            ljung_box = diagnostic.get("ljung_box", {})
            with summary_col1:
                if "error" in ljung_box:
                    error_msg = ljung_box.get("error", "Ошибка")
                    if "Недостаточно данных" in error_msg:
                        st.warning("🔍 Льюнга-Бокса: ⚠️ Недостаточно данных")
                    else:
                        st.error("🔍 Льюнга-Бокса: ❌ Ошибка")
                else:
                    # Проверяем, что статистика и p-value не NaN
                    statistic = ljung_box.get('statistic', np.nan)
                    pvalue = ljung_box.get('pvalue', np.nan)
                    if not np.isnan(statistic) and not np.isnan(pvalue):
                        if ljung_box.get("is_white_noise", False):
                            st.success("🔍 Льюнга-Бокса: ✅ Белый шум")
                        else:
                            st.warning("🔍 Льюнга-Бокса: ⚠️ Не белый шум")
                    else:
                        st.warning("🔍 Льюнга-Бокса: ⚠️ Недостаточно данных")
            
            # Тест Бройша-Пагана
            breusch_pagan = diagnostic.get("breusch_pagan", {})
            with summary_col2:
                if "error" in breusch_pagan:
                    error_msg = breusch_pagan.get("error", "Ошибка")
                    if "Недостаточно данных" in error_msg:
                        st.warning("📈 Бройша-Пагана: ⚠️ Недостаточно данных")
                    else:
                        st.error("📈 Бройша-Пагана: ❌ Ошибка")
                else:
                    # Проверяем, что статистика и p-value не NaN
                    statistic = breusch_pagan.get('statistic', np.nan)
                    pvalue = breusch_pagan.get('pvalue', np.nan)
                    if not np.isnan(statistic) and not np.isnan(pvalue):
                        if breusch_pagan.get("is_homoscedastic", False):
                            st.success("📈 Бройша-Пагана: ✅ Гомоскедастичность")
                        else:
                            st.warning("📈 Бройша-Пагана: ⚠️ Гетероскедастичность")
                    else:
                        st.warning("📈 Бройша-Пагана: ⚠️ Недостаточно данных")
            
            # Тест Шапиро-Уилка
            shapiro_wilk = diagnostic.get("shapiro_wilk", {})
            with summary_col3:
                if "error" in shapiro_wilk:
                    st.error("📊 Шапиро-Уилка: Ошибка")
                else:
                    if shapiro_wilk.get("is_normal", False):
                        st.success("📊 Шапиро-Уилка: ✅ Нормальность")
                    else:
                        st.warning("📊 Шапиро-Уилка: ⚠️ Ненормальность")
            
            # Детальные результаты тестов
            with st.expander("📊 Детальные результаты тестов", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                # Тест Льюнга-Бокса
                with col1:
                    st.markdown("**🔍 Тест Льюнга-Бокса**")
                    if "error" in ljung_box:
                        error_msg = ljung_box.get("error", "Ошибка")
                        if "Недостаточно данных" in error_msg:
                            st.warning(f"⚠️ {error_msg}")
                        else:
                            st.error(f"❌ {error_msg}")
                    else:
                        statistic = ljung_box.get('statistic', np.nan)
                        pvalue = ljung_box.get('pvalue', np.nan)
                        st.metric("Статистика", f"{statistic:.4f}" if not np.isnan(statistic) else "N/A")
                        st.metric("p-value", f"{pvalue:.4f}" if not np.isnan(pvalue) else "N/A")
                        if ljung_box.get('lags'):
                            st.caption(f"Лаги: {ljung_box.get('lags', 'N/A')}")
                        # Показываем результаты только если статистика и p-value не NaN
                        if not np.isnan(statistic) and not np.isnan(pvalue):
                            if ljung_box.get("is_white_noise", False):
                                st.success("✅ Остатки — белый шум (p > 0.05)")
                            else:
                                st.warning("⚠️ Остатки не являются белым шумом (p ≤ 0.05)")
                        else:
                            st.info("ℹ️ Недостаточно данных для интерпретации результатов теста")
                
                # Тест Бройша-Пагана
                with col2:
                    st.markdown("**📈 Тест Бройша-Пагана**")
                    if "error" in breusch_pagan:
                        error_msg = breusch_pagan.get("error", "Ошибка")
                        if "Недостаточно данных" in error_msg:
                            st.warning(f"⚠️ {error_msg}")
                        else:
                            st.error(f"❌ {error_msg}")
                    else:
                        statistic = breusch_pagan.get('statistic', np.nan)
                        pvalue = breusch_pagan.get('pvalue', np.nan)
                        st.metric("Статистика", f"{statistic:.4f}" if not np.isnan(statistic) else "N/A")
                        st.metric("p-value", f"{pvalue:.4f}" if not np.isnan(pvalue) else "N/A")
                        # Показываем результаты только если статистика и p-value не NaN
                        if not np.isnan(statistic) and not np.isnan(pvalue):
                            if breusch_pagan.get("is_homoscedastic", False):
                                st.success("✅ Гомоскедастичность (p > 0.05)")
                            else:
                                st.warning("⚠️ Гетероскедастичность (p ≤ 0.05)")
                        else:
                            st.info("ℹ️ Недостаточно данных для интерпретации результатов теста")
                
                # Тест Шапиро-Уилка
                with col3:
                    st.markdown("**📊 Тест Шапиро-Уилка**")
                    if "error" in shapiro_wilk:
                        st.error(f"Ошибка: {shapiro_wilk['error']}")
                    else:
                        st.metric("Статистика", f"{shapiro_wilk.get('statistic', np.nan):.4f}")
                        st.metric("p-value", f"{shapiro_wilk.get('pvalue', np.nan):.4f}")
                        if shapiro_wilk.get('sample_size'):
                            st.caption(f"Размер выборки: {shapiro_wilk.get('sample_size', 'N/A')}")
                        if shapiro_wilk.get("is_normal", False):
                            st.success("✅ Нормальность (p > 0.05)")
                        else:
                            st.warning("⚠️ Ненормальность (p ≤ 0.05)")
            
            # Описательная статистика остатков
            st.markdown("**📈 Описательная статистика остатков:**")
            residual_stats = diagnostic.get("residual_stats", {})
            if residual_stats:
                stats_col1, stats_col2, stats_col3, stats_col4, stats_col5, stats_col6 = st.columns(6)
                stats_col1.metric("Среднее", f"{residual_stats.get('mean', np.nan):.4f}")
                stats_col2.metric("Стд. отклонение", f"{residual_stats.get('std', np.nan):.4f}")
                stats_col3.metric("Медиана", f"{residual_stats.get('median', np.nan):.4f}")
                stats_col4.metric("Мин", f"{residual_stats.get('min', np.nan):.4f}")
                stats_col5.metric("Макс", f"{residual_stats.get('max', np.nan):.4f}")
                stats_col6.metric("Асимметрия", f"{residual_stats.get('skewness', np.nan):.4f}")
                
                # Эксцесс
                ex_col1, ex_col2 = st.columns([1, 3])
                with ex_col1:
                    st.metric("Эксцесс", f"{residual_stats.get('kurtosis', np.nan):.4f}", 
                             help="Эксцесс показывает 'толстохвостость' распределения. Нормальное распределение имеет эксцесс = 0")
            
            # Визуализация диагностики
            st.markdown("**📊 Визуализация диагностики:**")
            try:
                fig = plot_residuals_diagnostics(diagnostic, index=result.actual.index)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Ошибка при создании графиков диагностики: {e}")
                import traceback
                with st.expander("Подробности ошибки"):
                    st.code(traceback.format_exc(), language="python")
        
        # Сохраняем результаты диагностики
        analysis_data["diagnostics_results"] = diagnostics_results
        lab_state["stage6_completed"] = True
        
        st.success("Диагностика моделей завершена.")
    else:
        # Показываем информацию, если диагностика еще не запущена
        st.info("👆 Выберите модели и нажмите кнопку 'Запустить диагностику моделей' для начала диагностики.")
    
    return analysis_data


__all__ = [
    "stage6",
    "diagnose_model",
    "ljung_box_test",
    "breusch_pagan_test",
    "shapiro_wilk_test",
    "compute_acf_pacf",
    "plot_residuals_diagnostics",
]

