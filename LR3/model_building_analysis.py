"""
Модуль для построения и анализа моделей временных рядов (Этап 5).
Реализует классические модели (AR, MA, ARMA, ARIMA, SARIMA, SARIMAX, GARCH),
сезонные модели (TBATS, Prophet), многомерные модели (VAR, VECM),
бенчмарки (Naive, Seasonal Naive, SES) и ML модели (LinearRegression, RandomForest).

Оптимизация производительности:
- Горизонт прогнозирования по умолчанию: 10 дней (минимум для полной диагностики моделей)
- Медленные модели (TBATS, SARIMA) можно пропустить через чекбокс
- Автоматические ограничения размера обучающей выборки для ускорения
- Ограничения для каждой модели индивидуально настроены
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# Подавляем FutureWarning от sklearn (не критично, но засоряет вывод)
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*force_all_finite.*')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.special import inv_boxcox
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from forecasting_strategies_horizons import compute_metrics


@dataclass
class ForecastResult:
    name: str
    group: str
    forecast: pd.Series
    actual: pd.Series
    lower: Optional[pd.Series]
    upper: Optional[pd.Series]
    metrics: Dict[str, float]
    details: Dict[str, Any]


def apply_boxcox_transform(
    series: pd.Series,
    use_boxcox: bool,
    lambda_: Optional[float],
) -> Tuple[pd.Series, float]:
    values = series.astype(float)
    offset = 0.0
    if use_boxcox and lambda_ is not None:
        from scipy.stats import boxcox

        min_val = values.min()
        if min_val <= 0:
            offset = abs(min_val) + 1e-6
        adjusted = values + offset
        transformed_values = boxcox(adjusted, lmbda=lambda_)
        transformed_series = pd.Series(transformed_values, index=series.index, name=series.name)
        return transformed_series, offset
    return values, offset


def inverse_boxcox_values(
    values: np.ndarray | pd.Series | None,
    use_boxcox: bool,
    lambda_: Optional[float],
    offset: float,
) -> np.ndarray:
    if values is None:
        return np.array([])
    arr = np.asarray(values, dtype=float)
    if use_boxcox and lambda_ is not None:
        arr = inv_boxcox(arr, lambda_) - offset
    return arr


def finalize_forecast_result(
    name: str,
    group: str,
    actual: pd.Series,
    forecast_transformed: np.ndarray,
    lower_transformed: Optional[np.ndarray],
    upper_transformed: Optional[np.ndarray],
    use_boxcox: bool,
    lambda_: Optional[float],
    offset: float,
    details: Dict[str, Any],
) -> ForecastResult:
    forecast_original = inverse_boxcox_values(forecast_transformed, use_boxcox, lambda_, offset)
    forecast_series = pd.Series(forecast_original, index=actual.index, name="forecast")

    lower_series = (
        pd.Series(
            inverse_boxcox_values(lower_transformed, use_boxcox, lambda_, offset),
            index=actual.index,
            name="lower",
        )
        if lower_transformed is not None
        else None
    )
    upper_series = (
        pd.Series(
            inverse_boxcox_values(upper_transformed, use_boxcox, lambda_, offset),
            index=actual.index,
            name="upper",
        )
        if upper_transformed is not None
        else None
    )

    metrics = compute_metrics(actual.to_numpy(), forecast_series.to_numpy())
    return ForecastResult(
        name=name,
        group=group,
        forecast=forecast_series,
        actual=actual,
        lower=lower_series,
        upper=upper_series,
        metrics=metrics,
        details=details,
    )


def run_naive_forecast(
    train_series: pd.Series,
    horizon: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Naive прогноз: последнее значение повторяется на весь горизонт.
    """
    last_value = train_series.iloc[-1]
    forecast = np.full(horizon, float(last_value))
    return forecast, None, {"method": "Naive", "last_value": float(last_value)}


def run_seasonal_naive_forecast(
    train_series: pd.Series,
    horizon: int,
    seasonality: int = 7,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Seasonal Naive прогноз: последние значения сезона повторяются.
    """
    if len(train_series) < seasonality:
        # Если данных недостаточно, используем обычный Naive
        return run_naive_forecast(train_series, horizon)
    
    seasonal_values = train_series.iloc[-seasonality:].values
    # Повторяем сезонные значения на весь горизонт
    forecast = np.tile(seasonal_values, (horizon + seasonality - 1) // seasonality)[:horizon]
    return forecast, None, {"method": "Seasonal Naive", "seasonality": seasonality}


def run_ar_forecast(
    pm_module,
    train_series: pd.Series,
    horizon: int,
    p: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    AR(p) модель: авторегрессия порядка p.
    Использует statsmodels или pmdarima через auto_arima с фиксированными параметрами.
    """
    if pm_module is None:
        raise ImportError("pmdarima недоступен")
    
    # AR(p) = ARIMA(p, 0, 0) - используем auto_arima с фиксированными параметрами
    try:
        model = pm_module.auto_arima(
            train_series,
            start_p=p,
            max_p=p,
            start_q=0,
            max_q=0,
            d=0,
            seasonal=False,
            stepwise=False,
            suppress_warnings=True,
            error_action="ignore",
            information_criterion="aic",
        )
        
        forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
        
        # Получаем AIC/BIC из модели
        aic_value = None
        bic_value = None
        try:
            if hasattr(model, "aic"):
                aic_value = model.aic() if callable(getattr(model, "aic", None)) else model.aic
            if hasattr(model, "bic"):
                bic_value = model.bic() if callable(getattr(model, "bic", None)) else model.bic
        except Exception:
            pass
        
        details = {
            "model_type": f"AR({p})",
            "order": getattr(model, "order", (p, 0, 0)),
            "aic": aic_value,
            "bic": bic_value,
        }
        return np.asarray(forecast, dtype=float), np.asarray(conf_int, dtype=float), details
    except Exception as e:
        # Fallback: используем statsmodels если доступен
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(train_series, order=(p, 0, 0))
            fitted = model.fit()
            forecast_result = fitted.get_forecast(steps=horizon)
            forecast = forecast_result.predicted_mean.values
            conf_int = forecast_result.conf_int().values
            
            details = {
                "model_type": f"AR({p})",
                "order": (p, 0, 0),
                "aic": fitted.aic,
                "bic": fitted.bic,
            }
            return np.asarray(forecast, dtype=float), np.asarray(conf_int, dtype=float), details
        except ImportError:
            raise ImportError(f"Не удалось создать AR({p}) модель: {e}")


def run_ma_forecast(
    pm_module,
    train_series: pd.Series,
    horizon: int,
    q: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    MA(q) модель: скользящее среднее порядка q.
    """
    if pm_module is None:
        raise ImportError("pmdarima недоступен")
    
    # MA(q) = ARIMA(0, 0, q)
    try:
        model = pm_module.auto_arima(
            train_series,
            start_p=0,
            max_p=0,
            start_q=q,
            max_q=q,
            d=0,
            seasonal=False,
            stepwise=False,
            suppress_warnings=True,
            error_action="ignore",
            information_criterion="aic",
        )
        
        forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
        
        # Получаем AIC/BIC из модели
        aic_value = None
        bic_value = None
        try:
            if hasattr(model, "aic"):
                aic_value = model.aic() if callable(getattr(model, "aic", None)) else model.aic
            if hasattr(model, "bic"):
                bic_value = model.bic() if callable(getattr(model, "bic", None)) else model.bic
        except Exception:
            pass
        
        details = {
            "model_type": f"MA({q})",
            "order": getattr(model, "order", (0, 0, q)),
            "aic": aic_value,
            "bic": bic_value,
        }
        return np.asarray(forecast, dtype=float), np.asarray(conf_int, dtype=float), details
    except Exception as e:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(train_series, order=(0, 0, q))
            fitted = model.fit()
            forecast_result = fitted.get_forecast(steps=horizon)
            forecast = forecast_result.predicted_mean.values
            conf_int = forecast_result.conf_int().values
            
            details = {
                "model_type": f"MA({q})",
                "order": (0, 0, q),
                "aic": fitted.aic,
                "bic": fitted.bic,
            }
            return np.asarray(forecast, dtype=float), np.asarray(conf_int, dtype=float), details
        except ImportError:
            raise ImportError(f"Не удалось создать MA({q}) модель: {e}")


def run_arma_forecast(
    pm_module,
    train_series: pd.Series,
    horizon: int,
    p: int = 1,
    q: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    ARMA(p, q) модель: авторегрессия и скользящее среднее.
    """
    if pm_module is None:
        raise ImportError("pmdarima недоступен")
    
    # ARMA(p, q) = ARIMA(p, 0, q)
    try:
        model = pm_module.auto_arima(
            train_series,
            start_p=p,
            max_p=p,
            start_q=q,
            max_q=q,
            d=0,
            seasonal=False,
            stepwise=False,
            suppress_warnings=True,
            error_action="ignore",
            information_criterion="aic",
        )
        
        forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
        
        # Получаем AIC/BIC из модели
        aic_value = None
        bic_value = None
        try:
            if hasattr(model, "aic"):
                aic_value = model.aic() if callable(getattr(model, "aic", None)) else model.aic
            if hasattr(model, "bic"):
                bic_value = model.bic() if callable(getattr(model, "bic", None)) else model.bic
        except Exception:
            pass
        
        details = {
            "model_type": f"ARMA({p},{q})",
            "order": getattr(model, "order", (p, 0, q)),
            "aic": aic_value,
            "bic": bic_value,
        }
        return np.asarray(forecast, dtype=float), np.asarray(conf_int, dtype=float), details
    except Exception as e:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(train_series, order=(p, 0, q))
            fitted = model.fit()
            forecast_result = fitted.get_forecast(steps=horizon)
            forecast = forecast_result.predicted_mean.values
            conf_int = forecast_result.conf_int().values
            
            details = {
                "model_type": f"ARMA({p},{q})",
                "order": (p, 0, q),
                "aic": fitted.aic,
                "bic": fitted.bic,
            }
            return np.asarray(forecast, dtype=float), np.asarray(conf_int, dtype=float), details
        except ImportError:
            raise ImportError(f"Не удалось создать ARMA({p},{q}) модель: {e}")


def auto_arima_forecast(
    pm_module,
    train_series: pd.Series,
    horizon: int,
    exog_train: Optional[pd.DataFrame] = None,
    exog_test: Optional[pd.DataFrame] = None,
    seasonal: bool = False,
    max_p: int = 3,
    max_q: int = 3,
    max_P: int = 1,
    max_Q: int = 1,
    d: Optional[int] = None,
    D: Optional[int] = None,
    m: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Auto ARIMA/SARIMA/SARIMAX: автоматический подбор параметров через AIC/BIC.
    Оптимизировано для ускорения:
    - Ограничены параметры поиска (max_p, max_q) для больших данных
    - Ограничены итерации оптимизации (maxiter=50)
    - Используется 1 поток (n_jobs=1) для стабильности
    - Пропускается seasonal_test для больших данных (>1000 точек)
    """
    if pm_module is None:
        raise ImportError("pmdarima недоступен")

    # Оптимизировано: более агрессивное ограничение max_order для ускорения
    max_order = max(max_p + max_q, 5)
    
    # Дополнительное ограничение для больших датасетов (только параметры поиска, не размер данных)
    if len(train_series) > 800:
        # Для больших данных ограничиваем поиск параметров (не размер данных!)
        max_p = min(max_p, 2)
        max_q = min(max_q, 2)
        max_P = min(max_P, 1)
        max_Q = min(max_Q, 1)
        max_order = max(max_p + max_q, 3)
    # Для очень больших данных еще больше ограничиваем параметры поиска
    if len(train_series) > 1500:
        max_p = min(max_p, 1)  # Ограничиваем только параметры поиска
        max_q = min(max_q, 1)
        max_order = 2
    
    # Подготовка экзогенных переменных
    exog_train_array = None
    exog_test_array = None
    if exog_train is not None and not exog_train.empty:
        exog_train_array = exog_train.to_numpy()
    if exog_test is not None and not exog_test.empty:
        exog_test_array = exog_test.to_numpy()
    
    # Подготавливаем параметры в зависимости от типа модели
    # pmdarima.auto_arima ожидает позиционный аргумент 'y' для временного ряда
    if seasonal:
        # Для сезонных моделей (SARIMA/SARIMAX)
        # Оптимизировано: пропускаем seasonal_test для больших данных для ускорения
        seasonal_test_value = None if len(train_series) > 1000 else "ocsb"
        arima_params = {
            "y": train_series,  # Используем 'y' вместо 'train_series'
            "seasonal": True,
            "m": m,
            "trace": False,
            "start_p": 0,
            "max_p": max_p,
            "start_q": 0,
            "max_q": max_q,
            "start_P": 0,
            "max_P": max_P,
            "start_Q": 0,
            "max_Q": max_Q,
            "d": d,
            "D": D,
            "stepwise": True,
            "suppress_warnings": True,
            "error_action": "ignore",
            "seasonal_test": seasonal_test_value,  # Оптимизировано: None для больших данных
            "max_order": max_order,
            "exogenous": exog_train_array,
            "information_criterion": "aic",
            "n_jobs": 1,  # Оптимизировано: использовать 1 поток для стабильности
            "maxiter": 50,  # Оптимизировано: ограничить итерации оптимизации (было ~100+)
        }
    else:
        # Для несезонных моделей (ARIMA/ARIMAX) - не передаем сезонные параметры
        arima_params = {
            "y": train_series,  # Используем 'y' вместо 'train_series'
            "seasonal": False,
            "trace": False,
            "start_p": 0,
            "max_p": max_p,
            "start_q": 0,
            "max_q": max_q,
            "d": d,
            "stepwise": True,
            "suppress_warnings": True,
            "error_action": "ignore",
            "max_order": max_order,
            "exogenous": exog_train_array,
            "information_criterion": "aic",
            "n_jobs": 1,  # Оптимизировано: использовать 1 поток для стабильности
            "maxiter": 50,  # Оптимизировано: ограничить итерации оптимизации (было ~100+)
        }
    
    model = pm_module.auto_arima(**arima_params)

    forecast, conf_int = model.predict(
        n_periods=horizon,
        return_conf_int=True,
        alpha=0.05,
        exogenous=exog_test_array,
    )

    # Определяем тип модели на основе параметров
    if exog_train_array is not None and exog_train_array.size > 0:
        model_type = "SARIMAX" if seasonal else "ARIMAX"
    else:
        model_type = "SARIMA" if seasonal else "ARIMA"
    
    # Получаем AIC/BIC из модели
    aic_value = None
    bic_value = None
    try:
        if hasattr(model, "aic"):
            aic_value = model.aic() if callable(getattr(model, "aic", None)) else model.aic
        if hasattr(model, "bic"):
            bic_value = model.bic() if callable(getattr(model, "bic", None)) else model.bic
    except Exception:
        pass
    
    details = {
        "model_type": model_type,
        "order": getattr(model, "order", None),
        "seasonal_order": getattr(model, "seasonal_order", None),
        "aic": aic_value,
        "bic": bic_value,
        "information_criterion": "AIC",  # Используется AIC для подбора
    }
    return np.asarray(forecast, dtype=float), np.asarray(conf_int, dtype=float), details


def run_garch_forecast(
    pm_module,
    arch_model_cls,
    train_series: pd.Series,
    horizon: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    if pm_module is None or arch_model_cls is None:
        raise ImportError("Для GARCH требуется установить пакеты pmdarima и arch.")

    arima_model = pm_module.auto_arima(
        train_series.to_numpy(),
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    resid = arima_model.resid()
    if resid.size < 10:
        raise ValueError("Недостаточно данных для оценки GARCH (остатков < 10).")

    garch = arch_model_cls(resid, vol="Garch", p=1, q=1, dist="normal")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        garch_res = garch.fit(disp="off")

    variance_forecast = garch_res.forecast(horizon=horizon).variance.values[-1]
    sigma = np.sqrt(variance_forecast)
    mean_forecast = arima_model.predict(n_periods=horizon)
    lower = mean_forecast - 1.96 * sigma
    upper = mean_forecast + 1.96 * sigma

    details = {
        "arima_order": getattr(arima_model, "order", None),
        "garch_params": getattr(garch_res, "params", None),
    }
    return np.asarray(mean_forecast, dtype=float), np.column_stack([lower, upper]), details


def run_tbats_forecast(
    tbats_cls,
    train_series: pd.Series,
    horizon: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    TBATS: модель для сложной сезонности.
    Для больших данных ограничиваем размер обучающей выборки для ускорения.
    """
    if tbats_cls is None:
        raise ImportError("TBATS недоступен. Установите пакет tbats.")
    
    # Оптимизировано: более агрессивное ограничение для ускорения
    max_data_size = 1500  # Максимальный размер данных для TBATS (уменьшено с 2000)
    if len(train_series) > max_data_size:
        # Используем последние max_data_size точек для ускорения
        train_series = train_series.iloc[-max_data_size:]
    
    # Ограничиваем периоды сезонности для ускорения
    seasonal_periods = []
    if len(train_series) >= 30:
        seasonal_periods.append(7)  # Недельная сезонность
    if len(train_series) >= 90:
        seasonal_periods.append(12)  # Месячная сезонность
    
    if not seasonal_periods:
        seasonal_periods = [7]  # По умолчанию
    
    # Используем n_jobs=1 для стабильности (multiprocessing может вызывать проблемы)
    estimator = tbats_cls(
        seasonal_periods=seasonal_periods, 
        use_arma_errors=True, 
        show_warnings=False,
        n_jobs=1  # Отключаем multiprocessing для стабильности
    )
    model = estimator.fit(train_series.to_numpy())
    forecast = model.forecast(steps=horizon)
 
    return np.asarray(forecast, dtype=float), None, {"seasonal_periods": seasonal_periods}


def run_prophet_forecast(
    prophet_cls,
    train_df: pd.DataFrame,
    horizon: int,
    exog_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    if prophet_cls is None:
        raise ImportError("Facebook Prophet недоступен. Установите пакет prophet.")

    # Убеждаемся, что datetime не имеет часового пояса
    prophet_df = train_df[["datetime", "target"]].copy()
    prophet_df = prophet_df.rename(columns={"datetime": "ds", "target": "y"})
    
    # Преобразуем ds в naive datetime (без часового пояса)
    # Prophet требует naive datetime (без часового пояса)
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], utc=False, errors="coerce")
    # Если datetime имеет часовой пояс (tz-aware), убираем его
    if pd.api.types.is_datetime64tz_dtype(prophet_df["ds"]):
        # Преобразуем tz-aware datetime в naive datetime
        # Сначала конвертируем в UTC, затем удаляем tz
        prophet_df["ds"] = prophet_df["ds"].dt.tz_convert('UTC').dt.tz_localize(None)
    
    # Валидация и очистка данных
    # Удаляем строки с NaN в датах или целевой переменной
    prophet_df = prophet_df.dropna(subset=["ds", "y"])
    
    # Проверяем наличие данных
    if len(prophet_df) < 3:
        raise ValueError(f"Недостаточно данных для Prophet: требуется минимум 3 наблюдения, получено {len(prophet_df)}")
    
    # Проверяем на дубликаты дат
    if prophet_df["ds"].duplicated().any():
        # Если есть дубликаты, оставляем последнее значение
        prophet_df = prophet_df.drop_duplicates(subset=["ds"], keep="last").sort_values("ds")
    
    # Проверяем на Inf и заменяем на NaN, затем удаляем
    prophet_df = prophet_df.replace([np.inf, -np.inf], np.nan)
    prophet_df = prophet_df.dropna(subset=["y"])
    
    if len(prophet_df) < 3:
        raise ValueError(f"После очистки данных недостаточно данных для Prophet: требуется минимум 3 наблюдения, получено {len(prophet_df)}")
    
    # Проверяем на слишком большие/малые значения (может вызвать проблемы оптимизации)
    y_min, y_max = prophet_df["y"].min(), prophet_df["y"].max()
    if abs(y_min) > 1e10 or abs(y_max) > 1e10:
        raise ValueError(f"Значения слишком большие для Prophet: min={y_min:.2e}, max={y_max:.2e}. Попробуйте масштабировать данные.")
    
    # Проверяем на вариабельность (все значения одинаковые)
    if prophet_df["y"].std() == 0 or prophet_df["y"].nunique() < 2:
        raise ValueError("Все значения целевой переменной одинаковые. Prophet не может обучиться на константных данных.")
    
    # Подготовка данных для оптимизации Stan
    # Prophet может иметь проблемы с оптимизацией при очень маленьких или больших значениях
    # Используем масштабирование только если данные слишком маленькие
    scale_factor = 1.0
    original_y_mean = prophet_df["y"].mean()  # Сохраняем исходное среднее
    original_y_std = prophet_df["y"].std()  # Сохраняем исходное стандартное отклонение
    use_normalization = False
    
    # Проверяем диапазон данных
    y_min, y_max = prophet_df["y"].min(), prophet_df["y"].max()
    y_range = y_max - y_min
    y_mean_abs = abs(original_y_mean)
    
    # Для Prophet не масштабируем данные, если они уже в разумном диапазоне
    # Масштабирование может ухудшить оптимизацию Stan для стабильных данных
    # Оставляем данные как есть - Prophet должен работать с исходными значениями
    # Масштабируем только если значения действительно очень маленькие (< 0.1)
    if y_mean_abs > 0 and y_mean_abs < 0.1:
        # Только для очень маленьких значений (< 0.1) масштабируем
        scale_factor = 1000.0
        prophet_df["y"] = prophet_df["y"] * scale_factor
    # Для значений 0.7-0.71 оставляем как есть - это нормальный диапазон для Prophet
    
    # Сортируем по дате (обязательно для Prophet)
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)
    
    # Инициализируем список доступных экзогенных переменных
    available_exog = []
    
    # Создаем модель Prophet с максимально упрощенными параметрами для стабильности
    # Используем минимальную модель с самого начала для избежания ошибок оптимизации
    # НЕ добавляем экзогенные переменные сразу - попробуем сначала без них
    prophet_model = prophet_cls(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        growth='linear',
        n_changepoints=0,  # Отключаем changepoints для максимальной стабильности
        changepoint_prior_scale=0.001,  # Очень консервативный параметр
        seasonality_prior_scale=10.0,  # Увеличиваем для стабильности
        holidays_prior_scale=10.0,  # Увеличиваем для стабильности
        mcmc_samples=0,  # Отключаем MCMC для ускорения
        interval_width=0.95,
    )
    
    # Сохраняем экзогенные переменные для добавления позже, если основная модель обучится
    saved_exog_cols = exog_cols
    exog_cols = None  # Временно отключаем экзогенные переменные для упрощения

    # Экзогенные переменные временно отключены для упрощения и стабильности
    # Если модель успешно обучится, можно будет добавить их позже
    # Пока используем только ds и y для максимальной стабильности

    # Проверяем минимальный размер данных
    if len(prophet_df) < 3:
        raise ValueError(f"После обработки экзогенных переменных недостаточно данных для Prophet: требуется минимум 3 наблюдения, получено {len(prophet_df)}")
    
    # Обучение модели с обработкой ошибок
    # Пробуем обучить на меньшей выборке, если данных много (для стабильности)
    # Для стабильных данных используем еще меньшую выборку
    training_df = prophet_df.copy()
    y_cv = original_y_std / original_y_mean if original_y_mean > 0 else 0
    
    # Определяем размер выборки в зависимости от стабильности данных
    if y_cv < 0.02 or len(prophet_df) > 1000:
        # Для очень стабильных данных или больших выборок используем 200-300 точек
        max_training_size = 200 if y_cv < 0.01 else 300
        if len(prophet_df) > max_training_size:
            training_df = prophet_df.tail(max_training_size).reset_index(drop=True)
    elif len(prophet_df) > 500:
        # Для нормальных данных используем 500 точек
        training_df = prophet_df.tail(500).reset_index(drop=True)
    
    try:
        # Пробуем обучить с упрощенной моделью на меньшей выборке
        # Используем только ds и y для максимальной стабильности
        prophet_model.fit(training_df[["ds", "y"]])
    except (RuntimeError, ValueError) as e:
        # Ошибка оптимизации Stan - пробуем более простую модель
        error_msg = str(e).lower()
        if "optimization" in error_msg or "stan" in error_msg or "cmdstan" in error_msg or "failed" in error_msg:
            # Пробуем без сезонности и с минимальными параметрами
            try:
                # Используем модель без changepoints и с параметрами оптимизации
                # Пробуем на еще меньшей выборке (100 точек)
                small_training_df = training_df.tail(min(100, len(training_df))).reset_index(drop=True)
                prophet_model = prophet_cls(
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    growth='linear',
                    n_changepoints=0,
                    changepoint_prior_scale=0.001,
                    seasonality_prior_scale=10.0,
                    mcmc_samples=0,
                )
                # Убираем экзогенные переменные для упрощения
                prophet_model.fit(small_training_df[["ds", "y"]])
            except Exception as e2:
                # Если и это не помогло, пробуем модель без тренда (flat)
                try:
                    prophet_model = prophet_cls(
                        yearly_seasonality=False,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        growth='flat',  # Пробуем модель без тренда
                        n_changepoints=0,
                        changepoint_prior_scale=0.001,
                        mcmc_samples=0,
                    )
                    prophet_model.fit(prophet_df[["ds", "y"]])
                except Exception as e3:
                    # Последняя попытка - используем модель с константным трендом
                    try:
                        # Пробуем использовать только данные без оптимизации сложных параметров
                        # Используем простую линейную регрессию через Prophet
                        prophet_model = prophet_cls(
                            yearly_seasonality=False,
                            weekly_seasonality=False,
                            daily_seasonality=False,
                            growth='linear',
                            n_changepoints=0,
                            changepoint_prior_scale=0.0001,  # Еще более консервативный
                            seasonality_prior_scale=50.0,  # Очень большой для стабильности
                            mcmc_samples=0,
                        )
                        # Пробуем с меньшим количеством итераций оптимизации
                        prophet_model.fit(prophet_df[["ds", "y"]])
                    except Exception as e4:
                        # Если и это не помогло - используем fallback на простое среднее
                        # Это лучше, чем полностью пропускать модель
                        # Используем исходное среднее (до масштабирования)
                        forecast_values = np.full(horizon, original_y_mean)
                        
                        # Создаем простые доверительные интервалы (±10%)
                        lower_values = forecast_values * 0.9
                        upper_values = forecast_values * 1.1
                        
                        return (
                            forecast_values,
                            np.column_stack([lower_values, upper_values]),
                            {
                                "data_points": len(prophet_df),
                                "freq": "D",
                                "scale_factor": scale_factor,
                                "fallback": True,
                                "method": "mean",
                                "warning": "Prophet не смог обучиться, использовано среднее значение"
                            },
                        )
        else:
            # Другая RuntimeError - пробрасываем дальше
            raise
    except Exception as e:
        # Любая другая ошибка - пробуем упрощенную модель с параметрами оптимизации
        try:
            prophet_model = prophet_cls(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                growth='linear',
                n_changepoints=0,
                changepoint_prior_scale=0.001,
                seasonality_prior_scale=10.0,
                mcmc_samples=0,
            )
            prophet_model.fit(prophet_df[["ds", "y"]])
        except Exception as e2:
            # Используем fallback на простое среднее вместо ошибки
            # Используем исходное среднее (до масштабирования)
            forecast_values = np.full(horizon, original_y_mean)
            
            # Создаем простые доверительные интервалы (±10%)
            lower_values = forecast_values * 0.9
            upper_values = forecast_values * 1.1
            
            return (
                forecast_values,
                np.column_stack([lower_values, upper_values]),
                {
                    "data_points": len(prophet_df),
                    "freq": "D",
                    "scale_factor": scale_factor,
                    "fallback": True,
                    "method": "mean",
                    "warning": f"Prophet не смог обучиться (ошибка: {str(e2)[:100]}), использовано среднее значение"
                },
            )
    
    # Создаем будущие даты для прогноза
    try:
        # Определяем частоту данных автоматически
        if len(prophet_df) > 1:
            time_diff = (prophet_df["ds"].iloc[-1] - prophet_df["ds"].iloc[0]).days / (len(prophet_df) - 1)
            if time_diff < 1:
                freq = "H"  # Часовые данные
            elif time_diff < 7:
                freq = "D"  # Дневные данные
            elif time_diff < 30:
                freq = "W"  # Недельные данные
            else:
                freq = "M"  # Месячные данные
        else:
            freq = "D"  # По умолчанию дневные
        
        future = prophet_model.make_future_dataframe(periods=horizon, freq=freq)
    except Exception:
        # Если не удалось определить частоту, используем дневную
        future = prophet_model.make_future_dataframe(periods=horizon, freq="D")
    
    # Добавляем экзогенные переменные для будущих дат, если они есть
    if exog_cols and available_exog and len(available_exog) > 0:
        try:
            # Проверяем, что экзогенные переменные были добавлены в модель
            if hasattr(prophet_model, 'extra_regressors') and prophet_model.extra_regressors:
                # Используем последние значения экзогенных переменных
                # (это упрощение, в реальности нужно прогнозировать экзогенные переменные)
                if all(col in prophet_df.columns for col in available_exog):
                    last_exog_values = prophet_df[available_exog].iloc[-len(future):].values
                    if len(last_exog_values) >= len(future):
                        for idx, col in enumerate(available_exog):
                            if col in prophet_model.extra_regressors:
                                future[col] = last_exog_values[:len(future), idx]
        except Exception:
            # Если не удалось добавить экзогенные переменные, продолжаем без них
            pass

    # Прогнозирование
    try:
        forecast = prophet_model.predict(future).tail(horizon)
        forecast_values = forecast["yhat"].to_numpy()
        lower = forecast.get("yhat_lower")
        upper = forecast.get("yhat_upper")

        lower_values = lower.to_numpy() if lower is not None else None
        upper_values = upper.to_numpy() if upper is not None else None
        
        # Возвращаем масштабирование обратно
        if scale_factor != 1.0:
            forecast_values = forecast_values / scale_factor
            if lower_values is not None:
                lower_values = lower_values / scale_factor
            if upper_values is not None:
                upper_values = upper_values / scale_factor
        
        # Проверяем на NaN в прогнозах
        if np.isnan(forecast_values).any():
            # Заменяем NaN на исходное среднее (уже в исходном масштабе после обратного преобразования)
            forecast_values = np.where(np.isnan(forecast_values), 
                                     original_y_mean, 
                                     forecast_values)
    except Exception as e:
        raise ValueError(f"Ошибка при прогнозировании Prophet: {str(e)}") from e

    return (
        np.asarray(forecast_values, dtype=float),
        np.column_stack([lower_values, upper_values]) if lower_values is not None and upper_values is not None else None,
            {
                "data_points": len(prophet_df), 
                "freq": freq, 
                "scale_factor": scale_factor,
                "original_mean": original_y_mean,
                "original_std": original_y_std,
            },
    )


def prepare_prophet_dataframe(train_series: pd.Series, exog_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Подготавливает DataFrame для Prophet из временного ряда."""
    df = train_series.reset_index()
    df.columns = ["datetime", "target"] if len(df.columns) == 2 else ["target"]
    
    # Если нет колонки datetime, создаем ее из индекса
    if "datetime" not in df.columns:
        df.insert(0, "datetime", train_series.index)
    
    # Преобразуем datetime в naive datetime (без часового пояса) для Prophet
    # Prophet требует naive datetime (без часового пояса)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False, errors="coerce")
    # Если datetime имеет часовой пояс (tz-aware), убираем его
    if pd.api.types.is_datetime64tz_dtype(df["datetime"]):
        # Преобразуем tz-aware datetime в naive datetime
        # Сначала конвертируем в UTC, затем удаляем tz
        df["datetime"] = df["datetime"].dt.tz_convert('UTC').dt.tz_localize(None)
    
    # Выравниваем экзогенные переменные по индексу
    if exog_df is not None and not exog_df.empty:
        # Выравниваем по датам
        exog_aligned = exog_df.reset_index()
        if "datetime" in exog_aligned.columns:
            # Также обрабатываем datetime в exog_aligned
            exog_aligned["datetime"] = pd.to_datetime(exog_aligned["datetime"], utc=False, errors="coerce")
            if pd.api.types.is_datetime64tz_dtype(exog_aligned["datetime"]):
                # Преобразуем tz-aware datetime в naive datetime
                # Сначала конвертируем в UTC, затем удаляем tz
                exog_aligned["datetime"] = exog_aligned["datetime"].dt.tz_convert('UTC').dt.tz_localize(None)
            df = df.merge(exog_aligned, on="datetime", how="left")
        else:
            # Если нет колонки datetime, используем позиционное выравнивание
            exog_reset = exog_df.reset_index(drop=True)
            df = pd.concat([df.reset_index(drop=True), exog_reset], axis=1)
            # Удаляем дублирующиеся колонки
            df = df.loc[:, ~df.columns.duplicated()]
    
    return df


def run_var_model(train_df: pd.DataFrame, horizon: int, lag_order: int = 2) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    model = VAR(train_df)
    fitted = model.fit(lag_order)
    forecast = fitted.forecast(train_df.values[-lag_order:], steps=horizon)
    return forecast[:, 0], None, {"lag_order": lag_order}


def run_vecm_model(train_df: pd.DataFrame, horizon: int, det_order: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    if train_df.shape[1] < 2:
        raise ValueError("Для VECM требуется минимум две временные серии.")

    rank_test = coint_johansen(train_df, det_order=det_order, k_ar_diff=1)
    rank = int(np.argmax(rank_test.lr1 > rank_test.cvt[:, 1]))

    vecm = VECM(train_df, k_ar_diff=1, coint_rank=rank, deterministic="n")
    fitted = vecm.fit()
    forecast = fitted.predict(horizon)
    return forecast[:, 0], None, {"rank": rank}


def get_optimal_train_size(model_name: str, total_size: int, user_limit: Optional[int] = None) -> int:
    """
    Возвращает оптимальный размер обучающей выборки для модели.
    
    Args:
        model_name: Название модели (может содержать параметры, например "SARIMA(m=7)")
        total_size: Общий размер доступных данных
        user_limit: Пользовательское ограничение (если задано)
    
    Returns:
        Оптимальный размер обучающей выборки
    """
    # Если задано пользовательское ограничение, используем его
    if user_limit and user_limit > 0:
        return min(user_limit, total_size)
    
    # Нормализуем имя модели (убираем параметры в скобках для сопоставления)
    base_name = model_name.split("(")[0].strip()
    
    # Оптимизировано: более агрессивные ограничения для ускорения обучения
    limits = {
        "Naive": total_size,  # Быстрая модель, можно использовать все данные
        "Seasonal Naive": total_size,
        "SES": total_size,
        "AR": min(1000, total_size),  # Уменьшено с 2000 до 1000
        "MA": min(1000, total_size),  # Уменьшено с 2000 до 1000
        "ARMA": min(1000, total_size),  # Уменьшено с 2000 до 1000
        "ARIMA": min(1000, total_size),  # Уменьшено с 2000 до 1000
        "SARIMA": min(800, total_size),  # Уменьшено с 1500 до 800 (медленная модель)
        "SARIMAX": min(800, total_size),  # Уменьшено с 1500 до 800 (медленная модель)
        "GARCH": min(1000, total_size),  # Уменьшено с 2000 до 1000
        "TBATS": min(1500, total_size),  # Уменьшено с 2000 до 1500 (уже есть в run_tbats_forecast)
        "Prophet": min(2000, total_size),  # Уменьшено с 3000 до 2000
        "VAR": min(1000, total_size),  # Уменьшено с 2000 до 1000
        "VECM": min(1000, total_size),  # Уменьшено с 2000 до 1000
    }
    
    # Для ML моделей используем больше данных, но с разумным ограничением
    default_ml_limit = min(2000, total_size)  # Уменьшено с 5000 до 2000
    
    return limits.get(base_name, default_ml_limit)


def prepare_forecast_windows(
    series: pd.Series,
    horizon: int,
    max_train_size: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Подготавливает обучающую и тестовую выборки.
    
    Args:
        series: Временной ряд
        horizon: Горизонт прогнозирования
        max_train_size: Максимальный размер обучающей выборки (None = использовать все)
    
    Returns:
        Кортеж (train_series, test_series)
    """
    train_series = series.iloc[:-horizon]
    test_series = series.iloc[-horizon:]
    
    # Применяем ограничение размера, если задано
    if max_train_size and max_train_size > 0 and len(train_series) > max_train_size:
        train_series = train_series.iloc[-max_train_size:]
    
    return train_series, test_series


def run_ml_model_forecast(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    horizon: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Прогнозирование с использованием моделей машинного обучения (LinearRegression, RandomForest и т.д.).
    Использует признаки для прогнозирования на горизонте horizon.
    Использует прямую стратегию: модель обучается на train_df и предсказывает на test_df используя признаки.
    """
    # Подготовка данных для обучения
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    
    # Удаляем строки с NaN в признаках и целевой переменной
    train_valid_mask = X_train.notna().all(axis=1) & y_train.notna()
    X_train = X_train[train_valid_mask]
    y_train = y_train[train_valid_mask]
    
    if X_train.empty or y_train.empty:
        raise ValueError("Недостаточно данных для обучения модели.")
    
    # Обучаем модель
    model.fit(X_train, y_train)
    
    # Для прогнозирования используем признаки из test_df
    # Но не используем целевую переменную из test_df - она нужна только для оценки
    X_test = test_df[feature_cols].copy()
    
    # Удаляем строки с NaN в признаках
    test_valid_mask = X_test.notna().all(axis=1)
    X_test = X_test[test_valid_mask]
    
    if X_test.empty:
        raise ValueError("Недостаточно данных для прогнозирования.")
    
    # Делаем прогноз
    forecasts = model.predict(X_test)
    
    # Ограничиваем прогноз до горизонта
    forecast_array = forecasts[:horizon] if len(forecasts) > horizon else forecasts
    
    # Если прогнозов меньше горизонта, дополняем последним значением
    if len(forecast_array) < horizon:
        last_value = forecast_array[-1] if len(forecast_array) > 0 else float(y_train.iloc[-1])
        forecast_array = np.append(forecast_array, [last_value] * (horizon - len(forecast_array)))
    
    forecast_array = forecast_array[:horizon]
    details = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "features_used": len(feature_cols),
        "n_features": len(feature_cols),
    }
    
    return forecast_array, None, details


def stage5(
    analysis_data: Optional[Dict[str, Any]],
    lab_state: Dict[str, bool],
    source_df: Optional[pd.DataFrame],
    datetime_column: Optional[str],
    target_column: Optional[str],
    model_factories: Dict[str, Callable[[], object]],
    pm_module,
    arch_model_cls,
    tbats_cls,
    prophet_cls,
    pm_available: bool,
    arch_available: bool,
    tbats_available: bool,
    prophet_available: bool,
) -> Dict[str, Any]:
    if analysis_data is None:
        analysis_data = {}


    # Информация о доступных моделях
    with st.expander("📋 Информация о моделях", expanded=False):
        st.markdown("""
        **Группы моделей:**
        
        **🔷 Базовые модели:**
        - **AR(1)** - авторегрессия первого порядка
        - **MA(1)** - скользящее среднее первого порядка
        - **ARMA(1,1)** - авторегрессия и скользящее среднее
        - **ARIMA** - авторегрессивная интегрированная модель скользящего среднего (автоматический подбор через AIC/BIC)
        - **SARIMA** - сезонная ARIMA с адаптивным периодом сезонности (автоматический подбор через AIC/BIC)
        - **SARIMAX** - SARIMA с экзогенными переменными (требуются дополнительные признаки)
        
        **🔷 Волатильность:**
        - **GARCH(1,1)** - модель волатильности, обучается на остатках ARIMA
        
        **🔷 Сезонные модели:**
        - **TBATS** - для сложной сезонности (требуется библиотека `tbats`)
        - **Prophet** - для устойчивости к выбросам (требуется библиотека `prophet`)
        
        **🔷 Многомерные модели:**
        - **VAR** - векторная авторегрессия (требует ≥2 рядов)
        - **VECM** - векторная модель коррекции ошибок (только при коинтеграции)
        
        **🔷 Бенчмарки:**
        - **Naive** - последнее значение повторяется
        - **Seasonal Naive** - последние значения сезона повторяются
        - **SES** - простое экспоненциальное сглаживание
        
        **🔷 ML модели:**
        - **LinearRegression** - линейная регрессия с признаками
        - **RandomForestRegressor** - случайный лес с признаками
        
        **Особенности:**
        - Все модели поддерживают обратное преобразование Box-Cox
        - Доверительные интервалы доступны для моделей, которые их поддерживают
        - Подбор параметров через AIC/BIC для ARIMA моделей
        - Адаптивный период сезонности для SARIMA на основе данных
        """)

    if not lab_state.get("stage4_completed"):
        st.info("Завершите этап 4, чтобы перейти к построению моделей.")
        return analysis_data

    if source_df is None or datetime_column is None or target_column is None:
        st.error("Недостаточно данных для построения моделей. Проверьте предыдущие этапы.")
        return analysis_data

    # Получаем данные с признаками из этапа 2
    features_df = analysis_data.get("features_df")
    feature_cols = analysis_data.get("feature_cols", [])
    target_feature_name = analysis_data.get("target_feature_name", target_column)
    
    if features_df is None or features_df.empty:
        st.error("Не найдены данные с признаками. Завершите этап 2.")
        return analysis_data
    
    # Используем features_df как основной источник данных
    features_sorted = features_df.sort_values("datetime").copy()
    
    # Получаем целевую переменную из features_df
    if target_feature_name not in features_sorted.columns:
        st.error(f"Целевая переменная '{target_feature_name}' не найдена в данных с признаками.")
        return analysis_data
    
    target_series = features_sorted[target_feature_name].copy()
    target_series = target_series.dropna()
    
    if target_series.empty:
        st.error("Недостаточно данных в целевой переменной.")
        return analysis_data
    
    # Создаем временной ряд из исходных данных для классических моделей
    source_sorted = source_df.sort_values(datetime_column).copy()
    source_sorted[datetime_column] = pd.to_datetime(source_sorted[datetime_column], errors="coerce")
    source_sorted = source_sorted.dropna(subset=[datetime_column, target_column])
    source_sorted = source_sorted.set_index(datetime_column)
    
    # Получаем целевую переменную из исходных данных для классических моделей
    target_series_source = source_sorted[target_column].copy()
    target_series_source = target_series_source.dropna()
    
    if target_series_source.empty:
        st.error("Недостаточно данных в исходной целевой переменной.")
        return analysis_data
    
    # Определяем горизонт прогнозирования
    # По умолчанию 10 дней для обеспечения достаточного количества остатков для диагностики (минимум 10)
    default_horizon = 10
    
    # Пытаемся получить лучший горизонт из результатов этапа 3
    stage3_results = analysis_data.get("stage3_results")
    recommendation_text = ""
    if stage3_results is not None and not stage3_results.empty:
        try:
            best_row = stage3_results.sort_values("val_rmse").iloc[0]
            best_model_name = best_row["model"]
            best_horizon_from_stage3 = int(best_row["horizon"])
            recommendation_text = f"**Рекомендуемая конфигурация из этапа 3:** {best_model_name} с горизонтом {best_horizon_from_stage3}"
            # Используем горизонт из этапа 3, но не меньше 10 (минимум для диагностики)
            if best_horizon_from_stage3 < len(target_series) and best_horizon_from_stage3 < len(target_series_source):
                # Используем максимум из рекомендуемого горизонта и 10 для обеспечения диагностики
                default_horizon = max(best_horizon_from_stage3, 10)
                if best_horizon_from_stage3 < 10:
                    recommendation_text += f" (установлено минимум 10 для диагностики моделей)"
        except Exception:
            pass
    
    if recommendation_text:
        st.markdown(recommendation_text)
    
    # Проверяем, достаточно ли данных для горизонта по умолчанию
    min_available = min(len(target_series), len(target_series_source))
    if min_available <= default_horizon:
        # Если данных недостаточно, используем 10% от доступных данных
        original_default = default_horizon
        default_horizon = max(1, min_available // 10)
        st.warning(f"Недостаточно данных для горизонта {original_default}. Используется горизонт {default_horizon} (10% от доступных данных).")
    
    # Позволяем пользователю выбрать горизонт
    max_horizon = max(1, min_available - 10)  # Оставляем минимум 10 точек для обучения
    
    # Проверяем, есть ли сохраненное значение в session_state, но используем его только если >= 10
    saved_horizon = st.session_state.get("stage5_horizon", None)
    if saved_horizon is not None and saved_horizon < 10:
        # Если сохраненное значение меньше 10, сбрасываем его и используем default_horizon
        if "stage5_horizon" in st.session_state:
            del st.session_state["stage5_horizon"]
        initial_value = default_horizon
    elif saved_horizon is not None and saved_horizon >= 10:
        # Используем сохраненное значение, если оно >= 10
        initial_value = saved_horizon
    else:
        # Используем значение по умолчанию
        initial_value = default_horizon
    
    selected_horizon = st.number_input(
        "Горизонт прогнозирования",
        min_value=1,
        max_value=max_horizon,
        value=initial_value,
        step=1,
        help="Количество шагов вперед для прогнозирования. Рекомендуется 10-14 дней для курса доллара. "
             "Минимум 10 точек требуется для полной диагностики моделей (тесты Льюнга-Бокса и Бройша-Пагана). "
             "Большие значения (>30) значительно увеличивают время обучения.",
        key="stage5_horizon"
    )
    
    if min_available <= selected_horizon:
        st.error(f"Недостаточно данных для горизонта {selected_horizon}. Доступно только {min_available} точек.")
        return analysis_data
    
    # Предупреждение при выборе большого горизонта (оптимизация)
    if selected_horizon > 30:
        st.warning(f"⚠️ Выбран большой горизонт ({selected_horizon} дней). Это может значительно увеличить время обучения моделей. "
                  f"Для курса доллара рекомендуется использовать горизонт 7-14 дней.")
    
    # Опция для ограничения размера обучающей выборки
    train_size_available = len(target_series_source) - selected_horizon
    default_max_train_size = 0  # 0 = использовать все данные
    # Оптимизировано: более агрессивные ограничения для ускорения (1000 вместо 2000)
    if train_size_available > 1000:
        # Для больших датасетов предлагаем разумное ограничение для ускорения
        default_max_train_size = min(1000, train_size_available)
    
    max_train_size = st.number_input(
        "Максимальный размер обучающей выборки (0 = использовать все данные)",
        min_value=0,
        max_value=train_size_available,
        value=default_max_train_size,
        step=100,
        help="Ограничение размера обучающей выборки для ускорения обучения. "
             "Для больших датасетов рекомендуется 1000-2000 точек. "
             "0 означает использование всех доступных данных. "
             f"Доступно: {train_size_available} точек.",
        key="stage5_max_train_size"
    )
    
    if max_train_size > 0:
        st.info(f"📉 Будет использовано ограничение: последние {max_train_size} точек из {train_size_available} доступных")
    
    # Опция для пропуска медленных моделей
    skip_slow_models = st.checkbox(
        "⏱️ Пропустить медленные модели (TBATS, SARIMA при больших данных)",
        value=False,  # По умолчанию выключено - все модели запускаются
        help="TBATS и SARIMA могут обучаться очень долго (>5 минут) для больших данных. "
             "Включите этот флаг, чтобы пропустить их и ускорить обучение.",
        key="stage5_skip_slow"
    )
    
    # Подготавливаем данные для прогнозирования (для классических моделей из исходных данных)
    # Применяем общее ограничение размера, если задано
    train_series, test_series = prepare_forecast_windows(
        target_series_source, 
        selected_horizon,
        max_train_size=max_train_size if max_train_size > 0 else None
    )
    
    if max_train_size > 0 and len(train_series) < train_size_available:
        st.success(f"✅ Применено ограничение: используется {len(train_series)} точек вместо {train_size_available}")
    
    if train_series.empty or test_series.empty:
        st.error("Не удалось создать обучающую и тестовую выборки.")
        return analysis_data
    
    # Подготавливаем данные с признаками для ML моделей
    # Используем features_df, выровненный по датам
    if "datetime" in features_sorted.columns:
        features_sorted_aligned = features_sorted.set_index("datetime").copy()
    else:
        # Если нет колонки datetime, используем индекс
        features_sorted_aligned = features_sorted.copy()
    
    # Выравниваем train и test выборки
    train_features_df_full = features_sorted_aligned.iloc[:-selected_horizon].copy()
    test_features_df = features_sorted_aligned.iloc[-selected_horizon:].copy()
    
    # Применяем ограничение размера для ML моделей, если задано
    if max_train_size > 0 and len(train_features_df_full) > max_train_size:
        train_features_df = train_features_df_full.iloc[-max_train_size:].copy()
    else:
        train_features_df = train_features_df_full.copy()
    
    # Удаляем строки с NaN в целевой переменной из train
    train_features_df = train_features_df.dropna(subset=[target_feature_name])
    
    # Проверяем наличие признаков
    if not feature_cols:
        st.warning("Не найдены признаки для ML моделей. Будут использоваться только классические модели временных рядов.")
    
    # Проверяем, что в train_features_df есть достаточно данных
    if train_features_df.empty:
        st.error("Недостаточно данных для обучения ML моделей. Проверьте данные с признаками.")
        train_features_df = None
        test_features_df = None
    
    # Подготавливаем экзогенные переменные для классических моделей
    exog_cols = analysis_data.get("exog_selection", [])
    exog_train = None
    exog_test = None
    if exog_cols:
        # Проверяем, что экзогенные переменные есть в source_sorted
        available_exog = [col for col in exog_cols if col in source_sorted.columns]
        if available_exog:
            exog_df = source_sorted[available_exog].copy()
            exog_train = exog_df.iloc[:-selected_horizon].copy()
            exog_test = exog_df.iloc[-selected_horizon:].copy()
            
            # Удаляем строки с NaN
            exog_train = exog_train.dropna()
            exog_test = exog_test.dropna()
            
            # Выравниваем индексы
            train_idx = train_series.index
            test_idx = test_series.index
            exog_train = exog_train.reindex(train_idx).ffill().bfill()
            exog_test = exog_test.reindex(test_idx).ffill().bfill()

    use_boxcox = st.checkbox("Использовать Box-Cox трансформацию", value=False)
    lambda_default = analysis_data.get("boxcox_lambda", None)
    
    # Чекбокс для автоматического подбора λ
    auto_lambda = st.checkbox(
        "Автоматический подбор λ для Box-Cox",
        value=True,
        help="Если включено, λ будет автоматически оценен. Если выключено, можно указать значение вручную.",
        key="stage5_auto_lambda"
    )
    
    # Поле ввода λ показывается только если автоматический подбор выключен
    lambda_value = None
    if not auto_lambda:
        lambda_value = st.number_input(
            "Значение λ для Box-Cox",
            value=float(lambda_default or 0.0),
            step=0.05,
            help="Введите значение λ для Box-Cox трансформации вручную",
            key="stage5_lambda_value"
        )

    # Выбор моделей для запуска
    available_ml_models = list(model_factories.keys()) if model_factories else []
    selected_ml_models = st.multiselect(
        "Выберите ML модели для сравнения",
        available_ml_models,
        default=["LinearRegression"] if "LinearRegression" in available_ml_models else available_ml_models[:1] if available_ml_models else [],
        help="Выберите модели машинного обучения для прогнозирования с использованием признаков",
        key="stage5_ml_models"
    )

    # Показываем информацию о доступности опциональных моделей один раз
    unavailable_models = []
    install_commands = []
    
    if not pm_available:
        unavailable_models.append("SARIMA")
        if "pmdarima" not in install_commands:
            install_commands.append("pmdarima")
    
    if not arch_available or not pm_available:
        unavailable_models.append("GARCH")
        if "pmdarima" not in install_commands and not pm_available:
            install_commands.append("pmdarima")
        if "arch" not in install_commands and not arch_available:
            install_commands.append("arch")
    
    if not tbats_available:
        unavailable_models.append("TBATS")
        if "tbats" not in install_commands:
            install_commands.append("tbats")
    
    if not prophet_available:
        unavailable_models.append("Prophet")
        if "prophet" not in install_commands:
            install_commands.append("prophet")
    
    if unavailable_models:
        with st.expander("ℹ️ Информация о доступности моделей", expanded=False):
            st.markdown("**Недоступные модели:**")
            for model in set(unavailable_models):  # Убираем дубликаты
                st.markdown(f"- **{model}**")
            
            if install_commands:
                st.markdown("**Для установки:**")
                st.code(f"pip install {' '.join(install_commands)}", language="bash")

    if st.button("Запустить моделирование"):
        # Используем автоматический подбор, если чекбокс включен, иначе используем введенное значение
        is_lambda_auto = auto_lambda
        lambda_used = None if is_lambda_auto else lambda_value

        transformed_train, offset = apply_boxcox_transform(train_series, use_boxcox, lambda_used)
        if is_lambda_auto and use_boxcox:
            from scipy.stats import boxcox

            transformed_train, lambda_used = boxcox(transformed_train)
            transformed_train = pd.Series(transformed_train, index=train_series.index, name="target")

        models_to_run: List[Tuple[str, str, Callable[..., Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]], Dict[str, Any]]] = []

        # Добавляем ML модели (LinearRegression, RandomForest и т.д.)
        if selected_ml_models and feature_cols and train_features_df is not None and test_features_df is not None:
            if not train_features_df.empty and not test_features_df.empty:
                # Проверяем, что признаки есть в данных
                available_features = [col for col in feature_cols if col in train_features_df.columns and col in test_features_df.columns]
                if available_features:
                    for model_name in selected_ml_models:
                        if model_name in model_factories:
                            model_factory = model_factories[model_name]
                            try:
                                models_to_run.append(
                                    (
                                        model_name,
                                        "ML Models",
                                        run_ml_model_forecast,
                                        {
                                            "model": model_factory(),
                                            "train_df": train_features_df,
                                            "test_df": test_features_df,
                                            "feature_cols": available_features,
                                            "target_col": target_feature_name,
                                            "horizon": selected_horizon,
                                        },
                                    )
                                )
                            except Exception as e:
                                st.warning(f"Не удалось добавить модель {model_name}: {e}")
                else:
                    st.warning("Не найдены доступные признаки для ML моделей. Проверьте данные с признаками.")

        # Бенчмарки
        models_to_run.append(
            (
                "Naive",
                "Benchmarks",
                run_naive_forecast,
                {
                    "train_series": transformed_train,
                    "horizon": selected_horizon,
                },
            )
        )
        
        # Seasonal Naive (используем сезонность 7 дней, если данных достаточно)
        if len(transformed_train) >= 7:
            models_to_run.append(
                (
                    "Seasonal Naive",
                    "Benchmarks",
                    run_seasonal_naive_forecast,
                    {
                        "train_series": transformed_train,
                        "horizon": selected_horizon,
                        "seasonality": 7,
                    },
                )
            )
        
        # SES (Simple Exponential Smoothing)
        models_to_run.append(
            (
                "SES",
                "Benchmarks",
                lambda series, horizon: (
                    SimpleExpSmoothing(series).fit().forecast(horizon),
                    None,
                    {},
                ),
                {"series": transformed_train, "horizon": selected_horizon},
            )
        )

        # Базовые модели ARIMA через auto_arima
        if pm_available and pm_module is not None:
            # AR(1) - авторегрессия
            try:
                models_to_run.append(
                    (
                        "AR(1)",
                        "Basic",
                        run_ar_forecast,
                        {
                            "pm_module": pm_module,
                            "train_series": transformed_train,
                            "horizon": selected_horizon,
                            "p": 1,
                        },
                    )
                )
            except Exception:
                pass  # Пропускаем если не удалось создать модель
            
            # MA(1) - скользящее среднее
            try:
                models_to_run.append(
                    (
                        "MA(1)",
                        "Basic",
                        run_ma_forecast,
                        {
                            "pm_module": pm_module,
                            "train_series": transformed_train,
                            "horizon": selected_horizon,
                            "q": 1,
                        },
                    )
                )
            except Exception:
                pass
            
            # ARMA(1,1) - авторегрессия и скользящее среднее
            try:
                models_to_run.append(
                    (
                        "ARMA(1,1)",
                        "Basic",
                        run_arma_forecast,
                        {
                            "pm_module": pm_module,
                            "train_series": transformed_train,
                            "horizon": selected_horizon,
                            "p": 1,
                            "q": 1,
                        },
                    )
                )
            except Exception:
                pass
            
            # ARIMA (несезонная) - автоматический подбор через AIC/BIC
            # ARIMA без экзогенных переменных
            models_to_run.append(
                (
                    "ARIMA",
                    "Basic",
                    auto_arima_forecast,
                    {
                        "pm_module": pm_module,
                        "train_series": transformed_train,
                        "horizon": selected_horizon,
                        "seasonal": False,
                        "exog_train": None,  # ARIMA без экзогенных переменных
                        "exog_test": None,
                    },
                )
            )
            
            # SARIMA (сезонная) - автоматический подбор через AIC/BIC
            # SARIMA может быть медленной для больших данных - пропускаем если выбрано
            if skip_slow_models:
                st.info("⏭️ SARIMA пропущена (медленная модель). Чтобы включить, снимите флаг 'Пропустить медленные модели'.")
            else:
                # Определяем сезонность на основе данных (7 для недельной, 12 для месячной)
                # Если данных достаточно, пробуем оба варианта
                seasonal_periods = []
                if len(transformed_train) >= 30:
                    seasonal_periods.append(7)  # Недельная сезонность
                if len(transformed_train) >= 90:
                    seasonal_periods.append(12)  # Месячная сезонность
                if len(transformed_train) >= 365:
                    seasonal_periods.append(30)  # Месячная сезонность для дневных данных
                
                # Используем наиболее подходящий период сезонности
                if seasonal_periods:
                    # Выбираем наибольший период, для которого достаточно данных
                    best_m = max([m for m in seasonal_periods if len(transformed_train) >= 2 * m] or [7])
                else:
                    best_m = 7  # По умолчанию недельная сезонность
                
                # Для больших данных показываем предупреждение (оптимизировано: порог снижен)
                if len(transformed_train) > 500:
                    st.warning(f"⚠️ SARIMA может обучаться долго (>30 секунд) для {len(transformed_train)} точек. "
                              f"Если обучение занимает слишком много времени, включите флаг 'Пропустить медленные модели'.")
                
                models_to_run.append(
                    (
                        f"SARIMA(m={best_m})",
                        "Basic",
                        auto_arima_forecast,
                        {
                            "pm_module": pm_module,
                            "train_series": transformed_train,
                            "horizon": selected_horizon,
                            "seasonal": True,
                            "m": best_m,
                            "exog_train": None,  # SARIMA без экзогенных переменных
                            "exog_test": None,
                        },
                    )
                )
            
            # SARIMAX (с экзогенными переменными) - только если есть экзогенные переменные
            if exog_train is not None and exog_test is not None and not exog_train.empty and not exog_test.empty:
                # SARIMAX может быть медленной для больших данных - пропускаем если выбрано
                if skip_slow_models:
                    st.info("⏭️ SARIMAX пропущена (медленная модель). Чтобы включить, снимите флаг 'Пропустить медленные модели'.")
                else:
                    # Используем тот же период сезонности, что и для SARIMA
                    seasonal_periods = []
                    if len(transformed_train) >= 30:
                        seasonal_periods.append(7)
                    if len(transformed_train) >= 90:
                        seasonal_periods.append(12)
                    if len(transformed_train) >= 365:
                        seasonal_periods.append(30)
                    
                    best_m = max([m for m in seasonal_periods if len(transformed_train) >= 2 * m] or [7])
                    
                    # Для больших данных показываем предупреждение (оптимизировано: порог снижен)
                    if len(transformed_train) > 500:
                        st.warning(f"⚠️ SARIMAX может обучаться долго (>30 секунд) для {len(transformed_train)} точек.")
                    
                    models_to_run.append(
                        (
                            f"SARIMAX(m={best_m})",
                            "Basic",
                            auto_arima_forecast,
                            {
                                "pm_module": pm_module,
                                "train_series": transformed_train,
                                "horizon": selected_horizon,
                                "seasonal": True,
                                "m": best_m,
                                "exog_train": exog_train,
                                "exog_test": exog_test,
                            },
                        )
                    )

        # GARCH(1,1) - волатильность, обучается на остатках ARIMA
        if arch_available and pm_available and pm_module is not None and arch_model_cls is not None:
            models_to_run.append(
                (
                    "GARCH(1,1)",
                    "Volatility",
                    run_garch_forecast,
                    {
                        "pm_module": pm_module,
                        "arch_model_cls": arch_model_cls,
                        "train_series": transformed_train,
                        "horizon": selected_horizon,
                    },
                )
            )

        # TBATS - для сложной сезонности
        if tbats_available and tbats_cls is not None:
            # TBATS очень медленная для больших данных - пропускаем если выбрано или данных слишком много
            if skip_slow_models:
                st.info("⏭️ TBATS пропущена (медленная модель). Чтобы включить, снимите флаг 'Пропустить медленные модели'.")
            elif len(transformed_train) > 1500:
                st.warning(f"⚠️ TBATS может обучаться очень долго (>5 минут) для {len(transformed_train)} точек. "
                          f"Будет использовано последние 1500 точек для ускорения. "
                          f"Рекомендуется включить флаг 'Пропустить медленные модели'.")
                models_to_run.append(
                    (
                        "TBATS",
                        "Seasonal",
                        run_tbats_forecast,
                        {"tbats_cls": tbats_cls, "train_series": transformed_train, "horizon": selected_horizon},
                    )
                )
            else:
                models_to_run.append(
                    (
                        "TBATS",
                        "Seasonal",
                        run_tbats_forecast,
                        {"tbats_cls": tbats_cls, "train_series": transformed_train, "horizon": selected_horizon},
                    )
                )

        # Prophet - для устойчивости к выбросам
        if prophet_available and prophet_cls is not None:
            try:
                prophet_train_df = prepare_prophet_dataframe(train_series, exog_train)
                models_to_run.append(
                    (
                        "Prophet",
                        "Seasonal",
                        run_prophet_forecast,
                        {
                            "prophet_cls": prophet_cls,
                            "train_df": prophet_train_df,
                            "horizon": selected_horizon,
                            "exog_cols": exog_cols,
                        },
                    )
                )
            except Exception as e:
                st.warning(f"⚠️ Не удалось подготовить данные для Prophet: {e}")

        # Подготовка данных для мультивариантных моделей (VAR, VECM)
        multivariate_cols = [col for col in source_sorted.columns if col != target_column and pd.api.types.is_numeric_dtype(source_sorted[col])]
        if multivariate_cols:
            multivariate_df = source_sorted[[target_column] + multivariate_cols].copy()
            multivariate_df = multivariate_df.dropna()
            
            if not multivariate_df.empty and len(multivariate_df) > selected_horizon:
                # Выравниваем с train_series
                aligned_train = multivariate_df.iloc[:-selected_horizon]
                if aligned_train.shape[1] >= 2:
                    models_to_run.append(
                        (
                            "VAR",
                            "Multivariate",
                            run_var_model,
                            {"train_df": aligned_train, "horizon": selected_horizon},
                        )
                    )

                    try:
                        models_to_run.append(
                            (
                                "VECM",
                                "Multivariate",
                                run_vecm_model,
                                {"train_df": aligned_train, "horizon": selected_horizon},
                            )
                        )
                    except Exception as exc:
                        st.warning(f"Не удалось настроить VECM: {exc}")

        if not models_to_run:
            st.warning("⚠️ Нет доступных моделей для запуска. Проверьте установленные библиотеки.")
            return analysis_data

        forecast_results: List[ForecastResult] = []
        progress = st.progress(0)
        total_models = len(models_to_run)
        error_container = st.container()

        for idx, (name, group, func, params) in enumerate(models_to_run, start=1):
            progress.progress(idx / total_models)
            try:
                # Применяем автоматическое ограничение размера для медленных моделей
                # если пользователь не задал общее ограничение
                # Исключаем TBATS, так как у него уже есть встроенное ограничение
                if max_train_size == 0 and name != "TBATS":
                    # Определяем текущий размер данных
                    current_size = 0
                    if "train_series" in params:
                        current_size = len(params["train_series"])
                    elif "train_df" in params:
                        current_size = len(params["train_df"])
                    else:
                        current_size = len(transformed_train)
                    
                    optimal_size = get_optimal_train_size(name, current_size)
                    if optimal_size < current_size:
                        # Ограничиваем размер данных для этой модели
                        if "train_series" in params:
                            original_series = params["train_series"]
                            if len(original_series) > optimal_size:
                                params["train_series"] = original_series.iloc[-optimal_size:]
                                # Также ограничиваем экзогенные переменные, если они есть
                                if "exog_train" in params and params["exog_train"] is not None:
                                    exog_train = params["exog_train"]
                                    if len(exog_train) > optimal_size:
                                        params["exog_train"] = exog_train.iloc[-optimal_size:]
                                st.caption(f"⚡ {name}: используется {len(params['train_series'])} точек (оптимально для этой модели)")
                        elif "train_df" in params:
                            original_df = params["train_df"]
                            if len(original_df) > optimal_size:
                                params["train_df"] = original_df.iloc[-optimal_size:]
                                st.caption(f"⚡ {name}: используется {len(params['train_df'])} точек (оптимально для этой модели)")
                
                forecast, conf_int, details = func(**params)
                
                # Обработка доверительных интервалов
                lower = None
                upper = None
                if conf_int is not None:
                    conf_int_array = np.asarray(conf_int)
                    if conf_int_array.ndim == 2:
                        if conf_int_array.shape[1] >= 2:
                            # Если conf_int - это массив [lower, upper] для каждого шага
                            lower = conf_int_array[:, 0]
                            upper = conf_int_array[:, 1]
                        elif conf_int_array.shape[0] == 2 and conf_int_array.shape[1] > 1:
                            # Если conf_int - это массив [[lower1, lower2, ...], [upper1, upper2, ...]]
                            lower = conf_int_array[0, :]
                            upper = conf_int_array[1, :]
                    elif conf_int_array.ndim == 1:
                        if len(conf_int_array) == 2:
                            # Если conf_int - это одномерный массив [lower, upper] для одного шага
                            lower = np.full(selected_horizon, conf_int_array[0])
                            upper = np.full(selected_horizon, conf_int_array[1])
                        elif len(conf_int_array) == selected_horizon * 2:
                            # Если conf_int - это плоский массив [lower1, upper1, lower2, upper2, ...]
                            lower = conf_int_array[::2]
                            upper = conf_int_array[1::2]
                
                # Убеждаемся, что lower и upper имеют правильную длину
                if lower is not None and upper is not None:
                    if len(lower) != selected_horizon or len(upper) != selected_horizon:
                        # Если длины не совпадают, пытаемся привести к правильной длине
                        if len(lower) > selected_horizon:
                            lower = lower[:selected_horizon]
                        elif len(lower) < selected_horizon:
                            # Дополняем последним значением
                            last_lower = lower[-1] if len(lower) > 0 else forecast[-1] * 0.9
                            lower = np.append(lower, [last_lower] * (selected_horizon - len(lower)))
                        
                        if len(upper) > selected_horizon:
                            upper = upper[:selected_horizon]
                        elif len(upper) < selected_horizon:
                            # Дополняем последним значением
                            last_upper = upper[-1] if len(upper) > 0 else forecast[-1] * 1.1
                            upper = np.append(upper, [last_upper] * (selected_horizon - len(upper)))
                
                result = finalize_forecast_result(
                    name=name,
                    group=group,
                    actual=test_series,
                    forecast_transformed=forecast,
                    lower_transformed=lower,
                    upper_transformed=upper,
                    use_boxcox=use_boxcox,
                    lambda_=lambda_used,
                    offset=offset,
                    details=details,
                )
                
                # Показываем предупреждение, если Prophet использовал fallback
                if name == "Prophet" and details.get("fallback", False):
                    with error_container:
                        st.warning(f"⚠️ {name}: использован fallback метод ({details.get('method', 'mean')}) "
                                  f"из-за проблем с оптимизацией Stan. "
                                  f"Прогноз основан на среднем значении данных. "
                                  f"Детали: {details.get('warning', '')}")
                
                forecast_results.append(result)
            except ImportError as exc:
                with error_container:
                    st.error(f"❌ {name}: модель недоступна — {exc}. Установите необходимую библиотеку.")
            except ValueError as exc:
                with error_container:
                    # Prophet теперь не выбрасывает ValueError, а использует fallback
                    st.error(f"❌ {name}: ошибка данных — {exc}")
            except Exception as exc:
                with error_container:
                    # Prophet теперь использует fallback вместо ошибки
                    st.error(f"❌ {name}: ошибка построения прогноза — {exc}")
                    import traceback
                    with st.expander(f"Подробности ошибки для {name}"):
                        st.code(traceback.format_exc(), language="python")

        progress.empty()

        if forecast_results:
            st.success("Моделирование завершено.")
            analysis_data["forecast_results"] = forecast_results
            analysis_data["boxcox_lambda"] = lambda_used
            analysis_data["stage5_train_series"] = train_series
            analysis_data["stage5_test_series"] = test_series
            lab_state["stage5_completed"] = True
        else:
            st.warning("Не удалось построить ни одного прогноза.")

    # Сохраняем train_series и test_series для визуализации
    train_series_for_viz = analysis_data.get("stage5_train_series")
    test_series_for_viz = analysis_data.get("stage5_test_series")
    
    results: List[ForecastResult] = analysis_data.get("forecast_results", [])
    if results:
        st.markdown("#### 📊 Сравнение моделей")
        
        # Создаем таблицу сравнения метрик
        comparison_data = []
        for result in results:
            comparison_data.append({
                "Модель": result.name,
                "Группа": result.group,
                "MAE": result.metrics.get("mae", np.nan),
                "RMSE": result.metrics.get("rmse", np.nan),
                "MAPE": result.metrics.get("mape", np.nan),
            })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values("RMSE")
            st.dataframe(comparison_df, use_container_width=True)
            
            # Визуализация сравнения метрик
            st.markdown("#### 📈 Сравнение метрик моделей")
            fig_comparison = go.Figure()
            
            models = comparison_df["Модель"].values
            mae_values = comparison_df["MAE"].values
            rmse_values = comparison_df["RMSE"].values
            mape_values = comparison_df["MAPE"].values
            
            fig_comparison.add_trace(go.Bar(
                x=models,
                y=mae_values,
                name="MAE",
                marker_color="blue",
            ))
            fig_comparison.add_trace(go.Bar(
                x=models,
                y=rmse_values,
                name="RMSE",
                marker_color="red",
            ))
            fig_comparison.add_trace(go.Bar(
                x=models,
                y=mape_values,
                name="MAPE",
                marker_color="green",
            ))
            
            fig_comparison.update_layout(
                height=400,
                title="Сравнение метрик моделей",
                xaxis_title="Модель",
                yaxis_title="Значение метрики",
                barmode="group",
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Экспорт прогноза
        st.markdown("#### 💾 Экспорт прогноза")
        
        if results:
            # Выбор модели для экспорта
            model_names_for_export = [r.name for r in results]
            selected_model_for_export = st.selectbox(
                "Выберите модель для экспорта",
                model_names_for_export,
                index=0,
                help="Выберите модель, прогноз которой вы хотите экспортировать",
                key="stage5_export_model"
            )
            
            # Находим выбранную модель
            selected_result = next((r for r in results if r.name == selected_model_for_export), None)
            
            if selected_result:
                # Создаем DataFrame для экспорта
                export_df = pd.DataFrame({
                    "Дата": selected_result.forecast.index,
                    "Фактическое_значение": selected_result.actual.values,
                    "Прогноз": selected_result.forecast.values,
                })
                
                # Добавляем доверительные интервалы (если они есть, иначе NaN)
                if selected_result.lower is not None and len(selected_result.lower) == len(export_df):
                    export_df["Нижняя_граница"] = selected_result.lower.values
                else:
                    export_df["Нижняя_граница"] = np.nan
                    
                if selected_result.upper is not None and len(selected_result.upper) == len(export_df):
                    export_df["Верхняя_граница"] = selected_result.upper.values
                else:
                    export_df["Верхняя_граница"] = np.nan
                
                # Добавляем метрики в заголовок (как комментарий в CSV)
                metrics_text = f"Модель: {selected_result.name}, Группа: {selected_result.group}\n"
                metrics_text += f"MAE: {selected_result.metrics.get('mae', 'N/A'):.4f}, "
                metrics_text += f"RMSE: {selected_result.metrics.get('rmse', 'N/A'):.4f}, "
                metrics_text += f"MAPE: {selected_result.metrics.get('mape', 'N/A'):.2f}%"
                
                # Конвертируем DataFrame в CSV
                csv_data = export_df.to_csv(index=False, encoding='utf-8-sig')
                
                # Создаем имя файла
                model_name_safe = selected_result.name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
                try:
                    if isinstance(selected_result.forecast.index, pd.DatetimeIndex):
                        date_str = selected_result.forecast.index[0].strftime('%Y%m%d')
                    else:
                        date_str = 'export'
                except Exception:
                    date_str = 'export'
                filename = f"forecast_{model_name_safe}_{date_str}.csv"
                
                # Кнопка для скачивания
                st.download_button(
                    label="📥 Скачать прогноз в CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    help=f"Экспортировать прогноз модели {selected_result.name} в CSV файл",
                    key="stage5_download_csv"
                )
                
                # Показываем превью данных для экспорта
                with st.expander("👁️ Превью данных для экспорта", expanded=False):
                    st.dataframe(export_df, use_container_width=True)
                    st.caption(metrics_text)
            
            # Экспорт всех моделей (сводная таблица)
            st.markdown("**📊 Экспорт сводной таблицы всех моделей:**")
            
            # Создаем сводную таблицу со всеми моделями
            all_models_export = []
            for result in results:
                for idx in range(len(result.forecast)):
                    row = {
                        "Модель": result.name,
                        "Группа": result.group,
                        "Дата": result.forecast.index[idx],
                        "Фактическое_значение": result.actual.values[idx] if idx < len(result.actual) else np.nan,
                        "Прогноз": result.forecast.values[idx],
                    }
                    # Добавляем доверительные интервалы, если они есть
                    if result.lower is not None and idx < len(result.lower):
                        row["Нижняя_граница"] = result.lower.values[idx]
                    else:
                        row["Нижняя_граница"] = np.nan
                    
                    if result.upper is not None and idx < len(result.upper):
                        row["Верхняя_граница"] = result.upper.values[idx]
                    else:
                        row["Верхняя_граница"] = np.nan
                    
                    # Добавляем метрики для каждой модели (только в первой строке)
                    if idx == 0:
                        row["MAE"] = result.metrics.get('mae', np.nan)
                        row["RMSE"] = result.metrics.get('rmse', np.nan)
                        row["MAPE"] = result.metrics.get('mape', np.nan)
                    else:
                        row["MAE"] = np.nan
                        row["RMSE"] = np.nan
                        row["MAPE"] = np.nan
                    
                    all_models_export.append(row)
            
            if all_models_export:
                all_models_df = pd.DataFrame(all_models_export)
                all_models_csv = all_models_df.to_csv(index=False, encoding='utf-8-sig')
                all_models_filename = f"forecasts_all_models_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                st.download_button(
                    label="📥 Скачать все прогнозы в CSV",
                    data=all_models_csv,
                    file_name=all_models_filename,
                    mime="text/csv",
                    help="Экспортировать прогнозы всех моделей в один CSV файл",
                    key="stage5_download_all_csv"
                )
                
                with st.expander("👁️ Превью сводной таблицы", expanded=False):
                    st.dataframe(all_models_df.head(20), use_container_width=True)
                    st.caption(f"Всего записей: {len(all_models_df)}")
        
        st.markdown("---")
        st.markdown("#### 📈 Детальные результаты прогнозирования")

        for result in results:
            st.markdown(f"**{result.name} ({result.group})**")

            # Создаем детальную визуализацию с историческими данными
            fig = go.Figure()
            
            # Исторические данные (train) - используем сохраненные данные
            train_viz = train_series_for_viz if train_series_for_viz is not None else None
            if train_viz is not None and not train_viz.empty:
                fig.add_trace(
                    go.Scatter(
                        x=train_viz.index,
                        y=train_viz.values,
                        mode="lines",
                        name="История (train)",
                        line=dict(color="gray", width=1.5),
                        opacity=0.7,
                    )
                )
            
            # Фактические значения (test)
            fig.add_trace(
                go.Scatter(
                    x=result.actual.index,
                    y=result.actual.values,
                    mode="lines+markers",
                    name="Факт (test)",
                    line=dict(color="blue", width=2.5),
                    marker=dict(size=8, color="blue"),
                )
            )
            
            # Прогноз
            fig.add_trace(
                go.Scatter(
                    x=result.forecast.index,
                    y=result.forecast.values,
                    mode="lines+markers",
                    name="Прогноз",
                    line=dict(color="red", width=2.5),
                    marker=dict(size=8, color="red"),
                )
            )
            
            # Доверительные интервалы
            if result.lower is not None and result.upper is not None:
                fig.add_trace(
                    go.Scatter(
                        x=result.upper.index,
                        y=result.upper.values,
                        mode="lines",
                        name="Верхняя граница",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=result.lower.index,
                        y=result.lower.values,
                        mode="lines",
                        name="Доверительный интервал (95%)",
                        fill="tonexty",
                        fillcolor="rgba(255,0,0,0.2)",
                        line=dict(width=0),
                        showlegend=True,
                    )
                )
            
            # Добавляем вертикальную линию, разделяющую train и test
            if train_viz is not None and not train_viz.empty and not result.actual.empty:
                split_date = result.actual.index[0]
                
                # Находим min и max значения y для линии
                y_values = []
                if train_viz is not None and not train_viz.empty:
                    y_values.extend(train_viz.values)
                y_values.extend(result.actual.values)
                y_values.extend(result.forecast.values)
                
                if y_values:
                    y_min = min(y_values)
                    y_max = max(y_values)
                    
                    # Преобразуем Timestamp в строку для совместимости с plotly
                    # plotly автоматически распознает строковые даты
                    try:
                        if isinstance(split_date, pd.Timestamp):
                            # Преобразуем в строку ISO формата
                            split_date_for_plot = split_date.isoformat()
                        elif hasattr(split_date, 'strftime'):
                            split_date_for_plot = split_date.strftime('%Y-%m-%d')
                        else:
                            split_date_for_plot = str(split_date)
                    except Exception:
                        # Если преобразование не удалось, используем как есть
                        split_date_for_plot = split_date
                    
                    # Добавляем вертикальную линию через shape (более надежный способ)
                    fig.add_shape(
                        type="line",
                        x0=split_date_for_plot,
                        x1=split_date_for_plot,
                        y0=y_min,
                        y1=y_max,
                        line=dict(color="black", width=2, dash="dash"),
                        layer="below",
                    )
                    
                    # Добавляем аннотацию
                    fig.add_annotation(
                        x=split_date_for_plot,
                        y=y_max,
                        text="Разделение train/test",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="black",
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        font=dict(size=10),
                    )
            
            fig.update_layout(
                height=600,
                title=f"Прогноз модели {result.name} ({result.group})",
                xaxis_title="Дата",
                yaxis_title="Значение",
                hovermode="x unified",
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Метрики в виде карточек
            col1, col2, col3 = st.columns(3)
            mae = result.metrics.get('mae', np.nan)
            rmse = result.metrics.get('rmse', np.nan)
            mape = result.metrics.get('mape', np.nan)
            col1.metric("MAE", f"{mae:.4f}")
            col2.metric("RMSE", f"{rmse:.4f}")
            col3.metric("MAPE", f"{mape:.2f}%")
            
            # Проверяем, не равны ли метрики нулю (подозрительно для реальных данных)
            if not np.isnan(mae) and not np.isnan(rmse) and mae == 0.0 and rmse == 0.0:
                st.error("⚠️ **Внимание:** Метрики MAE и RMSE равны нулю. Это подозрительно и может указывать на: "
                        "1) Идеальное совпадение прогноза с фактическими значениями (маловероятно для реальных данных), "
                        "2) Проблему с данными (все значения одинаковые или данные не загружены правильно), "
                        "3) Ошибку в вычислении метрик или прогноза. "
                        "Рекомендуется проверить данные и процесс прогнозирования.")
            
            with st.expander("Подробнее о модели"):
                # Показываем предупреждение о fallback, если используется
                if result.details.get("fallback", False):
                    st.warning(f"⚠️ **Внимание:** {result.details.get('warning', 'Использован fallback метод')}")
                
                # Красивое отображение метрик
                st.markdown("**📊 Метрики:**")
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    if "mae" in result.metrics:
                        st.metric("MAE", f"{result.metrics.get('mae', np.nan):.4f}", 
                                 help="Среднее абсолютное отклонение")
                    if "rmse" in result.metrics:
                        st.metric("RMSE", f"{result.metrics.get('rmse', np.nan):.4f}", 
                                 help="Среднеквадратичное отклонение")
                
                with metrics_col2:
                    if "mape" in result.metrics:
                        st.metric("MAPE", f"{result.metrics.get('mape', np.nan):.2f}%", 
                                 help="Средняя абсолютная процентная ошибка")
                
                # Красивое отображение параметров модели
                if result.details:
                    st.markdown("**⚙️ Параметры модели:**")
                    if isinstance(result.details, dict):
                        details_col1, details_col2 = st.columns(2)
                        items = list(result.details.items())
                        mid = len(items) // 2
                        
                        with details_col1:
                            for key, value in items[:mid]:
                                if value is not None:
                                    st.markdown(f"**{key}:** `{value}`")
                        
                        with details_col2:
                            for key, value in items[mid:]:
                                if value is not None:
                                    st.markdown(f"**{key}:** `{value}`")
                    else:
                        st.markdown(f"`{result.details}`")

    return analysis_data


__all__ = [
    "stage5",
    "ForecastResult",
    "apply_boxcox_transform",
    "inverse_boxcox_values",
    "finalize_forecast_result",
    "auto_arima_forecast",
    "run_garch_forecast",
    "run_tbats_forecast",
    "run_prophet_forecast",
    "prepare_prophet_dataframe",
    "run_var_model",
    "run_vecm_model",
    "prepare_forecast_windows",
    "get_optimal_train_size",
]


