"""
Stage 6: classical exponential smoothing models (SES / Holt).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class ModelDiagnostics:
    ljung_box_pvalue: Optional[float]
    shapiro_pvalue: Optional[float]
    residuals: pd.Series
    fitted_values: pd.Series
    qq_theoretical: np.ndarray
    qq_sample: np.ndarray


@dataclass
class ModelResult:
    name: str
    forecast: pd.Series
    lower_ci: pd.Series
    upper_ci: pd.Series
    test_mae: float
    test_rmse: float
    test_mape: float
    runtime_seconds: float
    params: Dict[str, float]
    diagnostics: ModelDiagnostics


@dataclass
class BenchmarkResult:
    name: str
    forecast: pd.Series
    test_mae: float
    test_rmse: float
    test_mape: float


class ExponentialSmoothingRunner:
    """Run SES / Holt models and collect diagnostics."""

    def __init__(self, df: pd.DataFrame, date_column: str, value_column: str) -> None:
        if date_column not in df.columns:
            raise ValueError(f"Столбец с датой '{date_column}' не найден")
        if value_column not in df.columns:
            raise ValueError(f"Целевая переменная '{value_column}' не найдена")

        data = df[[date_column, value_column]].copy()
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.dropna(subset=[date_column, value_column]).sort_values(date_column)
        data = data.reset_index(drop=True)

        self.dates = data[date_column]
        self.series = data[value_column].astype(float)
        self.value_column = value_column

    def evaluate(
        self,
        horizon: int,
        seasonal_period: Optional[int] = None,
        include_multiplicative: bool = True,
    ) -> tuple[List[ModelResult], BenchmarkResult, List[str]]:
        if horizon < 1:
            raise ValueError("Горизонт прогноза должен быть ≥ 1")
        if len(self.series) <= horizon + 5:
            raise ValueError("Недостаточно данных для оценки. Уменьшите горизонт или загрузите больший ряд.")

        train = self.series.iloc[:-horizon]
        test = self.series.iloc[-horizon:]
        forecast_index = self.dates.iloc[-horizon:].reset_index(drop=True)

        results: List[ModelResult] = []
        failure_messages: List[str] = []

        baseline_forecast = pd.Series(np.repeat(train.iloc[-1], horizon), index=forecast_index)
        baseline_errors = baseline_forecast.values - test.values
        baseline_mae = float(np.mean(np.abs(baseline_errors)))
        baseline_rmse = float(np.sqrt(np.mean(baseline_errors ** 2)))
        with np.errstate(divide='ignore', invalid='ignore'):
            baseline_mape_array = np.abs(baseline_errors / test.values)
            baseline_mape_array[~np.isfinite(baseline_mape_array)] = np.nan
        baseline_mape = float(np.nanmean(baseline_mape_array)) if np.isnan(baseline_mape_array).sum() < len(baseline_mape_array) else float('nan')
        benchmark = BenchmarkResult("Наивный (последнее значение)", baseline_forecast, baseline_mae, baseline_rmse, baseline_mape)

        configs = [
            {"name": "SES", "trend": None, "seasonal": None, "min_obs": 2},
            {"name": "Holt (add)", "trend": "add", "seasonal": None, "min_obs": 4},
        ]

        if include_multiplicative and (self.series > 0).all():
            configs.append({"name": "Holt (mul)", "trend": "mul", "seasonal": None, "min_obs": 4})

        for cfg in configs:
            if len(train) < cfg.get("min_obs", 2):
                failure_messages.append(
                    f"{cfg['name']}: недостаточно наблюдений ({len(train)} < {cfg.get('min_obs', 2)})"
                )
                continue
            try:
                result = self._fit_and_forecast(
                    train=train,
                    test=test,
                    forecast_index=forecast_index,
                    trend=cfg["trend"],
                    seasonal=cfg.get("seasonal"),
                    seasonal_periods=seasonal_period,
                )
                results.append(ModelResult(name=cfg["name"], **result))
            except Exception as exc:
                failure_messages.append(f"{cfg['name']}: {exc}")
                continue

        return results, benchmark, failure_messages

    def _fit_and_forecast(
        self,
        train: pd.Series,
        test: pd.Series,
        forecast_index: pd.Index,
        trend: Optional[str],
        seasonal: Optional[str],
        seasonal_periods: Optional[int],
    ) -> Dict[str, object]:
        start_time = time.perf_counter()

        model = ExponentialSmoothing(
            train,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods if seasonal else None,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
        runtime = time.perf_counter() - start_time

        forecast_values = fit.forecast(len(test))
        
        # Убедимся, что forecast_values - это numpy array
        if isinstance(forecast_values, pd.Series):
            forecast_values = forecast_values.values
        
        forecast_series = pd.Series(forecast_values, index=forecast_index)

        resid = fit.resid
        resid = resid.dropna()

        residual_std = float(resid.std(ddof=1)) if len(resid) > 1 else 0.0
        critical_value = 1.96
        lower_ci = forecast_series - critical_value * residual_std
        upper_ci = forecast_series + critical_value * residual_std

        # Приводим test к тем же индексам для корректного вычисления
        test_values = test.values if hasattr(test, 'values') else np.array(test)
        forecast_vals = forecast_series.values if hasattr(forecast_series, 'values') else np.array(forecast_series)
        
        # Проверка на корректность размеров
        if len(forecast_vals) != len(test_values):
            raise ValueError(f"Несоответствие размеров: прогноз {len(forecast_vals)}, тест {len(test_values)}")
        
        # Вычисление метрик
        errors = forecast_vals - test_values
        
        # Проверка на NaN/Inf
        if np.any(~np.isfinite(errors)):
            valid_mask = np.isfinite(errors)
            if np.sum(valid_mask) == 0:
                mae = float('nan')
                rmse = float('nan')
                mape = float('nan')
            else:
                errors_valid = errors[valid_mask]
                test_values_valid = test_values[valid_mask]
                mae = float(np.mean(np.abs(errors_valid)))
                rmse = float(np.sqrt(np.mean(errors_valid ** 2)))
                with np.errstate(divide='ignore', invalid='ignore'):
                    mape_array = np.abs(errors_valid / test_values_valid)
                    mape_array[~np.isfinite(mape_array)] = np.nan
                mape = float(np.nanmean(mape_array)) if np.sum(np.isfinite(mape_array)) > 0 else float('nan')
        else:
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            with np.errstate(divide='ignore', invalid='ignore'):
                mape_array = np.abs(errors / test_values)
                mape_array[~np.isfinite(mape_array)] = np.nan
            mape = float(np.nanmean(mape_array)) if np.sum(np.isfinite(mape_array)) > 0 else float('nan')

        ljung_box_pvalue = None
        shapiro_pvalue = None

        if len(resid) > 5:
            try:
                lb = acorr_ljungbox(resid, lags=[min(10, len(resid) // 5)], return_df=True)
                ljung_box_pvalue = float(lb["lb_pvalue"].iloc[-1])
            except Exception:
                pass

            try:
                shapiro_pvalue = float(shapiro(resid)[1])
            except Exception:
                pass

        qq_values = stats.probplot(resid, dist="norm")
        qq_theoretical = np.array(qq_values[0][0])
        qq_sample = np.array(qq_values[0][1])

        fitted_values = pd.Series(fit.fittedvalues, index=train.index)

        # Преобразование параметров в float (обработка массивов numpy)
        params = {}
        for k, v in fit.params.items():
            if isinstance(v, (np.ndarray, pd.Series)):
                # Если это массив, берём первый элемент или среднее
                if len(v) == 1:
                    params[k] = float(v[0])
                else:
                    params[k] = float(np.mean(v))
            else:
                try:
                    params[k] = float(v)
                except (TypeError, ValueError):
                    # Если не можем конвертировать, сохраняем как строку
                    params[k] = str(v)

        diagnostics = ModelDiagnostics(
            ljung_box_pvalue=ljung_box_pvalue,
            shapiro_pvalue=shapiro_pvalue,
            residuals=resid,
            fitted_values=fitted_values,
            qq_theoretical=qq_theoretical,
            qq_sample=qq_sample,
        )

        result_dict = {
            "forecast": forecast_series,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "test_mae": mae,
            "test_rmse": rmse,
            "test_mape": mape,
            "runtime_seconds": runtime,
            "params": params,
            "diagnostics": diagnostics,
        }
        
        return result_dict
