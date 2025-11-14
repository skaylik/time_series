"""
Модуль для реализации стратегий многоперого прогнозирования (Этап 3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _create_model(model_name: str) -> RegressorMixin:
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(n_estimators=200, random_state=42)
    return LinearRegression()


def _build_lag_matrix(series: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    if len(series) <= max_lag:
        raise ValueError("Недостаточно данных для построения lag-фич. Увеличьте размер выборки или уменьшите максимальный лаг.")

    X, y = [], []
    for i in range(max_lag, len(series)):
        X.append(series[i - max_lag : i])
        y.append(series[i])
    return np.asarray(X), np.asarray(y)


def _build_direct_matrix(series: np.ndarray, max_lag: int, horizon: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(max_lag, len(series) - step + 1):
        X.append(series[i - max_lag : i])
        y.append(series[i + step - 1])
    return np.asarray(X), np.asarray(y)


@dataclass
class StrategyResult:
    name: str
    predictions: np.ndarray
    mae_per_step: np.ndarray
    rmse_per_step: np.ndarray
    mape_per_step: np.ndarray
    cumulative_mae: np.ndarray
    runtime_seconds: float
    test_mape: float


@dataclass
class BenchmarkResult:
    name: str
    forecast: np.ndarray
    mae: float
    rmse: float
    mape: float


class ForecastingStrategies:
    """Реализует рекурсивную, прямую и гибридную стратегии прогнозирования."""

    def __init__(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
    ) -> None:
        if date_column not in df.columns:
            raise ValueError(f"Столбец с датой '{date_column}' не найден")
        if value_column not in df.columns:
            raise ValueError(f"Целевая переменная '{value_column}' не найдена")

        data = df[[date_column, value_column]].copy()
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.dropna(subset=[date_column, value_column]).sort_values(date_column)
        data = data.reset_index(drop=True)

        self.dates = data[date_column]
        self.series = data[value_column].astype(float).to_numpy()
        self.value_column = value_column

    def evaluate(
        self,
        horizon: int = 7,
        max_lag: int = 30,
        model_name: str = "LinearRegression",
        hybrid_recursive_steps: int = 3,
    ) -> tuple[Dict[str, StrategyResult], np.ndarray, BenchmarkResult]:
        if horizon < 1:
            raise ValueError("Горизонт прогнозирования должен быть ≥ 1")
        if len(self.series) <= (max_lag + horizon):
            raise ValueError("Недостаточно точек в ряду для выбранных параметров. Уменьшите лаг или горизонт.")

        base_model = _create_model(model_name)
        train_series = self.series[:-horizon]
        actual = self.series[-horizon:]

        strategies: Dict[str, StrategyResult] = {}

        strategies['recursive'] = self._recursive_strategy(train_series, actual, base_model, max_lag)
        strategies['direct'] = self._direct_strategy(train_series, actual, base_model, max_lag)

        hybrid_steps = max(1, min(hybrid_recursive_steps, horizon))
        strategies['hybrid'] = self._hybrid_strategy(
            train_series,
            actual,
            base_model,
            max_lag,
            hybrid_steps,
        )

        baseline_forecast = np.repeat(train_series[-1], horizon)
        baseline_mae = float(np.mean(np.abs(baseline_forecast - actual)))
        baseline_rmse = float(np.sqrt(np.mean((baseline_forecast - actual) ** 2)))
        with np.errstate(divide='ignore', invalid='ignore'):
            baseline_mape_arr = np.abs((baseline_forecast - actual) / actual)
            baseline_mape_arr[~np.isfinite(baseline_mape_arr)] = np.nan
        baseline_mape = float(np.nanmean(baseline_mape_arr)) if np.isnan(baseline_mape_arr).sum() < len(baseline_mape_arr) else float('nan')
        benchmark = BenchmarkResult(
            name="Наивный (последнее значение)",
            forecast=baseline_forecast,
            mae=baseline_mae,
            rmse=baseline_rmse,
            mape=baseline_mape,
        )

        return strategies, actual, benchmark

    def _compute_metrics(self, predictions: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        mae = np.abs(predictions - actual)
        rmse = np.sqrt((predictions - actual) ** 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.abs((predictions - actual) / actual)
            mape[~np.isfinite(mape)] = np.nan
        cumulative = np.cumsum(mae)
        overall_mape = float(np.nanmean(mape)) if np.isnan(mape).sum() < len(mape) else float('nan')
        return mae, rmse, cumulative, mape, overall_mape

    def _recursive_strategy(
        self,
        train_series: np.ndarray,
        actual: np.ndarray,
        model: RegressorMixin,
        max_lag: int,
    ) -> StrategyResult:
        start_time = time.perf_counter()
        X_train, y_train = _build_lag_matrix(train_series, max_lag)
        model_rec = self._clone_model(model)
        model_rec.fit(X_train, y_train)

        history = list(train_series[-max_lag:])
        predictions = []
        for _ in range(len(actual)):
            input_vector = np.array(history[-max_lag:]).reshape(1, -1)
            next_pred = float(model_rec.predict(input_vector)[0])
            predictions.append(next_pred)
            history.append(next_pred)

        runtime = time.perf_counter() - start_time
        predictions_array = np.array(predictions)
        mae, rmse, cumulative, mape, overall_mape = self._compute_metrics(predictions_array, actual)
        return StrategyResult("Рекурсивная", predictions_array, mae, rmse, mape, cumulative, runtime, overall_mape)

    def _direct_strategy(
        self,
        train_series: np.ndarray,
        actual: np.ndarray,
        model: RegressorMixin,
        max_lag: int,
    ) -> StrategyResult:
        start_time = time.perf_counter()
        predictions: List[float] = []
        latest_lags = train_series[-max_lag:]

        for step in range(1, len(actual) + 1):
            X_train, y_train = _build_direct_matrix(train_series, max_lag, len(actual), step)
            model_direct = self._clone_model(model)
            model_direct.fit(X_train, y_train)
            pred = float(model_direct.predict(latest_lags.reshape(1, -1))[0])
            predictions.append(pred)

        runtime = time.perf_counter() - start_time
        predictions_array = np.array(predictions)
        mae, rmse, cumulative, mape, overall_mape = self._compute_metrics(predictions_array, actual)
        return StrategyResult("Прямая", predictions_array, mae, rmse, mape, cumulative, runtime, overall_mape)

    def _hybrid_strategy(
        self,
        train_series: np.ndarray,
        actual: np.ndarray,
        model: RegressorMixin,
        max_lag: int,
        recursive_steps: int,
    ) -> StrategyResult:
        start_time = time.perf_counter()

        X_train, y_train = _build_lag_matrix(train_series, max_lag)
        model_rec = self._clone_model(model)
        model_rec.fit(X_train, y_train)

        history = list(train_series[-max_lag:])
        predictions: List[float] = []

        # Рекурсивная часть
        for _ in range(recursive_steps):
            input_vector = np.array(history[-max_lag:]).reshape(1, -1)
            next_pred = float(model_rec.predict(input_vector)[0])
            predictions.append(next_pred)
            history.append(next_pred)

            if len(predictions) == len(actual):
                break

        # Прямая часть
        if len(predictions) < len(actual):
            remaining_steps = len(actual) - len(predictions)
            latest_lags = np.array(history[-max_lag:])

            for idx in range(len(predictions) + 1, len(actual) + 1):
                step = idx
                X_train_direct, y_train_direct = _build_direct_matrix(train_series, max_lag, len(actual), step)
                model_direct = self._clone_model(model)
                model_direct.fit(X_train_direct, y_train_direct)

                pred = float(model_direct.predict(latest_lags.reshape(1, -1))[0])
                predictions.append(pred)
                latest_lags = np.roll(latest_lags, -1)
                latest_lags[-1] = predictions[-1]

                if len(predictions) == len(actual):
                    break

        runtime = time.perf_counter() - start_time
        predictions_array = np.array(predictions)
        mae, rmse, cumulative, mape, overall_mape = self._compute_metrics(predictions_array, actual)
        return StrategyResult("Гибридная", predictions_array, mae, rmse, mape, cumulative, runtime, overall_mape)

    @staticmethod
    def _clone_model(model: RegressorMixin) -> RegressorMixin:
        if isinstance(model, RandomForestRegressor):
            return RandomForestRegressor(n_estimators=model.n_estimators, random_state=42)
        if isinstance(model, LinearRegression):
            return LinearRegression()
        raise ValueError("Неподдерживаемый тип модели")
