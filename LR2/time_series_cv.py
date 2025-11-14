"""
Time series cross-validation utilities (Stage 4).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit


def _clone_model(model: RegressorMixin) -> RegressorMixin:
    if isinstance(model, RandomForestRegressor):
        return RandomForestRegressor(n_estimators=model.n_estimators, random_state=42)
    if isinstance(model, LinearRegression):
        return LinearRegression()
    raise ValueError("Неподдерживаемый базовый регрессор")


def _create_model(model_name: str) -> RegressorMixin:
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(n_estimators=200, random_state=42)
    return LinearRegression()


def _build_lag_matrix(series: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(max_lag, len(series)):
        X.append(series[i - max_lag : i])
        y.append(series[i])
    return np.asarray(X), np.asarray(y)


@dataclass
class FoldResult:
    scheme: str
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    mae: float
    rmse: float
    runtime_seconds: float
    predictions: np.ndarray
    actuals: np.ndarray

    def to_dict(self) -> Dict[str, float | int | str]:
        return {
            "Схема": self.scheme,
            "Фолд": self.fold,
            "Train start": self.train_start,
            "Train end": self.train_end,
            "Test start": self.test_start,
            "Test end": self.test_end,
            "MAE": self.mae,
            "RMSE": self.rmse,
            "Время (сек.)": self.runtime_seconds,
        }


@dataclass
class CrossValidationSummary:
    scheme: str
    fold_results: List[FoldResult]

    @property
    def mean_mae(self) -> float:
        return float(np.mean([fold.mae for fold in self.fold_results]))

    @property
    def mean_rmse(self) -> float:
        return float(np.mean([fold.rmse for fold in self.fold_results]))

    @property
    def runtime_seconds(self) -> float:
        return float(np.sum([fold.runtime_seconds for fold in self.fold_results]))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([fold.to_dict() for fold in self.fold_results])


class TimeSeriesCrossValidator:
    """Кросс-валидация временных рядов без утечки будущего."""

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
        self.series = data[value_column].astype(float).to_numpy()
        self.value_column = value_column

    def evaluate(
        self,
        max_lag: int,
        model_name: str,
        sliding_train_window: int,
        sliding_test_window: int,
        expanding_initial_window: int,
        expanding_test_window: int,
        tss_splits: int,
    ) -> Dict[str, CrossValidationSummary]:
        model = _create_model(model_name)

        summaries: Dict[str, CrossValidationSummary] = {}

        summaries["Sliding window"] = self._sliding_window(
            model,
            max_lag,
            sliding_train_window,
            sliding_test_window,
        )

        summaries["Expanding window"] = self._expanding_window(
            model,
            max_lag,
            expanding_initial_window,
            expanding_test_window,
        )

        summaries["TimeSeriesSplit"] = self._time_series_split(
            model,
            max_lag,
            n_splits=tss_splits,
        )

        return summaries

    def _train_and_forecast(
        self,
        model: RegressorMixin,
        train_series: np.ndarray,
        test_series: np.ndarray,
        max_lag: int,
    ) -> tuple[np.ndarray, float, float, float]:
        if len(train_series) <= max_lag:
            raise ValueError("Размер обучающей выборки недостаточен для указанного максимального лага")

        X_train, y_train = _build_lag_matrix(train_series, max_lag)
        local_model = _clone_model(model)

        start_time = time.perf_counter()
        local_model.fit(X_train, y_train)

        history = list(train_series)
        predictions: List[float] = []

        for actual_value in test_series:
            input_vector = np.array(history[-max_lag:]).reshape(1, -1)
            pred_value = float(local_model.predict(input_vector)[0])
            predictions.append(pred_value)
            # walk-forward update with actual value to avoid future leakage
            history.append(actual_value)

        runtime = time.perf_counter() - start_time

        predictions_array = np.array(predictions)
        actual_array = np.array(test_series)
        errors = predictions_array - actual_array
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))

        return predictions_array, mae, rmse, runtime

    def _sliding_window(
        self,
        model: RegressorMixin,
        max_lag: int,
        train_window: int,
        test_window: int,
    ) -> CrossValidationSummary:
        if train_window <= max_lag:
            raise ValueError("Размер обучающего окна должен быть больше максимального лага")
        if test_window < 1:
            raise ValueError("Размер тестового окна должен быть ≥ 1")

        fold_results: List[FoldResult] = []
        total_length = len(self.series)
        fold_idx = 1

        for start in range(0, total_length - train_window - test_window + 1, test_window):
            train_start = start
            train_end = start + train_window
            test_end = train_end + test_window

            train_series = self.series[train_start:train_end]
            test_series = self.series[train_end:test_end]

            preds, mae, rmse, runtime = self._train_and_forecast(
                model,
                train_series,
                test_series,
                max_lag,
            )

            fold_results.append(
                FoldResult(
                    scheme="Sliding window",
                    fold=fold_idx,
                    train_start=self.dates.iloc[train_start],
                    train_end=self.dates.iloc[train_end - 1],
                    test_start=self.dates.iloc[train_end],
                    test_end=self.dates.iloc[test_end - 1],
                    mae=mae,
                    rmse=rmse,
                    runtime_seconds=runtime,
                    predictions=preds,
                    actuals=self.series[train_end:test_end],
                )
            )
            fold_idx += 1

        if not fold_results:
            raise ValueError("Не удалось сформировать ни одного фолда для скользящего окна. Проверьте параметры.")

        return CrossValidationSummary(scheme="Sliding window", fold_results=fold_results)

    def _expanding_window(
        self,
        model: RegressorMixin,
        max_lag: int,
        initial_train_window: int,
        test_window: int,
    ) -> CrossValidationSummary:
        if initial_train_window <= max_lag:
            raise ValueError("Начальное обучающее окно должно быть больше максимального лага")
        if test_window < 1:
            raise ValueError("Размер тестового окна должен быть ≥ 1")

        fold_results: List[FoldResult] = []
        total_length = len(self.series)
        fold_idx = 1

        train_end = initial_train_window
        while train_end + test_window <= total_length:
            train_series = self.series[:train_end]
            test_series = self.series[train_end : train_end + test_window]

            preds, mae, rmse, runtime = self._train_and_forecast(
                model,
                train_series,
                test_series,
                max_lag,
            )

            fold_results.append(
                FoldResult(
                    scheme="Expanding window",
                    fold=fold_idx,
                    train_start=self.dates.iloc[0],
                    train_end=self.dates.iloc[train_end - 1],
                    test_start=self.dates.iloc[train_end],
                    test_end=self.dates.iloc[train_end + test_window - 1],
                    mae=mae,
                    rmse=rmse,
                    runtime_seconds=runtime,
                    predictions=preds,
                    actuals=self.series[train_end : train_end + test_window],
                )
            )

            train_end += test_window
            fold_idx += 1

        if not fold_results:
            raise ValueError("Не удалось сформировать ни одного фолда для расширяющегося окна. Проверьте параметры.")

        return CrossValidationSummary(scheme="Expanding window", fold_results=fold_results)

    def _time_series_split(
        self,
        model: RegressorMixin,
        max_lag: int,
        n_splits: int,
    ) -> CrossValidationSummary:
        if n_splits < 2:
            raise ValueError("Количество фолдов TimeSeriesSplit должно быть ≥ 2")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results: List[FoldResult] = []

        for fold_idx, (train_index, test_index) in enumerate(tscv.split(self.series), start=1):
            train_series = self.series[train_index]
            test_series = self.series[test_index]

            if len(train_series) <= max_lag:
                raise ValueError(
                    "Слишком маленький размер обучающей выборки в одном из фолдов TimeSeriesSplit. "
                    "Уменьшите max_lag или n_splits."
                )

            preds, mae, rmse, runtime = self._train_and_forecast(
                model,
                train_series,
                test_series,
                max_lag,
            )

            fold_results.append(
                FoldResult(
                    scheme="TimeSeriesSplit",
                    fold=fold_idx,
                    train_start=self.dates.iloc[train_index[0]],
                    train_end=self.dates.iloc[train_index[-1]],
                    test_start=self.dates.iloc[test_index[0]],
                    test_end=self.dates.iloc[test_index[-1]],
                    mae=mae,
                    rmse=rmse,
                    runtime_seconds=runtime,
                    predictions=preds,
                    actuals=self.series[test_index],
                )
            )

        return CrossValidationSummary(scheme="TimeSeriesSplit", fold_results=fold_results)



