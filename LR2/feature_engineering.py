"""
Модуль для расширенного feature engineering временных рядов (Этап 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureEngineeringConfig:
    """Параметры генерации признаков."""

    lags: Sequence[int] = (1, 7, 30)
    rolling_windows: Sequence[int] = (7, 30, 90)
    include_cyclical: bool = True
    include_volatility: bool = True
    include_weekend_flag: bool = True
    include_holidays: bool = False
    holiday_dates: Optional[Iterable[pd.Timestamp]] = None
    drop_na: bool = True


@dataclass
class FeatureEngineeringResult:
    features: pd.DataFrame
    generated_columns: List[str] = field(default_factory=list)


class FeatureEngineer:
    """Генерация расширенных признаков для временного ряда."""

    def __init__(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
    ) -> None:
        self.original_df = df.copy()
        self.date_column = date_column
        self.value_column = value_column

        if date_column not in self.original_df.columns:
            raise ValueError(f"Столбец с датой '{date_column}' не найден")

        if value_column not in self.original_df.columns:
            raise ValueError(f"Столбец со значениями '{value_column}' не найден")

        self._prepare_dataframe()

    def _prepare_dataframe(self) -> None:
        df = self.original_df.copy()

        df[self.date_column] = pd.to_datetime(df[self.date_column], errors="coerce")
        df = df.dropna(subset=[self.date_column])
        df = df.sort_values(self.date_column).reset_index(drop=True)

        self.df = df

    def generate_features(
        self,
        config: Optional[FeatureEngineeringConfig] = None,
    ) -> FeatureEngineeringResult:
        if config is None:
            config = FeatureEngineeringConfig()

        df = self.df.copy()
        features_df = df[[self.date_column, self.value_column]].copy()
        generated_columns: List[str] = []

        timestamp = pd.to_datetime(features_df[self.date_column], errors="coerce")

        if timestamp.isna().all():
            raise ValueError(
                "Не удалось преобразовать столбец с датой в формат datetime."
            )

        features_df[self.date_column] = timestamp
        features_df = features_df.dropna(subset=[self.date_column]).reset_index(drop=True)
        timestamp = features_df[self.date_column]

        # Временные признаки
        features_df["day_of_week"] = timestamp.dt.dayofweek
        features_df["is_weekend"] = (features_df["day_of_week"] >= 5).astype(int)
        features_df["day_of_month"] = timestamp.dt.day
        features_df["day_of_year"] = timestamp.dt.dayofyear
        features_df["week_of_year"] = timestamp.dt.isocalendar().week.astype(int)
        features_df["month"] = timestamp.dt.month
        features_df["quarter"] = timestamp.dt.quarter
        features_df["year"] = timestamp.dt.year

        generated_columns.extend(
            [
                "day_of_week",
                "is_weekend",
                "day_of_month",
                "day_of_year",
                "week_of_year",
                "month",
                "quarter",
                "year",
            ]
        )

        if not config.include_weekend_flag:
            features_df = features_df.drop(columns=["is_weekend"])
            generated_columns.remove("is_weekend")

        # Циклические признаки
        if config.include_cyclical:
            features_df["day_of_week_sin"] = np.sin(2 * np.pi * features_df["day_of_week"] / 7)
            features_df["day_of_week_cos"] = np.cos(2 * np.pi * features_df["day_of_week"] / 7)
            features_df["month_sin"] = np.sin(2 * np.pi * features_df["month"] / 12)
            features_df["month_cos"] = np.cos(2 * np.pi * features_df["month"] / 12)
            features_df["day_of_year_sin"] = np.sin(2 * np.pi * features_df["day_of_year"] / 365.25)
            features_df["day_of_year_cos"] = np.cos(2 * np.pi * features_df["day_of_year"] / 365.25)

            generated_columns.extend(
                [
                    "day_of_week_sin",
                    "day_of_week_cos",
                    "month_sin",
                    "month_cos",
                    "day_of_year_sin",
                    "day_of_year_cos",
                ]
            )

        # Лаговые признаки
        for lag in config.lags:
            col_name = f"{self.value_column}_lag_{lag}"
            features_df[col_name] = features_df[self.value_column].shift(lag)
            generated_columns.append(col_name)

        # Скользящие статистики
        for window in config.rolling_windows:
            rolling_series = features_df[self.value_column].rolling(window=window, min_periods=1)

            mean_col = f"{self.value_column}_rolling_mean_{window}"
            std_col = f"{self.value_column}_rolling_std_{window}"
            min_col = f"{self.value_column}_rolling_min_{window}"
            max_col = f"{self.value_column}_rolling_max_{window}"

            features_df[mean_col] = rolling_series.mean()
            features_df[std_col] = rolling_series.std()
            features_df[min_col] = rolling_series.min()
            features_df[max_col] = rolling_series.max()

            generated_columns.extend([mean_col, std_col, min_col, max_col])

            if config.include_volatility:
                cv_col = f"{self.value_column}_rolling_cv_{window}"
                mean_values = features_df[mean_col]
                features_df[cv_col] = np.where(
                    mean_values.abs() > np.finfo(float).eps,
                    features_df[std_col] / mean_values,
                    np.nan,
                )
                generated_columns.append(cv_col)

        # Праздничные / событийные метки
        if config.include_holidays:
            features_df["is_holiday"] = 0
            if config.holiday_dates:
                holiday_index = pd.to_datetime(list(config.holiday_dates))
                holiday_index = holiday_index.normalize()
                features_df["is_holiday"] = (
                    features_df[self.date_column].dt.normalize().isin(holiday_index)
                ).astype(int)
            generated_columns.append("is_holiday")

        if config.drop_na:
            features_df = features_df.dropna().reset_index(drop=True)

        return FeatureEngineeringResult(features=features_df, generated_columns=generated_columns)
