"""
Модуль для декомпозиции временных рядов и анализа остатков.
Выполняет сезонную декомпозицию (аддитивную и мультипликативную),
анализирует остатки на стационарность (ADF, KPSS), нормальность и автокорреляцию.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss


class DecompositionAnalyzer:
    """Анализатор для декомпозиции временных рядов и анализа остатков."""

    def __init__(self, df: pd.DataFrame, date_column: str, value_column: str):
        """
        Инициализация анализатора.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame с данными временного ряда
        date_column : str
            Название столбца с датами
        value_column : str
            Название столбца со значениями временного ряда
        """
        self.df = df.copy()
        self.date_column = date_column
        self.value_column = value_column

        # Преобразование даты в datetime
        self.df[date_column] = pd.to_datetime(self.df[date_column], errors="coerce")

        # Создание временного ряда с датой в качестве индекса
        self.df = self.df.sort_values(date_column).dropna(subset=[date_column, value_column])
        self.df = self.df.set_index(date_column)

        # Извлечение временного ряда
        self.series = self.df[value_column].copy()
        self.series = self.series.dropna()

        if len(self.series) < 2:
            raise ValueError("Недостаточно данных для анализа временного ряда")

    def decompose(self, model: str = "additive", period: int = 7) -> Any:
        """
        Выполняет сезонную декомпозицию временного ряда.

        Parameters
        ----------
        model : str
            Тип модели: "additive" или "multiplicative"
        period : int
            Период сезонности

        Returns
        -------
        DecomposeResult
            Результат декомпозиции из statsmodels
        """
        if len(self.series) < 2 * period:
            raise ValueError(f"Недостаточно данных для периода {period}. Нужно минимум {2 * period} точек.")

        try:
            decomposition = seasonal_decompose(
                self.series,
                model=model,
                period=period,
                extrapolate_trend="freq",
            )
            return decomposition
        except Exception as e:
            raise ValueError(f"Ошибка при декомпозиции: {e}")

    def analyze_residuals(self, residuals: pd.Series) -> Dict[str, Any]:
        """
        Анализирует остатки декомпозиции.

        Parameters
        ----------
        residuals : pd.Series
            Остатки после декомпозиции

        Returns
        -------
        Dict[str, Any]
            Словарь с результатами анализа остатков
        """
        residuals_clean = residuals.dropna()

        if len(residuals_clean) < 10:
            return {
                "stationarity": {"adf": {"error": "Недостаточно данных"}, "kpss": {"error": "Недостаточно данных"}},
                "normality": {
                    "d_agostino": {"error": "Недостаточно данных"},
                    "jarque_bera": {"error": "Недостаточно данных"},
                },
            }

        result = {
            "stationarity": {},
            "normality": {},
            "autocorrelation": {},
        }

        # Тесты стационарности
        # ADF тест
        try:
            adf_result = adfuller(residuals_clean, autolag="AIC")
            # adf_result[4] содержит критические значения
            # Ключи могут быть числами (1, 5, 10) или строками ("1%", "5%", "10%")
            # Нормализуем формат ключей: если ключ не содержит "%", добавляем его
            critical_values = {}
            for k, v in adf_result[4].items():
                key_str = str(k)
                if "%" not in key_str:
                    critical_values[f"{key_str}%"] = v
                else:
                    critical_values[key_str] = v
            result["stationarity"]["adf"] = {
                "statistic": adf_result[0],
                "pvalue": adf_result[1],
                "critical_values": critical_values,
                "is_stationary": adf_result[1] < 0.05,
            }
        except Exception as e:
            result["stationarity"]["adf"] = {"error": str(e)}

        # KPSS тест
        try:
            kpss_result = kpss(residuals_clean, regression="c", nlags="auto")
            # kpss_result[3] содержит критические значения
            # Ключи могут быть числами (1, 5, 10) или строками ("1%", "5%", "10%")
            # Нормализуем формат ключей: если ключ не содержит "%", добавляем его
            critical_values = {}
            for k, v in kpss_result[3].items():
                key_str = str(k)
                if "%" not in key_str:
                    critical_values[f"{key_str}%"] = v
                else:
                    critical_values[key_str] = v
            result["stationarity"]["kpss"] = {
                "statistic": kpss_result[0],
                "pvalue": kpss_result[1],
                "critical_values": critical_values,
                "is_stationary": kpss_result[1] > 0.05,
            }
        except Exception as e:
            result["stationarity"]["kpss"] = {"error": str(e)}

        # Тесты нормальности
        # Тест Д'Агостино-Пирсона
        try:
            da_stat, da_pvalue = stats.normaltest(residuals_clean)
            result["normality"]["d_agostino"] = {
                "statistic": da_stat,
                "pvalue": da_pvalue,
                "is_normal": da_pvalue > 0.05,
            }
        except Exception as e:
            result["normality"]["d_agostino"] = {"error": str(e)}

        # Тест Жарке-Бера
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(residuals_clean)
            result["normality"]["jarque_bera"] = {
                "statistic": jb_stat,
                "pvalue": jb_pvalue,
                "is_normal": jb_pvalue > 0.05,
            }
        except Exception as e:
            result["normality"]["jarque_bera"] = {"error": str(e)}

        # Описательная статистика
        try:
            result["normality"]["descriptive"] = {
                "mean": float(residuals_clean.mean()),
                "std": float(residuals_clean.std()),
                "skewness": float(stats.skew(residuals_clean)),
                "kurtosis": float(stats.kurtosis(residuals_clean)),
            }
        except Exception:
            result["normality"]["descriptive"] = None

        # Тест автокорреляции (Ljung-Box)
        try:
            if len(residuals_clean) > 10:
                ljung_box = acorr_ljungbox(residuals_clean, lags=min(10, len(residuals_clean) // 4), return_df=True)
                lb_pvalue = ljung_box["lb_pvalue"].iloc[-1]
                result["autocorrelation"]["ljung_box"] = {
                    "pvalue": float(lb_pvalue),
                    "has_autocorrelation": lb_pvalue < 0.05,
                }
            else:
                result["autocorrelation"]["ljung_box"] = {"error": "Недостаточно данных"}
        except Exception as e:
            result["autocorrelation"]["ljung_box"] = {"error": str(e)}

        return result

    def _evaluate_decomposition(self, decomposition: Any, model: str, period: int) -> Dict[str, Any]:
        """
        Оценивает качество декомпозиции.

        Parameters
        ----------
        decomposition : DecomposeResult
            Результат декомпозиции
        model : str
            Тип модели
        period : int
            Период сезонности

        Returns
        -------
        Dict[str, Any]
            Словарь с оценкой качества
        """
        residuals = decomposition.resid.dropna()

        if len(residuals) == 0:
            return {
                "model": model,
                "period": period,
                "score": 0.0,
                "residual_stats": {"mean": 0.0, "std": 0.0, "variance": 0.0},
                "error": "Нет остатков для анализа",
            }

        # Оценка качества на основе дисперсии остатков (меньше = лучше)
        # Нормализуем оценку, чтобы большее значение означало лучшее качество
        residual_variance = residuals.var()
        residual_mean = abs(residuals.mean())
        residual_std = residuals.std()

        # Комплексная оценка: учитываем дисперсию, среднее и стандартное отклонение
        # Используем обратную функцию для дисперсии (меньше дисперсия = выше оценка)
        if residual_variance > 0:
            score = 100.0 / (1.0 + residual_variance + residual_mean + residual_std)
        else:
            score = 100.0

        return {
            "model": model,
            "period": period,
            "score": score,
            "residual_stats": {
                "mean": float(residual_mean),
                "std": float(residual_std),
                "variance": float(residual_variance),
            },
        }

    def get_best_decomposition(
        self, periods: List[int], models: List[str] = None
    ) -> Dict[str, Any]:
        """
        Находит лучшую декомпозицию среди всех комбинаций периодов и моделей.

        Parameters
        ----------
        periods : List[int]
            Список периодов для проверки
        models : List[str], optional
            Список моделей для проверки. По умолчанию ["additive", "multiplicative"]

        Returns
        -------
        Dict[str, Any]
            Словарь с информацией о лучшей декомпозиции
        """
        if models is None:
            models = ["additive", "multiplicative"]

        all_comparisons = {}
        best_score = -1
        best_decomposition = None
        best_model = None
        best_period = None

        for model in models:
            for period in periods:
                if len(self.series) < 2 * period:
                    continue

                try:
                    decomp = self.decompose(model=model, period=period)
                    evaluation = self._evaluate_decomposition(decomp, model, period)

                    if "error" not in evaluation:
                        key = f"{model}_{period}"
                        all_comparisons[key] = evaluation.copy()
                        all_comparisons[key]["decomposition"] = decomp

                        if evaluation["score"] > best_score:
                            best_score = evaluation["score"]
                            best_decomposition = decomp
                            best_model = model
                            best_period = period
                except Exception:
                    # Пропускаем комбинации, которые не работают
                    continue

        if best_decomposition is None:
            raise ValueError("Не удалось найти подходящую декомпозицию. Проверьте данные и параметры.")

        # Анализ остатков лучшей декомпозиции
        residual_analysis = self.analyze_residuals(best_decomposition.resid)

        return {
            "model": best_model,
            "period": best_period,
            "score": best_score,
            "decomposition": best_decomposition,
            "residual_analysis": residual_analysis,
            "all_comparisons": all_comparisons,
        }

