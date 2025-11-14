"""
Stationarity transformations (Stage 5) utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import inv_boxcox
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class TransformationStep:
    name: str
    params: Dict[str, float | int | List[float]]


@dataclass
class PipelineResult:
    name: str
    steps: List[TransformationStep]
    transformed_series: pd.Series
    adf_stat: Optional[float]
    adf_pvalue: Optional[float]
    kpss_stat: Optional[float]
    kpss_pvalue: Optional[float]
    adf_stationary: bool
    kpss_stationary: bool
    score: float


class StationarityTransformer:
    """Applies transformation pipelines and evaluates stationarity."""

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
        self._manual_boxcox_lambda: Optional[float] = None

    def evaluate_pipelines(
        self,
        seasonal_period: int = 7,
        use_boxcox: bool = True,
        use_log: bool = True,
        manual_boxcox_lambda: Optional[float] = None,
    ) -> tuple[List[PipelineResult], PipelineResult]:
        pipelines: Dict[str, List[str]] = {
            "Original": [],
        }

        base_transforms: List[str] = []
        if use_log:
            base_transforms.append("log")
        if use_boxcox:
            base_transforms.append("boxcox")

        for base in base_transforms:
            pipelines[f"{base.title()}"] = [base]

        pipelines["Diff-1"] = ["diff1"]
        pipelines[f"Diff-seasonal-{seasonal_period}"] = [f"diffs_{seasonal_period}"]
        pipelines[f"Diff-1+Diff-seasonal-{seasonal_period}"] = ["diff1", f"diffs_{seasonal_period}"]

        for base in base_transforms:
            pipelines[f"{base.title()}+Diff-1"] = [base, "diff1"]
            pipelines[f"{base.title()}+Diff-seasonal-{seasonal_period}"] = [base, f"diffs_{seasonal_period}"]
            pipelines[f"{base.title()}+Diff-1+Diff-seasonal-{seasonal_period}"] = [
                base,
                "diff1",
                f"diffs_{seasonal_period}",
            ]

        evaluated: List[PipelineResult] = []

        for name, steps_tags in pipelines.items():
            steps_info: List[TransformationStep] = []
            transformed = self.series.copy()

            try:
                for tag in steps_tags:
                    if tag == "log":
                        transformed, info = self._log_transform(transformed)
                        steps_info.append(info)
                    elif tag == "boxcox":
                        transformed, info = self._boxcox_transform(transformed)
                        steps_info.append(info)
                    elif tag == "diff1":
                        transformed, info = self._diff_transform(transformed, lag=1)
                        steps_info.append(info)
                    elif tag.startswith("diffs_"):
                        period = int(tag.split("_")[1])
                        transformed, info = self._diff_transform(transformed, lag=period)
                        steps_info.append(info)

                adf_stat, adf_p, kpss_stat, kpss_p = self._evaluate_stationarity(transformed.dropna())
                adf_stationary = adf_p is not None and adf_p < 0.05
                kpss_stationary = kpss_p is not None and kpss_p > 0.05

                score = self._score_pipeline(
                    len(steps_tags),
                    adf_p,
                    kpss_p,
                    adf_stationary,
                    kpss_stationary,
                )

                evaluated.append(
                    PipelineResult(
                        name=name,
                        steps=steps_info,
                        transformed_series=transformed,
                        adf_stat=adf_stat,
                        adf_pvalue=adf_p,
                        kpss_stat=kpss_stat,
                        kpss_pvalue=kpss_p,
                        adf_stationary=adf_stationary,
                        kpss_stationary=kpss_stationary,
                        score=score,
                    )
                )
            except Exception:
                continue

        if not evaluated:
            raise ValueError("Не удалось вычислить ни одну комбинацию преобразований")

        best = self._select_best_pipeline(evaluated)
        return evaluated, best

    def apply_steps(self, steps: List[TransformationStep]) -> pd.Series:
        series = self.series.copy()
        for step in steps:
            if step.name == "log":
                series = np.log(series + step.params.get("shift", 0.0))
            elif step.name == "boxcox":
                shift = step.params.get("shift", 0.0)
                lam = step.params.get("lambda", 1.0)
                series = pd.Series(
                    stats.boxcox(series + shift, lmbda=lam),
                    index=series.index,
                )
            elif step.name == "diff":
                lag = int(step.params.get("lag", 1))
                series = series.diff(lag)

        return series.dropna()

    def inverse_transform(
        self,
        transformed_values: pd.Series,
        steps: List[TransformationStep],
    ) -> pd.Series:
        series = transformed_values.copy()
        for step in reversed(steps):
            if step.name == "diff":
                lag = int(step.params.get("lag", 1))
                init_values = step.params.get("init_values", [])
                if len(init_values) < lag:
                    raise ValueError("Недостаточно исходных значений для обратного дифференцирования")

                restored = list(init_values)
                diffs = series.tolist()

                for diff_value in diffs:
                    restored.append(restored[-lag] + diff_value)

                series = pd.Series(restored)
            elif step.name == "boxcox":
                shift = step.params.get("shift", 0.0)
                lam = step.params.get("lambda", 1.0)
                series = pd.Series(inv_boxcox(series, lam) - shift)
            elif step.name == "log":
                shift = step.params.get("shift", 0.0)
                series = np.exp(series) - shift

        return series

    def _log_transform(self, series: pd.Series) -> tuple[pd.Series, TransformationStep]:
        min_value = series.min()
        shift = 0.0
        if min_value <= 0:
            shift = abs(min_value) + 1e-6
        transformed = np.log(series + shift)
        info = TransformationStep(name="log", params={"shift": shift})
        return transformed, info

    def _boxcox_transform(self, series: pd.Series) -> tuple[pd.Series, TransformationStep]:
        min_value = series.min()
        shift = 0.0
        if min_value <= 0:
            shift = abs(min_value) + 1e-6
        adjusted = series + shift
        if self._manual_boxcox_lambda is not None:
            lam = float(self._manual_boxcox_lambda)
        else:
            lam = stats.boxcox_normmax(adjusted, brack=(-5, 5))
        transformed = stats.boxcox(adjusted, lmbda=lam)
        info = TransformationStep(
            name="boxcox",
            params={"shift": shift, "lambda": lam},
        )
        return pd.Series(transformed, index=series.index), info

    def _diff_transform(self, series: pd.Series, lag: int) -> tuple[pd.Series, TransformationStep]:
        init_values = series.iloc[:lag].tolist()
        transformed = series.diff(lag)
        info = TransformationStep(
            name="diff",
            params={"lag": lag, "init_values": init_values},
        )
        return transformed.dropna(), info

    def _evaluate_stationarity(
        self, series: pd.Series
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        adf_stat = adf_p = kpss_stat = kpss_p = None

        try:
            adf_result = adfuller(series.dropna(), autolag="AIC")
            adf_stat, adf_p = adf_result[0], adf_result[1]
        except Exception:
            pass

        try:
            kpss_result = kpss(series.dropna(), regression="c", nlags="auto")
            kpss_stat, kpss_p = kpss_result[0], kpss_result[1]
        except Exception:
            pass

        return adf_stat, adf_p, kpss_stat, kpss_p

    def _score_pipeline(
        self,
        steps_count: int,
        adf_p: Optional[float],
        kpss_p: Optional[float],
        adf_stationary: bool,
        kpss_stationary: bool,
    ) -> float:
        score = 0.0
        if adf_p is not None:
            score += max(0.0, 1.0 - min(adf_p, 1.0))
        if kpss_p is not None:
            score += max(0.0, min(kpss_p, 1.0))
        if adf_stationary:
            score += 1.0
        if kpss_stationary:
            score += 1.0
        score -= 0.05 * steps_count
        return score

    def _select_best_pipeline(self, pipelines: List[PipelineResult]) -> PipelineResult:
        stationary = [
            p for p in pipelines if p.adf_stationary and p.kpss_stationary
        ]
        if stationary:
            stationary.sort(
                key=lambda p: (
                    len(p.steps),
                    p.adf_pvalue if p.adf_pvalue is not None else 1.0,
                    -p.score,
                )
            )
            return stationary[0]

        pipelines.sort(
            key=lambda p: (
                -(p.score),
                p.adf_pvalue if p.adf_pvalue is not None else 1.0,
            )
        )
        return pipelines[0]
