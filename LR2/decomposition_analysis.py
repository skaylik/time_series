# decomposition_analysis.py
"""
Модуль для углублённой декомпозиции временных рядов и анализа остатков.
Этап 1: Декомпозиция и анализ остатков
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
from scipy.stats import normaltest, jarque_bera
import warnings
warnings.filterwarnings('ignore')


class DecompositionAnalyzer:
    """
    Класс для декомпозиции временных рядов и анализа остатков.
    """
    
    def __init__(self, time_series, date_column=None, value_column=None):
        """
        Инициализация анализатора.
        
        Parameters:
        -----------
        time_series : pd.Series или pd.DataFrame
            Временной ряд для анализа
        date_column : str, optional
            Название столбца с датами (если передан DataFrame)
        value_column : str, optional
            Название столбца со значениями (если передан DataFrame)
        """
        if isinstance(time_series, pd.DataFrame):
            if date_column is None or value_column is None:
                raise ValueError("Для DataFrame необходимо указать date_column и value_column")
            
            # Устанавливаем дату как индекс
            df = time_series.copy()
            
            # Преобразуем дату в DatetimeIndex
            try:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            except Exception as e:
                raise ValueError(f"Ошибка при преобразовании даты: {e}")
            
            # Удаляем строки с некорректными датами
            df = df.dropna(subset=[date_column])
            
            if len(df) == 0:
                raise ValueError("После преобразования дат не осталось валидных данных")
            
            # Устанавливаем индекс
            df = df.set_index(date_column)
            
            # Явно преобразуем индекс в DatetimeIndex (даже если он уже datetime)
            # Это гарантирует, что индекс будет DatetimeIndex
            try:
                # Преобразуем индекс в DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    # Удаляем строки с некорректными датами
                    mask = pd.notna(df.index)
                    df = df[mask]
                else:
                    # Если уже DatetimeIndex, просто убеждаемся, что нет NaT
                    mask = pd.notna(df.index)
                    df = df[mask]
            except Exception as e:
                raise ValueError(f"Ошибка при преобразовании индекса в DatetimeIndex: {e}")
            
            # Финальная проверка после всех преобразований
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    f"Не удалось преобразовать столбец '{date_column}' в DatetimeIndex. "
                    f"Тип индекса: {type(df.index)}. "
                    f"Пример значений: {df.index[:5].tolist() if len(df) > 0 else 'нет данных'}"
                )
            
            self.series = df[value_column].dropna()
        else:
            self.series = time_series.dropna()
            
            # Если передан Series, проверяем индекс
            if not isinstance(self.series.index, pd.DatetimeIndex):
                # Пробуем преобразовать индекс
                try:
                    self.series.index = pd.to_datetime(self.series.index, errors='coerce')
                    # Удаляем строки с некорректными датами
                    mask = pd.notna(self.series.index)
                    self.series = self.series[mask]
                    
                    if not isinstance(self.series.index, pd.DatetimeIndex):
                        raise ValueError("Индекс временного ряда должен быть DatetimeIndex")
                except Exception as e:
                    raise ValueError(f"Не удалось преобразовать индекс в DatetimeIndex: {e}")
        
        # Финальная проверка
        if not isinstance(self.series.index, pd.DatetimeIndex):
            raise ValueError("Индекс временного ряда должен быть DatetimeIndex")
        
        # Сортируем по дате
        self.series = self.series.sort_index()
        
        self.results = {}
    
    def decompose(self, model='additive', period=None, extrapolate_trend='freq'):
        """
        Выполняет декомпозицию временного ряда.
        
        Parameters:
        -----------
        model : str
            Тип модели: 'additive' или 'multiplicative'
        period : int, optional
            Период сезонности. Если None, будет автоматически определен
        extrapolate_trend : str
            Метод экстраполяции тренда
        
        Returns:
        --------
        DecomposeResult : Результат декомпозиции
        """
        if period is None:
            # Автоматическое определение периода
            period = self._auto_detect_period()
        
        try:
            decomposition = seasonal_decompose(
                self.series,
                model=model,
                period=period,
                extrapolate_trend=extrapolate_trend
            )
            return decomposition
        except Exception as e:
            raise ValueError(f"Ошибка при декомпозиции: {e}")
    
    def _auto_detect_period(self):
        """
        Автоматическое определение периода сезонности.
        """
        # Пробуем стандартные периоды
        common_periods = [7, 30, 52, 365, 12]
        
        for period in common_periods:
            if len(self.series) >= 2 * period:
                return period
        
        # Если ничего не подошло, используем эвристику
        freq = pd.infer_freq(self.series.index)
        if freq:
            if 'D' in freq:
                return 7  # Недельная сезонность для дневных данных
            elif 'W' in freq:
                return 52  # Годовая сезонность для недельных данных
            elif 'M' in freq:
                return 12  # Годовая сезонность для месячных данных
        
        # По умолчанию
        return min(7, len(self.series) // 2)
    
    def test_stationarity(self, series, test_type='both'):
        """
        Проверяет стационарность временного ряда.
        
        Parameters:
        -----------
        series : pd.Series
            Временной ряд для проверки
        test_type : str
            'adf', 'kpss' или 'both'
        
        Returns:
        --------
        dict : Результаты тестов
        """
        results = {}
        
        if test_type in ['adf', 'both']:
            try:
                adf_result = adfuller(series.dropna())
                results['adf'] = {
                    'statistic': adf_result[0],
                    'pvalue': adf_result[1],
                    'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < 0.05
                }
            except Exception as e:
                results['adf'] = {'error': str(e)}
        
        if test_type in ['kpss', 'both']:
            try:
                kpss_result = kpss(series.dropna(), regression='c')
                results['kpss'] = {
                    'statistic': kpss_result[0],
                    'pvalue': kpss_result[1],
                    'critical_values': kpss_result[3],
                    'is_stationary': kpss_result[1] > 0.05
                }
            except Exception as e:
                results['kpss'] = {'error': str(e)}
        
        return results
    
    def test_normality(self, series):
        """
        Проверяет нормальность распределения остатков.
        
        Parameters:
        -----------
        series : pd.Series
            Остатки для проверки
        
        Returns:
        --------
        dict : Результаты тестов нормальности
        """
        series_clean = series.dropna()
        
        results = {}
        
        # Тест Д'Агостино-Пирсона
        try:
            stat, pvalue = normaltest(series_clean)
            results['d_agostino'] = {
                'statistic': stat,
                'pvalue': pvalue,
                'is_normal': pvalue > 0.05
            }
        except Exception as e:
            results['d_agostino'] = {'error': str(e)}
        
        # Тест Жарке-Бера
        try:
            stat, pvalue = jarque_bera(series_clean)
            results['jarque_bera'] = {
                'statistic': stat,
                'pvalue': pvalue,
                'is_normal': pvalue > 0.05
            }
        except Exception as e:
            results['jarque_bera'] = {'error': str(e)}
        
        # Описательная статистика
        results['descriptive'] = {
            'mean': series_clean.mean(),
            'std': series_clean.std(),
            'skewness': stats.skew(series_clean),
            'kurtosis': stats.kurtosis(series_clean),
            'min': series_clean.min(),
            'max': series_clean.max()
        }
        
        return results
    
    def analyze_residuals(self, residuals):
        """
        Полный анализ остатков.
        
        Parameters:
        -----------
        residuals : pd.Series
            Остатки для анализа
        
        Returns:
        --------
        dict : Результаты анализа остатков
        """
        residuals_clean = residuals.dropna()
        
        analysis = {
            'stationarity': self.test_stationarity(residuals_clean),
            'normality': self.test_normality(residuals_clean),
            'autocorrelation': self._check_autocorrelation(residuals_clean)
        }
        
        return analysis
    
    def _check_autocorrelation(self, series, lags=40):
        """
        Проверяет автокорреляцию в остатках.
        
        Parameters:
        -----------
        series : pd.Series
            Временной ряд
        lags : int
            Количество лагов для проверки
        
        Returns:
        --------
        dict : Результаты проверки автокорреляции
        """
        try:
            # Тест Льюнга-Бокса
            lb_result = acorr_ljungbox(series, lags=lags, return_df=True)
            
            return {
                'ljung_box': {
                    'statistics': lb_result['lb_stat'].tolist(),
                    'pvalues': lb_result['lb_pvalue'].tolist(),
                    'has_autocorrelation': (lb_result['lb_pvalue'] < 0.05).any()
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def compare_decompositions(self, periods=[7, 30, 365], models=['additive', 'multiplicative']):
        """
        Сравнивает различные варианты декомпозиции.
        
        Parameters:
        -----------
        periods : list
            Список периодов для проверки
        models : list
            Список моделей для проверки
        
        Returns:
        --------
        dict : Результаты сравнения всех вариантов
        """
        comparisons = {}
        
        for model in models:
            for period in periods:
                if len(self.series) < 2 * period:
                    continue
                
                key = f"{model}_{period}"
                
                try:
                    # Выполняем декомпозицию
                    decomp = self.decompose(model=model, period=period)
                    residuals = decomp.resid.dropna()
                    
                    # Анализируем остатки
                    residual_analysis = self.analyze_residuals(residuals)
                    
                    # Оцениваем качество
                    score = self._score_decomposition(residual_analysis, residuals)
                    
                    comparisons[key] = {
                        'model': model,
                        'period': period,
                        'decomposition': decomp,
                        'residual_analysis': residual_analysis,
                        'score': score,
                        'residual_stats': {
                            'mean': residuals.mean(),
                            'std': residuals.std(),
                            'variance': residuals.var()
                        }
                    }
                except Exception as e:
                    comparisons[key] = {'error': str(e)}
        
        return comparisons
    
    def _score_decomposition(self, residual_analysis, residuals):
        """
        Оценивает качество декомпозиции на основе анализа остатков.
        
        Parameters:
        -----------
        residual_analysis : dict
            Результаты анализа остатков
        residuals : pd.Series
            Остатки
        
        Returns:
        --------
        float : Оценка качества (чем выше, тем лучше)
        """
        score = 0.0
        
        # Стационарность остатков (важно!)
        if 'adf' in residual_analysis['stationarity']:
            adf = residual_analysis['stationarity']['adf']
            if 'is_stationary' in adf and adf['is_stationary']:
                score += 3.0
            elif 'pvalue' in adf:
                score += adf['pvalue'] * 3.0  # Частичные баллы
        
        if 'kpss' in residual_analysis['stationarity']:
            kpss = residual_analysis['stationarity']['kpss']
            if 'is_stationary' in kpss and kpss['is_stationary']:
                score += 2.0
            elif 'pvalue' in kpss:
                score += (1 - kpss['pvalue']) * 2.0
        
        # Нормальность остатков (желательно)
        if 'jarque_bera' in residual_analysis['normality']:
            jb = residual_analysis['normality']['jarque_bera']
            if 'is_normal' in jb and jb['is_normal']:
                score += 2.0
            elif 'pvalue' in jb:
                score += jb['pvalue'] * 2.0
        
        # Отсутствие автокорреляции (важно!)
        if 'autocorrelation' in residual_analysis:
            ac = residual_analysis['autocorrelation']
            if 'ljung_box' in ac:
                lb = ac['ljung_box']
                if 'has_autocorrelation' in lb and not lb['has_autocorrelation']:
                    score += 3.0
        
        # Низкая дисперсия остатков (желательно)
        residual_var = residuals.var()
        series_var = self.series.var()
        if series_var > 0:
            variance_ratio = 1 - (residual_var / series_var)
            score += variance_ratio * 2.0
        
        return score
    
    def get_best_decomposition(self, periods=[7, 30, 365], models=['additive', 'multiplicative']):
        """
        Находит лучшую декомпозицию на основе анализа остатков.
        
        Parameters:
        -----------
        periods : list
            Список периодов для проверки
        models : list
            Список моделей для проверки
        
        Returns:
        --------
        dict : Информация о лучшей декомпозиции
        """
        comparisons = self.compare_decompositions(periods=periods, models=models)
        
        # Фильтруем успешные декомпозиции
        valid_comparisons = {k: v for k, v in comparisons.items() if 'error' not in v}
        
        if not valid_comparisons:
            raise ValueError("Не удалось выполнить ни одну декомпозицию")
        
        # Находим лучшую по оценке
        best_key = max(valid_comparisons.keys(), key=lambda k: valid_comparisons[k]['score'])
        best = valid_comparisons[best_key]
        
        return {
            'key': best_key,
            'model': best['model'],
            'period': best['period'],
            'decomposition': best['decomposition'],
            'score': best['score'],
            'residual_analysis': best['residual_analysis'],
            'all_comparisons': comparisons
        }