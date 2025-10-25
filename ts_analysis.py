"""
Модуль для анализа временных рядов
Содержит функции для декомпозиции, расчета автокорреляций и тестирования стационарности
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from scipy import stats


def calculate_rolling_stats(series, window):
    """
    Вычисление скользящих статистик
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    window : int
        Размер окна
    
    Возвращает:
    ----------
    dict : словарь со скользящим средним и стандартным отклонением
    """
    rolling_mean = series.rolling(window=window, center=True).mean()
    rolling_std = series.rolling(window=window, center=True).std()
    
    return {
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std
    }


def perform_decomposition(series, period, model='additive'):
    """
    Выполнение сезонной декомпозиции временного ряда
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    period : int
        Период сезонности
    model : str
        Тип модели ('additive' или 'multiplicative')
    
    Возвращает:
    ----------
    DecomposeResult : результат декомпозиции
    """
    # Проверка на достаточность данных
    if len(series) < 2 * period:
        raise ValueError(f"Недостаточно данных для декомпозиции. "
                        f"Требуется минимум {2 * period} точек, имеется {len(series)}")
    
    # Удаление NaN значений
    series_clean = series.dropna()
    
    # Выполнение декомпозиции
    decomposition = seasonal_decompose(
        series_clean,
        model=model,
        period=period,
        extrapolate_trend='freq'
    )
    
    return decomposition


def calculate_acf_pacf(series, max_lags=40):
    """
    Расчет автокорреляционной функции (ACF) и частичной автокорреляции (PACF)
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    max_lags : int
        Максимальное количество лагов
    
    Возвращает:
    ----------
    tuple : (acf_values, pacf_values, acf_confint, pacf_confint)
    """
    # Удаление NaN значений
    series_clean = series.dropna()
    
    # Ограничение max_lags если данных недостаточно
    max_lags = min(max_lags, len(series_clean) // 2 - 1)
    
    # Расчет ACF
    acf_values, acf_confint = acf(
        series_clean,
        nlags=max_lags,
        alpha=0.05,
        fft=False
    )
    
    # Расчет PACF
    pacf_values, pacf_confint = pacf(
        series_clean,
        nlags=max_lags,
        alpha=0.05,
        method='ywm'
    )
    
    return acf_values, pacf_values, acf_confint, pacf_confint


def test_stationarity(series):
    """
    Проведение тестов на стационарность
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    
    Возвращает:
    ----------
    tuple : (adf_result, kpss_result)
    """
    # Удаление NaN значений
    series_clean = series.dropna()
    
    # Расширенный тест Дики-Фуллера (ADF)
    adf_result = adfuller(series_clean, autolag='AIC')
    adf_output = {
        'adf_stat': adf_result[0],
        'p_value': adf_result[1],
        'lags_used': adf_result[2],
        'n_obs': adf_result[3],
        'critical_values': adf_result[4],
        'ic_best': adf_result[5] if len(adf_result) > 5 else None
    }
    
    # Тест KPSS
    kpss_result = kpss(series_clean, regression='c', nlags='auto')
    kpss_output = {
        'kpss_stat': kpss_result[0],
        'p_value': kpss_result[1],
        'lags_used': kpss_result[2],
        'critical_values': kpss_result[3]
    }
    
    return adf_output, kpss_output


def calculate_correlations(dataframe):
    """
    Расчет корреляционной матрицы
    
    Параметры:
    ----------
    dataframe : pd.DataFrame
        Датафрейм с признаками
    
    Возвращает:
    ----------
    pd.DataFrame : корреляционная матрица
    """
    # Выбор только числовых столбцов
    numeric_df = dataframe.select_dtypes(include=[np.number])
    
    # Расчет корреляции
    correlation_matrix = numeric_df.corr()
    
    return correlation_matrix


def detect_outliers(series, method='iqr', threshold=1.5):
    """
    Обнаружение выбросов во временном ряде
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    method : str
        Метод обнаружения ('iqr' или 'zscore')
    threshold : float
        Порог для определения выбросов
    
    Возвращает:
    ----------
    pd.Series : булева маска с выбросами
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(series.dropna()))
        outliers = pd.Series(False, index=series.index)
        outliers[series.notna()] = z_scores > threshold
    
    else:
        raise ValueError(f"Неизвестный метод: {method}")
    
    return outliers


def calculate_trend(series, method='linear'):
    """
    Расчет тренда временного ряда
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    method : str
        Метод расчета тренда ('linear', 'polynomial')
    
    Возвращает:
    ----------
    np.ndarray : значения тренда
    """
    x = np.arange(len(series))
    y = series.values
    
    # Удаление NaN
    mask = ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if method == 'linear':
        # Линейная регрессия
        coeffs = np.polyfit(x_clean, y_clean, 1)
        trend = np.polyval(coeffs, x)
    
    elif method == 'polynomial':
        # Полиномиальная регрессия (2-я степень)
        coeffs = np.polyfit(x_clean, y_clean, 2)
        trend = np.polyval(coeffs, x)
    
    else:
        raise ValueError(f"Неизвестный метод: {method}")
    
    return trend


def calculate_seasonality_strength(decomposition):
    """
    Расчет силы сезонности
    
    Параметры:
    ----------
    decomposition : DecomposeResult
        Результат декомпозиции
    
    Возвращает:
    ----------
    float : сила сезонности (0-1)
    """
    # Удаление NaN
    seasonal_var = np.nanvar(decomposition.seasonal)
    residual_var = np.nanvar(decomposition.resid)
    
    if seasonal_var + residual_var == 0:
        return 0.0
    
    strength = max(0, 1 - (residual_var / (seasonal_var + residual_var)))
    
    return strength


def calculate_trend_strength(decomposition):
    """
    Расчет силы тренда
    
    Параметры:
    ----------
    decomposition : DecomposeResult
        Результат декомпозиции
    
    Возвращает:
    ----------
    float : сила тренда (0-1)
    """
    # Удаление NaN
    trend_var = np.nanvar(decomposition.trend)
    residual_var = np.nanvar(decomposition.resid)
    
    if trend_var + residual_var == 0:
        return 0.0
    
    strength = max(0, 1 - (residual_var / (trend_var + residual_var)))
    
    return strength

