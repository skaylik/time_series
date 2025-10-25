"""
Модуль для предобработки временных рядов
Включает очистку данных, обработку пропусков и выбросов
"""

import pandas as pd
import numpy as np
from scipy import interpolate
from datetime import datetime
import pytz


def standardize_timezone(df, date_column, target_timezone='Europe/Moscow'):
    """
    Приведение временных меток к единому формату и временной зоне
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    date_column : str
        Название столбца с датой
    target_timezone : str
        Целевая временная зона
    
    Возвращает:
    ----------
    pd.DataFrame : датафрейм с приведенными временными метками
    dict : отчёт о преобразовании
    """
    df_clean = df.copy()
    
    # Преобразование в datetime если еще не преобразовано
    if not pd.api.types.is_datetime64_any_dtype(df_clean[date_column]):
        df_clean[date_column] = pd.to_datetime(df_clean[date_column], errors='coerce')
    
    original_count = len(df_clean)
    
    # Удаление строк с невалидными датами
    df_clean = df_clean.dropna(subset=[date_column])
    invalid_dates = original_count - len(df_clean)
    
    # Приведение к временной зоне
    tz = pytz.timezone(target_timezone)
    
    # Если данные уже с временной зоной - конвертируем
    if df_clean[date_column].dt.tz is not None:
        df_clean[date_column] = df_clean[date_column].dt.tz_convert(tz)
        tz_converted = True
    else:
        # Локализуем в целевую зону
        df_clean[date_column] = df_clean[date_column].dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
        # Удаляем строки с NaT после локализации
        df_clean = df_clean.dropna(subset=[date_column])
        tz_converted = False
    
    report = {
        'original_count': original_count,
        'final_count': len(df_clean),
        'invalid_dates_removed': invalid_dates,
        'timezone': target_timezone,
        'tz_aware_input': tz_converted
    }
    
    return df_clean, report


def remove_duplicates(df, date_column, strategy='first'):
    """
    Удаление дубликатов по времени
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    date_column : str
        Название столбца с датой
    strategy : str
        Стратегия удаления: 'first', 'last', 'mean'
    
    Возвращает:
    ----------
    pd.DataFrame : датафрейм без дубликатов
    dict : отчёт об удалении
    """
    df_clean = df.copy()
    original_count = len(df_clean)
    
    if strategy in ['first', 'last']:
        # Удаляем дубликаты, оставляя первый или последний
        df_clean = df_clean.drop_duplicates(subset=[date_column], keep=strategy)
    
    elif strategy == 'mean':
        # Группируем по дате и берем среднее для числовых столбцов
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            agg_dict = {col: 'mean' for col in numeric_cols}
            df_clean = df_clean.groupby(date_column, as_index=False).agg(agg_dict)
        else:
            df_clean = df_clean.drop_duplicates(subset=[date_column], keep='first')
    
    duplicates_removed = original_count - len(df_clean)
    
    report = {
        'original_count': original_count,
        'final_count': len(df_clean),
        'duplicates_removed': duplicates_removed,
        'strategy': strategy
    }
    
    return df_clean, report


def check_monotonicity(df, date_column):
    """
    Проверка монотонности временного ряда
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    date_column : str
        Название столбца с датой
    
    Возвращает:
    ----------
    dict : отчёт о монотонности
    pd.DataFrame : датафрейм, отсортированный по времени
    """
    df_sorted = df.copy().sort_values(by=date_column).reset_index(drop=True)
    
    # Проверка на прыжки назад во времени
    dates = df_sorted[date_column]
    time_diffs = dates.diff()
    
    # Находим отрицательные разницы (возвраты во времени)
    backward_jumps = (time_diffs < pd.Timedelta(0)).sum()
    
    # Проверка равномерности интервалов
    if len(time_diffs) > 1:
        time_diffs_clean = time_diffs.dropna()
        if len(time_diffs_clean) > 0:
            median_diff = time_diffs_clean.median()
            std_diff = time_diffs_clean.std()
            min_diff = time_diffs_clean.min()
            max_diff = time_diffs_clean.max()
        else:
            median_diff = std_diff = min_diff = max_diff = pd.Timedelta(0)
    else:
        median_diff = std_diff = min_diff = max_diff = pd.Timedelta(0)
    
    is_monotonic = df[date_column].is_monotonic_increasing
    
    report = {
        'is_monotonic': is_monotonic,
        'backward_jumps': backward_jumps,
        'median_interval': str(median_diff),
        'std_interval': str(std_diff),
        'min_interval': str(min_diff),
        'max_interval': str(max_diff),
        'total_points': len(df_sorted)
    }
    
    return report, df_sorted


def handle_missing_values(df, date_column, value_column, method='linear', window=None):
    """
    Обработка пропущенных значений
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    date_column : str
        Название столбца с датой
    value_column : str
        Название столбца со значениями
    method : str
        Метод обработки: 'linear', 'polynomial', 'rolling_mean', 'ffill', 'bfill', 'drop'
    window : int
        Размер окна для rolling_mean
    
    Возвращает:
    ----------
    pd.DataFrame : датафрейм с обработанными пропусками
    dict : отчёт об обработке
    """
    df_clean = df.copy()
    
    original_count = len(df_clean)
    missing_count = df_clean[value_column].isnull().sum()
    missing_percentage = (missing_count / original_count) * 100
    
    if missing_count == 0:
        report = {
            'original_count': original_count,
            'missing_count': 0,
            'missing_percentage': 0.0,
            'method': 'none',
            'filled_count': 0
        }
        return df_clean, report
    
    # Создаем копию для индексации по дате
    df_indexed = df_clean.set_index(date_column)
    
    if method == 'drop':
        # Удаление строк с пропусками если их меньше 5%
        if missing_percentage < 5:
            df_clean = df_clean.dropna(subset=[value_column])
            filled_count = 0
        else:
            filled_count = 0
            print(f"Предупреждение: {missing_percentage:.2f}% пропусков - слишком много для удаления")
    
    elif method == 'linear':
        # Линейная интерполяция
        df_clean[value_column] = df_indexed[value_column].interpolate(method='linear')
        filled_count = missing_count
    
    elif method == 'polynomial':
        # Полиномиальная интерполяция
        df_clean[value_column] = df_indexed[value_column].interpolate(method='polynomial', order=2)
        filled_count = missing_count
    
    elif method == 'cubic':
        # Кубическая интерполяция
        df_clean[value_column] = df_indexed[value_column].interpolate(method='cubic')
        filled_count = missing_count
    
    elif method == 'rolling_mean':
        # Заполнение скользящим средним
        if window is None:
            window = 3
        
        # Вычисляем скользящее среднее
        rolling_mean = df_clean[value_column].rolling(window=window, center=True, min_periods=1).mean()
        
        # Заполняем только пропуски
        df_clean[value_column] = df_clean[value_column].fillna(rolling_mean)
        filled_count = missing_count
    
    elif method == 'ffill':
        # Forward fill - заполнение предыдущим значением
        df_clean[value_column] = df_clean[value_column].fillna(method='ffill')
        filled_count = missing_count
    
    elif method == 'bfill':
        # Backward fill - заполнение следующим значением
        df_clean[value_column] = df_clean[value_column].fillna(method='bfill')
        filled_count = missing_count
    
    else:
        raise ValueError(f"Неизвестный метод: {method}")
    
    remaining_missing = df_clean[value_column].isnull().sum()
    
    report = {
        'original_count': original_count,
        'missing_count': missing_count,
        'missing_percentage': missing_percentage,
        'method': method,
        'filled_count': filled_count,
        'remaining_missing': remaining_missing,
        'final_count': len(df_clean)
    }
    
    return df_clean, report


def detect_outliers_iqr(series, multiplier=1.5):
    """
    Обнаружение выбросов методом IQR
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    multiplier : float
        Множитель IQR для определения выбросов (стандартно 1.5)
    
    Возвращает:
    ----------
    pd.Series : булева маска с выбросами (True = выброс)
    dict : статистика по выбросам
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (series < lower_bound) | (series > upper_bound)
    
    stats = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'total_outliers': outliers.sum(),
        'outlier_percentage': (outliers.sum() / len(series)) * 100,
        'outlier_indices': series[outliers].index.tolist()
    }
    
    return outliers, stats


def handle_outliers(df, value_column, method='clip', iqr_multiplier=1.5, window=None):
    """
    Обработка выбросов
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    value_column : str
        Название столбца со значениями
    method : str
        Метод обработки: 'clip', 'remove', 'interpolate', 'median'
    iqr_multiplier : float
        Множитель IQR
    window : int
        Размер окна для медианной фильтрации
    
    Возвращает:
    ----------
    pd.DataFrame : датафрейм с обработанными выбросами
    dict : отчёт об обработке
    """
    df_clean = df.copy()
    
    # Обнаружение выбросов
    outliers, stats = detect_outliers_iqr(df_clean[value_column], multiplier=iqr_multiplier)
    
    if stats['total_outliers'] == 0:
        report = {
            'total_outliers': 0,
            'outlier_percentage': 0.0,
            'method': 'none',
            'handled_count': 0
        }
        report.update(stats)
        return df_clean, report
    
    original_values = df_clean[value_column].copy()
    
    if method == 'clip':
        # Ограничение выбросов границами
        df_clean.loc[outliers, value_column] = df_clean.loc[outliers, value_column].clip(
            lower=stats['lower_bound'],
            upper=stats['upper_bound']
        )
        handled_count = stats['total_outliers']
    
    elif method == 'remove':
        # Удаление строк с выбросами
        df_clean = df_clean[~outliers]
        handled_count = stats['total_outliers']
    
    elif method == 'interpolate':
        # Замена выбросов на NaN и интерполяция
        df_clean.loc[outliers, value_column] = np.nan
        df_clean[value_column] = df_clean[value_column].interpolate(method='linear')
        handled_count = stats['total_outliers']
    
    elif method == 'median':
        # Медианная фильтрация
        if window is None:
            window = 3
        
        rolling_median = df_clean[value_column].rolling(window=window, center=True, min_periods=1).median()
        df_clean.loc[outliers, value_column] = rolling_median[outliers]
        handled_count = stats['total_outliers']
    
    else:
        raise ValueError(f"Неизвестный метод: {method}")
    
    report = {
        'method': method,
        'handled_count': handled_count,
        'final_count': len(df_clean)
    }
    report.update(stats)
    
    return df_clean, report


def resample_timeseries(df, date_column, value_column, freq='D', agg_method='mean'):
    """
    Ресемплирование временного ряда до единой частоты
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    date_column : str
        Название столбца с датой
    value_column : str
        Название столбца со значениями
    freq : str
        Целевая частота: 'D' (день), 'H' (час), 'W' (неделя), 'M' (месяц) и т.д.
    agg_method : str
        Метод агрегации: 'mean', 'sum', 'median', 'min', 'max', 'first', 'last'
    
    Возвращает:
    ----------
    pd.DataFrame : ресемплированный датафрейм
    dict : отчёт о ресемплировании
    """
    df_resampled = df.copy()
    
    # Устанавливаем дату как индекс
    df_resampled = df_resampled.set_index(date_column)
    
    original_count = len(df_resampled)
    original_freq = pd.infer_freq(df_resampled.index)
    
    # Выбираем только числовые столбцы для ресемплирования
    numeric_cols = df_resampled.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        raise ValueError("Нет числовых столбцов для ресемплирования")
    
    # Создаем словарь методов агрегации
    if agg_method == 'mean':
        df_resampled = df_resampled[numeric_cols].resample(freq).mean()
    elif agg_method == 'sum':
        df_resampled = df_resampled[numeric_cols].resample(freq).sum()
    elif agg_method == 'median':
        df_resampled = df_resampled[numeric_cols].resample(freq).median()
    elif agg_method == 'min':
        df_resampled = df_resampled[numeric_cols].resample(freq).min()
    elif agg_method == 'max':
        df_resampled = df_resampled[numeric_cols].resample(freq).max()
    elif agg_method == 'first':
        df_resampled = df_resampled[numeric_cols].resample(freq).first()
    elif agg_method == 'last':
        df_resampled = df_resampled[numeric_cols].resample(freq).last()
    else:
        raise ValueError(f"Неизвестный метод агрегации: {agg_method}")
    
    # Сбрасываем индекс
    df_resampled = df_resampled.reset_index()
    
    final_count = len(df_resampled)
    
    report = {
        'original_count': original_count,
        'final_count': final_count,
        'original_freq': original_freq if original_freq else 'irregular',
        'target_freq': freq,
        'agg_method': agg_method,
        'count_change': final_count - original_count
    }
    
    return df_resampled, report


def preprocess_pipeline(df, date_column, value_column, config):
    """
    Полный пайплайн предобработки данных
    
    Параметры:
    ----------
    df : pd.DataFrame
        Исходный датафрейм
    date_column : str
        Название столбца с датой
    value_column : str
        Название столбца со значениями
    config : dict
        Конфигурация с параметрами предобработки
    
    Возвращает:
    ----------
    pd.DataFrame : обработанный датафрейм
    dict : полный отчёт о всех этапах
    """
    df_processed = df.copy()
    reports = {}
    
    # 1. Стандартизация временной зоны
    if config.get('standardize_tz', False):
        df_processed, reports['timezone'] = standardize_timezone(
            df_processed,
            date_column,
            config.get('target_timezone', 'Europe/Moscow')
        )
    
    # 2. Удаление дубликатов
    if config.get('remove_duplicates', False):
        df_processed, reports['duplicates'] = remove_duplicates(
            df_processed,
            date_column,
            config.get('duplicate_strategy', 'first')
        )
    
    # 3. Проверка монотонности и сортировка
    if config.get('check_monotonicity', True):
        reports['monotonicity'], df_processed = check_monotonicity(
            df_processed,
            date_column
        )
    
    # 4. Ресемплирование
    if config.get('resample', False):
        df_processed, reports['resample'] = resample_timeseries(
            df_processed,
            date_column,
            value_column,
            config.get('resample_freq', 'D'),
            config.get('resample_method', 'mean')
        )
    
    # 5. Обработка пропусков
    if config.get('handle_missing', False):
        df_processed, reports['missing'] = handle_missing_values(
            df_processed,
            date_column,
            value_column,
            config.get('missing_method', 'linear'),
            config.get('missing_window', None)
        )
    
    # 6. Обработка выбросов
    if config.get('handle_outliers', False):
        df_processed, reports['outliers'] = handle_outliers(
            df_processed,
            value_column,
            config.get('outlier_method', 'clip'),
            config.get('iqr_multiplier', 1.5),
            config.get('outlier_window', None)
        )
    
    return df_processed, reports

