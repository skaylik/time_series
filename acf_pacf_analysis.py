"""
Модуль для детального анализа ACF и PACF
Включает визуализацию и интерпретацию для определения параметров ARIMA
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats


def calculate_acf_pacf_detailed(series, nlags=40, alpha=0.05):
    """
    Детальный расчет ACF и PACF с доверительными интервалами
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    nlags : int
        Количество лагов
    alpha : float
        Уровень значимости для доверительных интервалов
    
    Возвращает:
    ----------
    dict : результаты расчета
    """
    series_clean = series.dropna()
    
    # Ограничиваем nlags если данных недостаточно
    max_lags = min(nlags, len(series_clean) // 2 - 1)
    
    if max_lags < 1:
        return {
            'acf_values': np.array([]),
            'pacf_values': np.array([]),
            'acf_confint': np.array([]),
            'pacf_confint': np.array([]),
            'lags': np.array([]),
            'n_obs': len(series_clean),
            'error': 'Недостаточно данных'
        }
    
    # Расчет ACF
    acf_values, acf_confint = acf(
        series_clean,
        nlags=max_lags,
        alpha=alpha,
        fft=False
    )
    
    # Расчет PACF
    pacf_values, pacf_confint = pacf(
        series_clean,
        nlags=max_lags,
        alpha=alpha,
        method='ywm'
    )
    
    lags = np.arange(len(acf_values))
    
    return {
        'acf_values': acf_values,
        'pacf_values': pacf_values,
        'acf_confint': acf_confint,
        'pacf_confint': pacf_confint,
        'lags': lags,
        'n_obs': len(series_clean),
        'alpha': alpha,
        'max_lags': max_lags
    }


def identify_significant_lags(values, confint, exclude_zero=True):
    """
    Определение статистически значимых лагов
    
    Параметры:
    ----------
    values : np.array
        Значения ACF или PACF
    confint : np.array
        Доверительные интервалы
    exclude_zero : bool
        Исключить нулевой лаг
    
    Возвращает:
    ----------
    dict : информация о значимых лагах
    """
    significant_lags = []
    
    start_idx = 1 if exclude_zero else 0
    
    for i in range(start_idx, len(values)):
        # Проверяем, выходит ли значение за доверительный интервал
        if values[i] > confint[i, 1] or values[i] < confint[i, 0]:
            significant_lags.append({
                'lag': i,
                'value': values[i],
                'lower_bound': confint[i, 0],
                'upper_bound': confint[i, 1],
                'exceeds_upper': values[i] > confint[i, 1],
                'exceeds_lower': values[i] < confint[i, 0]
            })
    
    return {
        'significant_lags': significant_lags,
        'count': len(significant_lags),
        'lags': [lag['lag'] for lag in significant_lags]
    }


def interpret_acf_pattern(acf_values, significant_lags_acf):
    """
    Интерпретация паттерна ACF для определения порядка MA
    
    Параметры:
    ----------
    acf_values : np.array
        Значения ACF
    significant_lags_acf : dict
        Значимые лаги ACF
    
    Возвращает:
    ----------
    dict : интерпретация
    """
    if len(acf_values) < 2:
        return {
            'pattern': 'Недостаточно данных',
            'suggested_ma': None,
            'interpretation': 'Недостаточно данных для интерпретации'
        }
    
    sig_lags = significant_lags_acf['lags']
    
    if not sig_lags:
        return {
            'pattern': 'Нет значимых лагов',
            'suggested_ma': 0,
            'interpretation': 'ACF не показывает значимых корреляций. Возможно, ряд близок к белому шуму или не требует MA компоненты.'
        }
    
    # Проверяем резкий обрыв (cutoff)
    # Резкий обрыв = все лаги до определенного значимы, после - нет
    max_sig_lag = max(sig_lags)
    consecutive_sig = []
    
    for i in range(1, len(acf_values)):
        if i in sig_lags:
            consecutive_sig.append(i)
        else:
            break
    
    has_cutoff = len(consecutive_sig) > 0 and consecutive_sig[-1] < max_sig_lag
    
    # Проверяем постепенное затухание
    # Затухание = значения постепенно уменьшаются
    abs_values = np.abs(acf_values[1:11])  # Первые 10 лагов
    is_decreasing = all(abs_values[i] >= abs_values[i+1] for i in range(len(abs_values)-1))
    
    # Проверяем экспоненциальное затухание
    if len(acf_values) > 5:
        # Пробуем линейную регрессию на логарифмах абсолютных значений
        valid_values = abs_values[abs_values > 0]
        if len(valid_values) > 3:
            x = np.arange(len(valid_values))
            y = np.log(valid_values)
            slope, _, r_value, _, _ = stats.linregress(x, y)
            exponential_decay = slope < 0 and r_value ** 2 > 0.7
        else:
            exponential_decay = False
    else:
        exponential_decay = False
    
    # Определяем паттерн
    if has_cutoff and len(consecutive_sig) <= 3:
        pattern = 'Резкий обрыв'
        suggested_ma = len(consecutive_sig)
        interpretation = f'ACF показывает резкий обрыв после лага {suggested_ma}. Это характерно для процесса MA({suggested_ma}). Рекомендуется использовать MA компоненту порядка {suggested_ma}.'
    
    elif exponential_decay:
        pattern = 'Экспоненциальное затухание'
        suggested_ma = 0
        interpretation = 'ACF показывает экспоненциальное затухание, что характерно для процесса AR. MA компонента, вероятно, не требуется (MA = 0), но проверьте PACF для определения порядка AR.'
    
    elif is_decreasing and len(sig_lags) > 3:
        pattern = 'Постепенное затухание'
        suggested_ma = 0
        interpretation = 'ACF показывает постепенное затухание, что характерно для процесса AR. MA компонента, вероятно, не требуется (MA = 0), но проверьте PACF для определения порядка AR.'
    
    else:
        pattern = 'Сложный паттерн'
        suggested_ma = None
        interpretation = f'ACF показывает сложный паттерн со значимыми лагами: {sig_lags[:5]}. Может потребоваться комбинация AR и MA компонент или сезонная ARIMA модель.'
    
    return {
        'pattern': pattern,
        'suggested_ma': suggested_ma,
        'interpretation': interpretation,
        'significant_lags': sig_lags[:10],
        'has_cutoff': has_cutoff,
        'exponential_decay': exponential_decay
    }


def interpret_pacf_pattern(pacf_values, significant_lags_pacf):
    """
    Интерпретация паттерна PACF для определения порядка AR
    
    Параметры:
    ----------
    pacf_values : np.array
        Значения PACF
    significant_lags_pacf : dict
        Значимые лаги PACF
    
    Возвращает:
    ----------
    dict : интерпретация
    """
    if len(pacf_values) < 2:
        return {
            'pattern': 'Недостаточно данных',
            'suggested_ar': None,
            'interpretation': 'Недостаточно данных для интерпретации'
        }
    
    sig_lags = significant_lags_pacf['lags']
    
    if not sig_lags:
        return {
            'pattern': 'Нет значимых лагов',
            'suggested_ar': 0,
            'interpretation': 'PACF не показывает значимых корреляций. Возможно, ряд близок к белому шуму или не требует AR компоненты.'
        }
    
    # Проверяем резкий обрыв (cutoff)
    max_sig_lag = max(sig_lags)
    consecutive_sig = []
    
    for i in range(1, len(pacf_values)):
        if i in sig_lags:
            consecutive_sig.append(i)
        else:
            break
    
    has_cutoff = len(consecutive_sig) > 0 and (len(consecutive_sig) == len(sig_lags) or consecutive_sig[-1] < max_sig_lag)
    
    # Определяем паттерн
    if has_cutoff and len(consecutive_sig) <= 5:
        pattern = 'Резкий обрыв'
        suggested_ar = len(consecutive_sig)
        interpretation = f'PACF показывает резкий обрыв после лага {suggested_ar}. Это характерно для процесса AR({suggested_ar}). Рекомендуется использовать AR компоненту порядка {suggested_ar}.'
    
    elif len(sig_lags) > 5:
        pattern = 'Постепенное затухание'
        # Если много значимых лагов, возможно это MA процесс
        suggested_ar = None
        interpretation = 'PACF показывает постепенное затухание со многими значимыми лагами, что характерно для процесса MA. AR компонента, вероятно, не требуется, но проверьте ACF для определения порядка MA.'
    
    elif len(sig_lags) <= 3 and max_sig_lag <= 5:
        pattern = 'Несколько значимых лагов'
        suggested_ar = max_sig_lag
        interpretation = f'PACF показывает значимые лаги: {sig_lags}. Рекомендуется AR порядка {suggested_ar} или меньше. Попробуйте разные порядки и сравните по AIC/BIC.'
    
    else:
        pattern = 'Сложный паттерн'
        suggested_ar = None
        interpretation = f'PACF показывает сложный паттерн со значимыми лагами: {sig_lags[:5]}. Может потребоваться комбинация AR и MA компонент или сезонная ARIMA модель.'
    
    return {
        'pattern': pattern,
        'suggested_ar': suggested_ar,
        'interpretation': interpretation,
        'significant_lags': sig_lags[:10],
        'has_cutoff': has_cutoff
    }


def suggest_arima_parameters(acf_interpretation, pacf_interpretation):
    """
    Предложение параметров ARIMA на основе ACF и PACF
    
    Параметры:
    ----------
    acf_interpretation : dict
        Интерпретация ACF
    pacf_interpretation : dict
        Интерпретация PACF
    
    Возвращает:
    ----------
    dict : рекомендации по параметрам
    """
    ar_order = pacf_interpretation['suggested_ar']
    ma_order = acf_interpretation['suggested_ma']
    
    suggestions = []
    
    # Основная рекомендация
    if ar_order is not None and ma_order is not None:
        suggestions.append({
            'model': f'ARIMA({ar_order}, d, {ma_order})',
            'p': ar_order,
            'q': ma_order,
            'confidence': 'высокая',
            'reason': 'Оба паттерна четко определены'
        })
    
    elif ar_order is not None:
        suggestions.append({
            'model': f'ARIMA({ar_order}, d, 0)',
            'p': ar_order,
            'q': 0,
            'confidence': 'средняя',
            'reason': 'Четко определен AR порядок, MA не требуется'
        })
        
        # Альтернатива с небольшой MA компонентой
        suggestions.append({
            'model': f'ARIMA({ar_order}, d, 1)',
            'p': ar_order,
            'q': 1,
            'confidence': 'низкая',
            'reason': 'Альтернатива: AR + минимальная MA компонента'
        })
    
    elif ma_order is not None:
        suggestions.append({
            'model': f'ARIMA(0, d, {ma_order})',
            'p': 0,
            'q': ma_order,
            'confidence': 'средняя',
            'reason': 'Четко определен MA порядок, AR не требуется'
        })
        
        # Альтернатива с небольшой AR компонентой
        suggestions.append({
            'model': f'ARIMA(1, d, {ma_order})',
            'p': 1,
            'q': ma_order,
            'confidence': 'низкая',
            'reason': 'Альтернатива: минимальная AR + MA компонента'
        })
    
    else:
        # Если паттерны неясны, предлагаем базовые модели
        suggestions.extend([
            {
                'model': 'ARIMA(1, d, 0)',
                'p': 1,
                'q': 0,
                'confidence': 'низкая',
                'reason': 'Базовая AR модель для тестирования'
            },
            {
                'model': 'ARIMA(0, d, 1)',
                'p': 0,
                'q': 1,
                'confidence': 'низкая',
                'reason': 'Базовая MA модель для тестирования'
            },
            {
                'model': 'ARIMA(1, d, 1)',
                'p': 1,
                'q': 1,
                'confidence': 'низкая',
                'reason': 'Базовая ARMA модель для тестирования'
            }
        ])
    
    return {
        'primary_suggestions': suggestions[:3],
        'note': 'Параметр d (порядок дифференцирования) должен быть определен на основе тестов стационарности (ADF, KPSS)',
        'recommendation': 'Сравните модели по критериям AIC и BIC для выбора лучшей'
    }


def create_acf_pacf_plot(acf_result, title_suffix=""):
    """
    Создание интерактивного графика ACF и PACF с Plotly
    
    Параметры:
    ----------
    acf_result : dict
        Результаты расчета ACF/PACF
    title_suffix : str
        Дополнение к заголовку
    
    Возвращает:
    ----------
    plotly.graph_objects.Figure : график
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'ACF (Autocorrelation Function){title_suffix}',
            f'PACF (Partial Autocorrelation Function){title_suffix}'
        ),
        vertical_spacing=0.12
    )
    
    lags = acf_result['lags']
    acf_values = acf_result['acf_values']
    pacf_values = acf_result['pacf_values']
    acf_confint = acf_result['acf_confint']
    pacf_confint = acf_result['pacf_confint']
    
    # ACF
    fig.add_trace(
        go.Bar(
            x=lags,
            y=acf_values,
            name='ACF',
            marker_color='steelblue',
            showlegend=False,
            hovertemplate='Лаг: %{x}<br>ACF: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Доверительные интервалы для ACF
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=acf_confint[:, 1],
            mode='lines',
            line=dict(color='red', dash='dash', width=1),
            name='Доверительный интервал',
            showlegend=True,
            hovertemplate='Верхняя граница: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=acf_confint[:, 0],
            mode='lines',
            line=dict(color='red', dash='dash', width=1),
            showlegend=False,
            hovertemplate='Нижняя граница: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # PACF
    fig.add_trace(
        go.Bar(
            x=lags,
            y=pacf_values,
            name='PACF',
            marker_color='darkorange',
            showlegend=False,
            hovertemplate='Лаг: %{x}<br>PACF: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Доверительные интервалы для PACF
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=pacf_confint[:, 1],
            mode='lines',
            line=dict(color='red', dash='dash', width=1),
            showlegend=False,
            hovertemplate='Верхняя граница: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=pacf_confint[:, 0],
            mode='lines',
            line=dict(color='red', dash='dash', width=1),
            showlegend=False,
            hovertemplate='Нижняя граница: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Настройка осей
    fig.update_xaxes(title_text="Лаг", row=1, col=1)
    fig.update_xaxes(title_text="Лаг", row=2, col=1)
    fig.update_yaxes(title_text="Корреляция", row=1, col=1)
    fig.update_yaxes(title_text="Частичная корреляция", row=2, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def comprehensive_acf_pacf_analysis(series, nlags=40, alpha=0.05):
    """
    Комплексный анализ ACF и PACF с интерпретацией
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    nlags : int
        Количество лагов
    alpha : float
        Уровень значимости
    
    Возвращает:
    ----------
    dict : полные результаты анализа
    """
    # Расчет ACF и PACF
    acf_pacf_result = calculate_acf_pacf_detailed(series, nlags, alpha)
    
    if 'error' in acf_pacf_result:
        return acf_pacf_result
    
    # Определение значимых лагов
    sig_acf = identify_significant_lags(
        acf_pacf_result['acf_values'],
        acf_pacf_result['acf_confint'],
        exclude_zero=True
    )
    
    sig_pacf = identify_significant_lags(
        acf_pacf_result['pacf_values'],
        acf_pacf_result['pacf_confint'],
        exclude_zero=True
    )
    
    # Интерпретация паттернов
    acf_interp = interpret_acf_pattern(acf_pacf_result['acf_values'], sig_acf)
    pacf_interp = interpret_pacf_pattern(acf_pacf_result['pacf_values'], sig_pacf)
    
    # Рекомендации по параметрам ARIMA
    arima_suggestions = suggest_arima_parameters(acf_interp, pacf_interp)
    
    return {
        'acf_pacf_values': acf_pacf_result,
        'significant_lags_acf': sig_acf,
        'significant_lags_pacf': sig_pacf,
        'acf_interpretation': acf_interp,
        'pacf_interpretation': pacf_interp,
        'arima_suggestions': arima_suggestions
    }

