"""
Модуль для детальной декомпозиции временных рядов
Включает анализ тренда, сезонности и остатков
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


def perform_decomposition(series, period, model='additive', extrapolate_trend='freq'):
    """
    Выполнение декомпозиции временного ряда
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд с datetime индексом
    period : int
        Период сезонности
    model : str
        Тип модели: 'additive' или 'multiplicative'
    extrapolate_trend : str
        Метод экстраполяции тренда
    
    Возвращает:
    ----------
    dict : результаты декомпозиции
    """
    try:
        # Удаляем NaN
        series_clean = series.dropna()
        
        if len(series_clean) < 2 * period:
            return {
                'error': f'Недостаточно данных. Требуется минимум {2 * period} точек для периода {period}',
                'min_required': 2 * period,
                'available': len(series_clean)
            }
        
        # Для мультипликативной модели проверяем наличие неположительных значений
        if model == 'multiplicative':
            if (series_clean <= 0).any():
                return {
                    'error': 'Мультипликативная декомпозиция требует положительных значений. Используйте аддитивную модель или трансформируйте данные.',
                    'negative_count': (series_clean <= 0).sum()
                }
        
        # Выполняем декомпозицию
        decomposition = seasonal_decompose(
            series_clean,
            model=model,
            period=period,
            extrapolate_trend=extrapolate_trend
        )
        
        return {
            'success': True,
            'model': model,
            'period': period,
            'observed': decomposition.observed,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'resid': decomposition.resid,
            'n_obs': len(series_clean)
        }
        
    except Exception as e:
        return {
            'error': f'Ошибка при декомпозиции: {str(e)}'
        }


def analyze_trend(trend_series):
    """
    Детальный анализ тренда
    
    Параметры:
    ----------
    trend_series : pd.Series
        Компонента тренда
    
    Возвращает:
    ----------
    dict : анализ тренда
    """
    trend_clean = trend_series.dropna()
    
    if len(trend_clean) < 3:
        return {'error': 'Недостаточно данных для анализа тренда'}
    
    # Основные характеристики
    trend_start = trend_clean.iloc[0]
    trend_end = trend_clean.iloc[-1]
    trend_change = trend_end - trend_start
    trend_change_pct = (trend_change / abs(trend_start) * 100) if trend_start != 0 else np.inf
    
    # Направление тренда
    if abs(trend_change_pct) < 5:
        direction = 'Стабильный (без тренда)'
        direction_emoji = '➡️'
    elif trend_change > 0:
        direction = 'Восходящий'
        direction_emoji = '📈'
    else:
        direction = 'Нисходящий'
        direction_emoji = '📉'
    
    # Сила тренда
    if abs(trend_change_pct) < 5:
        strength = 'Отсутствует'
    elif abs(trend_change_pct) < 20:
        strength = 'Слабый'
    elif abs(trend_change_pct) < 50:
        strength = 'Умеренный'
    else:
        strength = 'Сильный'
    
    # Линейная регрессия для оценки формы
    x = np.arange(len(trend_clean))
    y = trend_clean.values
    
    # Линейная модель
    linear_coeffs = np.polyfit(x, y, 1)
    linear_fit = np.polyval(linear_coeffs, x)
    linear_r2 = 1 - (np.sum((y - linear_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))
    
    # Квадратичная модель
    if len(trend_clean) > 3:
        quad_coeffs = np.polyfit(x, y, 2)
        quad_fit = np.polyval(quad_coeffs, x)
        quad_r2 = 1 - (np.sum((y - quad_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))
    else:
        quad_coeffs = None
        quad_r2 = 0
    
    # Экспоненциальная модель (если все значения положительные)
    if (trend_clean > 0).all():
        try:
            log_y = np.log(y)
            exp_coeffs = np.polyfit(x, log_y, 1)
            exp_fit = np.exp(np.polyval(exp_coeffs, x))
            exp_r2 = 1 - (np.sum((y - exp_fit) ** 2) / np.sum((y - np.mean(y)) ** 2))
        except:
            exp_coeffs = None
            exp_r2 = 0
    else:
        exp_coeffs = None
        exp_r2 = 0
    
    # Определение формы тренда
    if linear_r2 > 0.95:
        shape = 'Линейный'
        best_fit = 'linear'
        best_r2 = linear_r2
    elif quad_r2 > linear_r2 and quad_r2 > 0.90:
        shape = 'Квадратичный (параболический)'
        best_fit = 'quadratic'
        best_r2 = quad_r2
    elif exp_r2 > linear_r2 and exp_r2 > 0.90:
        shape = 'Экспоненциальный'
        best_fit = 'exponential'
        best_r2 = exp_r2
    elif linear_r2 > 0.80:
        shape = 'Приблизительно линейный'
        best_fit = 'linear'
        best_r2 = linear_r2
    else:
        shape = 'Сложный (нелинейный)'
        best_fit = 'complex'
        best_r2 = max(linear_r2, quad_r2, exp_r2)
    
    # Волатильность тренда
    trend_diff = trend_clean.diff().dropna()
    trend_volatility = trend_diff.std()
    
    # Точки изменения тренда (локальные максимумы и минимумы)
    peaks, _ = find_peaks(trend_clean.values)
    troughs, _ = find_peaks(-trend_clean.values)
    
    return {
        'direction': direction,
        'direction_emoji': direction_emoji,
        'strength': strength,
        'shape': shape,
        'best_fit': best_fit,
        'start_value': trend_start,
        'end_value': trend_end,
        'total_change': trend_change,
        'total_change_pct': trend_change_pct,
        'linear_r2': linear_r2,
        'quadratic_r2': quad_r2,
        'exponential_r2': exp_r2,
        'best_r2': best_r2,
        'volatility': trend_volatility,
        'turning_points': len(peaks) + len(troughs),
        'peaks': len(peaks),
        'troughs': len(troughs),
        'mean': trend_clean.mean(),
        'std': trend_clean.std()
    }


def analyze_seasonality(seasonal_series, period):
    """
    Детальный анализ сезонности
    
    Параметры:
    ----------
    seasonal_series : pd.Series
        Сезонная компонента
    period : int
        Период сезонности
    
    Возвращает:
    ----------
    dict : анализ сезонности
    """
    seasonal_clean = seasonal_series.dropna()
    
    if len(seasonal_clean) < period:
        return {'error': 'Недостаточно данных для анализа сезонности'}
    
    # Амплитуда сезонности
    amplitude = seasonal_clean.max() - seasonal_clean.min()
    mean_amplitude = amplitude / 2
    
    # Сила сезонности (отношение размаха к среднему)
    seasonal_range = seasonal_clean.max() - seasonal_clean.min()
    seasonal_strength = seasonal_range / abs(seasonal_clean.mean()) if seasonal_clean.mean() != 0 else np.inf
    
    # Паттерн одного периода
    seasonal_pattern = seasonal_clean.iloc[:period].values
    
    # Стабильность сезонности (проверяем повторяемость паттерна)
    num_periods = len(seasonal_clean) // period
    if num_periods > 1:
        periods_data = []
        for i in range(num_periods):
            start_idx = i * period
            end_idx = start_idx + period
            if end_idx <= len(seasonal_clean):
                periods_data.append(seasonal_clean.iloc[start_idx:end_idx].values)
        
        # Корреляция между периодами
        if len(periods_data) > 1:
            correlations = []
            for i in range(len(periods_data) - 1):
                if len(periods_data[i]) == len(periods_data[i+1]):
                    corr = np.corrcoef(periods_data[i], periods_data[i+1])[0, 1]
                    correlations.append(corr)
            
            avg_correlation = np.mean(correlations) if correlations else 0
            stability = 'Высокая' if avg_correlation > 0.9 else 'Средняя' if avg_correlation > 0.7 else 'Низкая'
        else:
            avg_correlation = None
            stability = 'Недостаточно данных'
    else:
        avg_correlation = None
        stability = 'Только один период'
    
    # Определение типа периодичности
    if period <= 24:
        periodicity = f'Суточная/Часовая ({period} точек)'
    elif period <= 31:
        periodicity = f'Месячная ({period} дней)'
    elif period <= 90:
        periodicity = f'Квартальная ({period} точек)'
    elif period <= 366:
        periodicity = f'Годовая ({period} дней)'
    else:
        periodicity = f'Долгосрочная ({period} точек)'
    
    # Пики и спады в сезонности
    peaks, _ = find_peaks(seasonal_pattern)
    troughs, _ = find_peaks(-seasonal_pattern)
    
    return {
        'period': period,
        'periodicity': periodicity,
        'amplitude': amplitude,
        'mean_amplitude': mean_amplitude,
        'seasonal_strength': seasonal_strength,
        'min_value': seasonal_clean.min(),
        'max_value': seasonal_clean.max(),
        'range': seasonal_range,
        'mean': seasonal_clean.mean(),
        'std': seasonal_clean.std(),
        'pattern': seasonal_pattern,
        'stability': stability,
        'avg_correlation': avg_correlation,
        'num_peaks': len(peaks),
        'num_troughs': len(troughs),
        'num_periods': num_periods
    }


def analyze_residuals(residual_series):
    """
    Детальный анализ остатков
    
    Параметры:
    ----------
    residual_series : pd.Series
        Остатки после декомпозиции
    
    Возвращает:
    ----------
    dict : анализ остатков
    """
    resid_clean = residual_series.dropna()
    
    if len(resid_clean) < 3:
        return {'error': 'Недостаточно данных для анализа остатков'}
    
    # Основные статистики
    mean = resid_clean.mean()
    std = resid_clean.std()
    
    # Проверка на случайность (должны быть близки к белому шуму)
    # 1. Среднее должно быть близко к нулю
    mean_close_to_zero = abs(mean) < 0.1 * std
    
    # 2. Тест на нормальность (Shapiro-Wilk)
    if len(resid_clean) >= 3 and len(resid_clean) <= 5000:
        try:
            shapiro_stat, shapiro_p = stats.shapiro(resid_clean)
            is_normal = shapiro_p > 0.05
        except:
            shapiro_stat, shapiro_p = None, None
            is_normal = None
    else:
        shapiro_stat, shapiro_p = None, None
        is_normal = None
    
    # 3. Тест на автокорреляцию (Ljung-Box)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    try:
        lb_result = acorr_ljungbox(resid_clean, lags=min(10, len(resid_clean) // 5), return_df=True)
        lb_pvalue = lb_result['lb_pvalue'].iloc[-1]
        no_autocorr = lb_pvalue > 0.05
    except:
        lb_pvalue = None
        no_autocorr = None
    
    # 4. Проверка на гетероскедастичность (постоянство дисперсии)
    if len(resid_clean) > 20:
        # Разделяем на две половины
        mid = len(resid_clean) // 2
        first_half = resid_clean.iloc[:mid]
        second_half = resid_clean.iloc[mid:]
        
        # F-тест на равенство дисперсий
        var1 = first_half.var()
        var2 = second_half.var()
        f_stat = var1 / var2 if var2 != 0 else np.inf
        
        # Примерная проверка (более строгий тест требует scipy.stats.f)
        homoscedastic = 0.5 < f_stat < 2.0
    else:
        homoscedastic = None
    
    # Выбросы в остатках
    Q1 = resid_clean.quantile(0.25)
    Q3 = resid_clean.quantile(0.75)
    IQR = Q3 - Q1
    outliers = resid_clean[(resid_clean < Q1 - 1.5 * IQR) | (resid_clean > Q3 + 1.5 * IQR)]
    outlier_pct = (len(outliers) / len(resid_clean)) * 100
    
    # Общая оценка качества декомпозиции
    quality_checks = {
        'mean_near_zero': mean_close_to_zero,
        'normally_distributed': is_normal,
        'no_autocorrelation': no_autocorr,
        'constant_variance': homoscedastic
    }
    
    passed_checks = sum([v for v in quality_checks.values() if v is True])
    total_checks = sum([v is not None for v in quality_checks.values()])
    
    if total_checks > 0:
        quality_score = (passed_checks / total_checks) * 100
        
        if quality_score >= 75:
            quality = 'Отличная'
            quality_emoji = '✅'
        elif quality_score >= 50:
            quality = 'Хорошая'
            quality_emoji = '✔️'
        elif quality_score >= 25:
            quality = 'Удовлетворительная'
            quality_emoji = '⚠️'
        else:
            quality = 'Плохая'
            quality_emoji = '❌'
    else:
        quality = 'Невозможно оценить'
        quality_emoji = '❓'
        quality_score = None
    
    return {
        'mean': mean,
        'std': std,
        'min': resid_clean.min(),
        'max': resid_clean.max(),
        'mean_near_zero': mean_close_to_zero,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'is_normal': is_normal,
        'ljung_box_p': lb_pvalue,
        'no_autocorrelation': no_autocorr,
        'homoscedastic': homoscedastic,
        'outlier_count': len(outliers),
        'outlier_pct': outlier_pct,
        'quality': quality,
        'quality_emoji': quality_emoji,
        'quality_score': quality_score,
        'quality_checks': quality_checks
    }


def create_decomposition_plot(decomp_result):
    """
    Создание визуализации декомпозиции
    
    Параметры:
    ----------
    decomp_result : dict
        Результаты декомпозиции
    
    Возвращает:
    ----------
    plotly.graph_objects.Figure : график
    """
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Исходный ряд (Observed)',
            'Тренд (Trend)',
            'Сезонность (Seasonal)',
            'Остатки (Residual)'
        ),
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )
    
    observed = decomp_result['observed']
    trend = decomp_result['trend']
    seasonal = decomp_result['seasonal']
    resid = decomp_result['resid']
    
    # Исходный ряд
    fig.add_trace(
        go.Scatter(
            x=observed.index,
            y=observed.values,
            mode='lines',
            name='Observed',
            line=dict(color='steelblue', width=1.5),
            hovertemplate='%{x}<br>Значение: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Тренд
    fig.add_trace(
        go.Scatter(
            x=trend.index,
            y=trend.values,
            mode='lines',
            name='Trend',
            line=dict(color='orangered', width=2),
            hovertemplate='%{x}<br>Тренд: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Сезонность
    fig.add_trace(
        go.Scatter(
            x=seasonal.index,
            y=seasonal.values,
            mode='lines',
            name='Seasonal',
            line=dict(color='green', width=1.5),
            hovertemplate='%{x}<br>Сезонность: %{y:.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Остатки
    fig.add_trace(
        go.Scatter(
            x=resid.index,
            y=resid.values,
            mode='lines',
            name='Residual',
            line=dict(color='gray', width=1),
            hovertemplate='%{x}<br>Остаток: %{y:.2f}<extra></extra>'
        ),
        row=4, col=1
    )
    
    # Добавляем нулевую линию для остатков
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=4, col=1)
    
    # Настройка осей
    for i in range(1, 5):
        fig.update_xaxes(title_text="Дата", row=i, col=1)
    
    fig.update_yaxes(title_text="Значение", row=1, col=1)
    fig.update_yaxes(title_text="Тренд", row=2, col=1)
    fig.update_yaxes(title_text="Сезонность", row=3, col=1)
    fig.update_yaxes(title_text="Остатки", row=4, col=1)
    
    fig.update_layout(
        height=1000,
        showlegend=False,
        hovermode='x unified',
        title_text=f"Декомпозиция временного ряда ({decomp_result['model'].capitalize()})",
        title_x=0.5
    )
    
    return fig


def create_seasonal_pattern_plot(seasonal_pattern, period):
    """
    Создание графика сезонного паттерна
    
    Параметры:
    ----------
    seasonal_pattern : np.array
        Паттерн одного сезонного периода
    period : int
        Период сезонности
    
    Возвращает:
    ----------
    plotly.graph_objects.Figure : график
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(seasonal_pattern) + 1)),
            y=seasonal_pattern,
            mode='lines+markers',
            line=dict(color='green', width=2),
            marker=dict(size=6),
            hovertemplate='Точка %{x}<br>Значение: %{y:.4f}<extra></extra>'
        )
    )
    
    fig.update_layout(
        title=f'Сезонный паттерн (период = {period})',
        xaxis_title='Точка в периоде',
        yaxis_title='Сезонная компонента',
        hovermode='closest',
        height=400
    )
    
    return fig


def create_residuals_analysis_plot(residuals):
    """
    Создание детального графика анализа остатков
    
    Параметры:
    ----------
    residuals : pd.Series
        Остатки
    
    Возвращает:
    ----------
    plotly.graph_objects.Figure : график
    """
    resid_clean = residuals.dropna()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Остатки во времени',
            'Гистограмма остатков',
            'Q-Q Plot (проверка нормальности)',
            'ACF остатков'
        ),
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "bar"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Остатки во времени
    fig.add_trace(
        go.Scatter(
            x=resid_clean.index,
            y=resid_clean.values,
            mode='lines',
            line=dict(color='gray', width=1),
            name='Остатки'
        ),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # 2. Гистограмма
    fig.add_trace(
        go.Histogram(
            x=resid_clean.values,
            nbinsx=30,
            marker_color='lightblue',
            name='Гистограмма',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Q-Q Plot
    from scipy.stats import probplot
    qq = probplot(resid_clean.values, dist="norm")
    
    fig.add_trace(
        go.Scatter(
            x=qq[0][0],
            y=qq[0][1],
            mode='markers',
            marker=dict(color='blue', size=4),
            name='Q-Q',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Добавляем теоретическую линию
    fig.add_trace(
        go.Scatter(
            x=qq[0][0],
            y=qq[1][1] + qq[1][0] * qq[0][0],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Теоретическая линия',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. ACF остатков
    from statsmodels.tsa.stattools import acf
    acf_values = acf(resid_clean, nlags=min(40, len(resid_clean) // 2 - 1))
    
    fig.add_trace(
        go.Bar(
            x=list(range(len(acf_values))),
            y=acf_values,
            marker_color='steelblue',
            name='ACF',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Доверительные интервалы
    conf_level = 1.96 / np.sqrt(len(resid_clean))
    fig.add_hline(y=conf_level, line_dash="dash", line_color="red", row=2, col=2)
    fig.add_hline(y=-conf_level, line_dash="dash", line_color="red", row=2, col=2)
    
    fig.update_xaxes(title_text="Дата", row=1, col=1)
    fig.update_xaxes(title_text="Остатки", row=1, col=2)
    fig.update_xaxes(title_text="Теоретические квантили", row=2, col=1)
    fig.update_xaxes(title_text="Лаг", row=2, col=2)
    
    fig.update_yaxes(title_text="Остатки", row=1, col=1)
    fig.update_yaxes(title_text="Частота", row=1, col=2)
    fig.update_yaxes(title_text="Выборочные квантили", row=2, col=1)
    fig.update_yaxes(title_text="ACF", row=2, col=2)
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Детальный анализ остатков",
        title_x=0.5
    )
    
    return fig


def comprehensive_decomposition_analysis(series, period, model='additive'):
    """
    Комплексный анализ декомпозиции
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    period : int
        Период сезонности
    model : str
        Тип модели
    
    Возвращает:
    ----------
    dict : полные результаты анализа
    """
    # Выполняем декомпозицию
    decomp_result = perform_decomposition(series, period, model)
    
    if 'error' in decomp_result:
        return decomp_result
    
    # Анализ компонент
    trend_analysis = analyze_trend(decomp_result['trend'])
    seasonal_analysis = analyze_seasonality(decomp_result['seasonal'], period)
    residual_analysis = analyze_residuals(decomp_result['resid'])
    
    return {
        'decomposition': decomp_result,
        'trend_analysis': trend_analysis,
        'seasonal_analysis': seasonal_analysis,
        'residual_analysis': residual_analysis
    }

