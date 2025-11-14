"""
Модуль для анализа стационарности временных рядов
Включает визуальный анализ, статистические тесты и дифференцирование
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from statsmodels.tsa.stattools import adfuller, kpss


def calculate_rolling_statistics(series, windows=[30, 60, 90]):
    """
    Расчет скользящих среднего и дисперсии для разных окон
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    windows : list
        Список размеров окон
    
    Возвращает:
    ----------
    dict : словарь со скользящими статистиками для каждого окна
    """
    results = {}
    
    for window in windows:
        rolling_mean = series.rolling(window=window, center=False).mean()
        rolling_std = series.rolling(window=window, center=False).std()
        rolling_var = series.rolling(window=window, center=False).var()
        
        results[window] = {
            'mean': rolling_mean,
            'std': rolling_std,
            'var': rolling_var
        }
    
    return results


def visual_trend_analysis(series, rolling_stats):
    """
    Визуальный анализ наличия тренда и изменения дисперсии
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    rolling_stats : dict
        Скользящие статистики
    
    Возвращает:
    ----------
    dict : результаты визуального анализа
    """
    analysis = {}
    
    # Проверка тренда по скользящему среднему
    for window, stats in rolling_stats.items():
        mean = stats['mean'].dropna()
        
        if len(mean) > 1:
            # Линейная регрессия для определения тренда
            x = np.arange(len(mean))
            coeffs = np.polyfit(x, mean, 1)
            slope = coeffs[0]
            
            # Определяем направление тренда
            if abs(slope) < 0.01:
                trend_direction = "Нет явного тренда"
            elif slope > 0:
                trend_direction = "Восходящий тренд"
            else:
                trend_direction = "Нисходящий тренд"
            
            # Проверка стабильности дисперсии
            variance = stats['var'].dropna()
            if len(variance) > 1:
                var_change = (variance.iloc[-1] - variance.iloc[0]) / variance.iloc[0] * 100
                
                if abs(var_change) < 10:
                    variance_stability = "Стабильная"
                elif var_change > 0:
                    variance_stability = "Возрастающая"
                else:
                    variance_stability = "Убывающая"
            else:
                var_change = 0
                variance_stability = "Недостаточно данных"
            
            analysis[window] = {
                'trend_direction': trend_direction,
                'trend_slope': slope,
                'variance_stability': variance_stability,
                'variance_change_pct': var_change,
                'mean_stability': mean.std() / mean.mean() if mean.mean() != 0 else 0
            }
    
    return analysis


def perform_adf_test(series):
    """
    Тест Дики-Фуллера (ADF) на стационарность
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    
    Возвращает:
    ----------
    dict : результаты теста
    """
    series_clean = series.dropna()
    
    if len(series_clean) < 3:
        return {
            'test_statistic': np.nan,
            'p_value': np.nan,
            'used_lag': np.nan,
            'n_obs': len(series_clean),
            'critical_values': {},
            'is_stationary': None,
            'interpretation': 'Недостаточно данных для теста'
        }
    
    result = adfuller(series_clean, autolag='AIC')
    
    # Интерпретация результатов
    is_stationary = result[1] < 0.05
    
    if result[1] < 0.01:
        confidence = "99%"
        interpretation = f"Ряд стационарен с высокой уверенностью (p < 0.01)"
    elif result[1] < 0.05:
        confidence = "95%"
        interpretation = f"Ряд стационарен (p < 0.05)"
    elif result[1] < 0.10:
        confidence = "90%"
        interpretation = f"Ряд стационарен с низкой уверенностью (p < 0.10)"
    else:
        confidence = "Нет"
        interpretation = f"Ряд нестационарен (p >= 0.10)"
    
    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'used_lag': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'ic_best': result[5] if len(result) > 5 else None,
        'is_stationary': is_stationary,
        'confidence': confidence,
        'interpretation': interpretation
    }


def perform_kpss_test(series, regression='c'):
    """
    Тест KPSS на стационарность
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    regression : str
        Тип регрессии: 'c' (константа) или 'ct' (константа + тренд)
    
    Возвращает:
    ----------
    dict : результаты теста
    """
    series_clean = series.dropna()
    
    if len(series_clean) < 3:
        return {
            'test_statistic': np.nan,
            'p_value': np.nan,
            'used_lag': np.nan,
            'critical_values': {},
            'is_stationary': None,
            'interpretation': 'Недостаточно данных для теста'
        }
    
    result = kpss(series_clean, regression=regression, nlags='auto')
    
    # Интерпретация результатов (для KPSS наоборот - большой p-value = стационарность)
    is_stationary = result[1] > 0.05
    
    if result[1] > 0.10:
        confidence = "90%+"
        interpretation = f"Ряд стационарен (p > 0.10)"
    elif result[1] > 0.05:
        confidence = "95%"
        interpretation = f"Ряд стационарен (p > 0.05)"
    elif result[1] > 0.01:
        confidence = "99%"
        interpretation = f"Ряд нестационарен с низкой уверенностью (p > 0.01)"
    else:
        confidence = "Высокая"
        interpretation = f"Ряд нестационарен (p <= 0.01)"
    
    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'used_lag': result[2],
        'critical_values': result[3],
        'is_stationary': is_stationary,
        'confidence': confidence,
        'interpretation': interpretation
    }


def apply_differencing(series, order=1):
    """
    Применение дифференцирования к временному ряду
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    order : int
        Порядок дифференцирования
    
    Возвращает:
    ----------
    pd.Series : дифференцированный ряд
    """
    differenced = series.copy()
    
    for i in range(order):
        differenced = differenced.diff().dropna()
    
    return differenced


def comprehensive_stationarity_test(series, max_diff_order=2):
    """
    Комплексный тест стационарности с автоматическим дифференцированием
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    max_diff_order : int
        Максимальный порядок дифференцирования
    
    Возвращает:
    ----------
    dict : результаты тестов для исходного и дифференцированных рядов
    """
    results = {}
    
    # Тест исходного ряда
    current_series = series.copy()
    
    for diff_order in range(max_diff_order + 1):
        if diff_order > 0:
            current_series = apply_differencing(series, order=diff_order)
        
        # ADF тест
        adf_result = perform_adf_test(current_series)
        
        # KPSS тест
        kpss_result = perform_kpss_test(current_series)
        
        # Согласованность тестов
        if adf_result['is_stationary'] is not None and kpss_result['is_stationary'] is not None:
            tests_agree = adf_result['is_stationary'] == kpss_result['is_stationary']
            
            if tests_agree and adf_result['is_stationary']:
                conclusion = "✅ Оба теста подтверждают стационарность"
            elif tests_agree and not adf_result['is_stationary']:
                conclusion = "❌ Оба теста указывают на нестационарность"
            else:
                conclusion = "⚠️ Результаты тестов противоречат друг другу"
        else:
            tests_agree = None
            conclusion = "Недостаточно данных"
        
        results[diff_order] = {
            'series': current_series,
            'adf': adf_result,
            'kpss': kpss_result,
            'tests_agree': tests_agree,
            'conclusion': conclusion
        }
        
        # Если ряд стал стационарным, можно прекратить дифференцирование
        if tests_agree and adf_result['is_stationary']:
            break
    
    return results


def create_stationarity_visualization(series, dates, rolling_stats, windows=[30, 60, 90]):
    """
    Создание визуализации для анализа стационарности
    
    Параметры:
    ----------
    series : pd.Series
        Временной ряд
    dates : pd.Series
        Временные метки
    rolling_stats : dict
        Скользящие статистики
    windows : list
        Размеры окон
    
    Возвращает:
    ----------
    plotly.graph_objects.Figure : график
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Временной ряд со скользящим средним',
            'Скользящее стандартное отклонение',
            'Скользящая дисперсия'
        ),
        vertical_spacing=0.10,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    colors = ['blue', 'orange', 'green']
    
    # График 1: Временной ряд со скользящим средним
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=series,
            mode='lines',
            name='Исходный ряд',
            line=dict(color='lightblue', width=1),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    for idx, window in enumerate(windows):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rolling_stats[window]['mean'],
                mode='lines',
                name=f'Скользящее среднее ({window})',
                line=dict(color=colors[idx], width=2)
            ),
            row=1, col=1
        )
    
    # График 2: Скользящее стандартное отклонение
    for idx, window in enumerate(windows):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rolling_stats[window]['std'],
                mode='lines',
                name=f'Скол. std ({window})',
                line=dict(color=colors[idx], width=2),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # График 3: Скользящая дисперсия
    for idx, window in enumerate(windows):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=rolling_stats[window]['var'],
                mode='lines',
                name=f'Скол. var ({window})',
                line=dict(color=colors[idx], width=2),
                showlegend=False
            ),
            row=3, col=1
        )
    
    fig.update_xaxes(title_text="Дата", row=3, col=1)
    fig.update_yaxes(title_text="Значение", row=1, col=1)
    fig.update_yaxes(title_text="Стд. откл.", row=2, col=1)
    fig.update_yaxes(title_text="Дисперсия", row=3, col=1)
    
    fig.update_layout(
        title="Анализ стационарности: Скользящие характеристики",
        height=900,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def create_differencing_comparison(original_series, diff_series, dates, order=1):
    """
    Создание графика сравнения исходного и дифференцированного ряда
    
    Параметры:
    ----------
    original_series : pd.Series
        Исходный ряд
    diff_series : pd.Series
        Дифференцированный ряд
    dates : pd.Series
        Временные метки
    order : int
        Порядок дифференцирования
    
    Возвращает:
    ----------
    plotly.graph_objects.Figure : график
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Исходный временной ряд',
            f'Дифференцированный ряд (порядок {order})'
        ),
        vertical_spacing=0.15
    )
    
    # Исходный ряд
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=original_series,
            mode='lines',
            name='Исходный ряд',
            line=dict(color='blue', width=1.5)
        ),
        row=1, col=1
    )
    
    # Дифференцированный ряд
    # Нужно выровнять даты с дифференцированным рядом
    diff_dates = dates[len(dates) - len(diff_series):]
    
    fig.add_trace(
        go.Scatter(
            x=diff_dates,
            y=diff_series,
            mode='lines',
            name=f'Diff({order})',
            line=dict(color='red', width=1.5)
        ),
        row=2, col=1
    )
    
    # Добавляем нулевую линию для дифференцированного ряда
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Дата", row=2, col=1)
    fig.update_yaxes(title_text="Значение", row=1, col=1)
    fig.update_yaxes(title_text="Разность", row=2, col=1)
    
    fig.update_layout(
        title=f"Сравнение: Исходный vs Дифференцированный ряд (порядок {order})",
        height=600,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


def get_stationarity_recommendation(test_results):
    """
    Получение рекомендации по стационарности
    
    Параметры:
    ----------
    test_results : dict
        Результаты комплексного теста стационарности
    
    Возвращает:
    ----------
    dict : рекомендация
    """
    recommendation = {
        'is_stationary': False,
        'required_differencing': 0,
        'message': '',
        'details': []
    }
    
    for diff_order, results in test_results.items():
        if results['tests_agree'] and results['adf']['is_stationary']:
            recommendation['is_stationary'] = True
            recommendation['required_differencing'] = diff_order
            
            if diff_order == 0:
                recommendation['message'] = "✅ Ряд стационарен, дифференцирование не требуется"
            else:
                recommendation['message'] = f"✅ Ряд стал стационарным после дифференцирования порядка {diff_order}"
            
            recommendation['details'] = [
                f"ADF p-value: {results['adf']['p_value']:.4f}",
                f"KPSS p-value: {results['kpss']['p_value']:.4f}",
                results['conclusion']
            ]
            
            return recommendation
    
    # Если не найден стационарный ряд
    last_order = max(test_results.keys())
    recommendation['message'] = f"⚠️ Ряд остается нестационарным даже после дифференцирования порядка {last_order}"
    recommendation['details'] = [
        "Рекомендуется:",
        "- Проверить наличие структурных сдвигов",
        "- Рассмотреть логарифмическое преобразование",
        "- Применить сезонное дифференцирование"
    ]
    
    return recommendation

