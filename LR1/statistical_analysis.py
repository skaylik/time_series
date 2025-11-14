"""
Модуль для статистического анализа и визуализации данных
Включает дескриптивную статистику, корреляционный анализ и визуализации
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy import stats


def calculate_descriptive_statistics(df, exclude_columns=None):
    """
    Расчет дескриптивной статистики для всех числовых столбцов
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    exclude_columns : list
        Список столбцов для исключения из анализа
    
    Возвращает:
    ----------
    pd.DataFrame : таблица со статистикой
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # Выбираем только числовые столбцы
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Исключаем указанные столбцы
    numeric_df = numeric_df.drop(columns=[col for col in exclude_columns if col in numeric_df.columns])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    # Базовая статистика
    stats_dict = {
        'Среднее': numeric_df.mean(),
        'Медиана': numeric_df.median(),
        'Стд. отклонение': numeric_df.std(),
        'Минимум': numeric_df.min(),
        'Максимум': numeric_df.max(),
        'Q1 (25%)': numeric_df.quantile(0.25),
        'Q2 (50%)': numeric_df.quantile(0.50),
        'Q3 (75%)': numeric_df.quantile(0.75),
        'IQR': numeric_df.quantile(0.75) - numeric_df.quantile(0.25),
        'Асимметрия': numeric_df.apply(lambda x: stats.skew(x.dropna())),
        'Эксцесс': numeric_df.apply(lambda x: stats.kurtosis(x.dropna())),
        'Количество': numeric_df.count(),
        'Пропуски': numeric_df.isnull().sum(),
        'Пропуски (%)': (numeric_df.isnull().sum() / len(numeric_df)) * 100
    }
    
    stats_df = pd.DataFrame(stats_dict).T
    
    return stats_df


def create_time_series_plots(df, date_column, value_columns, title="Временные ряды"):
    """
    Создание линейных графиков для временных рядов
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    date_column : str
        Название столбца с датой
    value_columns : list
        Список столбцов для визуализации
    title : str
        Заголовок графика
    
    Возвращает:
    ----------
    plotly.graph_objects.Figure : график
    """
    if not value_columns:
        return None
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for idx, col in enumerate(value_columns):
        fig.add_trace(go.Scatter(
            x=df[date_column],
            y=df[col],
            mode='lines',
            name=col,
            line=dict(color=colors[idx % len(colors)], width=2),
            hovertemplate=f'<b>{col}</b><br>Дата: %{{x}}<br>Значение: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Дата",
        yaxis_title="Значение",
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_histograms(df, columns, bins=30):
    """
    Создание гистограмм для признаков
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    columns : list
        Список столбцов для визуализации
    bins : int
        Количество бинов в гистограмме
    
    Возвращает:
    ----------
    plotly.graph_objects.Figure : график с гистограммами
    """
    if not columns:
        return None
    
    n_cols = len(columns)
    n_rows = (n_cols + 1) // 2  # 2 графика в ряд
    
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=[f"Гистограмма: {col}" for col in columns],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    for idx, col in enumerate(columns):
        row = idx // 2 + 1
        col_pos = idx % 2 + 1
        
        data = df[col].dropna()
        
        # Гистограмма
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=bins,
                name=col,
                marker=dict(
                    color='steelblue',
                    line=dict(color='white', width=1)
                ),
                showlegend=False,
                hovertemplate='Диапазон: %{x}<br>Частота: %{y}<extra></extra>'
            ),
            row=row,
            col=col_pos
        )
        
        # Добавляем кривую нормального распределения
        mean = data.mean()
        std = data.std()
        x_range = np.linspace(data.min(), data.max(), 100)
        y_normal = stats.norm.pdf(x_range, mean, std) * len(data) * (data.max() - data.min()) / bins
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_normal,
                mode='lines',
                name='Норм. распр.',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=(idx == 0),
                hovertemplate='Норм. распр.<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
            ),
            row=row,
            col=col_pos
        )
        
        # Обновляем оси
        fig.update_xaxes(title_text=col, row=row, col=col_pos)
        fig.update_yaxes(title_text="Частота", row=row, col=col_pos)
    
    fig.update_layout(
        title_text="Распределение признаков",
        height=300 * n_rows,
        showlegend=True
    )
    
    return fig


def create_boxplots(df, columns):
    """
    Создание box plots для признаков
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    columns : list
        Список столбцов для визуализации
    
    Возвращает:
    ----------
    plotly.graph_objects.Figure : график с box plots
    """
    if not columns:
        return None
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for idx, col in enumerate(columns):
        data = df[col].dropna()
        
        fig.add_trace(go.Box(
            y=data,
            name=col,
            marker_color=colors[idx % len(colors)],
            boxmean='sd',  # Показываем среднее и стд. откл.
            hovertemplate=(
                f'<b>{col}</b><br>'
                'Макс: %{y:.2f}<br>'
                '<extra></extra>'
            )
        ))
    
    fig.update_layout(
        title="Box Plots: Распределение и выбросы",
        yaxis_title="Значение",
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig


def create_correlation_heatmap(df, columns, method='pearson'):
    """
    Создание тепловой карты корреляций
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    columns : list
        Список столбцов для анализа
    method : str
        Метод корреляции: 'pearson' или 'spearman'
    
    Возвращает:
    ----------
    plotly.graph_objects.Figure : тепловая карта
    pd.DataFrame : матрица корреляций
    """
    if not columns or len(columns) < 2:
        return None, None
    
    # Выбираем только указанные столбцы
    data = df[columns].select_dtypes(include=[np.number])
    
    # Расчет корреляции
    if method == 'pearson':
        corr_matrix = data.corr(method='pearson')
        title = "Матрица корреляций (Pearson)"
    elif method == 'spearman':
        corr_matrix = data.corr(method='spearman')
        title = "Матрица корреляций (Spearman)"
    else:
        raise ValueError(f"Неизвестный метод: {method}")
    
    # Создание тепловой карты
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(
            title="Корреляция",
            tickmode='linear',
            tick0=-1,
            dtick=0.2
        ),
        hovertemplate='%{y} vs %{x}<br>Корреляция: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed'),
        height=max(500, len(columns) * 40),
        width=max(500, len(columns) * 40)
    )
    
    return fig, corr_matrix


def analyze_multicollinearity(corr_matrix, threshold=0.8):
    """
    Анализ мультиколлинеарности
    
    Параметры:
    ----------
    corr_matrix : pd.DataFrame
        Матрица корреляций
    threshold : float
        Порог для определения сильной корреляции
    
    Возвращает:
    ----------
    list : список пар признаков с сильной корреляцией
    """
    if corr_matrix is None or corr_matrix.empty:
        return []
    
    high_corr_pairs = []
    
    # Проходим по верхнему треугольнику матрицы (чтобы не дублировать пары)
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_value,
                    'abs_correlation': abs(corr_value)
                })
    
    # Сортируем по убыванию абсолютного значения корреляции
    high_corr_pairs = sorted(high_corr_pairs, key=lambda x: x['abs_correlation'], reverse=True)
    
    return high_corr_pairs


def detect_remaining_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Обнаружение аномальных значений, неуловленных на этапе очистки
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    columns : list
        Список столбцов для проверки
    method : str
        Метод обнаружения: 'iqr' или 'zscore'
    threshold : float
        Порог для определения выбросов
    
    Возвращает:
    ----------
    dict : словарь с информацией о выбросах по каждому столбцу
    """
    outliers_info = {}
    
    for col in columns:
        data = df[col].dropna()
        
        if len(data) == 0:
            continue
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > threshold]
            lower_bound = data.mean() - threshold * data.std()
            upper_bound = data.mean() + threshold * data.std()
        
        else:
            continue
        
        if len(outliers) > 0:
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(data)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min_outlier': outliers.min(),
                'max_outlier': outliers.max(),
                'outlier_values': outliers.tolist()[:10]  # Первые 10 для примера
            }
    
    return outliers_info


def create_scatter_matrix(df, columns, max_cols=5):
    """
    Создание матрицы scatter plots для анализа взаимосвязей
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    columns : list
        Список столбцов для визуализации
    max_cols : int
        Максимальное количество столбцов (для производительности)
    
    Возвращает:
    ----------
    plotly.graph_objects.Figure : матрица scatter plots
    """
    if not columns or len(columns) < 2:
        return None
    
    # Ограничиваем количество столбцов для производительности
    columns = columns[:max_cols]
    
    # Создаем scatter matrix
    fig = px.scatter_matrix(
        df,
        dimensions=columns,
        title="Матрица диаграмм рассеяния",
        height=max(600, len(columns) * 150),
        labels={col: col for col in columns}
    )
    
    fig.update_traces(
        diagonal_visible=False,
        showupperhalf=False,
        marker=dict(size=3, opacity=0.5)
    )
    
    return fig


def create_qq_plots(df, columns):
    """
    Создание Q-Q plots для проверки нормальности распределения
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    columns : list
        Список столбцов для визуализации
    
    Возвращает:
    ----------
    plotly.graph_objects.Figure : Q-Q plots
    """
    if not columns:
        return None
    
    n_cols = len(columns)
    n_rows = (n_cols + 1) // 2
    
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=[f"Q-Q Plot: {col}" for col in columns],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    for idx, col in enumerate(columns):
        row = idx // 2 + 1
        col_pos = idx % 2 + 1
        
        data = df[col].dropna()
        
        # Вычисляем квантили
        theoretical_quantiles = stats.probplot(data, dist="norm")[0][0]
        sample_quantiles = stats.probplot(data, dist="norm")[0][1]
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name=col,
                marker=dict(color='steelblue', size=5),
                showlegend=False,
                hovertemplate='Теор.: %{x:.2f}<br>Выборка: %{y:.2f}<extra></extra>'
            ),
            row=row,
            col=col_pos
        )
        
        # Линия нормального распределения
        fig.add_trace(
            go.Scatter(
                x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                y=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                mode='lines',
                line=dict(color='red', dash='dash'),
                showlegend=False,
                hovertemplate='y = x<extra></extra>'
            ),
            row=row,
            col=col_pos
        )
        
        fig.update_xaxes(title_text="Теоретические квантили", row=row, col=col_pos)
        fig.update_yaxes(title_text="Квантили выборки", row=row, col=col_pos)
    
    fig.update_layout(
        title_text="Q-Q Plots: Проверка нормальности",
        height=300 * n_rows
    )
    
    return fig


def perform_normality_tests(df, columns):
    """
    Тесты на нормальность распределения
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    columns : list
        Список столбцов для тестирования
    
    Возвращает:
    ----------
    pd.DataFrame : результаты тестов
    """
    if not columns:
        return pd.DataFrame()
    
    results = []
    
    for col in columns:
        data = df[col].dropna()
        
        if len(data) < 3:
            continue
        
        # Тест Шапиро-Уилка
        if len(data) <= 5000:  # Тест работает для выборок до 5000
            shapiro_stat, shapiro_p = stats.shapiro(data)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        # Тест Колмогорова-Смирнова
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        
        # Тест Андерсона-Дарлинга
        anderson_result = stats.anderson(data, dist='norm')
        
        results.append({
            'Признак': col,
            'Shapiro-Wilk stat': shapiro_stat,
            'Shapiro-Wilk p-value': shapiro_p,
            'K-S stat': ks_stat,
            'K-S p-value': ks_p,
            'Anderson stat': anderson_result.statistic,
            'Нормальное (Shapiro)': 'Да' if shapiro_p > 0.05 else 'Нет' if not np.isnan(shapiro_p) else 'N/A',
            'Нормальное (K-S)': 'Да' if ks_p > 0.05 else 'Нет'
        })
    
    return pd.DataFrame(results)

