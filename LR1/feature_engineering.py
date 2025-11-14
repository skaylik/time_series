"""
Модуль для инженерии признаков временных рядов
Включает создание лагов, скользящих статистик и анализ корреляций
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
from scipy import stats


def create_lag_features(df, column, lags=[1, 7, 30]):
    """
    Создание лаговых признаков для указанного столбца
    
    Параметры:
    ----------
    df : pd.DataFrame
        Исходный датафрейм
    column : str
        Название столбца для создания лагов
    lags : list
        Список лагов для создания
    
    Возвращает:
    ----------
    pd.DataFrame : датафрейм с новыми лаговыми признаками
    dict : информация о созданных признаках
    """
    df_with_lags = df.copy()
    created_features = []
    
    for lag in lags:
        feature_name = f"{column}_lag_{lag}"
        df_with_lags[feature_name] = df_with_lags[column].shift(lag)
        created_features.append({
            'name': feature_name,
            'lag': lag,
            'type': 'lag',
            'original_column': column,
            'missing_values': df_with_lags[feature_name].isnull().sum()
        })
    
    info = {
        'created_features': created_features,
        'total_created': len(created_features),
        'lags_used': lags
    }
    
    return df_with_lags, info


def create_rolling_features(df, column, windows=[7, 30], statistics=['mean', 'std', 'min', 'max']):
    """
    Создание скользящих статистик для указанного столбца
    
    Параметры:
    ----------
    df : pd.DataFrame
        Исходный датафрейм
    column : str
        Название столбца
    windows : list
        Список размеров окон
    statistics : list
        Список статистик для расчета ('mean', 'std', 'min', 'max', 'median')
    
    Возвращает:
    ----------
    pd.DataFrame : датафрейм с новыми признаками
    dict : информация о созданных признаках
    """
    df_with_rolling = df.copy()
    created_features = []
    
    for window in windows:
        for stat in statistics:
            feature_name = f"{column}_rolling_{stat}_{window}"
            
            if stat == 'mean':
                df_with_rolling[feature_name] = df_with_rolling[column].rolling(window=window, min_periods=1).mean()
            elif stat == 'std':
                df_with_rolling[feature_name] = df_with_rolling[column].rolling(window=window, min_periods=1).std()
            elif stat == 'min':
                df_with_rolling[feature_name] = df_with_rolling[column].rolling(window=window, min_periods=1).min()
            elif stat == 'max':
                df_with_rolling[feature_name] = df_with_rolling[column].rolling(window=window, min_periods=1).max()
            elif stat == 'median':
                df_with_rolling[feature_name] = df_with_rolling[column].rolling(window=window, min_periods=1).median()
            elif stat == 'sum':
                df_with_rolling[feature_name] = df_with_rolling[column].rolling(window=window, min_periods=1).sum()
            
            created_features.append({
                'name': feature_name,
                'window': window,
                'statistic': stat,
                'type': 'rolling',
                'original_column': column,
                'missing_values': df_with_rolling[feature_name].isnull().sum()
            })
    
    info = {
        'created_features': created_features,
        'total_created': len(created_features),
        'windows_used': windows,
        'statistics_used': statistics
    }
    
    return df_with_rolling, info


def create_all_features(df, target_column, feature_columns=None, 
                       target_lags=[1, 7, 30], feature_lags=[1, 7],
                       rolling_windows=[7, 30], rolling_stats=['mean', 'std']):
    """
    Создание всех признаков для временного ряда
    
    Параметры:
    ----------
    df : pd.DataFrame
        Исходный датафрейм
    target_column : str
        Целевая переменная
    feature_columns : list
        Список дополнительных признаков для лагов
    target_lags : list
        Лаги для целевой переменной
    feature_lags : list
        Лаги для дополнительных признаков
    rolling_windows : list
        Размеры окон для скользящих статистик
    rolling_stats : list
        Статистики для расчета
    
    Возвращает:
    ----------
    pd.DataFrame : датафрейм со всеми новыми признаками
    dict : полная информация о созданных признаках
    """
    df_engineered = df.copy()
    all_info = {
        'target_lags': {},
        'target_rolling': {},
        'feature_lags': {},
        'feature_rolling': {},
        'total_features_created': 0
    }
    
    # 1. Лаги целевой переменной
    df_engineered, lag_info = create_lag_features(df_engineered, target_column, lags=target_lags)
    all_info['target_lags'] = lag_info
    all_info['total_features_created'] += lag_info['total_created']
    
    # 2. Скользящие статистики целевой переменной
    df_engineered, rolling_info = create_rolling_features(
        df_engineered, target_column, 
        windows=rolling_windows, 
        statistics=rolling_stats
    )
    all_info['target_rolling'] = rolling_info
    all_info['total_features_created'] += rolling_info['total_created']
    
    # 3. Лаги для дополнительных признаков
    if feature_columns:
        feature_lag_info = {}
        for feature in feature_columns:
            if feature in df_engineered.columns and pd.api.types.is_numeric_dtype(df_engineered[feature]):
                df_engineered, f_lag_info = create_lag_features(
                    df_engineered, feature, lags=feature_lags
                )
                feature_lag_info[feature] = f_lag_info
                all_info['total_features_created'] += f_lag_info['total_created']
        
        all_info['feature_lags'] = feature_lag_info
    
    # 4. Скользящие статистики для дополнительных признаков (опционально)
    if feature_columns:
        feature_rolling_info = {}
        for feature in feature_columns:
            if feature in df_engineered.columns and pd.api.types.is_numeric_dtype(df_engineered[feature]):
                df_engineered, f_rolling_info = create_rolling_features(
                    df_engineered, feature, 
                    windows=rolling_windows[:1],  # Только первое окно для признаков
                    statistics=['mean']  # Только среднее для признаков
                )
                feature_rolling_info[feature] = f_rolling_info
                all_info['total_features_created'] += f_rolling_info['total_created']
        
        all_info['feature_rolling'] = feature_rolling_info
    
    return df_engineered, all_info


def calculate_lag_correlations(df, target_column, lag_columns):
    """
    Расчет корреляции между лаговыми признаками и целевой переменной
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    target_column : str
        Целевая переменная
    lag_columns : list
        Список лаговых признаков
    
    Возвращает:
    ----------
    pd.DataFrame : таблица корреляций
    """
    correlations = []
    
    for lag_col in lag_columns:
        if lag_col in df.columns:
            # Удаляем NaN одновременно из обоих столбцов
            valid_data = df[[target_column, lag_col]].dropna()
            
            # Проверяем, что есть достаточно данных
            if len(valid_data) < 3:
                continue
            
            try:
                # Pearson корреляция
                pearson_corr, pearson_p = stats.pearsonr(
                    valid_data[target_column], 
                    valid_data[lag_col]
                )
                
                # Spearman корреляция
                spearman_corr, spearman_p = stats.spearmanr(
                    valid_data[target_column], 
                    valid_data[lag_col]
                )
            except Exception as e:
                # Если не удается вычислить корреляцию, пропускаем
                continue
            
            # Извлекаем лаг из имени
            if '_lag_' in lag_col:
                lag_value = int(lag_col.split('_lag_')[-1])
            else:
                lag_value = None
            
            correlations.append({
                'Feature': lag_col,
                'Lag': lag_value,
                'Pearson_r': pearson_corr,
                'Pearson_p': pearson_p,
                'Spearman_r': spearman_corr,
                'Spearman_p': spearman_p,
                'Abs_Pearson_r': abs(pearson_corr),
                'Significant': 'Да' if pearson_p < 0.05 else 'Нет'
            })
    
    corr_df = pd.DataFrame(correlations)
    
    # Сортируем по абсолютной корреляции
    if not corr_df.empty:
        corr_df = corr_df.sort_values('Abs_Pearson_r', ascending=False)
    
    return corr_df


def check_multicollinearity_vif(df, feature_columns):
    """
    Проверка мультиколлинеарности через VIF (Variance Inflation Factor)
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    feature_columns : list
        Список признаков для проверки
    
    Возвращает:
    ----------
    pd.DataFrame : таблица с VIF значениями
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Выбираем только указанные признаки и удаляем NaN
    df_features = df[feature_columns].dropna()
    
    if len(df_features) < 10:
        return pd.DataFrame({
            'Feature': feature_columns,
            'VIF': ['Недостаточно данных'] * len(feature_columns)
        })
    
    vif_data = []
    
    for i, feature in enumerate(feature_columns):
        try:
            vif = variance_inflation_factor(df_features.values, i)
            vif_data.append({
                'Feature': feature,
                'VIF': vif,
                'Status': 'Высокая' if vif > 10 else ('Умеренная' if vif > 5 else 'Низкая')
            })
        except:
            vif_data.append({
                'Feature': feature,
                'VIF': np.nan,
                'Status': 'Ошибка расчета'
            })
    
    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values('VIF', ascending=False)
    
    return vif_df


def analyze_feature_importance_correlation(df, target_column, feature_columns, top_n=10):
    """
    Анализ важности признаков через корреляцию с целевой переменной
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    target_column : str
        Целевая переменная
    feature_columns : list
        Список признаков
    top_n : int
        Количество топ признаков для отображения
    
    Возвращает:
    ----------
    pd.DataFrame : таблица важности признаков
    """
    importance_data = []
    
    for feature in feature_columns:
        if feature in df.columns and feature != target_column:
            # Удаляем NaN для корректного расчета
            valid_data = df[[target_column, feature]].dropna()
            
            if len(valid_data) > 10:
                try:
                    corr, p_value = stats.pearsonr(valid_data[target_column], valid_data[feature])
                    
                    importance_data.append({
                        'Feature': feature,
                        'Correlation': corr,
                        'Abs_Correlation': abs(corr),
                        'P_value': p_value,
                        'Significant': 'Да' if p_value < 0.05 else 'Нет',
                        'Valid_samples': len(valid_data)
                    })
                except:
                    continue
    
    importance_df = pd.DataFrame(importance_data)
    
    if not importance_df.empty:
        importance_df = importance_df.sort_values('Abs_Correlation', ascending=False).head(top_n)
    
    return importance_df


def create_lag_correlation_plot(corr_df):
    """
    Создание графика корреляций лагов с целевой переменной
    
    Параметры:
    ----------
    corr_df : pd.DataFrame
        Таблица корреляций
    
    Возвращает:
    ----------
    plotly.graph_objects.Figure : график
    """
    if corr_df.empty:
        return None
    
    fig = go.Figure()
    
    # Pearson корреляция
    fig.add_trace(go.Bar(
        x=corr_df['Feature'],
        y=corr_df['Pearson_r'],
        name='Pearson',
        marker_color='steelblue',
        text=corr_df['Pearson_r'].round(3),
        textposition='outside',
        hovertemplate='%{x}<br>Корреляция: %{y:.4f}<extra></extra>'
    ))
    
    # Добавляем линии значимости
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                  annotation_text="Сильная корреляция (0.7)")
    fig.add_hline(y=0.3, line_dash="dash", line_color="orange", 
                  annotation_text="Умеренная корреляция (0.3)")
    fig.add_hline(y=-0.3, line_dash="dash", line_color="orange")
    fig.add_hline(y=-0.7, line_dash="dash", line_color="green")
    
    fig.update_layout(
        title="Корреляция лаговых признаков с целевой переменной",
        xaxis_title="Признак",
        yaxis_title="Корреляция (Pearson)",
        yaxis_range=[-1.1, 1.1],
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def create_feature_importance_plot(importance_df):
    """
    Создание графика важности признаков
    
    Параметры:
    ----------
    importance_df : pd.DataFrame
        Таблица важности признаков
    
    Возвращает:
    ----------
    plotly.graph_objects.Figure : график
    """
    if importance_df.empty:
        return None
    
    # Цветовая шкала в зависимости от корреляции
    colors = ['green' if x > 0.7 else 'orange' if x > 0.3 else 'gray' 
              for x in importance_df['Abs_Correlation']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=importance_df['Feature'],
        x=importance_df['Abs_Correlation'],
        orientation='h',
        marker_color=colors,
        text=importance_df['Abs_Correlation'].round(3),
        textposition='outside',
        hovertemplate='%{y}<br>|Корреляция|: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Топ признаков по корреляции с целевой переменной",
        xaxis_title="Абсолютная корреляция",
        yaxis_title="Признак",
        height=max(400, len(importance_df) * 30),
        showlegend=False
    )
    
    return fig


def create_rolling_features_plot(df, date_column, original_column, rolling_features):
    """
    Визуализация исходного ряда и скользящих статистик
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    date_column : str
        Столбец с датой
    original_column : str
        Исходный столбец
    rolling_features : list
        Список скользящих признаков для отображения
    
    Возвращает:
    ----------
    plotly.graph_objects.Figure : график
    """
    fig = go.Figure()
    
    # Исходный ряд
    fig.add_trace(go.Scatter(
        x=df[date_column],
        y=df[original_column],
        mode='lines',
        name='Исходный ряд',
        line=dict(color='lightblue', width=1),
        opacity=0.6
    ))
    
    # Скользящие признаки
    colors = px.colors.qualitative.Plotly
    for idx, feature in enumerate(rolling_features):
        if feature in df.columns:
            fig.add_trace(go.Scatter(
                x=df[date_column],
                y=df[feature],
                mode='lines',
                name=feature,
                line=dict(color=colors[idx % len(colors)], width=2)
            ))
    
    fig.update_layout(
        title=f"Исходный ряд и скользящие признаки: {original_column}",
        xaxis_title="Дата",
        yaxis_title="Значение",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def get_feature_statistics(df, feature_columns):
    """
    Получение статистики по созданным признакам
    
    Параметры:
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    feature_columns : list
        Список признаков
    
    Возвращает:
    ----------
    pd.DataFrame : таблица статистики
    """
    stats_data = []
    
    for feature in feature_columns:
        if feature in df.columns:
            data = df[feature]
            stats_data.append({
                'Feature': feature,
                'Mean': data.mean(),
                'Std': data.std(),
                'Min': data.min(),
                'Max': data.max(),
                'Missing': data.isnull().sum(),
                'Missing_%': (data.isnull().sum() / len(data)) * 100
            })
    
    stats_df = pd.DataFrame(stats_data)
    
    return stats_df

