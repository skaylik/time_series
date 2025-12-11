# feature_engineering.py - Модуль инжиниринга признаков (Этап 1)

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from scipy.special import boxcox, inv_boxcox
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Импорт для Streamlit интерфейса
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# ОСНОВНАЯ ЛОГИКА ИНЖИНИРИНГА ПРИЗНАКОВ
# ============================================================

class TimeSeriesFeatureEngineer:
    """
    Класс для инжиниринга признаков временных рядов
    """
    
    def __init__(self, date_col: str, target_col: str):
        """
        Инициализация класса
        
        Parameters:
        -----------
        date_col : str
            Название столбца с датой/временем
        target_col : str
            Название целевой переменной
        """
        self.date_col = date_col
        self.target_col = target_col
        self.lambda_boxcox = None
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание всех признаков
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходный DataFrame
            
        Returns:
        --------
        pd.DataFrame
            DataFrame с добавленными признаками
        """
        # Создаем копию DataFrame
        df_features = df.copy()
        
        # Преобразуем дату сразу, чтобы избежать ошибок
        df_features = self._ensure_datetime(df_features)
        
        # Сортируем по дате
        df_features = df_features.sort_values(self.date_col)
        
        # Сбрасываем индекс для корректных сдвигов
        df_features = df_features.reset_index(drop=True)
        
        # 1. Лаги
        df_features = self._create_lags(df_features)
        
        # 2. Скользящие окна
        df_features = self._create_rolling_features(df_features)
        
        # 3. Экспоненциальное сглаживание
        df_features = self._create_exponential_smoothing(df_features)
        
        # 4. Временные признаки
        df_features = self._create_time_features(df_features)
        
        # 5. Сезонные компоненты (Фурье)
        df_features = self._create_fourier_features(df_features)
        
        # УДАЛЯЕМ СТРОКУ СБРОСА ИНДЕКСА - ОНА УНИЧТОЖАЕТ ПРИЗНАКИ!
        # df_features = df_features.reset_index(drop=True)
        
        # После создания всех признаков выведем отладочную информацию
        print(f"Создано признаков: {len(df_features.columns)}")
        print(f"Имена признаков: {df_features.columns.tolist()}")

        return df_features
    
    def _ensure_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразование столбца с датой в datetime с обработкой временных зон
        """
        df_copy = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df_copy[self.date_col]):
            try:
                # Пробуем преобразовать с учетом временных зон
                df_copy[self.date_col] = pd.to_datetime(df_copy[self.date_col], utc=True)
            except Exception as e:
                # Если не получается с utc=True, пробуем без
                df_copy[self.date_col] = pd.to_datetime(df_copy[self.date_col])
        
        # Убираем временную зону если она есть
        if hasattr(df_copy[self.date_col].dt, 'tz') and df_copy[self.date_col].dt.tz is not None:
            df_copy[self.date_col] = df_copy[self.date_col].dt.tz_convert(None)
        
        return df_copy
    
    def _create_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание лаговых признаков
        """
        lags = [1, 2, 3, 7, 14, 30]
        
        for lag in lags:
            col_name = f'{self.target_col}_lag_{lag}'
            if col_name not in df.columns:  # Проверяем, не существует ли уже столбца
                df[col_name] = df[self.target_col].shift(lag)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание признаков скользящих окон
        """
        windows = [7, 14, 30]
        
        for window in windows:
            # Проверяем существование столбцов перед созданием
            mean_col = f'{self.target_col}_rolling_mean_{window}'
            std_col = f'{self.target_col}_rolling_std_{window}'
            min_col = f'{self.target_col}_rolling_min_{window}'
            max_col = f'{self.target_col}_rolling_max_{window}'
            median_col = f'{self.target_col}_rolling_median_{window}'
            range_col = f'{self.target_col}_rolling_range_{window}'
            cv_col = f'{self.target_col}_rolling_cv_{window}'
            
            if mean_col not in df.columns:
                df[mean_col] = df[self.target_col].rolling(
                    window=window, min_periods=1
                ).mean()
            
            if std_col not in df.columns:
                df[std_col] = df[self.target_col].rolling(
                    window=window, min_periods=1
                ).std()
            
            if min_col not in df.columns:
                df[min_col] = df[self.target_col].rolling(
                    window=window, min_periods=1
                ).min()
            
            if max_col not in df.columns:
                df[max_col] = df[self.target_col].rolling(
                    window=window, min_periods=1
                ).max()
            
            if median_col not in df.columns:
                df[median_col] = df[self.target_col].rolling(
                    window=window, min_periods=1
                ).median()
            
            if range_col not in df.columns:
                # Создаем только если созданы min и max
                if max_col in df.columns and min_col in df.columns:
                    df[range_col] = df[max_col] - df[min_col]
            
            if cv_col not in df.columns:
                # Создаем только если созданы std и mean
                if std_col in df.columns and mean_col in df.columns:
                    df[cv_col] = df[std_col] / df[mean_col]
                    # Заменяем бесконечные значения на NaN
                    df[cv_col] = df[cv_col].replace(
                        [np.inf, -np.inf], np.nan
                    )
        
        return df
    
    def _create_exponential_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание признаков экспоненциального сглаживания
        """
        alphas = [0.3, 0.5, 0.7]
        
        for alpha in alphas:
            col_name = f'{self.target_col}_exp_smooth_{alpha}'
            if col_name not in df.columns:
                df[col_name] = 0.0
                
                if len(df) > 0:
                    df.loc[0, col_name] = df.loc[0, self.target_col]
                
                for i in range(1, len(df)):
                    df.loc[i, col_name] = (
                        alpha * df.loc[i, self.target_col] + 
                        (1 - alpha) * df.loc[i-1, col_name]
                    )
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание временных признаков
        """
        # Убедимся, что дата в правильном формате
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
            df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Базовые временные признаки - создаем только если не существуют
        time_features = {
            'day_of_week': df[self.date_col].dt.dayofweek,  # 0-понедельник, 6-воскресенье
            'day_of_month': df[self.date_col].dt.day,
            'month': df[self.date_col].dt.month,
            'quarter': df[self.date_col].dt.quarter,
            'week_of_year': df[self.date_col].dt.isocalendar().week,
            'year': df[self.date_col].dt.year,
            'is_weekend': df[self.date_col].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0),
            'is_holiday': 0,  # Признак праздника (заглушка)
        }
        
        for col_name, values in time_features.items():
            if col_name not in df.columns:
                df[col_name] = values
        
        # Цикличные признаки
        cyclic_features = {
            'month_sin': np.sin(2 * np.pi * df['month'] / 12),
            'month_cos': np.cos(2 * np.pi * df['month'] / 12),
            'day_of_week_sin': np.sin(2 * np.pi * df['day_of_week'] / 7),
            'day_of_week_cos': np.cos(2 * np.pi * df['day_of_week'] / 7),
        }
        
        for col_name, values in cyclic_features.items():
            if col_name not in df.columns:
                df[col_name] = values
        
        return df
    
    def _create_fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание сезонных компонент Фурье
        """
        seasons = [7, 30]  # недельная и месячная сезонность
        t = np.arange(len(df))
        
        for season in seasons:
            fourier_cols = {
                f'fourier_sin_{season}': np.sin(2 * np.pi * t / season),
                f'fourier_cos_{season}': np.cos(2 * np.pi * t / season),
                f'fourier_sin_{season}_2': np.sin(4 * np.pi * t / season),
                f'fourier_cos_{season}_2': np.cos(4 * np.pi * t / season),
            }
            
            for col_name, values in fourier_cols.items():
                if col_name not in df.columns:
                    df[col_name] = values
        
        return df
    
    def apply_target_transformations(self, df: pd.DataFrame, 
                                     apply_log: bool = True, 
                                     apply_boxcox: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Применение трансформаций к целевой переменной
        """
        df_transformed = df.copy()
        transformation_params = {}
        
        # Логарифмическое преобразование
        if apply_log and (df_transformed[self.target_col] > 0).all():
            log_col = f'{self.target_col}_log'
            if log_col not in df_transformed.columns:
                df_transformed[log_col] = np.log(df_transformed[self.target_col])
                transformation_params['log_applied'] = True
                transformation_params['log_col'] = log_col
            else:
                transformation_params['log_applied'] = False
        else:
            transformation_params['log_applied'] = False
            
        # Преобразование Бокса-Кокса
        if apply_boxcox and (df_transformed[self.target_col] > 0).all():
            y_positive = df_transformed[self.target_col][df_transformed[self.target_col] > 0]
            if len(y_positive) > 0:
                # Добавляем маленькое значение для избежания нулей
                y_for_boxcox = df_transformed[self.target_col] + 1e-10
                self.lambda_boxcox = stats.boxcox_normmax(y_for_boxcox)
                transformation_params['lambda_boxcox'] = self.lambda_boxcox
                
                boxcox_col = f'{self.target_col}_boxcox'
                if boxcox_col not in df_transformed.columns:
                    # ИСПРАВЛЕНИЕ: stats.boxcox с параметром lmbda возвращает только преобразованные значения
                    df_transformed[boxcox_col] = stats.boxcox(
                        y_for_boxcox, 
                        lmbda=self.lambda_boxcox
                    )
                    transformation_params['boxcox_applied'] = True
                    transformation_params['boxcox_col'] = boxcox_col
                else:
                    transformation_params['boxcox_applied'] = False
            else:
                transformation_params['boxcox_applied'] = False
        else:
            transformation_params['boxcox_applied'] = False
        
        return df_transformed, transformation_params
    
    def inverse_target_transformations(self, predictions: np.ndarray,
                                       transformation_type: str = 'boxcox') -> np.ndarray:
        """
        Обратное преобразование целевой переменной
        """
        if transformation_type == 'boxcox' and self.lambda_boxcox is not None:
            return inv_boxcox(predictions, self.lambda_boxcox)
        elif transformation_type == 'log':
            return np.exp(predictions)
        else:
            return predictions
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """
        Получение категорий признаков
        """
        return {
            'Исходные признаки': [self.date_col, self.target_col],
            'Лаги': [f'{self.target_col}_lag_{lag}' for lag in [1, 2, 3, 7, 14, 30]],
            'Скользящие окна (mean)': [f'{self.target_col}_rolling_mean_{window}' for window in [7, 14, 30]],
            'Скользящие окна (std)': [f'{self.target_col}_rolling_std_{window}' for window in [7, 14, 30]],
            'Скользящие окна (min)': [f'{self.target_col}_rolling_min_{window}' for window in [7, 14, 30]],
            'Скользящие окна (max)': [f'{self.target_col}_rolling_max_{window}' for window in [7, 14, 30]],
            'Скользящие окна (median)': [f'{self.target_col}_rolling_median_{window}' for window in [7, 14, 30]],
            'Экспоненциальное сглаживание': [f'{self.target_col}_exp_smooth_{alpha}' for alpha in [0.3, 0.5, 0.7]],
            'Временные признаки': ['day_of_week', 'day_of_month', 'month', 'quarter', 
                                   'week_of_year', 'year', 'is_weekend', 'is_holiday'],
            'Цикличные признаки': ['month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos'],
            'Фурье компоненты': [f'fourier_sin_{season}' for season in [7, 30]] + 
                              [f'fourier_cos_{season}' for season in [7, 30]] +
                              [f'fourier_sin_{season}_2' for season in [7, 30]] +
                              [f'fourier_cos_{season}_2' for season in [7, 30]]
        }


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def create_time_series_features(df: pd.DataFrame, 
                                date_col: str, 
                                target_col: str,
                                apply_transformations: bool = False,
                                include_fourier: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Быстрое создание признаков временного ряда
    """
    print(f"🛠️ Начинаем создание признаков...")
    print(f"   - Размер исходных данных: {df.shape}")
    print(f"   - Дата: {date_col}, Цель: {target_col}")
    
    feature_engineer = TimeSeriesFeatureEngineer(date_col, target_col)
    df_features = feature_engineer.create_features(df)
    
    # Если не включать Фурье, удаляем эти колонки
    if not include_fourier:
        fourier_cols = [col for col in df_features.columns if 'fourier' in col]
        df_features = df_features.drop(columns=fourier_cols)
    
    transformation_info = {}
    if apply_transformations:
        df_features, transformation_info = feature_engineer.apply_target_transformations(df_features)
    
    feature_categories = feature_engineer.get_feature_categories()
    
    # Исключаем дубликаты из created_features
    created_features = []
    for col in df_features.columns:
        if col not in [date_col, target_col]:
            created_features.append(col)
    
    feature_info = {
        'original_features': [date_col, target_col],
        'created_features': created_features,
        'total_features': len(df_features.columns),
        'feature_categories': feature_categories,
        'transformation_info': transformation_info,
        'engineer': feature_engineer
    }
    
    # ВОТ КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ - ВОЗВРАЩАЕМ ДАННЫЕ!
    return df_features, feature_info


def analyze_feature_importance(df_features: pd.DataFrame, 
                               target_col: str,
                               date_col: str,
                               top_n: int = 20) -> pd.DataFrame:
    """
    Анализ значимости признаков с помощью корреляции
    """
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != target_col and col != date_col]
    
    correlations = []
    for col in feature_cols:
        valid_idx = df_features[col].notna() & df_features[target_col].notna()
        if valid_idx.sum() > 0:
            corr = np.corrcoef(df_features.loc[valid_idx, col], 
                               df_features.loc[valid_idx, target_col])[0, 1]
            correlations.append((col, corr))
    
    if correlations:
        corr_df = pd.DataFrame(correlations, columns=['Признак', 'Корреляция'])
        corr_df['Абс_корреляция'] = corr_df['Корреляция'].abs()
        corr_df = corr_df.sort_values('Абс_корреляция', ascending=False)
        
        return corr_df.head(top_n)
    else:
        return pd.DataFrame(columns=['Признак', 'Корреляция', 'Абс_корреляция'])


# ============================================================
# ИНТЕРФЕЙС ДЛЯ STREAMLIT
# ============================================================

def show_feature_engineering_ui(df, date_col, target_col):
    """
    Показать интерфейс инжиниринга признаков в Streamlit
    """
    
    # Проверяем, есть ли уже созданные признаки в session_state
    if 'df_features' in st.session_state and 'feature_info' in st.session_state:
        # Возвращаем существующие данные
        return st.session_state.df_features, st.session_state.feature_info
    
    # Если признаки еще не созданы, показываем интерфейс для их создания
    st.markdown("### ⚙️ Настройки создания признаков")
    
    col1, col2 = st.columns(2)
    with col1:
        apply_transformations = st.checkbox("Применить трансформации к целевой переменной", value=True)
        st.caption("Логарифм (если y > 0) и преобразование Бокса-Кокса")
    
    with col2:
        include_fourier = st.checkbox("Добавить Фурье-признаки", value=True)
        st.caption("Сезонные компоненты (недельная и месячная)")
    
    # Кнопка для создания признаков
    if st.button("🚀 Создать признаки", type="primary", use_container_width=True):
        with st.spinner("Создание признаков..."):
            try:
                # Создаем признаки
                df_features, feature_info = create_time_series_features(
                    df, 
                    date_col, 
                    target_col,
                    apply_transformations=apply_transformations,
                    include_fourier=include_fourier
                )
                
                st.success(f"✅ Создано {len(feature_info['created_features'])} признаков!")
                
                # Возвращаем данные
                return df_features, feature_info
                
            except Exception as e:
                st.error(f"Ошибка при создании признаков: {str(e)}")
                return None
    
    # Если кнопка не нажата, возвращаем None
    return None


def _display_feature_engineering_results(df_features, feature_info, date_col, target_col):
    """
    Вспомогательная функция для отображения результатов инжиниринга
    """
    # 1. Общая информация
    st.subheader("📊 Общая информация")
    
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    
    with info_col1:
        st.metric("Исходных признаков", len(feature_info['original_features']))
    
    with info_col2:
        st.metric("Созданных признаков", len(feature_info['created_features']))
    
    with info_col3:
        st.metric("Всего признаков", feature_info['total_features'])
    
    with info_col4:
        st.metric("Записей", len(df_features))
    
    # 2. Категории признаков
    st.subheader("🗂️ Категории признаков")
    
    category_stats = []
    for category, features in feature_info['feature_categories'].items():
        # Считаем только существующие признаки
        existing_features = [f for f in features if f in df_features.columns]
        if existing_features:
            category_stats.append({
                'Категория': category,
                'Количество признаков': len(existing_features),
                'Примеры': ', '.join(existing_features[:3]) + ('...' if len(existing_features) > 3 else '')
            })
    
    if category_stats:
        st.dataframe(pd.DataFrame(category_stats), width='stretch')
    
    # 3. Просмотр признаков
    st.subheader("👁️ Просмотр созданных признаков")
    
    if category_stats:
        selected_category = st.selectbox(
            "Выберите категорию для просмотра",
            options=[cat['Категория'] for cat in category_stats],
            key="category_selector"
        )
        
        if selected_category:
            # Находим признаки для выбранной категории
            features_to_show = []
            for cat in category_stats:
                if cat['Категория'] == selected_category:
                    # Извлекаем признаки из строки примеров
                    example_str = cat['Примеры']
                    if '...' in example_str:
                        features_to_show = example_str.replace('...', '').split(', ')
                    else:
                        features_to_show = example_str.split(', ')
                    break
            
            # Удаляем дубликаты и проверяем существование столбцов
            features_to_show = [f.strip() for f in features_to_show]
            features_to_show = list(set(features_to_show))  # Удаляем дубликаты
            
            # Формируем список столбцов для отображения
            cols_to_show = []
            seen_columns = set()
            
            # Сначала добавляем основные столбцы
            for col in [date_col, target_col]:
                if col in df_features.columns and col not in seen_columns:
                    cols_to_show.append(col)
                    seen_columns.add(col)
            
            # Затем добавляем признаки из выбранной категории
            for feature in features_to_show:
                if feature in df_features.columns and feature not in seen_columns:
                    cols_to_show.append(feature)
                    seen_columns.add(feature)
            
            if cols_to_show:
                st.dataframe(df_features[cols_to_show].head(10), width='stretch')
    
    # 4. Визуализация некоторых признаков
    st.subheader("📈 Визуализация созданных признаков")
    
    available_features = feature_info['created_features']
    # Выбираем интересные признаки для визуализации
    interesting_features = []
    for feat in available_features:
        if any(keyword in feat for keyword in ['lag', 'rolling', 'exp_smooth']):
            interesting_features.append(feat)
    
    # Ограничиваем количество
    interesting_features = interesting_features[:6]
    
    if interesting_features:
        num_features = len(interesting_features)
        rows = (num_features + 1) // 2
        
        fig = make_subplots(
            rows=rows, cols=2,
            subplot_titles=interesting_features,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        for i, feat in enumerate(interesting_features):
            row = i // 2 + 1
            col = i % 2 + 1
            fig.add_trace(
                go.Scatter(
                    x=df_features[date_col], 
                    y=df_features[feat], 
                    mode='lines',
                    name=feat,
                    line=dict(width=1)
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=300 * rows, 
            showlegend=False,
            title_text="Примеры созданных признаков"
        )
        st.plotly_chart(fig, width='stretch')
    
    # 5. Анализ корреляций
    st.subheader("🔗 Анализ корреляций с целевой переменной")
    
    try:
        corr_df = analyze_feature_importance(df_features, target_col, date_col, top_n=15)
        
        if not corr_df.empty:
            # Визуализация корреляций
            fig_corr = go.Figure()
            corr_df_sorted = corr_df.sort_values('Корреляция', ascending=True)
            colors = ['red' if x < 0 else 'green' for x in corr_df_sorted['Корреляция']]
            
            fig_corr.add_trace(go.Bar(
                x=corr_df_sorted['Корреляция'],
                y=corr_df_sorted['Признак'],
                orientation='h',
                marker_color=colors,
                text=corr_df_sorted['Корреляция'].round(3),
                textposition='auto'
            ))
            
            fig_corr.update_layout(
                title="Топ-15 признаков по корреляции с целевой переменной",
                height=500,
                xaxis_title="Корреляция",
                yaxis_title="Признак"
            )
            st.plotly_chart(fig_corr, width='stretch')
            
            # Таблица с корреляциями
            st.dataframe(corr_df, width='stretch')
        else:
            st.info("Не удалось вычислить корреляции. Возможно, недостаточно числовых признаков.")
            
    except Exception as e:
        st.warning(f"Не удалось проанализировать корреляции: {str(e)}")
    
    # 6. Информация о трансформациях
    if feature_info['transformation_info']:
        trans_info = feature_info['transformation_info']
        
        if trans_info.get('log_applied') or trans_info.get('boxcox_applied'):
            st.subheader("🔄 Информация о трансформациях")
            
            if trans_info.get('log_applied'):
                st.success(f"✅ Логарифмическое преобразование применено: {trans_info.get('log_col', 'N/A')}")
            
            if trans_info.get('boxcox_applied'):
                st.success(f"✅ Преобразование Бокса-Кокса применено: λ={trans_info.get('lambda_boxcox', 'N/A'):.4f}")
    
    # Не возвращаем ничего, т.к. это вспомогательная функция только для отображения