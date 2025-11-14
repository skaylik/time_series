# app.py
"""
Streamlit приложение для анализа временных рядов.
Углублённая декомпозиция и анализ остатков
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from decomposition_analysis import DecompositionAnalyzer
from feature_engineering import FeatureEngineer, FeatureEngineeringConfig
from forecasting_strategies import ForecastingStrategies
from time_series_cv import TimeSeriesCrossValidator, CrossValidationSummary
import warnings
warnings.filterwarnings('ignore')
from typing import Optional, List, Dict
from stationarity_transformations import StationarityTransformer, TransformationStep
from exp_smoothing_models import ExponentialSmoothingRunner, ModelResult, BenchmarkResult
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
import json

# Функции должны быть определены ДО их использования
def visualize_decomposition(decomp, title="Декомпозиция временного ряда"):
    """
    Визуализирует компоненты декомпозиции.
    """
    # Создаём subplot с 4 графиками
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Исходный ряд', 'Тренд', 'Сезонность', 'Остатки'),
        vertical_spacing=0.08,
        row_heights=[0.3, 0.3, 0.2, 0.2]
    )
    
    # Исходный ряд
    fig.add_trace(
        go.Scatter(
            x=decomp.observed.index,
            y=decomp.observed.values,
            mode='lines',
            name='Исходный ряд',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # Тренд
    fig.add_trace(
        go.Scatter(
            x=decomp.trend.index,
            y=decomp.trend.values,
            mode='lines',
            name='Тренд',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )
    
    # Сезонность
    fig.add_trace(
        go.Scatter(
            x=decomp.seasonal.index,
            y=decomp.seasonal.values,
            mode='lines',
            name='Сезонность',
            line=dict(color='orange', width=1)
        ),
        row=3, col=1
    )
    
    # Остатки
    fig.add_trace(
        go.Scatter(
            x=decomp.resid.index,
            y=decomp.resid.values,
            mode='lines',
            name='Остатки',
            line=dict(color='red', width=1)
        ),
        row=4, col=1
    )
    
    # Обновляем layout
    fig.update_layout(
        height=1000,
        title_text=title,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Дата", row=4, col=1)
    fig.update_yaxes(title_text="Значение", row=1, col=1)
    fig.update_yaxes(title_text="Тренд", row=2, col=1)
    fig.update_yaxes(title_text="Сезонность", row=3, col=1)
    fig.update_yaxes(title_text="Остатки", row=4, col=1)
    
    # Отображаем график
    st.plotly_chart(fig, use_container_width=True)


def display_residual_analysis(residual_analysis, residuals):
    """
    Отображает результаты анализа остатков.
    """
    # Стационарность
    st.markdown("#### 🔄 Проверка стационарности остатков")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Тест ADF (Augmented Dickey-Fuller)**")
        if 'adf' in residual_analysis['stationarity']:
            adf = residual_analysis['stationarity']['adf']
            if 'error' not in adf:
                st.write(f"- Статистика: {adf['statistic']:.4f}")
                st.write(f"- p-value: {adf['pvalue']:.4f}")
                is_stationary = adf.get('is_stationary', False)
                if is_stationary:
                    st.success("✅ Ряд стационарен (p < 0.05)")
                else:
                    st.warning("⚠️ Ряд нестационарен (p ≥ 0.05)")
                
                if 'critical_values' in adf:
                    st.write("**Критические значения:**")
                    for level, value in adf['critical_values'].items():
                        st.write(f"  {level}: {value:.4f}")
            else:
                st.error(f"Ошибка: {adf['error']}")
    
    with col2:
        st.markdown("**Тест KPSS (Kwiatkowski-Phillips-Schmidt-Shin)**")
        if 'kpss' in residual_analysis['stationarity']:
            kpss = residual_analysis['stationarity']['kpss']
            if 'error' not in kpss:
                st.write(f"- Статистика: {kpss['statistic']:.4f}")
                st.write(f"- p-value: {kpss['pvalue']:.4f}")
                is_stationary = kpss.get('is_stationary', False)
                if is_stationary:
                    st.success("✅ Ряд стационарен (p > 0.05)")
                else:
                    st.warning("⚠️ Ряд нестационарен (p ≤ 0.05)")
            else:
                st.error(f"Ошибка: {kpss['error']}")
    
    # Нормальность
    st.markdown("---")
    st.markdown("#### 📊 Проверка нормальности остатков")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Тест Д'Агостино-Пирсона**")
        if 'd_agostino' in residual_analysis['normality']:
            da = residual_analysis['normality']['d_agostino']
            if 'error' not in da:
                st.write(f"- Статистика: {da['statistic']:.4f}")
                st.write(f"- p-value: {da['pvalue']:.4f}")
                is_normal = da.get('is_normal', False)
                if is_normal:
                    st.success("✅ Распределение нормально (p > 0.05)")
                else:
                    st.warning("⚠️ Распределение ненормально (p ≤ 0.05)")
            else:
                st.error(f"Ошибка: {da['error']}")
    
    with col2:
        st.markdown("**Тест Жарке-Бера**")
        if 'jarque_bera' in residual_analysis['normality']:
            jb = residual_analysis['normality']['jarque_bera']
            if 'error' not in jb:
                st.write(f"- Статистика: {jb['statistic']:.4f}")
                st.write(f"- p-value: {jb['pvalue']:.4f}")
                is_normal = jb.get('is_normal', False)
                if is_normal:
                    st.success("✅ Распределение нормально (p > 0.05)")
                else:
                    st.warning("⚠️ Распределение ненормально (p ≤ 0.05)")
            else:
                st.error(f"Ошибка: {jb['error']}")
    
    # Описательная статистика
    if 'descriptive' in residual_analysis['normality']:
        desc = residual_analysis['normality']['descriptive']
        st.markdown("**Описательная статистика остатков:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Среднее", f"{desc['mean']:.4f}")
        with col2:
            st.metric("Ст. отклонение", f"{desc['std']:.4f}")
        with col3:
            st.metric("Асимметрия", f"{desc['skewness']:.4f}")
        with col4:
            st.metric("Эксцесс", f"{desc['kurtosis']:.4f}")
    
    # Визуализация остатков
    st.markdown("---")
    st.markdown("#### 📈 Визуализация остатков")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Гистограмма остатков
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=residuals.dropna().values,
            nbinsx=50,
            name='Остатки',
            marker_color='blue'
        ))
        fig_hist.update_layout(
            title='Гистограмма остатков',
            xaxis_title='Остатки',
            yaxis_title='Частота',
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Q-Q plot
        residuals_clean = residuals.dropna()
        qq_data = stats.probplot(residuals_clean, dist="norm")
        
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=qq_data[0][0],
            y=qq_data[0][1],
            mode='markers',
            name='Остатки',
            marker=dict(color='blue', size=4)
        ))
        fig_qq.add_trace(go.Scatter(
            x=qq_data[0][0],
            y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
            mode='lines',
            name='Теоретическая линия',
            line=dict(color='red', width=2)
        ))
        fig_qq.update_layout(
            title='Q-Q Plot (проверка нормальности)',
            xaxis_title='Теоретические квантили',
            yaxis_title='Выборочные квантили',
            height=400
        )
        st.plotly_chart(fig_qq, use_container_width=True)
    
    # ACF и PACF
    st.markdown("---")
    st.markdown("#### 🔄 ACF и PACF остатков")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ACF
        acf_values = acf(residuals.dropna(), nlags=40, fft=True)
        lags = range(len(acf_values))
        
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(
            x=list(lags),
            y=acf_values,
            name='ACF',
            marker_color='blue'
        ))
        # Добавляем доверительные интервалы
        conf_int = 1.96 / np.sqrt(len(residuals.dropna()))
        fig_acf.add_hline(y=conf_int, line_dash="dash", line_color="red", 
                         annotation_text="95% доверительный интервал")
        fig_acf.add_hline(y=-conf_int, line_dash="dash", line_color="red")
        fig_acf.update_layout(
            title='ACF остатков',
            xaxis_title='Лаг',
            yaxis_title='ACF',
            height=400
        )
        st.plotly_chart(fig_acf, use_container_width=True)
    
    with col2:
        # PACF
        pacf_values = pacf(residuals.dropna(), nlags=40)
        lags = range(len(pacf_values))
        
        fig_pacf = go.Figure()
        fig_pacf.add_trace(go.Bar(
            x=list(lags),
            y=pacf_values,
            name='PACF',
            marker_color='green'
        ))
        # Добавляем доверительные интервалы
        conf_int = 1.96 / np.sqrt(len(residuals.dropna()))
        fig_pacf.add_hline(y=conf_int, line_dash="dash", line_color="red",
                          annotation_text="95% доверительный интервал")
        fig_pacf.add_hline(y=-conf_int, line_dash="dash", line_color="red")
        fig_pacf.update_layout(
            title='PACF остатков',
            xaxis_title='Лаг',
            yaxis_title='PACF',
            height=400
        )
        st.plotly_chart(fig_pacf, use_container_width=True)
    
    # Автокорреляция
    if 'autocorrelation' in residual_analysis:
        st.markdown("---")
        st.markdown("#### 🔗 Проверка автокорреляции остатков")
        
        if 'ljung_box' in residual_analysis['autocorrelation']:
            lb = residual_analysis['autocorrelation']['ljung_box']
            if 'has_autocorrelation' in lb:
                has_ac = lb['has_autocorrelation']
                if has_ac:
                    st.warning("⚠️ Обнаружена автокорреляция в остатках")
                else:
                    st.success("✅ Автокорреляция в остатках отсутствует")


def display_comparison_table(comparisons):
    """
    Отображает таблицу сравнения всех вариантов декомпозиции.
    """
    # Создаём таблицу для сравнения
    comparison_data = []
    
    for key, value in comparisons.items():
        if 'error' not in value:
            comparison_data.append({
                'Модель': value['model'],
                'Период': value['period'],
                'Оценка': f"{value['score']:.2f}",
                'Среднее остатков': f"{value['residual_stats']['mean']:.4f}",
                'Ст. отклонение остатков': f"{value['residual_stats']['std']:.4f}",
                'Дисперсия остатков': f"{value['residual_stats']['variance']:.4f}"
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df['Оценка'] = pd.to_numeric(comparison_df['Оценка'], errors='coerce')
        for col in ['Среднее остатков', 'Ст. отклонение остатков', 'Дисперсия остатков']:
            comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce')

        comparison_df = comparison_df.sort_values('Оценка', ascending=False)

        st.dataframe(
            comparison_df,
            use_container_width=True
        )
        
        # Визуализация сравнения
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"{row['Модель']}_{row['Период']}" for _, row in comparison_df.iterrows()],
            y=comparison_df['Оценка'].astype(float),
            marker_color='steelblue'
        ))
        fig.update_layout(
            title='Сравнение оценок качества декомпозиций',
            xaxis_title='Вариант декомпозиции',
            yaxis_title='Оценка качества',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Нет данных для сравнения")

# Настройка страницы
st.set_page_config(
    page_title="🧪 Лабораторный практикум № 2 — Прогнозирование временных рядов",
    page_icon="📊",
    layout="wide"
)

# Заголовок
st.title("🧪 Лабораторный практикум № 2")
st.markdown("### Прогнозирование временных рядов: стратегии, валидация и модели экспоненциального сглаживания")

# Краткая теория
with st.expander("📚 Краткая теория", expanded=False):
    st.markdown("""
    **Декомпозиция временного ряда** — это разложение ряда на компоненты:
    
    - **Тренд (Trend)** — долгосрочная направленность данных (рост, падение, стабильность)
    - **Сезонность (Seasonal)** — регулярные повторяющиеся паттерны с фиксированным периодом
    - **Остатки (Residual)** — случайная составляющая, не объясняемая трендом и сезонностью
    
    **Типы моделей:**
    - **Аддитивная**: `Y(t) = Trend(t) + Seasonal(t) + Residual(t)`
    - **Мультипликативная**: `Y(t) = Trend(t) × Seasonal(t) × Residual(t)`
    
    **Анализ остатков** позволяет оценить качество декомпозиции:
    - **Стационарность** (ADF, KPSS тесты) — остатки должны быть стационарными
    - **Нормальность** (тесты Д'Агостино-Пирсона, Жарке-Бера) — желательно нормальное распределение
    - **Автокорреляция** (ACF/PACF, тест Льюнга-Бокса) — остатки не должны иметь автокорреляции
    
    **Цель блока:** Выбрать оптимальную декомпозицию на основе анализа остатков.
    """)

# Боковая панель - только загрузка файла
st.sidebar.markdown("### 📁 Загрузка данных")
uploaded_file = st.sidebar.file_uploader(
    "Выберите файл",
    type=['csv', 'parquet'],
    help="Поддерживаются файлы CSV и Parquet",
    label_visibility="collapsed"
)

# Основная область - загрузка и настройки
if uploaded_file is None:
    st.info("👆 Загрузите файл в боковой панели или используйте пример данных")
    
    # Кнопка для загрузки примера
    if st.button("📥 Загрузить пример данных", type="primary"):
        try:
            df_example = pd.read_csv('Dollar-Exchange.csv')
            st.session_state['df'] = df_example
            st.session_state['file_loaded'] = True
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка при загрузке примера: {e}")
else:
    # Загружаем файл
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        else:
            st.error("Неподдерживаемый формат файла")
            st.stop()
        
        st.session_state['df'] = df
        st.session_state['file_loaded'] = True
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
        st.stop()

# Если данные загружены
if st.session_state.get('file_loaded', False):
    df = st.session_state['df']
    
    st.markdown("---")
    
    # Информация о загруженном файле
    st.success(f"✅ Файл загружен: {uploaded_file.name if uploaded_file else 'Пример данных'}")
    
    # Информация о данных
    st.subheader("📋 Информация о данных")
    
    # Определяем столбец с датой для метрик
    date_columns_for_metrics = [col for col in df.columns if df[col].dtype == 'object' or 
                                pd.api.types.is_datetime64_any_dtype(df[col])]
    date_col_for_metrics = date_columns_for_metrics[0] if date_columns_for_metrics else None
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Строк", f"{df.shape[0]:,}")
    with col2:
        st.metric("📑 Столбцов", f"{df.shape[1]:,}")
    with col3:
        if date_col_for_metrics:
            try:
                date_start = str(df[date_col_for_metrics].min())[:10]
            except:
                date_start = "N/A"
        else:
            date_start = "N/A"
        st.metric("📅 Дата начала", date_start)
    with col4:
        if date_col_for_metrics:
            try:
                date_end = str(df[date_col_for_metrics].max())[:10]
            except:
                date_end = "N/A"
        else:
            date_end = "N/A"
        st.metric("📅 Дата конца", date_end)
    
    st.markdown("---")
    
    # Настройки анализа
    st.subheader("⚙️ Настройки анализа")
    
    # Выбор столбцов в колонках
    col1, col2 = st.columns(2)
    
    with col1:
        # Выбор столбца с датой
        date_columns = [col for col in df.columns if df[col].dtype == 'object' or 
                       pd.api.types.is_datetime64_any_dtype(df[col])]
        
        if not date_columns:
            st.error("❌ Не найден столбец с датами. Пожалуйста, проверьте данные.")
            st.stop()
        
        date_column = st.selectbox(
            "📅 Выберите столбец с датой",
            date_columns,
            index=0,
            help="Выберите столбец, содержащий даты временного ряда"
        )
    
    with col2:
        # Выбор переменной для временного ряда
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            st.error("❌ Не найдены числовые столбцы. Пожалуйста, проверьте данные.")
            st.stop()
        
        value_column = st.selectbox(
            "📈 Выберите переменную для временного ряда",
            numeric_columns,
            index=0,
            help="Выберите числовой столбец, для которого будет выполнен анализ временного ряда"
        )
    
    st.markdown("---")

    # Предпросмотр данных
    st.subheader("👀 Предпросмотр данных")
    
    with st.expander("📋 Просмотр данных и статистика", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**📅 Диапазон дат:**")
            st.write(f"{df[date_column].min()} - {df[date_column].max()}")
        
        with col2:
            st.write(f"**📊 Выбранная переменная:** {value_column}")
        
        st.markdown("---")
        
        # Статистика по всем числовым столбцам
        st.markdown("#### 📈 Статистика по всем числовым столбцам")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        else:
            st.info("Нет числовых столбцов для статистики")

        st.markdown("---")

    # Этап 1: Декомпозиция временного ряда
    st.markdown("---")
    st.subheader("📊 Этап 1: Декомпозиция временного ряда")
    st.caption("Разложение временного ряда на компоненты: тренд, сезонность и остатки")

    stage1_completed = False

    # Инициализируем анализатор
    try:
        analyzer = DecompositionAnalyzer(
            df,
            date_column=date_column,
            value_column=value_column
        )
        
        st.success(f"✅ Анализатор инициализирован успешно!")
        st.info(f"📊 **Анализируемая переменная:** {value_column} | **Столбец с датой:** {date_column}")
        stage1_completed = True
        st.session_state['stage1_completed'] = True
        
        # Выбор режима работы
        st.markdown("---")
        st.subheader("🎯 Выберите режим анализа")
        
        analysis_mode = st.radio(
            "",
            ["Автоматический поиск лучшей декомпозиции", "Ручной выбор параметров"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if analysis_mode == "Автоматический поиск лучшей декомпозиции":
            # Автоматический поиск лучшей декомпозиции
            st.markdown("---")
            st.subheader("🔍 Автоматический поиск лучшей декомпозиции")
            st.caption("Система автоматически найдет оптимальную декомпозицию на основе анализа остатков")

            auto_col1, auto_col2 = st.columns(2)

            with auto_col1:
                model_type = st.radio(
                    "🎯 Тип модели",
                    ["Автоматический выбор", "Аддитивная", "Мультипликативная"],
                    index=0,
                    help="Выберите тип сезонной модели для анализа",
                    key="auto_model_type"
                )

            with auto_col2:
                periods_options = ["7", "30", "365"]
                periods_input = st.text_input(
                    "📊 Периоды сезонности (через запятую)",
                    value=", ".join(periods_options),
                    help="Например: 7, 30, 365",
                    key="auto_periods_input"
                )

                try:
                    periods = [int(p.strip()) for p in periods_input.split(',') if p.strip()]
                    periods = [p for p in periods if p > 1]
                    if not periods:
                        raise ValueError
                except Exception:
                    periods = [7, 30, 365]
                    st.warning("⚠️ Используются периоды по умолчанию: 7, 30, 365")
            
            if st.button("🚀 Начать анализ", type="primary"):
                with st.spinner("⏳ Выполняется анализ всех вариантов декомпозиции... Это может занять некоторое время."):
                    try:
                        # Определяем модели для проверки
                        if model_type == "Автоматический выбор":
                            models = ['additive', 'multiplicative']
                        elif model_type == "Аддитивная":
                            models = ['additive']
                        else:
                            models = ['multiplicative']
                        
                        # Находим лучшую декомпозицию
                        best_result = analyzer.get_best_decomposition(
                            periods=periods,
                            models=models
                        )
                        
                        st.session_state['best_decomposition'] = best_result
                        st.success("✅ Анализ завершён!")
                        
                    except Exception as e:
                        st.error(f"Ошибка при анализе: {e}")
                        st.exception(e)
            
            # Показываем результаты
            if 'best_decomposition' in st.session_state:
                best = st.session_state['best_decomposition']
                
                st.markdown("---")
                st.subheader("🏆 Лучшая декомпозиция")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Модель", best['model'].upper())
                with col2:
                    st.metric("Период", best['period'])
                with col3:
                    st.metric("Оценка качества", f"{best['score']:.2f}")
                
                # Визуализация компонентов
                decomp = best['decomposition']
                visualize_decomposition(decomp, "Лучшая декомпозиция")
                
                # Анализ остатков
                st.markdown("---")
                st.subheader("📈 Анализ остатков")
                display_residual_analysis(best['residual_analysis'], decomp.resid)
                
                # Сравнение всех вариантов
                st.markdown("---")
                st.subheader("📊 Сравнение всех вариантов")
                display_comparison_table(best['all_comparisons'])
        
        else:
            # Ручной выбор параметров
            st.markdown("---")
            st.subheader("⚙️ Ручной выбор параметров")
            st.caption("Выберите параметры декомпозиции вручную")
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_model = st.selectbox(
                    "Тип модели",
                    ["additive", "multiplicative"],
                    index=0,
                    help="Аддитивная или мультипликативная модель"
                )
            
            with col2:
                selected_period = st.number_input(
                    "Период сезонности",
                    min_value=2,
                    max_value=len(analyzer.series) // 2,
                    value=7,
                    step=1,
                    help="Период сезонности для декомпозиции"
                )
            
            if st.button("🔍 Выполнить декомпозицию", type="primary"):
                with st.spinner("⏳ Выполняется декомпозиция..."):
                    try:
                        decomp = analyzer.decompose(
                            model=selected_model,
                            period=selected_period
                        )
                        
                        st.session_state['manual_decomposition'] = decomp
                        st.session_state['manual_model'] = selected_model
                        st.session_state['manual_period'] = selected_period
                        st.success("✅ Декомпозиция выполнена!")
                        
                    except Exception as e:
                        st.error(f"Ошибка при декомпозиции: {e}")
                        st.exception(e)
            
            # Показываем результаты ручной декомпозиции
            if 'manual_decomposition' in st.session_state:
                decomp = st.session_state['manual_decomposition']
                
                st.markdown("---")
                st.subheader("📊 Результаты декомпозиции")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Модель", st.session_state['manual_model'].upper())
                with col2:
                    st.metric("Период", st.session_state['manual_period'])
                
                # Визуализация компонентов
                visualize_decomposition(decomp, "Декомпозиция")
                
                # Анализ остатков
                st.markdown("---")
                st.subheader("📈 Анализ остатков")
                residual_analysis = analyzer.analyze_residuals(decomp.resid)
                display_residual_analysis(residual_analysis, decomp.resid)
        
    except Exception as e:
        st.error(f"Ошибка при инициализации анализатора: {e}")
        st.exception(e)
        st.session_state['stage1_completed'] = False

    # Проверяем состояние первого блока, чтобы управлять последовательностью
    stage1_ready = st.session_state.get('stage1_completed', False) or stage1_completed

    if not stage1_ready:
        st.session_state['stage2_completed'] = False
        st.session_state['stage3_completed'] = False
        st.session_state['stage4_completed'] = False
        st.session_state['stage5_completed'] = False
        st.session_state['stage6_completed'] = False
        st.session_state.pop('stage3_results', None)
        st.session_state.pop('stage4_results', None)
        st.session_state.pop('stage5_results', None)
        st.session_state.pop('stage6_results', None)

    # Этап 2: Расширенный feature engineering
    st.markdown("---")
    st.subheader("🧮 Этап 2: Расширенный feature engineering")
    st.caption(
        "Генерируем временные, лаговые и скользящие признаки для улучшения качества моделей."
    )

    with st.expander("📚 Теория по feature engineering", expanded=False):
        st.markdown(
            """
            **Зачем нужен feature engineering?**

            - Добавляя *временные признаки* (день недели, месяц, квартал и др.), мы позволяем модели учитывать календарные закономерности.
            - *Циклические признаки* на основе `sin`/`cos` сохраняют информацию о повторяющихся паттернах (например, понедельник и воскресенье близки по значению).
            - *Лаги* (`lag_1`, `lag_7`, `lag_30`) дают модели доступ к прошлым значениям ряда.
            - *Скользящие статистики* (среднее, стандартное отклонение, минимум, максимум) отражают локальный контекст и динамику.
            - *Волатильность* (коэффициент вариации) показывает стабильность/изменчивость ряда в выбранном окне.
            - Праздничные и событийные метки помогают учесть внешние факторы, влияющие на поведение временного ряда.
            """
        )

    if not stage1_ready:
        st.info("Сначала завершите предыдущий блок, чтобы продолжить к feature engineering.")
    else:
        default_config = FeatureEngineeringConfig()

        with st.form("feature_engineering_form"):
            include_cyclical = st.checkbox(
                "Добавить циклические признаки (sin/cos)", value=default_config.include_cyclical
            )
            include_volatility = st.checkbox(
                "Добавить признаки волатильности (скользящий коэффициент вариации)",
                value=default_config.include_volatility,
            )
            include_weekend_flag = st.checkbox(
                "Сохранять признак выходного дня",
                value=default_config.include_weekend_flag,
            )
            include_holidays = st.checkbox(
                "Добавить праздничные/событийные метки",
                value=default_config.include_holidays,
            )
            holidays_input = st.text_input(
                "Праздничные даты (YYYY-MM-DD, через запятую)",
                value="",
                help="Например: 2024-01-01, 2024-05-09",
            )
            drop_na = st.checkbox(
                "Удалить строки с пропусками после генерации",
                value=default_config.drop_na,
            )

            feature_submit = st.form_submit_button("Создать признаки")

        if feature_submit:
            holiday_dates: Optional[List[pd.Timestamp]] = None
            invalid_dates: List[str] = []

            if include_holidays and holidays_input.strip():
                holiday_dates = []
                for item in holidays_input.split(','):
                    candidate = item.strip()
                    if not candidate:
                        continue
                    try:
                        holiday_dates.append(pd.to_datetime(candidate))
                    except Exception:
                        invalid_dates.append(candidate)

            config = FeatureEngineeringConfig(
                include_cyclical=include_cyclical,
                include_volatility=include_volatility,
                include_weekend_flag=include_weekend_flag,
                include_holidays=include_holidays,
                holiday_dates=holiday_dates,
                drop_na=drop_na,
            )

            try:
                engineer = FeatureEngineer(
                    df=df,
                    date_column=date_column,
                    value_column=value_column,
                )
                feature_result = engineer.generate_features(config=config)

                features_df = feature_result.features
                st.session_state['feature_engineering_result'] = features_df
                st.session_state['feature_engineering_columns'] = feature_result.generated_columns
                st.session_state['stage2_completed'] = True

                rows_before = len(df)
                rows_after = len(features_df)
                created_features = len(feature_result.generated_columns)

                st.success(
                    f"Создано {created_features} дополнительных признаков. "
                    f"Размер данных: {rows_after} строк (до генерации: {rows_before})."
                )

                if invalid_dates:
                    st.warning(
                        "Некоторые даты не удалось распознать и они пропущены: "
                        + ", ".join(invalid_dates)
                    )

                st.dataframe(features_df.head(20), use_container_width=True)

                csv_data = features_df.to_csv(index=False).encode('utf-8')
                file_name = f"features_{value_column}.csv"
                st.download_button(
                    label="📥 Скачать сгенерированные признаки",
                    data=csv_data,
                    file_name=file_name,
                    mime="text/csv",
                )

                with st.expander("🔎 Перечень сгенерированных признаков", expanded=False):
                    st.write(feature_result.generated_columns)

            except Exception as feature_error:
                st.error(f"Не удалось сгенерировать признаки: {feature_error}")
                st.exception(feature_error)
                st.session_state['stage2_completed'] = False

    if st.session_state.get('feature_engineering_result') is not None and stage1_ready:
        st.info("Признаки уже сгенерированы — вы можете пересоздать их, изменив настройки выше.")

    stage2_ready = st.session_state.get('stage2_completed', False)

    if not stage2_ready:
        st.session_state['stage3_completed'] = False
        st.session_state['stage4_completed'] = False
        st.session_state['stage5_completed'] = False
        st.session_state['stage6_completed'] = False
        st.session_state.pop('stage3_results', None)
        st.session_state.pop('stage4_results', None)
        st.session_state.pop('stage5_results', None)
        st.session_state.pop('stage6_results', None)

    # Этап 3: Стратегии многошагового прогнозирования
    st.markdown("---")
    st.subheader("🔮 Этап 3: Стратегии многошагового прогнозирования")
    st.caption("Сравниваем рекурсивную, прямую и гибридную стратегии для горизонта h ≥ 7.")

    with st.expander("📚 Теория по стратегиям", expanded=False):
        st.markdown(
            """
            **Подходы к многошаговому прогнозу:**

            - **Рекурсивная стратегия**: обучается одна модель для шага `t+1`, затем её прогнозы
              последовательно подаются на вход, что может приводить к накоплению ошибок, но требует минимальных ресурсов.
            - **Прямая стратегия**: обучается отдельная модель для каждого шага `t+h`. Такой подход устойчивее к ошибкам,
              но требует больше вычислений (h моделей).
            - **Гибридная стратегия**: комбинирует подходы, используя рекурсию для ближайших шагов и прямые модели
              для дальних горизонтв, снижая суммарную ошибку.

            **Метрики сравнения:**
            - `MAE` и `RMSE` на каждом шаге горизонта.
            - Время обучения/прогноза каждой стратегии.
            - Накопление ошибки (сумма абсолютных ошибок по шагам).
            """
        )

    if not stage2_ready:
        st.info("Сначала завершите предыдущий блок, чтобы перейти к стратегиям прогнозирования.")
    else:
        max_possible_lag = max(3, min(120, len(df) - 1))
        available_horizons = [h for h in (7, 30, 90) if h < len(df)]
        if not available_horizons:
            available_horizons = [max(2, len(df) // 4)]
            st.warning("Недостаточно данных для стандартных горизонтов (7/30/90). Используется ближайшее возможное значение.")
        horizon = int(st.selectbox("Горизонт прогноза (h)", available_horizons, index=0, key="stage3_horizon"))
        lag_upper_bound = max(3, min(max_possible_lag, len(df) - horizon - 1))
        max_lag_stage3 = int(
            st.slider(
                "Максимальный лаг для обучения",
                min_value=3,
                max_value=lag_upper_bound,
                value=min(30, lag_upper_bound),
            )
        )
        base_model_stage3 = st.selectbox(
            "Базовая модель",
            ["LinearRegression", "RandomForestRegressor"],
            index=0,
            help="Используется как базовый регрессор внутри стратегий",
        )
        hybrid_steps = int(
            st.slider(
                "Количество рекурсивных шагов для гибридной стратегии",
                min_value=1,
                max_value=max(1, horizon - 1),
                value=min(3, horizon - 1),
            )
        )

        run_stage3 = st.button("🔮 Запустить прогноз", type="primary")

        if run_stage3:
            try:
                strategies_runner = ForecastingStrategies(df, date_column, value_column)
                stage3_results, actual_values, stage3_benchmark = strategies_runner.evaluate(
                    horizon=horizon,
                    max_lag=max_lag_stage3,
                    model_name=base_model_stage3,
                    hybrid_recursive_steps=hybrid_steps,
                )

                st.session_state['stage3_results'] = {
                    'results': stage3_results,
                    'actual': actual_values,
                    'benchmark': stage3_benchmark,
                    'horizon': horizon,
                    'config': {
                        'max_lag': max_lag_stage3,
                        'model': base_model_stage3,
                        'hybrid_steps': hybrid_steps,
                    },
                }
                st.success("Прогноз успешно выполнен!")
            except Exception as forecast_error:
                st.error(f"Не удалось выполнить прогноз: {forecast_error}")
                st.exception(forecast_error)
                st.session_state['stage3_completed'] = False

    if stage2_ready and st.session_state.get('stage3_results'):
        stage3_state = st.session_state['stage3_results']
        results_dict = stage3_state['results']
        actual_values = stage3_state['actual']
        stage3_benchmark = stage3_state.get('benchmark')
        horizon_steps = np.arange(1, len(actual_values) + 1)

        st.session_state['stage3_completed'] = True

        st.markdown("---")
        st.subheader("📊 Результаты сравнения стратегий")

        metrics_frames = []
        runtime_rows = []

        for key, res in results_dict.items():
            strategy_name = res.name
            metrics_frames.append(
                pd.DataFrame(
                    {
                        'Стратегия': strategy_name,
                        'Шаг': horizon_steps,
                        'MAE': res.mae_per_step,
                        'RMSE': res.rmse_per_step,
                        'MAPE': res.mape_per_step,
                        'Накопленная ошибка (MAE)': res.cumulative_mae,
                    }
                )
            )

            runtime_rows.append(
                {
                    'Стратегия': strategy_name,
                    'Время (сек.)': res.runtime_seconds,
                    'Средний MAE': float(np.mean(res.mae_per_step)),
                    'Средний RMSE': float(np.mean(res.rmse_per_step)),
                    'Средний MAPE': res.test_mape,
                }
            )

        metrics_df = pd.concat(metrics_frames, ignore_index=True)
        st.dataframe(metrics_df.round(4), use_container_width=True)

        runtime_df = pd.DataFrame(runtime_rows)
        st.dataframe(runtime_df.round(4), use_container_width=True)

        metrics_csv = runtime_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Скачать метрики стратегий",
            data=metrics_csv,
            file_name="strategies_metrics.csv",
            mime="text/csv",
        )

        # Графики
        fig_forecast = go.Figure()
        fig_forecast.add_trace(
            go.Scatter(
                x=horizon_steps,
                y=actual_values,
                mode='lines+markers',
                name='Фактические значения',
            )
        )

        for res in results_dict.values():
            fig_forecast.add_trace(
                go.Scatter(
                    x=horizon_steps,
                    y=res.predictions,
                    mode='lines+markers',
                    name=res.name,
                )
            )

        fig_forecast.update_layout(
            title='Фактические значения и прогнозы',
            xaxis_title='Шаг горизонта',
            yaxis_title=value_column,
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        fig_cumulative = go.Figure()
        for res in results_dict.values():
            fig_cumulative.add_trace(
                go.Scatter(
                    x=horizon_steps,
                    y=res.cumulative_mae,
                    mode='lines+markers',
                    name=res.name,
                )
            )

        fig_cumulative.update_layout(
            title='Накопленная абсолютная ошибка (MAE)',
            xaxis_title='Шаг горизонта',
            yaxis_title='Сумма абсолютных ошибок',
        )
        st.plotly_chart(fig_cumulative, use_container_width=True)

        forecasts_export = pd.DataFrame({'Шаг': horizon_steps, 'Фактическое': actual_values})
        for res in results_dict.values():
            forecasts_export[res.name] = res.predictions
        if stage3_benchmark is not None:
            forecasts_export[stage3_benchmark.name] = stage3_benchmark.forecast

        forecasts_csv = forecasts_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Скачать прогнозы стратегий",
            data=forecasts_csv,
            file_name="strategies_forecasts.csv",
            mime="text/csv",
        )

    stage3_ready = st.session_state.get('stage3_completed', False)

    if not stage3_ready:
        st.session_state['stage4_completed'] = False
        st.session_state['stage5_completed'] = False
        st.session_state['stage6_completed'] = False
        st.session_state.pop('stage4_results', None)
        st.session_state.pop('stage5_results', None)
        st.session_state.pop('stage6_results', None)

    # Этап 4: Кросс-валидация временных рядов
    st.markdown("---")
    st.subheader("🧪 Этап 4: Кросс-валидация для временных рядов")
    st.caption("Оцениваем стабильность качества моделей без утечки будущего.")

    with st.expander("📚 Теория по кросс-валидации", expanded=False):
        st.markdown(
            """
            **Схемы кросс-валидации:**

            - **Скользящее окно**: используем фиксированное обучающее окно и последовательно сдвигаем его вперёд.
            - **Расширяющееся окно**: обучающее окно растёт со временем, добавляя новые наблюдения.
            - **TimeSeriesSplit (sklearn)**: реализует последовательное разделение на обучающую и тестовую выборки без перемешивания.

            Для каждой схемы мы оцениваем метрики (MAE, RMSE), суммарное время, а также анализируем динамику ошибок по фолдам
            — это помогает понять стабильность модели и чувствительность к разным временным интервалам.
            """
        )

    total_points = len(df)

    if not stage3_ready:
        st.info("Сначала завершите предыдущий блок, чтобы перейти к кросс-валидации.")
    elif total_points < 30:
        st.warning("Для кросс-валидации требуется минимум 30 наблюдений. Загрузите более длинный ряд.")
    else:
        cv_cols = st.columns(2)

        with cv_cols[0]:
            cv_max_lag = int(
                st.slider(
                    "Максимальный лаг",
                    min_value=3,
                    max_value=min(120, max(4, total_points - 5)),
                    value=min(14, max(4, total_points // 6)),
                )
            )

            sliding_train_window = int(
                st.number_input(
                    "Обучающее окно (скользящее)",
                    min_value=cv_max_lag + 1,
                    max_value=max(cv_max_lag + 2, total_points - 2),
                    value=min(max(cv_max_lag + 1, total_points // 2), total_points - 2),
                    step=1,
                )
            )

            sliding_test_window = int(
                st.number_input(
                    "Тестовое окно (скользящее)",
                    min_value=1,
                    max_value=max(1, total_points - sliding_train_window - 1),
                    value=min(14, max(1, (total_points - sliding_train_window) // 3)),
                    step=1,
                )
            )

            base_model_cv = st.selectbox(
                "Базовая модель",
                ["LinearRegression", "RandomForestRegressor"],
                index=0,
                key="cv_base_model",
            )

        with cv_cols[1]:
            expanding_initial_window = int(
                st.number_input(
                    "Начальное обучающее окно (расширяющееся)",
                    min_value=cv_max_lag + 1,
                    max_value=max(cv_max_lag + 2, total_points - 2),
                    value=min(max(cv_max_lag + 1, total_points // 3), total_points - 2),
                    step=1,
                )
            )

            expanding_test_window = int(
                st.number_input(
                    "Тестовое окно (расширяющееся)",
                    min_value=1,
                    max_value=max(1, total_points - expanding_initial_window - 1),
                    value=min(14, max(1, (total_points - expanding_initial_window) // 3)),
                    step=1,
                )
            )

            tss_splits = int(
                st.slider(
                    "Число фолдов TimeSeriesSplit",
                    min_value=2,
                    max_value=min(10, max(2, total_points // 5)),
                    value=min(5, max(2, total_points // 10)),
                )
            )

        run_cv = st.button("🧪 Запустить кросс-валидацию", type="primary")

        if run_cv:
            try:
                cv_runner = TimeSeriesCrossValidator(df, date_column, value_column)
                cv_summaries = cv_runner.evaluate(
                    max_lag=cv_max_lag,
                    model_name=base_model_cv,
                    sliding_train_window=sliding_train_window,
                    sliding_test_window=sliding_test_window,
                    expanding_initial_window=expanding_initial_window,
                    expanding_test_window=expanding_test_window,
                    tss_splits=tss_splits,
                )

                st.session_state['stage4_results'] = {
                    'summaries': cv_summaries,
                    'config': {
                        'max_lag': cv_max_lag,
                        'base_model': base_model_cv,
                        'sliding_train': sliding_train_window,
                        'sliding_test': sliding_test_window,
                        'expanding_initial': expanding_initial_window,
                        'expanding_test': expanding_test_window,
                        'tss_splits': tss_splits,
                    },
                }
                st.session_state['stage4_completed'] = True
                st.success("Кросс-валидация успешно выполнена!")
            except Exception as cv_error:
                st.error(f"Не удалось выполнить кросс-валидацию: {cv_error}")
                st.exception(cv_error)
                st.session_state['stage4_completed'] = False

        if st.session_state.get('stage4_results'):
            cv_state = st.session_state['stage4_results']
            summaries: Dict[str, CrossValidationSummary] = cv_state['summaries']

            all_folds_frames = []
            overview_rows = []

            for scheme, summary in summaries.items():
                folds_df = summary.to_dataframe()
                folds_df["MAE"] = folds_df["MAE"].astype(float)
                folds_df["RMSE"] = folds_df["RMSE"].astype(float)
                folds_df["Время (сек.)"] = folds_df["Время (сек.)"].astype(float)
                all_folds_frames.append(folds_df)

                mae_values = [fold.mae for fold in summary.fold_results]
                rmse_values = [fold.rmse for fold in summary.fold_results]

                overview_rows.append(
                    {
                        "Схема": scheme,
                        "Средний MAE": summary.mean_mae,
                        "Std(MAE)": float(np.std(mae_values, ddof=1)) if len(mae_values) > 1 else 0.0,
                        "Средний RMSE": summary.mean_rmse,
                        "Std(RMSE)": float(np.std(rmse_values, ddof=1)) if len(rmse_values) > 1 else 0.0,
                        "Суммарное время (сек.)": summary.runtime_seconds,
                    }
                )

            folds_summary_df = pd.concat(all_folds_frames, ignore_index=True)
            st.markdown("#### Результаты по фолдам")
            st.dataframe(folds_summary_df.round(4), use_container_width=True)

            overview_df = pd.DataFrame(overview_rows)
            st.markdown("#### Средние метрики по стратегиям")
            st.dataframe(overview_df.round(4), use_container_width=True)

            # Визуализация динамики ошибок по фолдам
            mae_plot_df = folds_summary_df[["Схема", "Фолд", "MAE"]]
            fig_mae = go.Figure()
            for scheme in mae_plot_df["Схема"].unique():
                scheme_data = mae_plot_df[mae_plot_df["Схема"] == scheme]
                fig_mae.add_trace(
                    go.Scatter(
                        x=scheme_data["Фолд"],
                        y=scheme_data["MAE"],
                        mode='lines+markers',
                        name=scheme,
                    )
                )
            fig_mae.update_layout(
                title="MAE по фолдам",
                xaxis_title="Фолд",
                yaxis_title="MAE",
            )
            st.plotly_chart(fig_mae, use_container_width=True)

            rmse_plot_df = folds_summary_df[["Схема", "Фолд", "RMSE"]]
            fig_rmse = go.Figure()
            for scheme in rmse_plot_df["Схема"].unique():
                scheme_data = rmse_plot_df[rmse_plot_df["Схема"] == scheme]
                fig_rmse.add_trace(
                    go.Scatter(
                        x=scheme_data["Фолд"],
                        y=scheme_data["RMSE"],
                        mode='lines+markers',
                        name=scheme,
                    )
                )
            fig_rmse.update_layout(
                title="RMSE по фолдам",
                xaxis_title="Фолд",
                yaxis_title="RMSE",
            )
            st.plotly_chart(fig_rmse, use_container_width=True)

    stage4_ready = st.session_state.get('stage4_completed', False)

    if not stage4_ready:
        st.session_state['stage5_completed'] = False
        st.session_state['stage6_completed'] = False
        st.session_state.pop('stage5_results', None)
        st.session_state.pop('stage6_results', None)

    # Этап 5: Приведение к стационарности и преобразования
    st.markdown("---")
    st.subheader("🔄 Этап 5: Приведение к стационарности")
    st.caption("Стабилизируем дисперсию, убираем тренд/сезонность и подбираем оптимальные преобразования.")

    with st.expander("📚 Теория по преобразованиям", expanded=False):
        st.markdown(
            """
            **Инструменты стационирования:**

            - **Логарифмирование** и **Box-Cox** стабилизируют дисперсию, приближая распределение к нормальному.
            - **Дифференцирование** первого порядка устраняет тренд, а сезонное дифференцирование — сезонные колебания.
            - Комбинируя преобразования, важно контролировать стационарность тестами **ADF** (p-value < 0.05) и **KPSS** (p-value > 0.05).
            - Для оценки моделей в исходных единицах необходимо уметь выполнять **обратное преобразование**.
            """
        )

    if not stage4_ready:
        st.info("Сначала завершите предыдущий блок, чтобы перейти к преобразованиям стационарности.")
    else:
        stage5_cols = st.columns(3)
        with stage5_cols[0]:
            seasonal_period = int(
                st.number_input(
                    "Сезонный период",
                    min_value=2,
                    max_value= max(2, len(df) // 2),
                    value=7,
                )
            )
        with stage5_cols[1]:
            use_log = st.checkbox("Включить лог-трансформацию", value=True)
        with stage5_cols[2]:
            use_boxcox = st.checkbox("Включить Box-Cox", value=True)
            manual_lambda = None
            if use_boxcox:
                use_manual_lambda = st.checkbox("Указать λ вручную", value=False)
                if use_manual_lambda:
                    manual_lambda = st.slider("λ для Box-Cox", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
            else:
                manual_lambda = None
 
        run_stage5 = st.button("🔄 Подобрать цепочку преобразований", type="primary")

        if run_stage5:
            try:
                transformer = StationarityTransformer(df, date_column, value_column)
                pipelines, best_pipeline = transformer.evaluate_pipelines(
                    seasonal_period=seasonal_period,
                    use_boxcox=use_boxcox,
                    use_log=use_log,
                    manual_boxcox_lambda=manual_lambda,
                )

                def serialize_step(step: TransformationStep) -> Dict[str, object]:
                    return {
                        'name': step.name,
                        'params': step.params,
                    }

                pipelines_payload = []
                for res in pipelines:
                    pipelines_payload.append(
                        {
                            'name': res.name,
                            'steps': [serialize_step(step) for step in res.steps],
                            'adf_stat': res.adf_stat,
                            'adf_pvalue': res.adf_pvalue,
                            'kpss_stat': res.kpss_stat,
                            'kpss_pvalue': res.kpss_pvalue,
                            'adf_stationary': res.adf_stationary,
                            'kpss_stationary': res.kpss_stationary,
                            'score': res.score,
                        }
                    )

                st.session_state['stage5_results'] = {
                    'pipelines': pipelines_payload,
                    'best_name': best_pipeline.name,
                    'best_steps': [serialize_step(step) for step in best_pipeline.steps],
                    'seasonal_period': seasonal_period,
                    'use_log': use_log,
                    'use_boxcox': use_boxcox,
                    'manual_lambda': manual_lambda,
                }
                st.session_state['stage5_completed'] = True
                st.success("Преобразования успешно рассчитаны!")
            except Exception as stage5_error:
                st.error(f"Не удалось подобрать преобразования: {stage5_error}")
                st.exception(stage5_error)
                st.session_state['stage5_completed'] = False

        if stage4_ready and st.session_state.get('stage5_results'):
            stage5_state = st.session_state['stage5_results']
            pipelines_info = stage5_state['pipelines']
            st.session_state['stage5_completed'] = True

            def describe_steps(steps: List[Dict[str, object]]) -> str:
                parts = []
                for step in steps:
                    name = step['name']
                    params = step['params']
                    if name == 'log':
                        parts.append("log")
                    elif name == 'boxcox':
                        lam = params.get('lambda', None)
                        parts.append(f"boxcox(λ={lam:.3f})" if lam is not None else "boxcox")
                    elif name == 'diff':
                        lag = params.get('lag', 1)
                        parts.append(f"diff(lag={lag})")
                    else:
                        parts.append(name)
                return " -> ".join(parts) if parts else "Нет"

            summary_rows = []
            for item in pipelines_info:
                summary_rows.append(
                    {
                        "Цепочка": item['name'],
                        "Преобразования": describe_steps(item['steps']),
                        "ADF p-value": item['adf_pvalue'],
                        "KPSS p-value": item['kpss_pvalue'],
                        "ADF стац.": item['adf_stationary'],
                        "KPSS стац.": item['kpss_stationary'],
                        "Score": item['score'],
                    }
                )

            pipelines_df = pd.DataFrame(summary_rows)
            st.markdown("#### Результаты преобразований")
            st.dataframe(pipelines_df.round(4), use_container_width=True)

            pipelines_csv = pipelines_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать результаты преобразований",
                data=pipelines_csv,
                file_name="transformations_summary.csv",
                mime="text/csv",
            )

            best_name = stage5_state['best_name']
            st.success(f"Рекомендуемая цепочка: **{best_name}**")

            pipeline_names = [item['name'] for item in pipelines_info]
            default_index = pipeline_names.index(best_name) if best_name in pipeline_names else 0
            selected_pipeline_name = st.selectbox(
                "Выберите цепочку для просмотра",
                pipeline_names,
                index=default_index,
            )

            selected_item = next(item for item in pipelines_info if item['name'] == selected_pipeline_name)
            selected_steps = [TransformationStep(step['name'], step['params']) for step in selected_item['steps']]

            transformer_view = StationarityTransformer(df, date_column, value_column)
            transformed_series = transformer_view.apply_steps(selected_steps)
            aligned_dates = transformer_view.dates.iloc[transformed_series.index]

            fig_transformed = go.Figure()
            fig_transformed.add_trace(
                go.Scatter(
                    x=aligned_dates,
                    y=transformed_series.values,
                    mode='lines',
                    name='Преобразованный ряд',
                )
            )
            fig_transformed.update_layout(
                title=f"Ряд после преобразований: {selected_pipeline_name}",
                xaxis_title="Дата",
                yaxis_title=value_column,
            )
            st.plotly_chart(fig_transformed, use_container_width=True)

            preview_df = pd.DataFrame(
                {
                    "date": aligned_dates,
                    f"{value_column}_transformed": transformed_series.values,
                }
            ).set_index("date")
            st.dataframe(preview_df.tail(20).round(4))

            csv_transformed = preview_df.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Скачать преобразованный ряд",
                data=csv_transformed,
                file_name=f"stationary_{selected_pipeline_name}.csv",
                mime="text/csv",
            )

            st.info(
                "Для обратного преобразования используйте метод `inverse_transform` из модуля `stationarity_transformations`."
            )

    # Этап 6: Модели экспоненциального сглаживания
    st.markdown("---")
    st.subheader("📉 Этап 6: Модели экспоненциального сглаживания")
    st.caption("Сравниваем SES и модели Хольта, оцениваем прогнозы и остатки.")

    with st.expander("📚 Теория по EXP-сглаживанию", expanded=False):
        st.markdown(
            """
            - **SES (Simple Exponential Smoothing)** подходит для рядов без тренда и сезонности.
            - **Модель Хольта (аддитивная)** учитывает линейный тренд; **мультипликативная** форма требует положительных данных
              и хорошо работает, когда тренд растёт/падает пропорционально уровню.
            - Для диагностики проверяем остатки: отсутствие автокорреляции (тест Льюнга–Бокса), нормальность (Shapiro–Wilk, Q-Q plot)
              и гомоскедастичность (график остатки vs прогнозы).
            """
        )

    stage5_ready = st.session_state.get('stage5_completed', False)

    if not stage5_ready:
        st.info("Сначала завершите предыдущий блок, чтобы перейти к моделям экспоненциального сглаживания.")
    else:
        total_points = len(df)
        if total_points < 20:
            st.warning("Для экспоненциального сглаживания требуется минимум 20 наблюдений.")
        else:
            stage6_cols = st.columns(3)

            with stage6_cols[0]:
                available_horizons_stage6 = [h for h in (7, 30, 90) if h < total_points]
                if not available_horizons_stage6:
                    available_horizons_stage6 = [max(2, total_points // 4)]
                    st.warning("Недостаточно данных для стандартных горизонтов (7/30/90). Используется ближайшее возможное значение.")
                stage6_horizon = int(st.selectbox(
                    "Горизонт прогноза (h)",
                    available_horizons_stage6,
                    index=0,
                    key="stage6_horizon",
                ))

            with stage6_cols[1]:
                include_multiplicative = st.checkbox(
                    "Включить мультипликативную модель Хольта",
                    value=(df[value_column] > 0).all(),
                    help="Данные должны быть > 0 для мультипликативного тренда",
                )

            with stage6_cols[2]:
                seasonal_period_stage6 = st.number_input(
                    "Сезонный период (опционально)",
                    min_value=0,
                    max_value=max(0, total_points // 2),
                    value=0,
                    help="0 = без сезонности"
                )
                seasonal_period_stage6 = int(seasonal_period_stage6) if seasonal_period_stage6 > 1 else None

            run_stage6 = st.button("📉 Запустить экспоненциальное сглаживание", type="primary")

            if run_stage6:
                try:
                    runner = ExponentialSmoothingRunner(df, date_column, value_column)
                    model_results, benchmark, failed_models = runner.evaluate(
                        horizon=stage6_horizon,
                        seasonal_period=seasonal_period_stage6,
                        include_multiplicative=include_multiplicative,
                    )

                    actual_series = runner.series.iloc[-stage6_horizon:]
                    actual_series.index = runner.dates.iloc[-stage6_horizon:]

                    st.session_state['stage6_results'] = {
                        'models': model_results,
                        'benchmark': benchmark,
                        'failures': failed_models,
                        'actual': actual_series,
                        'config': {
                            'horizon': stage6_horizon,
                            'seasonal_period': seasonal_period_stage6,
                            'include_multiplicative': include_multiplicative,
                        },
                    }
                    st.session_state['stage6_completed'] = True
                    st.success("Модели экспоненциального сглаживания успешно обучены!")
                except Exception as stage6_error:
                    st.error(f"Не удалось обучить модели: {stage6_error}")
                    st.exception(stage6_error)
                    st.session_state['stage6_completed'] = False

            if st.session_state.get('stage6_results'):
                stage6_state = st.session_state['stage6_results']
                model_results: List[ModelResult] = stage6_state['models']
                benchmark: BenchmarkResult = stage6_state['benchmark']
                failed_models: List[str] = stage6_state.get('failures', [])
                config_stage6 = stage6_state['config']
                cfg_horizon = int(config_stage6['horizon'])

                if failed_models:
                    for msg in failed_models:
                        st.warning(f"Модель пропущена: {msg}")

                runner_display = ExponentialSmoothingRunner(df, date_column, value_column)
                full_series = runner_display.series
                full_dates = runner_display.dates

                actual_series = full_series.iloc[-cfg_horizon:]
                actual_series.index = full_dates.iloc[-cfg_horizon:]
                st.session_state['stage6_completed'] = True

                summary_rows = [
                    {
                        "Модель": benchmark.name,
                        "MAE": benchmark.test_mae,
                        "RMSE": benchmark.test_rmse,
                        "MAPE": benchmark.test_mape,
                        "Время (сек.)": np.nan,
                    }
                ]

                for res in model_results:
                    summary_rows.append(
                        {
                            "Модель": res.name,
                            "MAE": res.test_mae,
                            "RMSE": res.test_rmse,
                            "MAPE": res.test_mape,
                            "Время (сек.)": res.runtime_seconds,
                        }
                    )

                summary_df = pd.DataFrame(summary_rows)
                
                st.markdown("#### Сравнение моделей")
                # Форматируем числовые столбцы (используем научную нотацию для очень малых чисел)
                summary_df_display = summary_df.copy()
                
                # Функция для форматирования чисел
                def format_metric(val):
                    if pd.isna(val):
                        return "None"
                    if abs(val) < 0.0001 and val != 0:
                        return f"{val:.4e}"  # Научная нотация
                    elif abs(val) < 1:
                        return f"{val:.6f}"  # 6 знаков для маленьких чисел
                    else:
                        return f"{val:.4f}"  # 4 знака для больших
                
                for col in ['MAE', 'RMSE', 'MAPE']:
                    if col in summary_df_display.columns:
                        summary_df_display[col] = summary_df_display[col].apply(format_metric)
                
                if 'Время (сек.)' in summary_df_display.columns:
                    summary_df_display['Время (сек.)'] = summary_df_display['Время (сек.)'].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) else "None"
                    )
                
                st.dataframe(summary_df_display, use_container_width=True)

                summary_csv = summary_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать метрики моделей",
                    data=summary_csv,
                    file_name="exp_smoothing_metrics.csv",
                    mime="text/csv",
                )

                model_names = [benchmark.name] + [res.name for res in model_results]
                selected_model_name = st.selectbox(
                    "Выберите модель для деталей",
                    model_names,
                )

                selected_result = None
                if selected_model_name == benchmark.name:
                    forecast_series = benchmark.forecast
                    if not isinstance(forecast_series, pd.Series):
                        forecast_series = pd.Series(forecast_series, index=actual_series.index)
                    lower_ci = upper_ci = None
                    diagnostics = None
                else:
                    selected_result = next(res for res in model_results if res.name == selected_model_name)
                    forecast_series = selected_result.forecast
                    lower_ci = selected_result.lower_ci
                    upper_ci = selected_result.upper_ci
                    diagnostics = selected_result.diagnostics

                history_available = len(full_series) - cfg_horizon
                tail_length = min(history_available, cfg_horizon * 3)
                history_start = max(0, history_available - tail_length)
                tail_history = full_series.iloc[history_start:history_available]
                tail_history.index = full_dates.iloc[history_start:history_start + len(tail_history)]

                fig_forecast_stage6 = go.Figure()
                fig_forecast_stage6.add_trace(
                    go.Scatter(
                        x=tail_history.index,
                        y=tail_history.values,
                        mode='lines',
                        name='История',
                    )
                )
                fig_forecast_stage6.add_trace(
                    go.Scatter(
                        x=actual_series.index,
                        y=actual_series.values,
                        mode='lines+markers',
                        name='Фактические значения',
                    )
                )
                fig_forecast_stage6.add_trace(
                    go.Scatter(
                        x=forecast_series.index,
                        y=forecast_series.values,
                        mode='lines+markers',
                        name=f'Прогноз ({selected_model_name})',
                    )
                )

                if lower_ci is not None and upper_ci is not None:
                    fig_forecast_stage6.add_trace(
                        go.Scatter(
                            x=forecast_series.index.tolist() + forecast_series.index[::-1].tolist(),
                            y=upper_ci.values.tolist() + lower_ci.values[::-1].tolist(),
                            fill='toself',
                            fillcolor='rgba(31, 119, 180, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo='skip',
                            showlegend=True,
                            name='Доверительный интервал',
                        )
                    )

                fig_forecast_stage6.update_layout(
                    title=f"Прогноз модели {selected_model_name}",
                    xaxis_title="Дата",
                    yaxis_title=value_column,
                )
                st.plotly_chart(fig_forecast_stage6, use_container_width=True)

                # Диагностика остатков (только для моделей экспоненциального сглаживания)
                if diagnostics is not None:
                    resid = diagnostics.residuals
                    fitted = diagnostics.fitted_values

                    st.markdown("#### Диагностика остатков")
                    diag_cols = st.columns(2)

                    with diag_cols[0]:
                        fig_resid_scatter = go.Figure()
                        fig_resid_scatter.add_trace(
                            go.Scatter(
                                x=fitted.values,
                                y=resid.values,
                                mode='markers',
                                name='Остатки',
                            )
                        )
                        fig_resid_scatter.add_hline(y=0, line=dict(color='red', dash='dash'))
                        fig_resid_scatter.update_layout(
                            title='Остатки vs прогнозы',
                            xaxis_title='Прогноз',
                            yaxis_title='Остаток',
                        )
                        st.plotly_chart(fig_resid_scatter, use_container_width=True)

                    with diag_cols[1]:
                        fig_qq = go.Figure()
                        fig_qq.add_trace(
                            go.Scatter(
                                x=diagnostics.qq_theoretical,
                                y=diagnostics.qq_sample,
                                mode='markers',
                                name='Q-Q точки',
                            )
                        )
                        slope, intercept, _ = stats.probplot(resid, dist="norm")[1]
                        fig_qq.add_trace(
                            go.Scatter(
                                x=diagnostics.qq_theoretical,
                                y=slope * diagnostics.qq_theoretical + intercept,
                                mode='lines',
                                name='Идеальная линия',
                                line=dict(color='red'),
                            )
                        )
                        fig_qq.update_layout(
                            title='Q-Q Plot остатков',
                            xaxis_title='Теоретические квантили',
                            yaxis_title='Выборочные квантили',
                        )
                        st.plotly_chart(fig_qq, use_container_width=True)

                    st.write(
                        f"**Ljung-Box p-value:** {diagnostics.ljung_box_pvalue if diagnostics.ljung_box_pvalue is not None else 'н/д'}, "
                        f"**Shapiro-Wilk p-value:** {diagnostics.shapiro_pvalue if diagnostics.shapiro_pvalue is not None else 'н/д'}"
                    )

                # Подготовка данных для экспорта (для всех моделей)
                lower_ci_values = lower_ci.values if lower_ci is not None else np.full(len(forecast_series), np.nan)
                upper_ci_values = upper_ci.values if upper_ci is not None else np.full(len(forecast_series), np.nan)

                # Скачать параметры модели (только для не-benchmark моделей)
                if selected_model_name != benchmark.name:
                    params_json = json.dumps(selected_result.params, indent=2).encode('utf-8')
                    st.download_button(
                        label="📥 Скачать параметры модели",
                        data=params_json,
                        file_name=f"params_{selected_model_name}.json",
                        mime="application/json",
                    )

                # Скачать прогноз (для всех моделей)
                csv_forecast = pd.DataFrame(
                    {
                        "date": forecast_series.index,
                        "forecast": forecast_series.values,
                        "lower_ci": lower_ci_values,
                        "upper_ci": upper_ci_values,
                    }
                ).to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="📥 Скачать прогноз",
                    data=csv_forecast,
                    file_name=f"forecast_{selected_model_name}.csv",
                    mime="text/csv",
                )

                if not model_results:
                    st.warning("Не удалось обучить модели экспоненциального сглаживания; доступен только наивный прогноз.")

    # Этап 7: Обоснование выбора лучшей модели
    st.markdown("---")
    st.subheader("🎯 Этап 7: Обоснование выбора лучшей модели")
    st.caption("Комплексный анализ всех моделей и стратегий для выбора оптимального решения")

    with st.expander("📚 Теория по выбору модели", expanded=False):
        st.markdown(
            """
            **Критерии выбора модели:**

            - **Точность прогноза**: Оценка по метрикам MAE, RMSE, MAPE на тестовой выборке
            - **Горизонт прогноза**: Различные модели могут лучше работать на коротких или длинных горизонтах
            - **Стабильность**: Оценка через кросс-валидацию - модель должна показывать стабильные результаты
            - **Преобразования**: Влияние стационаризации (Box-Cox, логарифмирование) на качество прогноза
            - **Вычислительная сложность**: Время обучения и прогнозирования
            - **Интерпретируемость**: Насколько легко объяснить результаты модели

            **Подход к выбору:**
            1. Сравнить метрики по всем моделям
            2. Проанализировать поведение на разных горизонтах
            3. Проверить согласованность с кросс-валидацией
            4. Учесть влияние преобразований данных
            5. Выбрать модель с лучшим балансом точности и стабильности
            """
        )

    stage6_ready = st.session_state.get('stage6_completed', False)

    if not stage6_ready:
        st.info("Сначала завершите все предыдущие блоки, чтобы получить сводный анализ.")
    else:
        # Проверяем наличие данных из всех блоков
        has_stage3 = st.session_state.get('stage3_completed', False)
        has_stage4 = st.session_state.get('stage4_completed', False)
        has_stage5 = st.session_state.get('stage5_completed', False)
        has_stage6 = st.session_state.get('stage6_completed', False)

        if not all([has_stage3, has_stage4, has_stage5, has_stage6]):
            st.warning("Для полного анализа необходимо завершить все блоки (3-6). Некоторые разделы могут быть недоступны.")

        st.markdown("---")
        st.subheader("📊 Сводная таблица метрик всех моделей")

        # Собираем данные из всех блоков
        all_models_data = []

        # Данные из блока "Стратегии прогнозирования"
        if has_stage3 and st.session_state.get('stage3_results'):
            stage3_state = st.session_state['stage3_results']
            results_dict = stage3_state['results']
            horizon_used = stage3_state['horizon']
            
            for key, res in results_dict.items():
                all_models_data.append({
                    'Этап': 'Стратегии',
                    'Модель': res.name,
                    'Горизонт': horizon_used,
                    'MAE': float(np.mean(res.mae_per_step)),
                    'RMSE': float(np.mean(res.rmse_per_step)),
                    'MAPE': res.test_mape,
                    'Время (сек.)': res.runtime_seconds,
                    'Категория': 'ML-стратегии'
                })
            
            # Добавляем benchmark из блока "Стратегии"
            if stage3_state.get('benchmark'):
                bench = stage3_state['benchmark']
                all_models_data.append({
                    'Этап': 'Стратегии',
                    'Модель': bench.name,
                    'Горизонт': horizon_used,
                    'MAE': bench.mae,
                    'RMSE': bench.rmse,
                    'MAPE': bench.mape,
                    'Время (сек.)': np.nan,
                    'Категория': 'Benchmark'
                })

        # Данные из блока "Кросс-валидация"
        if has_stage4 and st.session_state.get('stage4_results'):
            stage4_state = st.session_state['stage4_results']
            summaries = stage4_state['summaries']
            
            for scheme, summary in summaries.items():
                all_models_data.append({
                    'Этап': 'CV',
                    'Модель': f"CV: {scheme}",
                    'Горизонт': 'Variable',
                    'MAE': summary.mean_mae,
                    'RMSE': summary.mean_rmse,
                    'MAPE': np.nan,
                    'Время (сек.)': summary.runtime_seconds,
                    'Категория': 'Кросс-валидация'
                })

        # Данные из блока "Экспоненциальное сглаживание"
        if has_stage6 and st.session_state.get('stage6_results'):
            stage6_state = st.session_state['stage6_results']
            model_results = stage6_state['models']
            benchmark = stage6_state['benchmark']
            config_stage6 = stage6_state['config']
            horizon_stage6 = config_stage6['horizon']
            
            # Benchmark модель
            all_models_data.append({
                'Этап': 'EXP',
                'Модель': benchmark.name,
                'Горизонт': horizon_stage6,
                'MAE': benchmark.test_mae,
                'RMSE': benchmark.test_rmse,
                'MAPE': benchmark.test_mape,
                'Время (сек.)': np.nan,
                'Категория': 'Benchmark'
            })
            
            # Модели экспоненциального сглаживания
            for res in model_results:
                all_models_data.append({
                    'Этап': 'EXP',
                    'Модель': res.name,
                    'Горизонт': horizon_stage6,
                    'MAE': res.test_mae,
                    'RMSE': res.test_rmse,
                    'MAPE': res.test_mape,
                    'Время (сек.)': res.runtime_seconds,
                    'Категория': 'Статистические'
                })

        if all_models_data:
            summary_all_df = pd.DataFrame(all_models_data)
            
            # Функция форматирования для сводной таблицы
            def format_summary_metric(val):
                if pd.isna(val):
                    return "N/A"
                if abs(val) < 0.0001 and val != 0:
                    return f"{val:.4e}"
                elif abs(val) < 1:
                    return f"{val:.6f}"
                else:
                    return f"{val:.4f}"
            
            summary_display = summary_all_df.copy()
            for col in ['MAE', 'RMSE', 'MAPE', 'Время (сек.)']:
                if col in summary_display.columns:
                    summary_display[col] = summary_display[col].apply(format_summary_metric)
            
            st.dataframe(summary_display, use_container_width=True, height=400)
            
            # Кнопка экспорта сводной таблицы
            summary_csv = summary_all_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать сводную таблицу метрик",
                data=summary_csv,
                file_name="all_models_comparison.csv",
                mime="text/csv",
            )

            # Визуализация сравнения моделей
            st.markdown("---")
            st.subheader("📈 Визуальное сравнение моделей")
            
            viz_cols = st.columns(2)
            
            with viz_cols[0]:
                # График MAE по моделям
                fig_mae = go.Figure()
                
                for category in summary_all_df['Категория'].unique():
                    cat_data = summary_all_df[summary_all_df['Категория'] == category]
                    fig_mae.add_trace(go.Bar(
                        name=category,
                        x=cat_data['Модель'],
                        y=cat_data['MAE'],
                        text=cat_data['MAE'].apply(lambda x: f"{x:.4f}" if pd.notna(x) and abs(x) >= 0.0001 else f"{x:.2e}" if pd.notna(x) else "N/A"),
                        textposition='auto',
                    ))
                
                fig_mae.update_layout(
                    title='Сравнение MAE по моделям',
                    xaxis_title='Модель',
                    yaxis_title='MAE',
                    barmode='group',
                    height=500,
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig_mae, use_container_width=True)
            
            with viz_cols[1]:
                # График RMSE по моделям
                fig_rmse = go.Figure()
                
                for category in summary_all_df['Категория'].unique():
                    cat_data = summary_all_df[summary_all_df['Категория'] == category]
                    fig_rmse.add_trace(go.Bar(
                        name=category,
                        x=cat_data['Модель'],
                        y=cat_data['RMSE'],
                        text=cat_data['RMSE'].apply(lambda x: f"{x:.4f}" if pd.notna(x) and abs(x) >= 0.0001 else f"{x:.2e}" if pd.notna(x) else "N/A"),
                        textposition='auto',
                    ))
                
                fig_rmse.update_layout(
                    title='Сравнение RMSE по моделям',
                    xaxis_title='Модель',
                    yaxis_title='RMSE',
                    barmode='group',
                    height=500,
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig_rmse, use_container_width=True)

            # Анализ по горизонтам
            st.markdown("---")
            st.subheader("🔍 Анализ по горизонтам прогноза")
            
            # Группируем по горизонтам
            horizons_analysis = summary_all_df[summary_all_df['Горизонт'] != 'Variable'].copy()
            if len(horizons_analysis) > 0:
                horizons_analysis['Горизонт'] = pd.to_numeric(horizons_analysis['Горизонт'], errors='coerce')
                horizons_analysis = horizons_analysis.dropna(subset=['Горизонт'])
                
                if len(horizons_analysis) > 0:
                    # Категоризация горизонтов
                    horizons_analysis['Категория горизонта'] = horizons_analysis['Горизонт'].apply(
                        lambda h: 'Короткий (≤7)' if h <= 7 else 'Средний (8-30)' if h <= 30 else 'Длинный (>30)'
                    )
                    
                    horizon_summary = horizons_analysis.groupby(['Категория горизонта', 'Модель']).agg({
                        'MAE': 'mean',
                        'RMSE': 'mean',
                        'MAPE': 'mean'
                    }).reset_index()
                    
                    st.markdown("**Средние метрики по категориям горизонта:**")
                    
                    horizon_display = horizon_summary.copy()
                    for col in ['MAE', 'RMSE', 'MAPE']:
                        if col in horizon_display.columns:
                            horizon_display[col] = horizon_display[col].apply(format_summary_metric)
                    
                    st.dataframe(horizon_display, use_container_width=True)
                    
                    # Находим лучшие модели по горизонтам
                    st.markdown("**Лучшие модели по горизонтам:**")
                    
                    for horizon_cat in horizon_summary['Категория горизонта'].unique():
                        cat_data = horizon_summary[horizon_summary['Категория горизонта'] == horizon_cat]
                        best_mae = cat_data.loc[cat_data['MAE'].idxmin()]
                        best_rmse = cat_data.loc[cat_data['RMSE'].idxmin()]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"**{horizon_cat}** (по MAE): {best_mae['Модель']} (MAE={best_mae['MAE']:.6f})")
                        with col2:
                            st.info(f"**{horizon_cat}** (по RMSE): {best_rmse['Модель']} (RMSE={best_rmse['RMSE']:.6f})")
                else:
                    st.info("Недостаточно данных для анализа по горизонтам")
            else:
                st.info("Данные по разным горизонтам отсутствуют")

            # Анализ влияния преобразований
            st.markdown("---")
            st.subheader("🔄 Анализ влияния преобразований (Box-Cox, Log)")
            
            if has_stage5 and st.session_state.get('stage5_results'):
                stage5_state = st.session_state['stage5_results']
                pipelines_info = stage5_state['pipelines']
                best_pipeline_name = stage5_state['best_name']
                
                st.markdown("**Результаты тестирования преобразований:**")
                
                transform_summary = []
                for item in pipelines_info:
                    transform_summary.append({
                        'Цепочка': item['name'],
                        'ADF стац.': '✅' if item['adf_stationary'] else '❌',
                        'KPSS стац.': '✅' if item['kpss_stationary'] else '❌',
                        'Score': item['score'],
                        'Рекомендуется': '⭐' if item['name'] == best_pipeline_name else ''
                    })
                
                transform_df = pd.DataFrame(transform_summary)
                st.dataframe(transform_df, use_container_width=True)
                
                st.success(f"**Рекомендуемая цепочка преобразований:** {best_pipeline_name}")
                
                st.markdown("""
                **Рекомендации по использованию преобразований:**
                - Если данные имеют растущую дисперсию → используйте **Box-Cox** или **Log**
                - Если ряд нестационарен → примените **дифференцирование**
                - Для сезонных данных → используйте **сезонное дифференцирование**
                - После преобразований проверьте стационарность (ADF/KPSS тесты)
                """)
            else:
                st.info("Данные о преобразованиях недоступны. Завершите соответствующий блок.")

            # Согласованность с кросс-валидацией
            st.markdown("---")
            st.subheader("✅ Согласованность с кросс-валидацией")
            
            if has_stage4 and st.session_state.get('stage4_results'):
                stage4_state = st.session_state['stage4_results']
                cv_summaries = stage4_state['summaries']
                
                st.markdown("**Результаты кросс-валидации:**")
                
                cv_comparison = []
                for scheme, summary in cv_summaries.items():
                    mae_values = [fold.mae for fold in summary.fold_results]
                    rmse_values = [fold.rmse for fold in summary.fold_results]
                    
                    cv_comparison.append({
                        'Схема CV': scheme,
                        'Средний MAE': summary.mean_mae,
                        'Std(MAE)': float(np.std(mae_values, ddof=1)) if len(mae_values) > 1 else 0.0,
                        'Средний RMSE': summary.mean_rmse,
                        'Std(RMSE)': float(np.std(rmse_values, ddof=1)) if len(rmse_values) > 1 else 0.0,
                    })
                
                cv_df = pd.DataFrame(cv_comparison)
                st.dataframe(cv_df.round(6), use_container_width=True)
                
                st.markdown("""
                **Интерпретация результатов CV:**
                - **Низкий Std(MAE/RMSE)** → модель стабильна на разных временных интервалах
                - **Высокий Std(MAE/RMSE)** → модель чувствительна к выбору обучающей выборки
                - Сравните средние метрики CV с финальными оценками на тесте
                - Если есть большое расхождение → возможна переподгонка или особенности тестового периода
                """)
            else:
                st.info("Данные кросс-валидации недоступны. Завершите соответствующий блок.")

            # Финальные рекомендации
            st.markdown("---")
            st.subheader("🎯 Финальные рекомендации")
            
            # Находим лучшую модель по MAE
            best_model_idx = summary_all_df['MAE'].idxmin()
            best_model = summary_all_df.loc[best_model_idx]
            
            # Топ-3 модели
            top3 = summary_all_df.nsmallest(3, 'MAE')
            
            # Форматируем MAPE
            best_mape = best_model['MAPE']
            best_mape_str = f"{best_mape:.6f}" if pd.notna(best_mape) else 'N/A'
            
            st.success(f"""
            **Лучшая модель по MAE:**
            - **Название:** {best_model['Модель']}
            - **Блок:** {best_model['Этап']}
            - **MAE:** {best_model['MAE']:.6f}
            - **RMSE:** {best_model['RMSE']:.6f}
            - **MAPE:** {best_mape_str}
            """)
            
            st.markdown("**Топ-3 модели:**")
            top3_display = top3[['Модель', 'Этап', 'MAE', 'RMSE', 'MAPE']].copy()
            for col in ['MAE', 'RMSE', 'MAPE']:
                top3_display[col] = top3_display[col].apply(format_summary_metric)
            st.dataframe(top3_display.reset_index(drop=True), use_container_width=True)
            
            st.markdown("""
            **Критерии окончательного выбора:**
            
            1. **Точность прогноза** (MAE, RMSE) - основной критерий
            2. **Стабильность** - проверьте согласованность с кросс-валидацией
            3. **Горизонт прогноза** - выберите модель, лучшую для вашего горизонта
            4. **Вычислительные ресурсы** - учтите время обучения
            5. **Интерпретируемость** - статистические модели обычно легче объяснить
            6. **Преобразования данных** - не забудьте применить рекомендуемые преобразования
            
            💡 **Совет:** Для продакшена рассмотрите ансамбль из топ-3 моделей!
            """)
            
            # Экспорт финального отчета
            st.markdown("---")
            
            # Создаём финальный отчет
            report_lines = [
                "# ФИНАЛЬНЫЙ ОТЧЕТ ПО АНАЛИЗУ ВРЕМЕННЫХ РЯДОВ\n",
                f"\n## Лучшая модель: {best_model['Модель']}\n",
                f"- Блок: {best_model['Этап']}\n",
                f"- MAE: {best_model['MAE']:.6f}\n",
                f"- RMSE: {best_model['RMSE']:.6f}\n",
                f"- MAPE: {best_mape_str}\n",
                f"- Горизонт: {best_model['Горизонт']}\n",
                f"\n## Топ-3 модели:\n"
            ]
            
            for idx, row in top3.iterrows():
                row_mape = row['MAPE']
                row_mape_str = f"{row_mape:.6f}" if pd.notna(row_mape) else 'N/A'
                report_lines.append(f"\n### {row['Модель']}\n")
                report_lines.append(f"- MAE: {row['MAE']:.6f}\n")
                report_lines.append(f"- RMSE: {row['RMSE']:.6f}\n")
                report_lines.append(f"- MAPE: {row_mape_str}\n")
            
            if has_stage5:
                report_lines.append(f"\n## Рекомендуемые преобразования:\n")
                report_lines.append(f"{best_pipeline_name}\n")
            
            report_text = "".join(report_lines)
            
            st.download_button(
                label="📥 Скачать финальный отчет",
                data=report_text.encode('utf-8'),
                file_name="final_model_selection_report.md",
                mime="text/markdown",
            )
        else:
            st.warning("Нет данных для анализа. Пожалуйста, завершите блоки 3-6.")