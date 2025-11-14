import streamlit as st
import pandas as pd
import os
import sys

# Отключение шаблонов Plotly для избежания ошибок рекурсии
os.environ['PLOTLY_RENDERER'] = 'browser'

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
from datetime import datetime
import io

# Увеличение лимита рекурсии для избежания ошибок с Plotly
sys.setrecursionlimit(5000)

# Очистка кэша шаблонов Plotly
try:
    pio.templates.default = None
except:
    pass

# Импорт модулей анализа
from ts_analysis import (
    calculate_acf_pacf,
    test_stationarity,
    calculate_rolling_stats,
    calculate_correlations
)
from report_generator import generate_html_report
from data_preprocessing import (
    standardize_timezone,
    remove_duplicates,
    check_monotonicity,
    handle_missing_values,
    handle_outliers,
    resample_timeseries,
    preprocess_pipeline
)
from statistical_analysis import (
    calculate_descriptive_statistics,
    create_time_series_plots,
    create_histograms,
    create_boxplots,
    create_correlation_heatmap,
    analyze_multicollinearity,
    detect_remaining_outliers,
    create_scatter_matrix,
    create_qq_plots,
    perform_normality_tests
)
from stationarity_analysis import (
    calculate_rolling_statistics,
    visual_trend_analysis,
    perform_adf_test,
    perform_kpss_test,
    apply_differencing,
    comprehensive_stationarity_test,
    create_stationarity_visualization,
    create_differencing_comparison,
    get_stationarity_recommendation
)
from feature_engineering import (
    create_lag_features,
    create_rolling_features,
    create_all_features,
    calculate_lag_correlations,
    check_multicollinearity_vif,
    analyze_feature_importance_correlation,
    create_lag_correlation_plot,
    create_feature_importance_plot,
    create_rolling_features_plot,
    get_feature_statistics
)
from acf_pacf_analysis import (
    calculate_acf_pacf_detailed,
    identify_significant_lags,
    interpret_acf_pattern,
    interpret_pacf_pattern,
    suggest_arima_parameters,
    create_acf_pacf_plot,
    comprehensive_acf_pacf_analysis
)
from decomposition_analysis import (
    perform_decomposition,
    analyze_trend,
    analyze_seasonality,
    analyze_residuals,
    create_decomposition_plot,
    create_seasonal_pattern_plot,
    create_residuals_analysis_plot,
    comprehensive_decomposition_analysis
)

# Конфигурация страницы
st.set_page_config(
    page_title="Анализ временных рядов",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("📈 Анализ временных рядов")
st.markdown("---")

# Инициализация состояния сессии
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_preprocessed' not in st.session_state:
    st.session_state.df_preprocessed = None
if 'preprocessing_reports' not in st.session_state:
    st.session_state.preprocessing_reports = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'engineered_features' not in st.session_state:
    st.session_state.engineered_features = None
if 'feature_engineering_info' not in st.session_state:
    st.session_state.feature_engineering_info = {}

# Боковая панель для загрузки данных
with st.sidebar:
    st.header("⚙️ Настройки")
    
    st.markdown("---")
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "Выберите файл",
        type=['csv', 'parquet'],
        help="Поддерживаются форматы CSV и Parquet"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                st.session_state.df = pd.read_parquet(uploaded_file)
            st.success("✅ Файл успешно загружен!")
        except Exception as e:
            st.error(f"❌ Ошибка загрузки файла: {str(e)}")

# Основное содержимое
if st.session_state.df is not None:
    df = st.session_state.df
    
    st.header("📊 Просмотр данных")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Информация о датасете
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Количество строк", df.shape[0])
    with col2:
        st.metric("Количество столбцов", df.shape[1])
    with col3:
        st.metric("Пропущенных значений", df.isnull().sum().sum())
    
    st.markdown("---")
    
    # Выбор переменных для анализа
    st.header("🎯 Выбор переменных")
    
    col1, col2 = st.columns(2)
    
    with col1:
        date_column = st.selectbox(
            "Выберите столбец с датой/временем:",
            df.columns.tolist(),
            help="Столбец, содержащий временные метки"
        )
    
    with col2:
        target_column = st.selectbox(
            "Выберите целевую переменную:",
            [col for col in df.columns if col != date_column],
            help="Переменная для анализа временного ряда"
        )
    
    # Дополнительные признаки для корреляционного анализа
    feature_columns = st.multiselect(
        "Выберите дополнительные признаки для анализа корреляций:",
        [col for col in df.columns if col not in [date_column, target_column]],
        help="Опционально: выберите переменные для корреляционного анализа"
    )
    
    # Преобразование столбца даты
    try:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
    except:
        st.error("❌ Не удалось преобразовать выбранный столбец в формат даты/времени")
        st.stop()
    
    st.markdown("---")
    
    # Раздел предобработки данных
    st.header("🧹 Предобработка данных")
    
    with st.expander("⚙️ Настройки предобработки", expanded=False):
        st.markdown("""
        Предобработка данных включает очистку, нормализацию временных меток, 
        обработку пропусков и выбросов для повышения качества анализа.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Временные метки")
            
            standardize_tz = st.checkbox(
                "Привести к единой временной зоне",
                value=False,
                help="Преобразование всех временных меток в выбранную зону"
            )
            
            if standardize_tz:
                target_timezone = st.selectbox(
                    "Временная зона:",
                    ["Europe/Moscow", "UTC", "Europe/London", "America/New_York", "Asia/Tokyo"],
                    help="Целевая временная зона для всех данных"
                )
            else:
                target_timezone = "Europe/Moscow"
            
            remove_dups = st.checkbox(
                "Удалить дубликаты по времени",
                value=True,
                help="Удаление строк с одинаковыми временными метками"
            )
            
            if remove_dups:
                duplicate_strategy = st.selectbox(
                    "Стратегия удаления:",
                    ["first", "last", "mean"],
                    format_func=lambda x: {"first": "Оставить первый", "last": "Оставить последний", "mean": "Среднее значение"}[x]
                )
            else:
                duplicate_strategy = "first"
            
            do_resample = st.checkbox(
                "Ресемплировать к единой частоте",
                value=False,
                help="Приведение временного ряда к регулярной частоте"
            )
            
            if do_resample:
                col_a, col_b = st.columns(2)
                with col_a:
                    resample_freq = st.selectbox(
                        "Частота:",
                        ["H", "D", "W", "M"],
                        format_func=lambda x: {"H": "Час", "D": "День", "W": "Неделя", "M": "Месяц"}[x]
                    )
                with col_b:
                    resample_method = st.selectbox(
                        "Метод агрегации:",
                        ["mean", "sum", "median", "min", "max", "first", "last"],
                        format_func=lambda x: {
                            "mean": "Среднее", "sum": "Сумма", "median": "Медиана",
                            "min": "Минимум", "max": "Максимум", 
                            "first": "Первое", "last": "Последнее"
                        }[x]
                    )
            else:
                resample_freq = "D"
                resample_method = "mean"
        
        with col2:
            st.subheader("Качество данных")
            
            handle_missing = st.checkbox(
                "Обработать пропущенные значения",
                value=True,
                help="Заполнение или удаление пропусков в данных"
            )
            
            if handle_missing:
                missing_method = st.selectbox(
                    "Метод обработки пропусков:",
                    ["linear", "polynomial", "cubic", "rolling_mean", "ffill", "bfill", "drop"],
                    format_func=lambda x: {
                        "linear": "Линейная интерполяция",
                        "polynomial": "Полиномиальная интерполяция",
                        "cubic": "Кубическая интерполяция",
                        "rolling_mean": "Скользящее среднее",
                        "ffill": "Заполнение вперед",
                        "bfill": "Заполнение назад",
                        "drop": "Удалить (<5%)"
                    }[x]
                )
                
                if missing_method == "rolling_mean":
                    missing_window = st.slider(
                        "Окно для скользящего среднего:",
                        min_value=2,
                        max_value=20,
                        value=3
                    )
                else:
                    missing_window = None
            else:
                missing_method = "linear"
                missing_window = None
            
            handle_out = st.checkbox(
                "Обработать выбросы",
                value=True,
                help="Обнаружение и обработка аномальных значений методом IQR"
            )
            
            if handle_out:
                col_c, col_d = st.columns(2)
                with col_c:
                    outlier_method = st.selectbox(
                        "Метод обработки выбросов:",
                        ["clip", "remove", "interpolate", "median"],
                        format_func=lambda x: {
                            "clip": "Ограничить границами",
                            "remove": "Удалить",
                            "interpolate": "Интерполировать",
                            "median": "Заменить медианой"
                        }[x]
                    )
                with col_d:
                    iqr_multiplier = st.slider(
                        "Множитель IQR:",
                        min_value=1.0,
                        max_value=3.0,
                        value=1.5,
                        step=0.1,
                        help="Чувствительность обнаружения (меньше = строже)"
                    )
                
                if outlier_method == "median":
                    outlier_window = st.slider(
                        "Окно для медианы:",
                        min_value=2,
                        max_value=20,
                        value=3
                    )
                else:
                    outlier_window = None
            else:
                outlier_method = "clip"
                iqr_multiplier = 1.5
                outlier_window = None
        
        # Кнопка запуска предобработки
        if st.button("🧹 Запустить предобработку", use_container_width=True):
            with st.spinner("Выполняется предобработка данных..."):
                try:
                    # Конфигурация предобработки
                    config = {
                        'standardize_tz': standardize_tz,
                        'target_timezone': target_timezone,
                        'remove_duplicates': remove_dups,
                        'duplicate_strategy': duplicate_strategy,
                        'check_monotonicity': True,
                        'resample': do_resample,
                        'resample_freq': resample_freq,
                        'resample_method': resample_method,
                        'handle_missing': handle_missing,
                        'missing_method': missing_method,
                        'missing_window': missing_window,
                        'handle_outliers': handle_out,
                        'outlier_method': outlier_method,
                        'iqr_multiplier': iqr_multiplier,
                        'outlier_window': outlier_window
                    }
                    
                    # Выполнение предобработки
                    df_processed, reports = preprocess_pipeline(
                        df,
                        date_column,
                        target_column,
                        config
                    )
                    
                    # Сохранение результатов
                    st.session_state.df_preprocessed = df_processed
                    st.session_state.preprocessing_reports = reports
                    
                    st.success("✅ Предобработка успешно выполнена!")
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при предобработке: {str(e)}")
                    st.exception(e)
    
    # Отображение результатов предобработки
    if st.session_state.df_preprocessed is not None:
        st.subheader("📊 Результаты предобработки")
        
        reports = st.session_state.preprocessing_reports
        
        # Создание метрик
        metric_cols = st.columns(4)
        
        # Определяем, какие данные использовать
        df_to_analyze = st.session_state.df_preprocessed
        
        with metric_cols[0]:
            original = len(df)
            final = len(df_to_analyze)
            delta = final - original
            st.metric("Строк данных", final, delta=delta)
        
        with metric_cols[1]:
            if 'duplicates' in reports:
                st.metric(
                    "Удалено дубликатов",
                    reports['duplicates']['duplicates_removed']
                )
            else:
                st.metric("Удалено дубликатов", 0)
        
        with metric_cols[2]:
            if 'missing' in reports:
                st.metric(
                    "Заполнено пропусков",
                    reports['missing']['filled_count'],
                    delta=f"-{reports['missing']['missing_percentage']:.1f}%"
                )
            else:
                st.metric("Заполнено пропусков", 0)
        
        with metric_cols[3]:
            if 'outliers' in reports:
                st.metric(
                    "Обработано выбросов",
                    reports['outliers']['handled_count'],
                    delta=f"{reports['outliers']['outlier_percentage']:.1f}%"
                )
            else:
                st.metric("Обработано выбросов", 0)
        
        # Детальные отчёты
        with st.expander("📋 Детальные отчёты предобработки"):
            for step_name, report in reports.items():
                st.markdown(f"**{step_name.upper()}**")
                st.json(report)
        
        # Сравнение до и после
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**До предобработки**")
            st.dataframe(df[[date_column, target_column]].head(10), use_container_width=True)
        
        with col2:
            st.markdown("**После предобработки**")
            st.dataframe(df_to_analyze[[date_column, target_column]].head(10), use_container_width=True)
        
        # График сравнения
        fig_compare = go.Figure(layout=go.Layout(template=None))
        
        fig_compare.add_trace(go.Scatter(
            x=df[date_column],
            y=df[target_column],
            mode='lines',
            name='До предобработки',
            line=dict(color='lightblue', width=1),
            opacity=0.7
        ))
        
        fig_compare.add_trace(go.Scatter(
            x=df_to_analyze[date_column],
            y=df_to_analyze[target_column],
            mode='lines',
            name='После предобработки',
            line=dict(color='darkblue', width=2)
        ))
        
        # Отметка выбросов если они были обнаружены
        if 'outliers' in reports and reports['outliers']['total_outliers'] > 0:
            outlier_stats = reports['outliers']
            # Визуализация границ выбросов
            fig_compare.add_hline(
                y=outlier_stats['upper_bound'],
                line_dash="dash",
                line_color="red",
                annotation_text="Верхняя граница"
            )
            fig_compare.add_hline(
                y=outlier_stats['lower_bound'],
                line_dash="dash",
                line_color="red",
                annotation_text="Нижняя граница"
            )
        
        fig_compare.update_layout(
            title="Сравнение данных до и после предобработки",
            xaxis_title="Дата",
            yaxis_title=target_column,
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Обновляем df для дальнейшего анализа
        df = df_to_analyze.copy()
    
    st.markdown("---")
    
    # Раздел статистического анализа
    st.header("📊 Статистический анализ и визуализация")
    
    # Получаем числовые столбцы для анализа
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_columns:
        # Дескриптивная статистика
        st.subheader("📈 Дескриптивная статистика")
        
        with st.expander("🔍 Просмотреть статистику", expanded=False):
            stats_df = calculate_descriptive_statistics(df, exclude_columns=[])
            
            if not stats_df.empty:
                st.dataframe(
                    stats_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn_r', axis=1),
                    use_container_width=True
                )
                
                # Интерпретация статистики
                st.markdown("""
                **Интерпретация показателей:**
                - **Асимметрия**: 0 = симметричное, >0 = правый хвост, <0 = левый хвост
                - **Эксцесс**: 0 = нормальное, >0 = острое, <0 = плоское распределение
                - **IQR**: Межквартильный размах, показывает разброс средних 50% данных
                """)
                
                # Кнопка скачивания статистики
                csv = stats_df.to_csv(encoding='utf-8-sig')
                st.download_button(
                    label="📥 Скачать статистику (CSV)",
                    data=csv,
                    file_name=f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("Нет числовых столбцов для анализа")
        
        # Визуализации
        st.subheader("📉 Визуализация данных")
        
        # Выбор столбцов для визуализации
        viz_columns = st.multiselect(
            "Выберите признаки для визуализации:",
            numeric_columns,
            default=numeric_columns[:min(3, len(numeric_columns))],
            help="Выберите признаки для построения графиков"
        )
        
        if viz_columns:
            tabs = st.tabs([
                "📈 Временные ряды",
                "📊 Гистограммы",
                "📦 Box Plots",
                "🔥 Корреляции",
                "🔍 Q-Q Plots",
                "🎯 Scatter Matrix"
            ])
            
            # Вкладка 1: Временные ряды
            with tabs[0]:
                st.markdown("### Линейные графики по времени")
                fig_ts = create_time_series_plots(df, date_column, viz_columns)
                if fig_ts:
                    st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.info("Недостаточно данных для визуализации")
            
            # Вкладка 2: Гистограммы
            with tabs[1]:
                st.markdown("### Распределение признаков")
                
                bins = st.slider(
                    "Количество бинов:",
                    min_value=10,
                    max_value=100,
                    value=30,
                    help="Количество интервалов в гистограмме"
                )
                
                fig_hist = create_histograms(df, viz_columns, bins=bins)
                if fig_hist:
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Анализ нормальности
                    st.markdown("#### Тесты на нормальность распределения")
                    normality_tests = perform_normality_tests(df, viz_columns)
                    if not normality_tests.empty:
                        st.dataframe(
                            normality_tests.style.format({
                                'Shapiro-Wilk stat': '{:.4f}',
                                'Shapiro-Wilk p-value': '{:.4f}',
                                'K-S stat': '{:.4f}',
                                'K-S p-value': '{:.4f}',
                                'Anderson stat': '{:.4f}'
                            }),
                            use_container_width=True
                        )
                        st.caption("💡 p-value > 0.05 указывает на нормальное распределение")
            
            # Вкладка 3: Box Plots
            with tabs[2]:
                st.markdown("### Выбросы и квартили")
                
                fig_box = create_boxplots(df, viz_columns)
                if fig_box:
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Анализ выбросов
                    st.markdown("#### Обнаруженные аномальные значения")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        outlier_method = st.selectbox(
                            "Метод обнаружения:",
                            ["iqr", "zscore"],
                            format_func=lambda x: "IQR (Межквартильный размах)" if x == "iqr" else "Z-score"
                        )
                    with col2:
                        outlier_threshold = st.slider(
                            "Порог чувствительности:",
                            min_value=1.0,
                            max_value=3.0,
                            value=1.5 if outlier_method == "iqr" else 3.0,
                            step=0.1
                        )
                    
                    outliers_info = detect_remaining_outliers(
                        df,
                        viz_columns,
                        method=outlier_method,
                        threshold=outlier_threshold
                    )
                    
                    if outliers_info:
                        for col_name, info in outliers_info.items():
                            with st.expander(f"📌 {col_name}: {info['count']} выбросов ({info['percentage']:.2f}%)"):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Нижняя граница", f"{info['lower_bound']:.2f}")
                                    st.metric("Минимальный выброс", f"{info['min_outlier']:.2f}")
                                with col_b:
                                    st.metric("Верхняя граница", f"{info['upper_bound']:.2f}")
                                    st.metric("Максимальный выброс", f"{info['max_outlier']:.2f}")
                                
                                st.markdown("**Примеры значений-выбросов:**")
                                st.write(info['outlier_values'])
                    else:
                        st.success("✅ Аномальные значения не обнаружены")
            
            # Вкладка 4: Корреляции
            with tabs[3]:
                st.markdown("### Корреляционный анализ")
                
                col1, col2 = st.columns(2)
                with col1:
                    corr_method = st.selectbox(
                        "Метод корреляции:",
                        ["pearson", "spearman"],
                        format_func=lambda x: "Pearson (линейная)" if x == "pearson" else "Spearman (ранговая)"
                    )
                with col2:
                    multicollinearity_threshold = st.slider(
                        "Порог мультиколлинеарности:",
                        min_value=0.5,
                        max_value=1.0,
                        value=0.8,
                        step=0.05,
                        help="Значение корреляции для определения сильной связи"
                    )
                
                # Выбор признаков для корреляции
                corr_columns = st.multiselect(
                    "Признаки для корреляционного анализа:",
                    numeric_columns,
                    default=numeric_columns,
                    help="Выберите признаки для расчета корреляций"
                )
                
                if len(corr_columns) >= 2:
                    fig_corr, corr_matrix = create_correlation_heatmap(
                        df,
                        corr_columns,
                        method=corr_method
                    )
                    
                    if fig_corr:
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Анализ мультиколлинеарности
                        st.markdown("#### 🔍 Анализ мультиколлинеарности")
                        
                        high_corr = analyze_multicollinearity(
                            corr_matrix,
                            threshold=multicollinearity_threshold
                        )
                        
                        if high_corr:
                            st.warning(f"⚠️ Обнаружено {len(high_corr)} пар признаков с высокой корреляцией")
                            
                            corr_df = pd.DataFrame(high_corr)
                            st.dataframe(
                                corr_df[['feature_1', 'feature_2', 'correlation']].style.format({
                                    'correlation': '{:.4f}'
                                }).background_gradient(cmap='RdYlGn_r', subset=['correlation']),
                                use_container_width=True
                            )
                            
                            st.markdown("""
                            **⚡ Рекомендации:**
                            - Признаки с высокой корреляцией могут дублировать информацию
                            - Рассмотрите возможность удаления одного из коррелирующих признаков
                            - Или используйте методы снижения размерности (PCA)
                            """)
                        else:
                            st.success(f"✅ Сильная мультиколлинеарность не обнаружена (|r| < {multicollinearity_threshold})")
                else:
                    st.info("Выберите минимум 2 признака для корреляционного анализа")
            
            # Вкладка 5: Q-Q Plots
            with tabs[4]:
                st.markdown("### Проверка нормальности распределения (Q-Q Plots)")
                
                st.info("""
                📊 **Q-Q Plot** сравнивает квантили данных с квантилями нормального распределения.
                Если точки лежат близко к диагональной линии - распределение близко к нормальному.
                """)
                
                fig_qq = create_qq_plots(df, viz_columns)
                if fig_qq:
                    st.plotly_chart(fig_qq, use_container_width=True)
            
            # Вкладка 6: Scatter Matrix
            with tabs[5]:
                st.markdown("### Матрица диаграмм рассеяния")
                
                st.info("📌 Для производительности отображается максимум 5 признаков")
                
                scatter_columns = viz_columns[:5]
                
                if len(scatter_columns) >= 2:
                    fig_scatter = create_scatter_matrix(df, scatter_columns)
                    if fig_scatter:
                        st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Выберите минимум 2 признака для матрицы рассеяния")
        else:
            st.info("👆 Выберите признаки для визуализации")
    else:
        st.warning("⚠️ В данных нет числовых столбцов для анализа")
    
    st.markdown("---")
    
    # Раздел анализа стационарности
    st.header("🔬 Анализ стационарности")
    
    st.markdown("""
    **Стационарность** - важное свойство временного ряда, означающее что статистические характеристики 
    (среднее, дисперсия) не изменяются во времени. Стационарные ряды проще моделировать и прогнозировать.
    """)
    
    # Выбор переменной для анализа стационарности
    stationarity_column = st.selectbox(
        "Выберите переменную для анализа стационарности:",
        [col for col in df.columns if col != date_column and pd.api.types.is_numeric_dtype(df[col])],
        key="stationarity_column",
        help="Выберите числовую переменную для тестирования стационарности"
    )
    
    if stationarity_column:
        with st.expander("⚙️ Параметры анализа стационарности", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Параметры скользящих окон:**")
                window_30 = st.number_input("Малое окно:", min_value=5, max_value=100, value=30, key="win30")
                window_60 = st.number_input("Среднее окно:", min_value=10, max_value=200, value=60, key="win60")
                window_90 = st.number_input("Большое окно:", min_value=20, max_value=300, value=90, key="win90")
                windows = [window_30, window_60, window_90]
            
            with col2:
                st.markdown("**Параметры дифференцирования:**")
                max_diff_order = st.slider(
                    "Макс. порядок дифференцирования:",
                    min_value=1,
                    max_value=3,
                    value=2,
                    help="Максимальный порядок дифференцирования для проверки"
                )
                
                kpss_regression = st.selectbox(
                    "Тип регрессии для KPSS:",
                    ["c", "ct"],
                    format_func=lambda x: "Константа" if x == "c" else "Константа + Тренд",
                    help="c - проверка стационарности относительно константы, ct - относительно тренда"
                )
            
            # Кнопка запуска анализа
            if st.button("🔬 Запустить анализ стационарности", type="primary", use_container_width=True):
                with st.spinner("Выполняется анализ стационарности..."):
                    try:
                        # Подготовка данных
                        series = df[stationarity_column].dropna()
                        dates = df[date_column][series.index]
                        
                        if len(series) < 10:
                            st.error("❌ Недостаточно данных для анализа (минимум 10 точек)")
                            st.stop()
                        
                        # Расчет скользящих статистик
                        rolling_stats = calculate_rolling_statistics(series, windows=windows)
                        
                        # Визуальный анализ тренда
                        trend_analysis = visual_trend_analysis(series, rolling_stats)
                        
                        # Комплексный тест стационарности
                        stationarity_tests = comprehensive_stationarity_test(series, max_diff_order=max_diff_order)
                        
                        # Рекомендация
                        recommendation = get_stationarity_recommendation(stationarity_tests)
                        
                        # Сохранение результатов в session state
                        st.session_state.stationarity_results = {
                            'series': series,
                            'dates': dates,
                            'rolling_stats': rolling_stats,
                            'trend_analysis': trend_analysis,
                            'tests': stationarity_tests,
                            'recommendation': recommendation,
                            'column': stationarity_column,
                            'windows': windows
                        }
                        
                        st.success("✅ Анализ стационарности завершён!")
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка при анализе: {str(e)}")
                        st.exception(e)
        
        # Отображение результатов анализа стационарности
        if 'stationarity_results' in st.session_state and st.session_state.stationarity_results:
            results = st.session_state.stationarity_results
            
            st.markdown("---")
            st.subheader("📊 Результаты анализа стационарности")
            
            # Рекомендация
            recommendation = results['recommendation']
            
            if recommendation['is_stationary'] and recommendation['required_differencing'] == 0:
                st.success(recommendation['message'])
            elif recommendation['is_stationary']:
                st.info(recommendation['message'])
            else:
                st.warning(recommendation['message'])
            
            with st.expander("📋 Детали рекомендации", expanded=False):
                for detail in recommendation['details']:
                    st.write(f"- {detail}")
            
            # Вкладки с результатами
            tabs = st.tabs([
                "📈 Визуальный анализ",
                "📊 Скользящие характеристики", 
                "🧪 Тесты стационарности",
                "🔄 Дифференцирование"
            ])
            
            # Вкладка 1: Визуальный анализ тренда
            with tabs[0]:
                st.markdown("### Визуальный анализ тренда и дисперсии")
                
                trend_analysis = results['trend_analysis']
                
                # Таблица с результатами анализа
                trend_data = []
                for window, analysis in trend_analysis.items():
                    trend_data.append({
                        'Окно': window,
                        'Направление тренда': analysis['trend_direction'],
                        'Наклон тренда': f"{analysis['trend_slope']:.6f}",
                        'Стабильность дисперсии': analysis['variance_stability'],
                        'Изменение дисперсии (%)': f"{analysis['variance_change_pct']:.2f}%",
                        'Коэфф. вариации среднего': f"{analysis['mean_stability']:.4f}"
                    })
                
                trend_df = pd.DataFrame(trend_data)
                st.dataframe(trend_df, use_container_width=True)
                
                st.markdown("""
                **Интерпретация:**
                - **Тренд**: стабильное скользящее среднее → нет тренда, растущее/падающее → есть тренд
                - **Дисперсия**: стабильная дисперсия → гомоскедастичность, изменяющаяся → гетероскедастичность
                - **Коэффициент вариации**: чем меньше, тем стабильнее среднее
                """)
            
            # Вкладка 2: Скользящие характеристики
            with tabs[1]:
                st.markdown("### Скользящие среднее, стандартное отклонение и дисперсия")
                
                fig_rolling = create_stationarity_visualization(
                    results['series'],
                    results['dates'],
                    results['rolling_stats'],
                    results['windows']
                )
                
                st.plotly_chart(fig_rolling, use_container_width=True)
                
                st.markdown("""
                **Как интерпретировать:**
                - **Скользящее среднее**: 
                  - Горизонтальная линия → нет тренда
                  - Растущая/падающая → есть тренд
                - **Скользящее стандартное отклонение**:
                  - Постоянное → стабильная дисперсия
                  - Изменяющееся → изменяющаяся волатильность
                - **Скользящая дисперсия**:
                  - Аналогично стд. откл., но в квадрате
                """)
            
            # Вкладка 3: Тесты стационарности
            with tabs[2]:
                st.markdown("### Статистические тесты стационарности")
                
                tests = results['tests']
                
                for diff_order, test_result in tests.items():
                    if diff_order == 0:
                        st.markdown(f"#### Исходный ряд")
                    else:
                        st.markdown(f"#### После дифференцирования порядка {diff_order}")
                    
                    col1, col2 = st.columns(2)
                    
                    # ADF тест
                    with col1:
                        st.markdown("**Тест Дики-Фуллера (ADF)**")
                        
                        adf = test_result['adf']
                        
                        # Метрики
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("ADF-статистика", f"{adf['test_statistic']:.4f}")
                        with metric_col2:
                            p_value_color = "green" if adf['p_value'] < 0.05 else "red"
                            st.metric("p-value", f"{adf['p_value']:.4f}")
                        
                        # Критические значения
                        st.markdown("*Критические значения:*")
                        for level, value in adf['critical_values'].items():
                            st.write(f"- {level}: {value:.4f}")
                        
                        # Интерпретация
                        if adf['is_stationary']:
                            st.success(f"✅ {adf['interpretation']}")
                        else:
                            st.error(f"❌ {adf['interpretation']}")
                    
                    # KPSS тест
                    with col2:
                        st.markdown("**Тест KPSS**")
                        
                        kpss_test = test_result['kpss']
                        
                        # Метрики
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("KPSS-статистика", f"{kpss_test['test_statistic']:.4f}")
                        with metric_col2:
                            st.metric("p-value", f"{kpss_test['p_value']:.4f}")
                        
                        # Критические значения
                        st.markdown("*Критические значения:*")
                        for level, value in kpss_test['critical_values'].items():
                            st.write(f"- {level}: {value:.4f}")
                        
                        # Интерпретация
                        if kpss_test['is_stationary']:
                            st.success(f"✅ {kpss_test['interpretation']}")
                        else:
                            st.error(f"❌ {kpss_test['interpretation']}")
                    
                    # Общий вывод
                    if test_result['tests_agree'] is not None:
                        if test_result['tests_agree']:
                            st.info(f"ℹ️ {test_result['conclusion']}")
                        else:
                            st.warning(f"⚠️ {test_result['conclusion']}")
                    
                    st.markdown("---")
                
                st.markdown("""
                **Интерпретация тестов:**
                
                **ADF (Augmented Dickey-Fuller):**
                - H₀: ряд имеет единичный корень (нестационарен)
                - H₁: ряд стационарен
                - **p < 0.05** → отвергаем H₀ → ряд **стационарен**
                - **p ≥ 0.05** → не отвергаем H₀ → ряд **нестационарен**
                
                **KPSS (Kwiatkowski-Phillips-Schmidt-Shin):**
                - H₀: ряд стационарен
                - H₁: ряд нестационарен
                - **p > 0.05** → не отвергаем H₀ → ряд **стационарен**
                - **p ≤ 0.05** → отвергаем H₀ → ряд **нестационарен**
                
                ⚠️ **Важно:** Если результаты тестов противоречат друг другу, рекомендуется:
                - Визуальный анализ графиков
                - Проверка данных на выбросы
                - Рассмотрение других типов преобразований
                """)
            
            # Вкладка 4: Дифференцирование
            with tabs[3]:
                st.markdown("### Дифференцирование временного ряда")
                
                st.info("""
                **Дифференцирование** - метод преобразования нестационарного ряда в стационарный 
                путем вычисления разностей между соседними значениями.
                """)
                
                # Выбор порядка для визуализации
                available_orders = list(results['tests'].keys())
                
                if len(available_orders) > 1:
                    viz_order = st.selectbox(
                        "Выберите порядок дифференцирования для визуализации:",
                        available_orders[1:],  # Пропускаем 0 (исходный ряд)
                        format_func=lambda x: f"Порядок {x}"
                    )
                    
                    diff_series = results['tests'][viz_order]['series']
                    
                    # График сравнения
                    fig_diff = create_differencing_comparison(
                        results['series'],
                        diff_series,
                        results['dates'],
                        order=viz_order
                    )
                    
                    st.plotly_chart(fig_diff, use_container_width=True)
                    
                    # Статистика дифференцированного ряда
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Среднее", f"{diff_series.mean():.4f}")
                    with col2:
                        st.metric("Стд. откл.", f"{diff_series.std():.4f}")
                    with col3:
                        st.metric("Минимум", f"{diff_series.min():.4f}")
                    with col4:
                        st.metric("Максимум", f"{diff_series.max():.4f}")
                    
                    st.markdown("""
                    **Формула дифференцирования:**
                    - Порядок 1: `y'(t) = y(t) - y(t-1)`
                    - Порядок 2: `y''(t) = y'(t) - y'(t-1) = [y(t) - y(t-1)] - [y(t-1) - y(t-2)]`
                    
                    **Применение:**
                    - Удаление линейного тренда → дифференцирование 1-го порядка
                    - Удаление квадратичного тренда → дифференцирование 2-го порядка
                    - Сезонное дифференцирование → разность с лагом = период сезонности
                    """)
                else:
                    st.info("Дифференцирование не применялось или не требуется")
    else:
        st.info("👆 Выберите переменную для анализа стационарности")
    
    st.markdown("---")
    
    # Раздел инженерии признаков
    st.header("⚙️ Инженерия признаков")
    
    st.markdown("""
    **Инженерия признаков** - создание новых информативных признаков из существующих данных.
    Лаги и скользящие статистики помогают захватить временную зависимость в данных.
    """)
    
    # Выбор целевой переменной для создания признаков
    fe_target_column = st.selectbox(
        "Выберите целевую переменную:",
        [col for col in df.columns if col != date_column and pd.api.types.is_numeric_dtype(df[col])],
        key="fe_target",
        help="Переменная, для которой будут созданы признаки"
    )
    
    if fe_target_column:
        with st.expander("⚙️ Настройки инженерии признаков", expanded=True):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Лаговые признаки**")
                
                # Выбор лагов для целевой переменной
                st.markdown("*Лаги целевой переменной:*")
                use_lag_1 = st.checkbox("Лаг 1", value=True, key="lag1")
                use_lag_7 = st.checkbox("Лаг 7", value=True, key="lag7")
                use_lag_30 = st.checkbox("Лаг 30", value=True, key="lag30")
                
                custom_lags = st.text_input(
                    "Дополнительные лаги (через запятую):",
                    placeholder="например: 14, 21, 60",
                    help="Введите дополнительные лаги через запятую"
                )
                
                target_lags = []
                if use_lag_1:
                    target_lags.append(1)
                if use_lag_7:
                    target_lags.append(7)
                if use_lag_30:
                    target_lags.append(30)
                
                if custom_lags:
                    try:
                        custom_lags_list = [int(x.strip()) for x in custom_lags.split(',') if x.strip()]
                        target_lags.extend(custom_lags_list)
                        target_lags = sorted(list(set(target_lags)))
                    except:
                        st.warning("⚠️ Некорректный формат дополнительных лагов")
                
                st.info(f"Будет создано лагов для целевой переменной: {len(target_lags)}")
            
            with col2:
                st.markdown("**Скользящие статистики**")
                
                # Выбор окон
                use_window_7 = st.checkbox("Окно 7", value=True, key="win7_fe")
                use_window_30 = st.checkbox("Окно 30", value=True, key="win30_fe")
                
                custom_windows = st.text_input(
                    "Дополнительные окна (через запятую):",
                    placeholder="например: 14, 60, 90",
                    help="Введите дополнительные размеры окон"
                )
                
                rolling_windows = []
                if use_window_7:
                    rolling_windows.append(7)
                if use_window_30:
                    rolling_windows.append(30)
                
                if custom_windows:
                    try:
                        custom_windows_list = [int(x.strip()) for x in custom_windows.split(',') if x.strip()]
                        rolling_windows.extend(custom_windows_list)
                        rolling_windows = sorted(list(set(rolling_windows)))
                    except:
                        st.warning("⚠️ Некорректный формат дополнительных окон")
                
                # Выбор статистик
                st.markdown("*Статистики для расчета:*")
                use_mean = st.checkbox("Среднее (mean)", value=True, key="stat_mean")
                use_std = st.checkbox("Стд. откл. (std)", value=True, key="stat_std")
                use_min = st.checkbox("Минимум (min)", value=False, key="stat_min")
                use_max = st.checkbox("Максимум (max)", value=False, key="stat_max")
                
                rolling_stats = []
                if use_mean:
                    rolling_stats.append('mean')
                if use_std:
                    rolling_stats.append('std')
                if use_min:
                    rolling_stats.append('min')
                if use_max:
                    rolling_stats.append('max')
            
            # Дополнительные признаки для лагов
            st.markdown("**Дополнительные признаки**")
            
            available_features = [col for col in df.columns 
                                 if col not in [date_column, fe_target_column] 
                                 and pd.api.types.is_numeric_dtype(df[col])]
            
            feature_columns = st.multiselect(
                "Выберите признаки для создания лагов:",
                available_features,
                help="Для выбранных признаков будут созданы лаговые признаки"
            )
            
            if feature_columns:
                feature_lags_input = st.text_input(
                    "Лаги для дополнительных признаков (через запятую):",
                    value="1, 7",
                    help="Лаги, которые будут созданы для каждого дополнительного признака"
                )
                
                try:
                    feature_lags = [int(x.strip()) for x in feature_lags_input.split(',') if x.strip()]
                except:
                    feature_lags = [1, 7]
                    st.warning("⚠️ Используются лаги по умолчанию: 1, 7")
            else:
                feature_lags = [1, 7]
            
            # Кнопка создания признаков
            col_btn1, col_btn2 = st.columns([3, 1])
            
            with col_btn1:
                if st.button("🛠️ Создать признаки", type="primary", use_container_width=True):
                    if not target_lags and not rolling_windows:
                        st.error("❌ Выберите хотя бы один лаг или окно для создания признаков")
                    else:
                        with st.spinner("Создание признаков..."):
                            try:
                                # Создаем все признаки
                                df_engineered, fe_info = create_all_features(
                                    df,
                                    target_column=fe_target_column,
                                    feature_columns=feature_columns if feature_columns else None,
                                    target_lags=target_lags,
                                    feature_lags=feature_lags,
                                    rolling_windows=rolling_windows,
                                    rolling_stats=rolling_stats
                                )
                                
                                # Сохраняем результаты
                                st.session_state.engineered_features = df_engineered
                                st.session_state.feature_engineering_info = fe_info
                                
                                st.success(f"✅ Создано признаков: {fe_info['total_features_created']}")
                                
                            except Exception as e:
                                st.error(f"❌ Ошибка при создании признаков: {str(e)}")
                                st.exception(e)
            
            with col_btn2:
                if st.session_state.engineered_features is not None:
                    if st.button("🗑️ Очистить", use_container_width=True):
                        st.session_state.engineered_features = None
                        st.session_state.feature_engineering_info = {}
        
        # Отображение результатов инженерии признаков
        if st.session_state.engineered_features is not None:
            df_engineered = st.session_state.engineered_features
            fe_info = st.session_state.feature_engineering_info
            
            st.markdown("---")
            st.subheader("📊 Результаты инженерии признаков")
            
            # Метрики
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Всего создано признаков", fe_info['total_features_created'])
            with col2:
                st.metric("Лаговых признаков", 
                         fe_info['target_lags']['total_created'] + 
                         sum(v['total_created'] for v in fe_info.get('feature_lags', {}).values()))
            with col3:
                st.metric("Скользящих признаков", 
                         fe_info['target_rolling']['total_created'] +
                         sum(v['total_created'] for v in fe_info.get('feature_rolling', {}).values()))
            with col4:
                original_cols = len(df.columns)
                new_cols = len(df_engineered.columns)
                st.metric("Столбцов в данных", new_cols, delta=new_cols - original_cols)
            
            # Вкладки с результатами
            tabs = st.tabs([
                "📋 Обзор признаков",
                "📊 Корреляции лагов",
                "🎯 Важность признаков",
                "⚠️ Мультиколлинеарность",
                "📈 Визуализация"
            ])
            
            # Вкладка 1: Обзор созданных признаков
            with tabs[0]:
                st.markdown("### Созданные признаки")
                
                # Список всех созданных признаков
                created_features_list = []
                
                # Лаги целевой переменной
                for feat_info in fe_info['target_lags']['created_features']:
                    created_features_list.append(feat_info)
                
                # Скользящие целевой переменной
                for feat_info in fe_info['target_rolling']['created_features']:
                    created_features_list.append(feat_info)
                
                # Лаги дополнительных признаков
                if 'feature_lags' in fe_info:
                    for feature_lags in fe_info['feature_lags'].values():
                        for feat_info in feature_lags['created_features']:
                            created_features_list.append(feat_info)
                
                # Скользящие дополнительных признаков
                if 'feature_rolling' in fe_info:
                    for feature_rolling in fe_info['feature_rolling'].values():
                        for feat_info in feature_rolling['created_features']:
                            created_features_list.append(feat_info)
                
                features_df = pd.DataFrame(created_features_list)
                
                if not features_df.empty:
                    st.dataframe(
                        features_df.style.background_gradient(
                            subset=['missing_values'], 
                            cmap='Reds'
                        ),
                        use_container_width=True
                    )
                    
                    # Статистика по созданным признакам
                    st.markdown("### Статистика новых признаков")
                    
                    new_feature_names = features_df['name'].tolist()
                    stats_df = get_feature_statistics(df_engineered, new_feature_names)
                    
                    st.dataframe(
                        stats_df.style.format({
                            'Mean': '{:.4f}',
                            'Std': '{:.4f}',
                            'Min': '{:.4f}',
                            'Max': '{:.4f}',
                            'Missing_%': '{:.2f}%'
                        }).background_gradient(subset=['Missing_%'], cmap='YlOrRd'),
                        use_container_width=True
                    )
            
            # Вкладка 2: Корреляции лагов
            with tabs[1]:
                st.markdown("### Корреляция лаговых признаков с целевой переменной")
                
                # Получаем все лаговые признаки
                lag_features = [feat['name'] for feat in fe_info['target_lags']['created_features']]
                
                if feature_columns and 'feature_lags' in fe_info:
                    for feature_lags in fe_info['feature_lags'].values():
                        lag_features.extend([feat['name'] for feat in feature_lags['created_features']])
                
                if lag_features:
                    # Расчет корреляций
                    corr_df = calculate_lag_correlations(
                        df_engineered.dropna(subset=[fe_target_column]),
                        fe_target_column,
                        lag_features
                    )
                    
                    if not corr_df.empty:
                        # График
                        fig_lag_corr = create_lag_correlation_plot(corr_df)
                        if fig_lag_corr:
                            st.plotly_chart(fig_lag_corr, use_container_width=True)
                        
                        # Таблица
                        st.markdown("#### Детальная таблица корреляций")
                        st.dataframe(
                            corr_df.style.format({
                                'Pearson_r': '{:.4f}',
                                'Pearson_p': '{:.4f}',
                                'Spearman_r': '{:.4f}',
                                'Spearman_p': '{:.4f}',
                                'Abs_Pearson_r': '{:.4f}'
                            }).background_gradient(subset=['Abs_Pearson_r'], cmap='RdYlGn'),
                            use_container_width=True
                        )
                        
                        # Наиболее информативные лаги
                        st.markdown("#### 💡 Наиболее информативные лаги")
                        top_lags = corr_df.head(5)
                        for idx, row in top_lags.iterrows():
                            significance = "✅" if row['Significant'] == 'Да' else "⚠️"
                            st.write(f"{significance} **{row['Feature']}**: корреляция = {row['Pearson_r']:.4f}, p-value = {row['Pearson_p']:.4f}")
                    else:
                        st.info("Недостаточно данных для расчета корреляций")
                else:
                    st.info("Нет лаговых признаков для анализа")
            
            # Вкладка 3: Важность всех признаков
            with tabs[2]:
                st.markdown("### Важность всех признаков")
                
                all_feature_names = [col for col in df_engineered.columns 
                                    if col not in [date_column, fe_target_column] 
                                    and pd.api.types.is_numeric_dtype(df_engineered[col])]
                
                top_n = st.slider("Количество топ признаков:", 5, 50, 15, key="top_n_features")
                
                importance_df = analyze_feature_importance_correlation(
                    df_engineered,
                    fe_target_column,
                    all_feature_names,
                    top_n=top_n
                )
                
                if not importance_df.empty:
                    # График
                    fig_importance = create_feature_importance_plot(importance_df)
                    if fig_importance:
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Таблица
                    st.dataframe(
                        importance_df.style.format({
                            'Correlation': '{:.4f}',
                            'Abs_Correlation': '{:.4f}',
                            'P_value': '{:.4f}'
                        }).background_gradient(subset=['Abs_Correlation'], cmap='RdYlGn'),
                        use_container_width=True
                    )
                else:
                    st.info("Недостаточно данных для анализа важности")
            
            # Вкладка 4: Мультиколлинеарность
            with tabs[3]:
                st.markdown("### Проверка мультиколлинеарности (VIF)")
                
                st.info("""
                **VIF (Variance Inflation Factor)** - показатель мультиколлинеарности:
                - VIF < 5: низкая мультиколлинеарность ✅
                - 5 ≤ VIF ≤ 10: умеренная мультиколлинеарность ⚠️
                - VIF > 10: высокая мультиколлинеарность ❌
                """)
                
                # Выбор признаков для проверки
                features_to_check = st.multiselect(
                    "Выберите признаки для проверки VIF:",
                    all_feature_names,
                    default=all_feature_names[:min(10, len(all_feature_names))],
                    help="Рекомендуется выбрать не более 10-15 признаков"
                )
                
                if features_to_check and len(features_to_check) >= 2:
                    if st.button("🔍 Рассчитать VIF", key="calc_vif"):
                        with st.spinner("Расчет VIF..."):
                            try:
                                vif_df = check_multicollinearity_vif(
                                    df_engineered,
                                    features_to_check
                                )
                                
                                st.dataframe(
                                    vif_df,
                                    use_container_width=True
                                )
                                
                                # Предупреждения
                                if not vif_df.empty and 'VIF' in vif_df.columns:
                                    high_vif = vif_df[pd.to_numeric(vif_df['VIF'], errors='coerce') > 10]
                                    if not high_vif.empty:
                                        st.warning(f"⚠️ Обнаружено {len(high_vif)} признаков с высокой мультиколлинеарностью (VIF > 10)")
                                        st.dataframe(high_vif, use_container_width=True)
                                        st.markdown("""
                                        **Рекомендации:**
                                        - Удалите один из сильно коррелирующих признаков
                                        - Используйте PCA для снижения размерности
                                        - Примените регуляризацию (Ridge, Lasso) при моделировании
                                        """)
                            except Exception as e:
                                st.error(f"Ошибка расчета VIF: {str(e)}")
                else:
                    st.info("Выберите минимум 2 признака для проверки VIF")
            
            # Вкладка 5: Визуализация
            with tabs[4]:
                st.markdown("### Визуализация скользящих признаков")
                
                # Выбор скользящих признаков для визуализации
                rolling_feature_names = [feat['name'] for feat in fe_info['target_rolling']['created_features']]
                
                if rolling_feature_names:
                    selected_rolling = st.multiselect(
                        "Выберите скользящие признаки для отображения:",
                        rolling_feature_names,
                        default=rolling_feature_names[:min(3, len(rolling_feature_names))]
                    )
                    
                    if selected_rolling:
                        fig_rolling = create_rolling_features_plot(
                            df_engineered,
                            date_column,
                            fe_target_column,
                            selected_rolling
                        )
                        
                        st.plotly_chart(fig_rolling, use_container_width=True)
                    else:
                        st.info("Выберите признаки для визуализации")
                else:
                    st.info("Нет скользящих признаков для визуализации")
                
                # Превью данных
                st.markdown("### Превью данных с новыми признаками")
                
                cols_to_show = [date_column, fe_target_column] + [feat['name'] for feat in created_features_list[:10]]
                cols_to_show = [col for col in cols_to_show if col in df_engineered.columns]
                
                st.dataframe(
                    df_engineered[cols_to_show].head(20),
                    use_container_width=True
                )
                
                # Кнопка экспорта
                csv = df_engineered.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 Скачать данные с признаками (CSV)",
                    data=csv,
                    file_name=f"engineered_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    else:
        st.info("👆 Выберите целевую переменную для инженерии признаков")
    
    st.markdown("---")
    
    # Раздел ACF/PACF анализа
    st.header("📊 ACF/PACF Анализ для ARIMA")
    
    st.markdown("""
    **ACF и PACF** - ключевые инструменты для определения параметров ARIMA моделей.
    Анализ паттернов помогает выбрать оптимальные порядки AR и MA компонент.
    """)
    
    # Выбор переменной для ACF/PACF анализа
    acf_target_column = st.selectbox(
        "Выберите переменную для ACF/PACF анализа:",
        [col for col in df.columns if col != date_column and pd.api.types.is_numeric_dtype(df[col])],
        key="acf_target",
        help="Переменная для анализа автокорреляций"
    )
    
    if acf_target_column:
        with st.expander("⚙️ Параметры ACF/PACF анализа", expanded=True):
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                nlags = st.number_input(
                    "Количество лагов:",
                    min_value=5,
                    max_value=200,
                    value=40,
                    help="Максимальное количество лагов для анализа"
                )
            
            with col2:
                confidence_level = st.slider(
                    "Уровень доверия (%):",
                    min_value=90,
                    max_value=99,
                    value=95,
                    help="Уровень доверия для доверительных интервалов"
                )
                alpha = 1 - (confidence_level / 100)
            
            with col3:
                apply_diff = st.checkbox(
                    "Применить дифференцирование",
                    value=False,
                    help="Применить дифференцирование 1-го порядка перед анализом"
                )
            
            # Кнопка запуска ACF/PACF анализа
            if st.button("📊 Запустить ACF/PACF анализ", type="primary", use_container_width=True):
                with st.spinner("Выполняется ACF/PACF анализ..."):
                    try:
                        # Подготовка данных
                        series = df[acf_target_column].dropna()
                        
                        if len(series) < 10:
                            st.error("❌ Недостаточно данных для анализа (минимум 10 точек)")
                            st.stop()
                        
                        # Применяем дифференцирование если нужно
                        if apply_diff:
                            series = series.diff().dropna()
                            title_suffix = " (после дифференцирования)"
                        else:
                            title_suffix = ""
                        
                        # Комплексный анализ
                        acf_pacf_results = comprehensive_acf_pacf_analysis(
                            series,
                            nlags=nlags,
                            alpha=alpha
                        )
                        
                        # Сохранение результатов
                        st.session_state.acf_pacf_results = acf_pacf_results
                        st.session_state.acf_target_column = acf_target_column
                        st.session_state.acf_title_suffix = title_suffix
                        
                        st.success("✅ ACF/PACF анализ завершён!")
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка при анализе: {str(e)}")
                        st.exception(e)
        
        # Отображение результатов ACF/PACF
        if 'acf_pacf_results' in st.session_state and st.session_state.acf_pacf_results:
            results = st.session_state.acf_pacf_results
            
            if 'error' in results:
                st.error(f"❌ {results['error']}")
            else:
                st.markdown("---")
                st.subheader("📊 Результаты ACF/PACF анализа")
                
                # Вкладки с результатами
                tabs = st.tabs([
                    "📈 Графики ACF/PACF",
                    "🔍 Значимые лаги",
                    "💡 Интерпретация",
                    "🎯 Рекомендации ARIMA"
                ])
                
                # Вкладка 1: Графики
                with tabs[0]:
                    st.markdown("### Графики автокорреляционных функций")
                    
                    # Создаем график
                    fig_acf_pacf = create_acf_pacf_plot(
                        results['acf_pacf_values'],
                        title_suffix=st.session_state.acf_title_suffix
                    )
                    
                    st.plotly_chart(fig_acf_pacf, use_container_width=True)
                    
                    st.markdown("""
                    **Как читать графики:**
                    
                    **ACF (Autocorrelation Function):**
                    - Показывает корреляцию ряда с его лагами (включая косвенные зависимости)
                    - Значения за пределами красных пунктирных линий статистически значимы
                    - **Резкий обрыв** → указывает на MA процесс
                    - **Постепенное затухание** → указывает на AR процесс
                    
                    **PACF (Partial Autocorrelation Function):**
                    - Показывает "чистую" корреляцию с лагом, исключая влияние промежуточных лагов
                    - **Резкий обрыв** → указывает на AR процесс
                    - **Постепенное затухание** → указывает на MA процесс
                    """)
                
                # Вкладка 2: Значимые лаги
                with tabs[1]:
                    st.markdown("### Статистически значимые лаги")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### ACF - Значимые лаги")
                        sig_acf = results['significant_lags_acf']
                        
                        st.metric("Количество значимых лагов", sig_acf['count'])
                        
                        if sig_acf['significant_lags']:
                            sig_acf_df = pd.DataFrame(sig_acf['significant_lags'])
                            st.dataframe(
                                sig_acf_df[['lag', 'value']].style.format({
                                    'value': '{:.4f}'
                                }).background_gradient(subset=['value'], cmap='RdYlGn'),
                                use_container_width=True
                            )
                        else:
                            st.info("Нет статистически значимых лагов в ACF")
                    
                    with col2:
                        st.markdown("#### PACF - Значимые лаги")
                        sig_pacf = results['significant_lags_pacf']
                        
                        st.metric("Количество значимых лагов", sig_pacf['count'])
                        
                        if sig_pacf['significant_lags']:
                            sig_pacf_df = pd.DataFrame(sig_pacf['significant_lags'])
                            st.dataframe(
                                sig_pacf_df[['lag', 'value']].style.format({
                                    'value': '{:.4f}'
                                }).background_gradient(subset=['value'], cmap='RdYlGn'),
                                use_container_width=True
                            )
                        else:
                            st.info("Нет статистически значимых лагов в PACF")
                    
                    st.markdown("""
                    **💡 Значимость лагов:**
                    
                    Лаг считается статистически значимым, если его значение выходит 
                    за доверительный интервал (красные пунктирные линии на графике).
                    
                    - **Положительные значимые лаги** → положительная автокорреляция
                    - **Отрицательные значимые лаги** → отрицательная автокорреляция
                    """)
                
                # Вкладка 3: Интерпретация
                with tabs[2]:
                    st.markdown("### Интерпретация паттернов")
                    
                    acf_interp = results['acf_interpretation']
                    pacf_interp = results['pacf_interpretation']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📈 Интерпретация ACF")
                        
                        st.info(f"**Паттерн:** {acf_interp['pattern']}")
                        
                        if acf_interp['suggested_ma'] is not None:
                            st.success(f"**Предложенный порядок MA:** {acf_interp['suggested_ma']}")
                        else:
                            st.warning("**Порядок MA:** Не определен")
                        
                        st.markdown("**Интерпретация:**")
                        st.write(acf_interp['interpretation'])
                        
                        if acf_interp.get('significant_lags'):
                            st.markdown("**Значимые лаги ACF:**")
                            st.write(', '.join(map(str, acf_interp['significant_lags'][:10])))
                    
                    with col2:
                        st.markdown("#### 📊 Интерпретация PACF")
                        
                        st.info(f"**Паттерн:** {pacf_interp['pattern']}")
                        
                        if pacf_interp['suggested_ar'] is not None:
                            st.success(f"**Предложенный порядок AR:** {pacf_interp['suggested_ar']}")
                        else:
                            st.warning("**Порядок AR:** Не определен")
                        
                        st.markdown("**Интерпретация:**")
                        st.write(pacf_interp['interpretation'])
                        
                        if pacf_interp.get('significant_lags'):
                            st.markdown("**Значимые лаги PACF:**")
                            st.write(', '.join(map(str, pacf_interp['significant_lags'][:10])))
                    
                    st.markdown("---")
                    st.markdown("""
                    ### 📚 Теоретическая справка
                    
                    **Типичные паттерны:**
                    
                    | Процесс | ACF | PACF |
                    |---------|-----|------|
                    | **AR(p)** | Экспоненциальное затухание | Резкий обрыв после лага p |
                    | **MA(q)** | Резкий обрыв после лага q | Экспоненциальное затухание |
                    | **ARMA(p,q)** | Экспоненциальное затухание | Экспоненциальное затухание |
                    
                    **Примеры:**
                    - **AR(1)**: PACF обрывается после лага 1, ACF затухает экспоненциально
                    - **MA(1)**: ACF обрывается после лага 1, PACF затухает экспоненциально
                    - **ARMA(1,1)**: И ACF, и PACF затухают экспоненциально
                    """)
                
                # Вкладка 4: Рекомендации ARIMA
                with tabs[3]:
                    st.markdown("### 🎯 Рекомендации по параметрам ARIMA")
                    
                    arima_sugg = results['arima_suggestions']
                    
                    st.info(arima_sugg['note'])
                    
                    st.markdown("#### Предложенные модели:")
                    
                    for idx, suggestion in enumerate(arima_sugg['primary_suggestions'], 1):
                        with st.container():
                            col_a, col_b, col_c = st.columns([2, 1, 3])
                            
                            with col_a:
                                if suggestion['confidence'] == 'высокая':
                                    st.success(f"**{idx}. {suggestion['model']}**")
                                elif suggestion['confidence'] == 'средняя':
                                    st.info(f"**{idx}. {suggestion['model']}**")
                                else:
                                    st.warning(f"**{idx}. {suggestion['model']}**")
                            
                            with col_b:
                                st.write(f"**Уверенность:** {suggestion['confidence']}")
                            
                            with col_c:
                                st.write(f"*{suggestion['reason']}*")
                            
                            st.markdown(f"**Параметры:** p={suggestion['p']}, q={suggestion['q']}")
                    
                    st.markdown("---")
                    st.markdown("### 📝 Следующие шаги")
                    
                    st.markdown("""
                    1. **Определите порядок дифференцирования (d):**
                       - Используйте тесты стационарности (ADF, KPSS)
                       - Если ряд нестационарен: d=1 или d=2
                       - Если ряд стационарен: d=0
                    
                    2. **Обучите модели с предложенными параметрами**
                    
                    3. **Сравните модели по критериям:**
                       - **AIC** (Akaike Information Criterion) - меньше лучше
                       - **BIC** (Bayesian Information Criterion) - меньше лучше
                       - **RMSE** на тестовой выборке
                    
                    4. **Проверьте остатки модели:**
                       - Остатки должны быть белым шумом
                       - ACF остатков не должен показывать значимых лагов
                       - Тест Льюнга-Бокса для проверки независимости остатков
                    
                    5. **Если модель неудовлетворительна:**
                       - Попробуйте сезонную ARIMA (SARIMA)
                       - Рассмотрите трансформации данных (логарифм, Box-Cox)
                       - Добавьте экзогенные переменные (ARIMAX)
                    """)
                    
                    st.markdown(arima_sugg['recommendation'])
    else:
        st.info("👆 Выберите переменную для ACF/PACF анализа")
    
    st.markdown("---")
    
    # Раздел декомпозиции временного ряда
    st.header("🧩 Декомпозиция временного ряда")
    
    st.markdown("""
    **Декомпозиция** разделяет временной ряд на компоненты для понимания структуры данных:
    - **Тренд** - долгосрочное направление изменения
    - **Сезонность** - повторяющиеся паттерны
    - **Остатки** - случайные колебания
    """)
    
    # Выбор переменной для декомпозиции
    decomp_target_column = st.selectbox(
        "Выберите переменную для декомпозиции:",
        [col for col in df.columns if col != date_column and pd.api.types.is_numeric_dtype(df[col])],
        key="decomp_target",
        help="Переменная для декомпозиции на компоненты"
    )
    
    if decomp_target_column:
        with st.expander("⚙️ Параметры декомпозиции", expanded=True):
            
            col1, col2 = st.columns(2)
            
            with col1:
                decomp_period = st.number_input(
                    "Период сезонности:",
                    min_value=2,
                    max_value=365,
                    value=7,
                    help="Период повторения сезонного паттерна (например, 7 для недельной сезонности)"
                )
            
            with col2:
                decomp_model = st.selectbox(
                    "Модель декомпозиции:",
                    ["additive", "multiplicative"],
                    format_func=lambda x: "Аддитивная (value = trend + seasonal + residual)" if x == "additive" else "Мультипликативная (value = trend × seasonal × residual)",
                    help="Аддитивная: амплитуда сезонности постоянна. Мультипликативная: амплитуда растет со временем"
                )
            
            st.markdown("""
            **💡 Выбор модели:**
            - **Аддитивная**: когда амплитуда сезонных колебаний постоянна
            - **Мультипликативная**: когда амплитуда сезонности растёт вместе с трендом
            """)
            
            # Кнопка запуска декомпозиции
            if st.button("🧩 Запустить декомпозицию", type="primary", use_container_width=True):
                with st.spinner("Выполняется декомпозиция..."):
                    try:
                        # Подготовка данных
                        if date_column:
                            series = df.set_index(date_column)[decomp_target_column]
                        else:
                            series = df[decomp_target_column]
                        
                        # Комплексный анализ декомпозиции
                        decomp_results = comprehensive_decomposition_analysis(
                            series,
                            period=decomp_period,
                            model=decomp_model
                        )
                        
                        if 'error' in decomp_results:
                            st.error(f"❌ {decomp_results['error']}")
                            if 'min_required' in decomp_results:
                                st.info(f"ℹ️ Требуется минимум {decomp_results['min_required']} точек, доступно {decomp_results['available']}")
                            if 'negative_count' in decomp_results:
                                st.info(f"ℹ️ Найдено {decomp_results['negative_count']} неположительных значений")
                        else:
                            # Сохранение результатов
                            st.session_state.decomp_results = decomp_results
                            st.session_state.decomp_target_column = decomp_target_column
                            
                            st.success("✅ Декомпозиция завершена!")
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка при декомпозиции: {str(e)}")
                        st.exception(e)
        
        # Отображение результатов декомпозиции
        if 'decomp_results' in st.session_state and st.session_state.decomp_results:
            results = st.session_state.decomp_results
            
            if 'error' not in results:
                st.markdown("---")
                st.subheader("📊 Результаты декомпозиции")
                
                # Метрики
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Модель", results['decomposition']['model'].capitalize())
                
                with col2:
                    st.metric("Период", results['decomposition']['period'])
                
                with col3:
                    st.metric("Наблюдений", results['decomposition']['n_obs'])
                
                with col4:
                    quality = results['residual_analysis']['quality']
                    quality_emoji = results['residual_analysis']['quality_emoji']
                    st.metric("Качество", f"{quality_emoji} {quality}")
                
                # Вкладки с результатами
                tabs = st.tabs([
                    "📈 Визуализация компонент",
                    "📊 Анализ тренда",
                    "🔄 Анализ сезонности",
                    "🎲 Анализ остатков",
                    "📋 Сводка"
                ])
                
                # Вкладка 1: Визуализация компонент
                with tabs[0]:
                    st.markdown("### Декомпозиция временного ряда")
                    
                    fig_decomp = create_decomposition_plot(results['decomposition'])
                    st.plotly_chart(fig_decomp, use_container_width=True)
                    
                    st.markdown(f"""
                    **Модель декомпозиции:** {results['decomposition']['model'].upper()}
                    
                    - **Observed (Исходный ряд):** Исходные данные
                    - **Trend (Тренд):** Долгосрочное направление изменения
                    - **Seasonal (Сезонность):** Повторяющийся паттерн с периодом {results['decomposition']['period']}
                    - **Residual (Остатки):** Случайные колебания после удаления тренда и сезонности
                    """)
                
                # Вкладка 2: Анализ тренда
                with tabs[1]:
                    st.markdown("### 📊 Детальный анализ тренда")
                    
                    trend_anal = results['trend_analysis']
                    
                    # Основные метрики тренда
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Направление",
                            f"{trend_anal['direction_emoji']} {trend_anal['direction']}"
                        )
                    
                    with col2:
                        st.metric("Сила тренда", trend_anal['strength'])
                    
                    with col3:
                        st.metric("Форма тренда", trend_anal['shape'])
                    
                    with col4:
                        st.metric(
                            "Качество аппроксимации (R²)",
                            f"{trend_anal['best_r2']:.4f}"
                        )
                    
                    # Детальная информация
                    st.markdown("#### Характеристики тренда")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**Значения:**")
                        st.write(f"- Начальное: {trend_anal['start_value']:.2f}")
                        st.write(f"- Конечное: {trend_anal['end_value']:.2f}")
                        st.write(f"- Изменение: {trend_anal['total_change']:.2f} ({trend_anal['total_change_pct']:.2f}%)")
                        st.write(f"- Среднее: {trend_anal['mean']:.2f}")
                        st.write(f"- Ст. отклонение: {trend_anal['std']:.2f}")
                    
                    with col_b:
                        st.markdown("**Качество моделей:**")
                        st.write(f"- Линейная R²: {trend_anal['linear_r2']:.4f}")
                        st.write(f"- Квадратичная R²: {trend_anal['quadratic_r2']:.4f}")
                        st.write(f"- Экспоненциальная R²: {trend_anal['exponential_r2']:.4f}")
                        st.write(f"- Волатильность: {trend_anal['volatility']:.4f}")
                    
                    st.markdown("**Точки перелома:**")
                    st.write(f"- Локальных максимумов: {trend_anal['peaks']}")
                    st.write(f"- Локальных минимумов: {trend_anal['troughs']}")
                    st.write(f"- Всего точек изменения: {trend_anal['turning_points']}")
                    
                    st.markdown("""
                    **💡 Интерпретация:**
                    
                    - **Линейный тренд (R² > 0.95):** Стабильный рост/падение с постоянной скоростью
                    - **Квадратичный тренд:** Ускоряющийся или замедляющийся рост
                    - **Экспоненциальный тренд:** Взрывной рост (характерно для вирусных процессов)
                    - **Сложный тренд:** Несколько фаз роста/падения, требует дополнительного анализа
                    """)
                
                # Вкладка 3: Анализ сезонности
                with tabs[2]:
                    st.markdown("### 🔄 Детальный анализ сезонности")
                    
                    seasonal_anal = results['seasonal_analysis']
                    
                    # Основные метрики сезонности
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Период", seasonal_anal['period'])
                    
                    with col2:
                        st.metric("Периодичность", seasonal_anal['periodicity'])
                    
                    with col3:
                        st.metric("Амплитуда", f"{seasonal_anal['amplitude']:.4f}")
                    
                    with col4:
                        st.metric("Стабильность", seasonal_anal['stability'])
                    
                    # Паттерн сезонности
                    st.markdown("#### Сезонный паттерн (один период)")
                    
                    fig_seasonal = create_seasonal_pattern_plot(
                        seasonal_anal['pattern'],
                        seasonal_anal['period']
                    )
                    st.plotly_chart(fig_seasonal, use_container_width=True)
                    
                    # Детальная информация
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**Характеристики паттерна:**")
                        st.write(f"- Минимум: {seasonal_anal['min_value']:.4f}")
                        st.write(f"- Максимум: {seasonal_anal['max_value']:.4f}")
                        st.write(f"- Размах: {seasonal_anal['range']:.4f}")
                        st.write(f"- Среднее: {seasonal_anal['mean']:.4f}")
                        st.write(f"- Ст. отклонение: {seasonal_anal['std']:.4f}")
                    
                    with col_b:
                        st.markdown("**Структура паттерна:**")
                        st.write(f"- Количество пиков: {seasonal_anal['num_peaks']}")
                        st.write(f"- Количество спадов: {seasonal_anal['num_troughs']}")
                        st.write(f"- Полных периодов: {seasonal_anal['num_periods']}")
                        if seasonal_anal['avg_correlation'] is not None:
                            st.write(f"- Корреляция периодов: {seasonal_anal['avg_correlation']:.4f}")
                    
                    st.markdown("""
                    **💡 Интерпретация:**
                    
                    - **Высокая стабильность (корр > 0.9):** Сезонность предсказуема и постоянна
                    - **Средняя стабильность (0.7 < корр < 0.9):** Есть вариации в сезонном паттерне
                    - **Низкая стабильность (корр < 0.7):** Сезонность нестабильна или слаба
                    - **Сила сезонности:** Отношение амплитуды к среднему значению
                    """)
                
                # Вкладка 4: Анализ остатков
                with tabs[3]:
                    st.markdown("### 🎲 Детальный анализ остатков")
                    
                    resid_anal = results['residual_analysis']
                    
                    # Основные метрики остатков
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Качество декомпозиции",
                            f"{resid_anal['quality_emoji']} {resid_anal['quality']}"
                        )
                    
                    with col2:
                        score_val = resid_anal['quality_score']
                        score_str = f"{score_val:.0f}%" if score_val is not None else "N/A"
                        st.metric("Оценка", score_str)
                    
                    with col3:
                        st.metric("Среднее остатков", f"{resid_anal['mean']:.4f}")
                    
                    with col4:
                        st.metric("Ст. отклонение", f"{resid_anal['std']:.4f}")
                    
                    # График анализа остатков
                    st.markdown("#### Визуальный анализ остатков")
                    
                    fig_resid = create_residuals_analysis_plot(results['decomposition']['resid'])
                    st.plotly_chart(fig_resid, use_container_width=True)
                    
                    # Тесты качества
                    st.markdown("#### Проверки качества остатков")
                    
                    checks = resid_anal['quality_checks']
                    
                    check_data = []
                    
                    if checks['mean_near_zero'] is not None:
                        check_data.append({
                            'Проверка': 'Среднее ≈ 0',
                            'Результат': '✅ Да' if checks['mean_near_zero'] else '❌ Нет',
                            'Значение': f"{resid_anal['mean']:.4f}"
                        })
                    
                    if checks['normally_distributed'] is not None:
                        check_data.append({
                            'Проверка': 'Нормальное распределение',
                            'Результат': '✅ Да' if checks['normally_distributed'] else '❌ Нет',
                            'Значение': f"p-value = {resid_anal['shapiro_p']:.4f}" if resid_anal['shapiro_p'] else "N/A"
                        })
                    
                    if checks['no_autocorrelation'] is not None:
                        check_data.append({
                            'Проверка': 'Нет автокорреляции',
                            'Результат': '✅ Да' if checks['no_autocorrelation'] else '❌ Нет',
                            'Значение': f"p-value = {resid_anal['ljung_box_p']:.4f}" if resid_anal['ljung_box_p'] else "N/A"
                        })
                    
                    if checks['constant_variance'] is not None:
                        check_data.append({
                            'Проверка': 'Постоянная дисперсия',
                            'Результат': '✅ Да' if checks['constant_variance'] else '❌ Нет',
                            'Значение': 'Гомоскедастичность' if checks['constant_variance'] else 'Гетероскедастичность'
                        })
                    
                    if check_data:
                        checks_df = pd.DataFrame(check_data)
                        st.dataframe(checks_df, use_container_width=True, hide_index=True)
                    
                    # Выбросы
                    st.markdown("**Выбросы в остатках:**")
                    st.write(f"- Количество: {resid_anal['outlier_count']}")
                    st.write(f"- Процент: {resid_anal['outlier_pct']:.2f}%")
                    
                    st.markdown("""
                    **💡 Интерпретация остатков:**
                    
                    **Идеальные остатки (белый шум):**
                    - ✅ Среднее близко к нулю
                    - ✅ Нормальное распределение (тест Шапиро-Уилка, p > 0.05)
                    - ✅ Нет автокорреляции (тест Льюнга-Бокса, p > 0.05)
                    - ✅ Постоянная дисперсия (гомоскедастичность)
                    
                    **Если проверки не прошли:**
                    - ❌ Среднее ≠ 0 → декомпозиция неполная, есть систематическое смещение
                    - ❌ Не нормальные → возможны выбросы или пропущена нелинейность
                    - ❌ Есть автокорреляция → в остатках есть структура, декомпозиция неполная
                    - ❌ Нестабильная дисперсия → рассмотрите мультипликативную модель или трансформацию
                    
                    **Рекомендации при плохих остатках:**
                    1. Попробуйте другой тип декомпозиции (аддитивную/мультипликативную)
                    2. Измените период сезонности
                    3. Рассмотрите STL декомпозицию (более гибкая)
                    4. Примените трансформации данных (логарифм, Box-Cox)
                    """)
                
                # Вкладка 5: Сводка
                with tabs[4]:
                    st.markdown("### 📋 Сводка декомпозиции")
                    
                    st.markdown(f"""
                    **Модель:** {results['decomposition']['model'].upper()}  
                    **Период сезонности:** {results['decomposition']['period']}  
                    **Наблюдений:** {results['decomposition']['n_obs']}
                    """)
                    
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("#### 📈 Тренд")
                        trend_anal = results['trend_analysis']
                        st.write(f"**Направление:** {trend_anal['direction_emoji']} {trend_anal['direction']}")
                        st.write(f"**Сила:** {trend_anal['strength']}")
                        st.write(f"**Форма:** {trend_anal['shape']}")
                        st.write(f"**Изменение:** {trend_anal['total_change_pct']:.2f}%")
                    
                    with col2:
                        st.markdown("#### 🔄 Сезонность")
                        seasonal_anal = results['seasonal_analysis']
                        st.write(f"**Период:** {seasonal_anal['period']}")
                        st.write(f"**Тип:** {seasonal_anal['periodicity']}")
                        st.write(f"**Амплитуда:** {seasonal_anal['amplitude']:.4f}")
                        st.write(f"**Стабильность:** {seasonal_anal['stability']}")
                    
                    with col3:
                        st.markdown("#### 🎲 Остатки")
                        resid_anal = results['residual_analysis']
                        st.write(f"**Качество:** {resid_anal['quality_emoji']} {resid_anal['quality']}")
                        score = resid_anal['quality_score']
                        if score is not None:
                            st.write(f"**Оценка:** {score:.0f}%")
                        st.write(f"**Среднее:** {resid_anal['mean']:.4f}")
                        st.write(f"**Выбросов:** {resid_anal['outlier_pct']:.2f}%")
                    
                    st.markdown("---")
                    
                    st.markdown("### 🎯 Выводы и рекомендации")
                    
                    # Автоматические выводы
                    conclusions = []
                    
                    # По тренду
                    if trend_anal['strength'] in ['Сильный', 'Умеренный']:
                        conclusions.append(f"✅ Обнаружен {trend_anal['direction'].lower()} тренд ({trend_anal['strength'].lower()}). Необходимо учитывать при прогнозировании.")
                    else:
                        conclusions.append("ℹ️ Тренд слабый или отсутствует. Ряд относительно стабилен.")
                    
                    # По сезонности
                    if seasonal_anal['stability'] == 'Высокая':
                        conclusions.append(f"✅ Сезонность стабильна с периодом {seasonal_anal['period']}. Паттерн предсказуем.")
                    elif seasonal_anal['stability'] == 'Средняя':
                        conclusions.append(f"⚠️ Сезонность умеренно стабильна. Возможны вариации в паттерне.")
                    else:
                        conclusions.append("⚠️ Сезонность нестабильна или слаба.")
                    
                    # По остаткам
                    if resid_anal['quality'] in ['Отличная', 'Хорошая']:
                        conclusions.append("✅ Остатки близки к белому шуму. Декомпозиция качественная.")
                    elif resid_anal['quality'] == 'Удовлетворительная':
                        conclusions.append("⚠️ Остатки удовлетворительные, но есть незначительные проблемы.")
                    else:
                        conclusions.append("❌ Остатки имеют структуру. Рекомендуется улучшить декомпозицию.")
                    
                    for conclusion in conclusions:
                        st.markdown(conclusion)
                    
                    st.markdown("---")
                    
                    st.markdown("""
                    **📚 Дополнительные шаги:**
                    
                    1. **Для прогнозирования:**
                       - Используйте обнаруженный тренд для долгосрочных прогнозов
                       - Учитывайте сезонность при краткосрочном прогнозировании
                       - Модели: ARIMA, SARIMA, Prophet, ExponentialSmoothing
                    
                    2. **Если декомпозиция неудовлетворительна:**
                       - Попробуйте другой период сезонности
                       - Переключитесь между аддитивной/мультипликативной моделью
                       - Рассмотрите STL декомпозицию (более гибкая)
                       - Примените трансформации (логарифм для стабилизации дисперсии)
                    
                    3. **Для анализа:**
                       - Изучите причины тренда (внешние факторы, рост рынка)
                       - Определите источники сезонности (календарь, погода, события)
                       - Исследуйте выбросы в остатках (аномалии, события)
                    """)
    else:
        st.info("👆 Выберите переменную для декомпозиции")
    
    st.markdown("---")
    
    # Параметры анализа временных рядов
    st.header("🔧 Параметры анализа временных рядов")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        seasonal_period = st.number_input(
            "Период сезонности:",
            min_value=2,
            max_value=365,
            value=7,
            help="Количество точек в одном сезонном цикле"
        )
    
    with col2:
        max_lags = st.number_input(
            "Количество лагов (ACF/PACF):",
            min_value=5,
            max_value=100,
            value=40,
            help="Максимальное количество лагов для автокорреляции"
        )
    
    with col3:
        rolling_window = st.number_input(
            "Окно скользящего среднего:",
            min_value=2,
            max_value=100,
            value=7,
            help="Размер окна для скользящего среднего"
        )
    
    with col4:
        decomposition_model = st.selectbox(
            "Модель декомпозиции:",
            ["additive", "multiplicative"],
            format_func=lambda x: "Аддитивная" if x == "additive" else "Мультипликативная",
            help="Тип декомпозиции временного ряда"
        )
    
    # Кнопка запуска анализа
    if st.button("🚀 Запустить анализ", type="primary", use_container_width=True):
        with st.spinner("Выполняется анализ..."):
            try:
                # Подготовка данных
                ts_data = df[[date_column, target_column]].copy()
                ts_data.columns = ['date', 'value']
                ts_data = ts_data.dropna()
                
                # Скользящее среднее и тренд
                rolling_stats = calculate_rolling_stats(ts_data['value'], rolling_window)
                
                # Декомпозиция (нужна серия с datetime индексом)
                ts_series = ts_data.set_index('date')['value']
                decomposition = perform_decomposition(
                    ts_series, 
                    period=seasonal_period,
                    model=decomposition_model
                )
                
                # Проверка на ошибку декомпозиции
                if 'error' in decomposition:
                    raise ValueError(f"Ошибка декомпозиции: {decomposition['error']}")
                
                # ACF и PACF
                acf_values, pacf_values, acf_confint, pacf_confint = calculate_acf_pacf(
                    ts_data['value'],
                    max_lags=max_lags
                )
                
                # Тесты на стационарность
                adf_result, kpss_result = test_stationarity(ts_data['value'])
                
                # Корреляции
                correlation_matrix = None
                if feature_columns:
                    corr_data = df[[target_column] + feature_columns].copy()
                    correlation_matrix = calculate_correlations(corr_data)
                
                # Сохранение результатов
                st.session_state.analysis_results = {
                    'ts_data': ts_data,
                    'rolling_stats': rolling_stats,
                    'decomposition': decomposition,
                    'acf': acf_values,
                    'pacf': pacf_values,
                    'acf_confint': acf_confint,
                    'pacf_confint': pacf_confint,
                    'adf_result': adf_result,
                    'kpss_result': kpss_result,
                    'correlation_matrix': correlation_matrix,
                    'params': {
                        'target_column': target_column,
                        'seasonal_period': seasonal_period,
                        'max_lags': max_lags,
                        'rolling_window': rolling_window,
                        'decomposition_model': decomposition_model
                    }
                }
                
                st.success("✅ Анализ успешно выполнен!")
                
            except Exception as e:
                st.error(f"❌ Ошибка при выполнении анализа: {str(e)}")
                st.exception(e)
    
    # Отображение результатов
    if st.session_state.analysis_results:
        st.markdown("---")
        st.header("📈 Результаты анализа")
        
        results = st.session_state.analysis_results
        ts_data = results['ts_data']
        
        # 1. График временного ряда с трендом
        st.subheader("1. Временной ряд с трендом и скользящим средним")
        fig = go.Figure(layout=go.Layout(template=None))
        
        fig.add_trace(go.Scatter(
            x=ts_data['date'],
            y=ts_data['value'],
            mode='lines',
            name='Исходный ряд',
            line=dict(color='lightblue', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=ts_data['date'],
            y=results['rolling_stats']['rolling_mean'],
            mode='lines',
            name=f'Скользящее среднее ({results["params"]["rolling_window"]})',
            line=dict(color='orange', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=ts_data['date'],
            y=results['rolling_stats']['rolling_std'],
            mode='lines',
            name=f'Скользящее стд. откл. ({results["params"]["rolling_window"]})',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"Временной ряд: {results['params']['target_column']}",
            xaxis_title="Дата",
            yaxis_title="Значение",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. Корреляционная матрица
        if results['correlation_matrix'] is not None:
            st.subheader("2. Корреляционная матрица")
            fig = go.Figure(
                data=go.Heatmap(
                    z=results['correlation_matrix'].values,
                    x=results['correlation_matrix'].columns,
                    y=results['correlation_matrix'].columns,
                    colorscale='RdBu_r',
                    zmid=0,
                    text=np.round(results['correlation_matrix'].values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="Корреляция")
                ),
                layout=go.Layout(template=None)
            )
            
            fig.update_layout(
                title="Тепловая карта корреляций",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 3. ACF и PACF
        st.subheader("3. Автокорреляционная функция (ACF) и частичная автокорреляция (PACF)")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('ACF', 'PACF')
        )
        
        # ACF
        lags = np.arange(len(results['acf']))
        fig.add_trace(
            go.Bar(x=lags, y=results['acf'], name='ACF', marker_color='steelblue'),
            row=1, col=1
        )
        
        # Доверительные интервалы для ACF
        upper_bound = results['acf_confint'][:, 1] - results['acf']
        lower_bound = results['acf'] - results['acf_confint'][:, 0]
        
        fig.add_trace(
            go.Scatter(
                x=lags, y=results['acf_confint'][:, 1],
                mode='lines', line=dict(color='red', dash='dash'),
                showlegend=False, name='Верхняя граница'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=lags, y=results['acf_confint'][:, 0],
                mode='lines', line=dict(color='red', dash='dash'),
                showlegend=False, name='Нижняя граница'
            ),
            row=1, col=1
        )
        
        # PACF
        lags = np.arange(len(results['pacf']))
        fig.add_trace(
            go.Bar(x=lags, y=results['pacf'], name='PACF', marker_color='darkorange'),
            row=1, col=2
        )
        
        # Доверительные интервалы для PACF
        fig.add_trace(
            go.Scatter(
                x=lags, y=results['pacf_confint'][:, 1],
                mode='lines', line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=lags, y=results['pacf_confint'][:, 0],
                mode='lines', line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Лаг", row=1, col=1)
        fig.update_xaxes(title_text="Лаг", row=1, col=2)
        fig.update_yaxes(title_text="Корреляция", row=1, col=1)
        fig.update_yaxes(title_text="Частичная корреляция", row=1, col=2)
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. Декомпозиция
        st.subheader("4. Декомпозиция временного ряда")
        
        decomp = results['decomposition']
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Исходный ряд', 'Тренд', 'Сезонность', 'Остатки'),
            vertical_spacing=0.08
        )
        
        # Исходный ряд
        fig.add_trace(
            go.Scatter(x=decomp['observed'].index, y=decomp['observed'].values, name='Исходный', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Тренд
        fig.add_trace(
            go.Scatter(x=decomp['trend'].index, y=decomp['trend'].values, name='Тренд', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Сезонность
        fig.add_trace(
            go.Scatter(x=decomp['seasonal'].index, y=decomp['seasonal'].values, name='Сезонность', line=dict(color='green')),
            row=3, col=1
        )
        
        # Остатки
        fig.add_trace(
            go.Scatter(x=decomp['resid'].index, y=decomp['resid'].values, name='Остатки', line=dict(color='red')),
            row=4, col=1
        )
        
        fig.update_layout(height=800, showlegend=False)
        fig.update_xaxes(title_text="Дата", row=4, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 5. Тесты на стационарность
        st.subheader("5. Тесты на стационарность")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Расширенный тест Дики-Фуллера (ADF)**")
            adf = results['adf_result']
            st.write(f"- ADF-статистика: {adf['adf_stat']:.4f}")
            st.write(f"- p-значение: {adf['p_value']:.4f}")
            st.write("- Критические значения:")
            for key, value in adf['critical_values'].items():
                st.write(f"  - {key}: {value:.4f}")
            
            if adf['p_value'] < 0.05:
                st.success("✅ Ряд стационарный (p < 0.05)")
            else:
                st.warning("⚠️ Ряд нестационарный (p >= 0.05)")
        
        with col2:
            st.write("**Тест Квятковского-Филлипса-Шмидта-Шина (KPSS)**")
            kpss = results['kpss_result']
            st.write(f"- KPSS-статистика: {kpss['kpss_stat']:.4f}")
            st.write(f"- p-значение: {kpss['p_value']:.4f}")
            st.write("- Критические значения:")
            for key, value in kpss['critical_values'].items():
                st.write(f"  - {key}: {value:.4f}")
            
            if kpss['p_value'] > 0.05:
                st.success("✅ Ряд стационарный (p > 0.05)")
            else:
                st.warning("⚠️ Ряд нестационарный (p <= 0.05)")
        
        # Экспорт отчёта
        st.markdown("---")
        st.header("📥 Экспорт отчёта")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Сгенерировать HTML-отчёт", use_container_width=True):
                with st.spinner("Генерация отчёта..."):
                    try:
                        html_report = generate_html_report(results)
                        
                        # Создание кнопки скачивания
                        st.download_button(
                            label="⬇️ Скачать HTML-отчёт",
                            data=html_report,
                            file_name=f"time_series_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html",
                            use_container_width=True
                        )
                        
                        st.success("✅ HTML-отчёт успешно сгенерирован!")
                    except Exception as e:
                        st.error(f"❌ Ошибка генерации отчёта: {str(e)}")
        
        with col2:
            st.info("📝 PDF экспорт будет доступен в следующей версии")

else:
    # Стартовая страница
    st.info("👈 Загрузите данные через боковую панель для начала работы")
    
    st.markdown("""
    ### 🎯 Возможности приложения:
    
    - 📊 **Загрузка данных**: CSV, Parquet
    - 📈 **Визуализация**: интерактивные графики временных рядов
    - 🔍 **Анализ**: декомпозиция, ACF/PACF, тесты на стационарность
    - 📉 **Корреляции**: тепловая карта взаимосвязей признаков
    - 📥 **Экспорт**: генерация HTML-отчётов
    
    ### 🚀 Начните с:
    1. Загрузки своих данных
    2. Выбора целевой переменной и признаков
    3. Настройки параметров анализа
    4. Запуска анализа и просмотра результатов
    """)

# Футер
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Разработано с использованием Streamlit и Plotly | © 2025
    </div>
    """,
    unsafe_allow_html=True
)