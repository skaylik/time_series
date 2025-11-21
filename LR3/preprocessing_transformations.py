"""
Модуль для предобработки и преобразований временных рядов (Этап 1).
Выполняет сезонную декомпозицию, анализ лог/Box–Cox трансформаций и дифференцирований,
проверку стационарности (ADF, KPSS тесты) для подготовки данных к моделированию.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf

from decomposition_analysis import DecompositionAnalyzer


def _visualize_decomposition(decomp, title: str = "Декомпозиция временного ряда") -> None:
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=("Исходный ряд", "Тренд", "Сезонность", "Остатки"),
        vertical_spacing=0.08,
        row_heights=[0.3, 0.3, 0.2, 0.2],
    )

    fig.add_trace(
        go.Scatter(
            x=decomp.observed.index,
            y=decomp.observed.values,
            mode="lines",
            name="Исходный ряд",
            line=dict(color="blue", width=1),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=decomp.trend.index,
            y=decomp.trend.values,
            mode="lines",
            name="Тренд",
            line=dict(color="green", width=2),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=decomp.seasonal.index,
            y=decomp.seasonal.values,
            mode="lines",
            name="Сезонность",
            line=dict(color="orange", width=1),
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=decomp.resid.index,
            y=decomp.resid.values,
            mode="lines",
            name="Остатки",
            line=dict(color="red", width=1),
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        height=1000,
        title_text=title,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Дата", row=4, col=1)
    fig.update_yaxes(title_text="Значение", row=1, col=1)
    fig.update_yaxes(title_text="Тренд", row=2, col=1)
    fig.update_yaxes(title_text="Сезонность", row=3, col=1)
    fig.update_yaxes(title_text="Остатки", row=4, col=1)
    st.plotly_chart(fig, use_container_width=True)


def _display_residual_analysis(residual_analysis: Dict[str, Any], residuals: pd.Series) -> None:
    st.markdown("#### 🔄 Проверка стационарности остатков")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Тест ADF (Augmented Dickey-Fuller)**")
        adf = residual_analysis.get("stationarity", {}).get("adf")
        if adf is not None:
            if "error" in adf:
                st.error(f"Ошибка: {adf['error']}")
            else:
                # Красивое отображение статистики
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    st.metric("Статистика", f"{adf['statistic']:.4f}")
                with stat_col2:
                    p_value = adf['pvalue']
                    st.metric("p-value", f"{p_value:.4f}")
                
                if adf.get("is_stationary", False):
                    st.success("✅ Ряд стационарен (p < 0.05)")
                else:
                    st.warning("⚠️ Ряд нестационарен (p ≥ 0.05)")
                
                if "critical_values" in adf and adf["critical_values"]:
                    st.markdown("**Критические значения:**")
                    crit_items = list(adf["critical_values"].items())
                    if crit_items:
                        # Разбиваем на колонки в зависимости от количества значений
                        n_cols = min(3, len(crit_items))
                        crit_cols = st.columns(n_cols)
                        for idx, (level, value) in enumerate(crit_items):
                            col_idx = idx % n_cols
                            with crit_cols[col_idx]:
                                # level уже в формате "1%", "5%", "10%"
                                # Используем его напрямую для заголовка метрики
                                level_str = str(level)
                                st.metric(level_str, f"{value:.4f}")

    with col2:
        st.markdown("**Тест KPSS (Kwiatkowski-Phillips-Schmidt-Shin)**")
        kpss = residual_analysis.get("stationarity", {}).get("kpss")
        if kpss is not None:
            if "error" in kpss:
                st.error(f"Ошибка: {kpss['error']}")
            else:
                # Красивое отображение статистики
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    st.metric("Статистика", f"{kpss['statistic']:.4f}")
                with stat_col2:
                    p_value = kpss['pvalue']
                    st.metric("p-value", f"{p_value:.4f}")
                
                if kpss.get("is_stationary", False):
                    st.success("✅ Ряд стационарен (p > 0.05)")
                else:
                    st.warning("⚠️ Ряд нестационарен (p ≤ 0.05)")

    st.markdown("---")
    st.markdown("#### 📊 Проверка нормальности остатков")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Тест Д'Агостино-Пирсона**")
        da = residual_analysis.get("normality", {}).get("d_agostino")
        if da is not None:
            if "error" in da:
                st.error(f"Ошибка: {da['error']}")
            else:
                # Красивое отображение статистики
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    st.metric("Статистика", f"{da['statistic']:.4f}")
                with stat_col2:
                    p_value = da['pvalue']
                    st.metric("p-value", f"{p_value:.4f}")
                
                if da.get("is_normal", False):
                    st.success("✅ Распределение нормально (p > 0.05)")
                else:
                    st.warning("⚠️ Распределение ненормально (p ≤ 0.05)")

    with col2:
        st.markdown("**Тест Жарке-Бера**")
        jb = residual_analysis.get("normality", {}).get("jarque_bera")
        if jb is not None:
            if "error" in jb:
                st.error(f"Ошибка: {jb['error']}")
            else:
                # Красивое отображение статистики
                stat_col1, stat_col2 = st.columns(2)
                with stat_col1:
                    st.metric("Статистика", f"{jb['statistic']:.4f}")
                with stat_col2:
                    p_value = jb['pvalue']
                    st.metric("p-value", f"{p_value:.4f}")
                
                if jb.get("is_normal", False):
                    st.success("✅ Распределение нормально (p > 0.05)")
                else:
                    st.warning("⚠️ Распределение ненормально (p ≤ 0.05)")

    desc = residual_analysis.get("normality", {}).get("descriptive")
    if desc:
        st.markdown("**Описательная статистика остатков:**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Среднее", f"{desc['mean']:.4f}")
        col2.metric("Ст. отклонение", f"{desc['std']:.4f}")
        col3.metric("Асимметрия", f"{desc['skewness']:.4f}")
        col4.metric("Эксцесс", f"{desc['kurtosis']:.4f}")

    st.markdown("---")
    st.markdown("#### 📈 Визуализация остатков")
    col1, col2 = st.columns(2)

    with col1:
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=residuals.dropna().values,
                nbinsx=50,
                name="Остатки",
                marker_color="blue",
            )
        )
        fig_hist.update_layout(
            title="Гистограмма остатков",
            xaxis_title="Остатки",
            yaxis_title="Частота",
            height=400,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        residuals_clean = residuals.dropna()
        qq_data = stats.probplot(residuals_clean, dist="norm")
        fig_qq = go.Figure()
        fig_qq.add_trace(
            go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode="markers",
                name="Остатки",
                marker=dict(color="blue", size=4),
            )
        )
        fig_qq.add_trace(
            go.Scatter(
                x=qq_data[0][0],
                y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                mode="lines",
                name="Теоретическая линия",
                line=dict(color="red", width=2),
            )
        )
        fig_qq.update_layout(
            title="Q-Q Plot (проверка нормальности)",
            xaxis_title="Теоретические квантили",
            yaxis_title="Выборочные квантили",
            height=400,
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔄 ACF и PACF остатков")
    col1, col2 = st.columns(2)

    with col1:
        acf_values = acf(residuals.dropna(), nlags=40, fft=True)
        fig_acf = go.Figure()
        fig_acf.add_trace(
            go.Bar(
                x=list(range(len(acf_values))),
                y=acf_values,
                name="ACF",
                marker_color="blue",
            )
        )
        conf_int = 1.96 / np.sqrt(len(residuals.dropna()))
        fig_acf.add_hline(y=conf_int, line_dash="dash", line_color="red", annotation_text="95% доверительный интервал")
        fig_acf.add_hline(y=-conf_int, line_dash="dash", line_color="red")
        fig_acf.update_layout(
            title="ACF остатков",
            xaxis_title="Лаг",
            yaxis_title="ACF",
            height=400,
        )
        st.plotly_chart(fig_acf, use_container_width=True)

    with col2:
        pacf_values = pacf(residuals.dropna(), nlags=40)
        fig_pacf = go.Figure()
        fig_pacf.add_trace(
            go.Bar(
                x=list(range(len(pacf_values))),
                y=pacf_values,
                name="PACF",
                marker_color="green",
            )
        )
        conf_int = 1.96 / np.sqrt(len(residuals.dropna()))
        fig_pacf.add_hline(y=conf_int, line_dash="dash", line_color="red", annotation_text="95% доверительный интервал")
        fig_pacf.add_hline(y=-conf_int, line_dash="dash", line_color="red")
        fig_pacf.update_layout(
            title="PACF остатков",
            xaxis_title="Лаг",
            yaxis_title="PACF",
            height=400,
        )
        st.plotly_chart(fig_pacf, use_container_width=True)

    if "autocorrelation" in residual_analysis:
        st.markdown("---")
        st.markdown("#### 🔗 Проверка автокорреляции остатков")
        lb = residual_analysis["autocorrelation"].get("ljung_box")
        if lb and "has_autocorrelation" in lb:
            if lb["has_autocorrelation"]:
                st.warning("⚠️ Обнаружена автокорреляция в остатках")
            else:
                st.success("✅ Автокорреляция в остатках отсутствует")


def _display_comparison_table(comparisons: Dict[str, Dict[str, Any]]) -> None:
    comparison_data = []
    for value in comparisons.values():
        if "error" not in value:
            comparison_data.append(
                {
                    "Модель": value["model"],
                    "Период": value["period"],
                    "Оценка": f"{value['score']:.2f}",
                    "Среднее остатков": f"{value['residual_stats']['mean']:.4f}",
                    "Ст. отклонение остатков": f"{value['residual_stats']['std']:.4f}",
                    "Дисперсия остатков": f"{value['residual_stats']['variance']:.4f}",
                }
            )

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        numeric_cols = ["Оценка", "Среднее остатков", "Ст. отклонение остатков", "Дисперсия остатков"]
        for col in numeric_cols:
            comparison_df[col] = pd.to_numeric(comparison_df[col], errors="coerce")
        comparison_df = comparison_df.sort_values("Оценка", ascending=False)

        st.dataframe(comparison_df, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[f"{row['Модель']}_{row['Период']}" for _, row in comparison_df.iterrows()],
                y=comparison_df["Оценка"].astype(float),
                marker_color="steelblue",
            )
        )
        fig.update_layout(
            title="Сравнение оценок качества декомпозиций",
            xaxis_title="Вариант декомпозиции",
            yaxis_title="Оценка качества",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Нет данных для сравнения")


def preprocessing_transformations(
    df: pd.DataFrame,
    analysis_data: Optional[Dict[str, Any]],
    lab_state: Dict[str, bool],
    alpha: float,
) -> Dict[str, Any]:
    if analysis_data is None:
        analysis_data = {}

    date_columns_for_metrics = [
        col for col in df.columns if df[col].dtype == "object" or pd.api.types.is_datetime64_any_dtype(df[col])
    ]
    date_col_for_metrics = date_columns_for_metrics[0] if date_columns_for_metrics else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Строк", f"{df.shape[0]:,}")
    col2.metric("📑 Столбцов", f"{df.shape[1]:,}")
    date_start = str(df[date_col_for_metrics].min())[:10] if date_col_for_metrics else "N/A"
    date_end = str(df[date_col_for_metrics].max())[:10] if date_col_for_metrics else "N/A"
    col3.metric("📅 Дата начала", date_start)
    col4.metric("📅 Дата конца", date_end)

    col1, col2 = st.columns(2)
    with col1:
        date_columns = [
            col for col in df.columns if df[col].dtype == "object" or pd.api.types.is_datetime64_any_dtype(df[col])
        ]
        if not date_columns:
            st.error("❌ Не найден столбец с датами. Пожалуйста, проверьте данные.")
            lab_state["stage1_completed"] = False
            return analysis_data
        date_column = st.selectbox(
            "📅 Выберите столбец с датой",
            date_columns,
            index=0,
            help="Выберите столбец, содержащий даты временного ряда",
        )
    with col2:
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_columns:
            st.error("❌ Не найдены числовые столбцы. Пожалуйста, проверьте данные.")
            lab_state["stage1_completed"] = False
            return analysis_data
        value_column = st.selectbox(
            "📈 Выберите переменную для временного ряда",
            numeric_columns,
            index=0,
            help="Выберите числовой столбец, для которого будет выполнен анализ временного ряда",
        )

    st.markdown("---")
    with st.expander("📋 Просмотр данных и статистика", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        col1, col2 = st.columns(2)
        col1.write(f"**📅 Диапазон дат:** {df[date_column].min()} - {df[date_column].max()}")
        col2.write(f"**📊 Выбранная переменная:** {value_column}")
        st.markdown("---")
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            st.markdown("#### 📈 Статистика по всем числовым столбцам")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        else:
            st.info("Нет числовых столбцов для статистики")

    try:
        analyzer = DecompositionAnalyzer(df, date_column=date_column, value_column=value_column)
    except Exception as exc:
        st.error(f"Ошибка при инициализации анализатора: {exc}")
        lab_state["stage1_completed"] = False
        return analysis_data

    st.success("✅ Анализатор инициализирован успешно!")
    st.info(f"📊 **Анализируемая переменная:** {value_column} | **Столбец с датой:** {date_column}")
    lab_state["stage1_completed"] = True

    analysis_mode = st.radio(
        "",
        ["Автоматический поиск лучшей декомпозиции", "Ручной выбор параметров"],
        horizontal=True,
        label_visibility="collapsed",
    )

    datetime_series = pd.to_datetime(df[date_column], errors="coerce")

    if analysis_mode == "Автоматический поиск лучшей декомпозиции":
        auto_col1, auto_col2 = st.columns(2)
        with auto_col1:
            model_type = st.radio(
                "🎯 Тип модели",
                ["Автоматический выбор", "Аддитивная", "Мультипликативная"],
                index=0,
                help="Выберите тип сезонной модели для анализа",
                key="auto_model_type",
            )
        with auto_col2:
            periods_input = st.text_input(
                "📊 Периоды сезонности (через запятую)",
                value="7, 30, 365",
                help="Например: 7, 30, 365",
                key="auto_periods_input",
            )
            try:
                periods = [int(p.strip()) for p in periods_input.split(",") if p.strip()]
                periods = [p for p in periods if p > 1]
                if not periods:
                    raise ValueError
            except Exception:
                periods = [7, 30, 365]
                st.warning("⚠️ Используются периоды по умолчанию: 7, 30, 365")

        if st.button("🚀 Начать анализ", type="primary"):
            with st.spinner("⏳ Выполняется анализ всех вариантов декомпозиции..."):
                try:
                    models = (
                        ["additive", "multiplicative"]
                        if model_type == "Автоматический выбор"
                        else ["additive"] if model_type == "Аддитивная" else ["multiplicative"]
                    )
                    best_result = analyzer.get_best_decomposition(periods=periods, models=models)
                    analysis_data["best_decomposition"] = best_result
                    st.success("✅ Анализ завершён!")
                except Exception as exc:
                    st.error(f"Ошибка при анализе: {exc}")
                    st.exception(exc)

        if "best_decomposition" in analysis_data:
            best = analysis_data["best_decomposition"]
            st.markdown("---")
            st.subheader("🏆 Лучшая декомпозиция")
            col1, col2, col3 = st.columns(3)
            col1.metric("Модель", best["model"].upper())
            col2.metric("Период", best["period"])
            col3.metric("Оценка качества", f"{best['score']:.2f}")
            decomp = best["decomposition"]
            _visualize_decomposition(decomp, "Лучшая декомпозиция")
            st.markdown("---")
            st.subheader("📈 Анализ остатков")
            _display_residual_analysis(best["residual_analysis"], decomp.resid)
            st.markdown("---")
            st.subheader("📊 Сравнение всех вариантов")
            _display_comparison_table(best["all_comparisons"])
    else:
        col1, col2 = st.columns(2)
        with col1:
            selected_model = st.selectbox(
                "Тип модели",
                ["additive", "multiplicative"],
                index=0,
                help="Аддитивная или мультипликативная модель",
            )
        with col2:
            selected_period = st.number_input(
                "Период сезонности",
                min_value=2,
                max_value=len(analyzer.series) // 2,
                value=7,
                step=1,
                help="Период сезонности для декомпозиции",
            )

        if st.button("🔍 Выполнить декомпозицию", type="primary"):
            with st.spinner("⏳ Выполняется декомпозиция..."):
                try:
                    decomp = analyzer.decompose(model=selected_model, period=selected_period)
                    analysis_data["manual_decomposition"] = decomp
                    analysis_data["manual_model"] = selected_model
                    analysis_data["manual_period"] = selected_period
                    st.success("✅ Декомпозиция выполнена!")
                except Exception as exc:
                    st.error(f"Ошибка при декомпозиции: {exc}")
                    st.exception(exc)

        if "manual_decomposition" in analysis_data:
            decomp = analysis_data["manual_decomposition"]
            st.markdown("---")
            st.subheader("📊 Результаты декомпозиции")
            col1, col2 = st.columns(2)
            col1.metric("Модель", analysis_data["manual_model"].upper())
            col2.metric("Период", analysis_data["manual_period"])
            _visualize_decomposition(decomp, "Декомпозиция")
            st.markdown("---")
            st.subheader("📈 Анализ остатков")
            residual_analysis = analyzer.analyze_residuals(decomp.resid)
            _display_residual_analysis(residual_analysis, decomp.resid)

    selected_series = analyzer.series.copy()
    selected_series.name = value_column

    # Создаем datetime_series с тем же индексом, что и selected_series
    # чтобы они правильно выравнивались при объединении
    if isinstance(selected_series.index, pd.DatetimeIndex):
        datetime_series_aligned = pd.Series(selected_series.index, index=selected_series.index, name=date_column)
    else:
        # Если индекс не DatetimeIndex, используем исходный datetime_series
        datetime_series_aligned = datetime_series

    analysis_data["source_df"] = df
    analysis_data["date_column"] = date_column
    analysis_data["datetime_column"] = date_column
    analysis_data["value_column"] = value_column
    analysis_data["target_column"] = value_column
    analysis_data["datetime_series"] = datetime_series_aligned
    analysis_data["selected_series"] = selected_series

    return analysis_data


stage1 = preprocessing_transformations

__all__ = ["stage1", "preprocessing_transformations"]

