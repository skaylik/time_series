"""
Модуль для формирования выводов и рекомендаций по моделям (Этап 9).
Выполняет комплексную оценку моделей, обосновывает выбор лучшей модели,
предоставляет рекомендации по продакшену (Prophet vs SARIMAX, TBATS, обновление модели).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def compute_model_complexity_score(
    train_time: float,
    predict_time: float,
    has_seasonality: bool,
    has_exogenous: bool,
    model_type: str,
) -> float:
    """
    Вычисляет оценку вычислительной сложности модели (меньше = лучше).
    
    Args:
        train_time: Время обучения в секундах
        predict_time: Время прогноза в секундах
        has_seasonality: Есть ли сезонность
        has_exogenous: Есть ли экзогенные переменные
        model_type: Тип модели
    
    Returns:
        Оценка сложности (0-10, где 0 = самая простая)
    """
    complexity = 0.0
    
    # Базовая сложность по типу модели
    model_complexity_map = {
        "Naive": 0.0,
        "Seasonal Naive": 0.5,
        "SES": 1.0,
        "AR": 2.0,
        "MA": 2.0,
        "ARMA": 3.0,
        "ARIMA": 4.0,
        "SARIMA": 5.0,
        "SARIMAX": 6.0,
        "GARCH": 6.5,
        "VAR": 7.0,
        "VECM": 7.5,
        "TBATS": 8.0,
        "Prophet": 7.0,
        "LinearRegression": 3.0,
        "RandomForestRegressor": 5.0,
    }
    
    # Определяем базовую сложность
    base_complexity = model_complexity_map.get(model_type, 5.0)
    complexity += base_complexity
    
    # Учитываем время обучения (нормализованное)
    if not np.isnan(train_time) and train_time > 0:
        # Логарифмическая шкала для времени
        time_complexity = min(2.0, np.log10(max(1, train_time * 100)))
        complexity += time_complexity
    
    # Учитываем сезонность
    if has_seasonality:
        complexity += 1.0
    
    # Учитываем экзогенные переменные
    if has_exogenous:
        complexity += 1.0
    
    return min(10.0, complexity)


def compute_interpretability_score(model_type: str, has_details: bool, details: Optional[Dict[str, Any]]) -> float:
    """
    Вычисляет оценку интерпретируемости модели (больше = лучше, 0-10).
    
    Args:
        model_type: Тип модели
        has_details: Есть ли детали модели
        details: Детали модели
    
    Returns:
        Оценка интерпретируемости (0-10, где 10 = самая интерпретируемая)
    """
    interpretability_map = {
        "Naive": 10.0,  # Максимально простая
        "Seasonal Naive": 10.0,
        "SES": 9.0,
        "AR": 8.0,
        "MA": 8.0,
        "ARMA": 7.0,
        "ARIMA": 7.0,
        "SARIMA": 6.0,
        "SARIMAX": 5.0,
        "GARCH": 6.0,
        "VAR": 5.0,
        "VECM": 4.0,
        "TBATS": 3.0,  # Сложная для интерпретации
        "Prophet": 6.0,
        "LinearRegression": 9.0,  # Очень интерпретируемая
        "RandomForestRegressor": 2.0,  # Черный ящик
    }
    
    base_interpretability = interpretability_map.get(model_type, 5.0)
    
    # Если есть детали (параметры), увеличиваем интерпретируемость
    if has_details and details:
        if "order" in details or "seasonal_order" in details:
            base_interpretability += 0.5
        if "aic" in details or "bic" in details:
            base_interpretability += 0.5
    
    return min(10.0, base_interpretability)


def compute_comprehensive_score(
    mase: float,
    ljung_box_pvalue: float,
    shapiro_wilk_pvalue: float,
    complexity: float,
    interpretability: float,
    weight_quality: float = 0.4,
    weight_adequacy: float = 0.3,
    weight_interpretability: float = 0.2,
    weight_complexity: float = 0.1,
) -> float:
    """
    Вычисляет комплексную оценку модели.
    
    Args:
        mase: MASE метрика (меньше = лучше)
        ljung_box_pvalue: p-value теста Льюнга-Бокса (больше = лучше)
        shapiro_wilk_pvalue: p-value теста Шапиро-Уилка (больше = лучше)
        complexity: Оценка сложности (меньше = лучше, 0-10)
        interpretability: Оценка интерпретируемости (больше = лучше, 0-10)
        weight_quality: Вес качества (по умолчанию 0.4)
        weight_adequacy: Вес адекватности (по умолчанию 0.3)
        weight_interpretability: Вес интерпретируемости (по умолчанию 0.2)
        weight_complexity: Вес сложности (по умолчанию 0.1)
    
    Returns:
        Комплексная оценка (меньше = лучше)
    """
    # Нормализуем метрики
    # MASE: меньше = лучше, нормализуем (предполагаем максимум 5)
    mase_normalized = min(1.0, mase / 5.0) if not np.isnan(mase) else 1.0
    
    # Ljung-Box: больше = лучше (белый шум), нормализуем через (1 - pvalue)
    lb_score = (1 - ljung_box_pvalue) if not np.isnan(ljung_box_pvalue) else 1.0
    
    # Shapiro-Wilk: больше = лучше (нормальность), нормализуем через (1 - pvalue)
    sw_score = (1 - shapiro_wilk_pvalue) if not np.isnan(shapiro_wilk_pvalue) else 1.0
    
    # Adequacy: среднее из тестов
    adequacy_score = (lb_score + sw_score) / 2.0
    
    # Complexity: меньше = лучше, нормализуем (0-10 -> 0-1)
    complexity_normalized = complexity / 10.0
    
    # Interpretability: больше = лучше, нормализуем через (10 - interpretability) / 10
    interpretability_normalized = (10 - interpretability) / 10.0
    
    # Взвешенная сумма
    comprehensive_score = (
        weight_quality * mase_normalized +
        weight_adequacy * adequacy_score +
        weight_interpretability * interpretability_normalized +
        weight_complexity * complexity_normalized
    )
    
    return comprehensive_score


def get_production_recommendations(
    best_model: str,
    best_group: str,
    has_seasonality: bool,
    has_exogenous: bool,
    data_length: int,
) -> Dict[str, Any]:
    """
    Генерирует рекомендации по продакшену на основе выбранной модели.
    
    Args:
        best_model: Название лучшей модели
        best_group: Группа модели
        has_seasonality: Есть ли сезонность в данных
        has_exogenous: Есть ли экзогенные переменные
        data_length: Длина данных
    
    Returns:
        Словарь с рекомендациями
    """
    recommendations = {
        "model_selection": {},
        "when_to_use": {},
        "updating_strategy": {},
        "general_notes": [],
    }
    
    # Рекомендации по выбору модели
    # Prophet vs SARIMAX (показываем всегда, если есть сезонность)
    if has_seasonality:
        if "Prophet" in best_model:
            recommendations["model_selection"]["prophet_vs_sarimax"] = """
            **Используйте Prophet, если:**
            - Есть множественная сезонность (дневная, недельная, месячная)
            - Данные содержат выбросы и пропуски
            - Нужна автоматическая обработка праздников
            - Требуется быстрая настройка без глубоких знаний временных рядов
            - Данные имеют нелинейные тренды
            
            **Используйте SARIMAX, если:**
            - Есть одна четкая сезонность
            - Нужна высокая интерпретируемость параметров
            - Требуется контроль над параметрами модели
            - Есть экзогенные переменные с известным воздействием
            - Нужна статистическая обоснованность модели
            """
            
            recommendations["when_to_use"]["prophet"] = "Рекомендуется для бизнес-метрик с множественной сезонностью и выбросами"
            recommendations["when_to_use"]["sarimax"] = "Рекомендуется для технических метрик с четкой сезонностью"
        
        elif "SARIMA" in best_model or "SARIMAX" in best_model:
            recommendations["model_selection"]["prophet_vs_sarimax"] = """
            **Используйте SARIMAX, если:**
            - Есть одна четкая сезонность (недельная, месячная)
            - Нужна высокая интерпретируемость параметров
            - Требуется контроль над параметрами модели (p, d, q, P, D, Q, m)
            - Есть экзогенные переменные с известным воздействием
            - Нужна статистическая обоснованность модели
            - Данные стационарны после дифференцирования
            
            **Используйте Prophet, если:**
            - Есть множественная сезонность
            - Данные содержат много выбросов
            - Нужна автоматическая обработка праздников
            - Требуется быстрая настройка
            - Данные имеют нелинейные тренды
            """
            
            recommendations["when_to_use"]["prophet"] = "Рекомендуется при множественной сезонности и выбросах"
            recommendations["when_to_use"]["sarimax"] = "Рекомендуется для текущих данных с четкой сезонностью"
        else:
            # Для других сезонных моделей тоже показываем рекомендации
            recommendations["model_selection"]["prophet_vs_sarimax"] = """
            **Общие рекомендации по выбору модели для сезонных данных:**
            
            **Используйте SARIMAX, если:**
            - Есть одна четкая сезонность (недельная, месячная)
            - Нужна высокая интерпретируемость параметров
            - Требуется контроль над параметрами модели
            - Есть экзогенные переменные с известным воздействием
            - Нужна статистическая обоснованность модели
            
            **Используйте Prophet, если:**
            - Есть множественная сезонность
            - Данные содержат много выбросов
            - Нужна автоматическая обработка праздников
            - Требуется быстрая настройка
            - Данные имеют нелинейные тренды
            
            **Текущая лучшая модель ({best_model})** показывает хорошие результаты для ваших данных.
            """
    
    # Рекомендации по TBATS
    if "TBATS" in best_model:
        recommendations["model_selection"]["tbats_usage"] = """
        **TBATS рекомендуется использовать, если:**
        - Есть сложная сезонность (нестандартные периоды)
        - Множественная сезонность не решается Prophet
        - Нужна точная обработка нелинейных трендов
        - Данные имеют большой объем
        
        **TBATS НЕ нужен, если:**
        - Есть простая сезонность (одна периодичность)
        - Данных мало (менее 2-3 сезонов)
        - Нужна быстрая модель
        - Требуется высокая интерпретируемость
        """
    else:
        if has_seasonality:
            recommendations["model_selection"]["tbats_usage"] = f"""
            **Для ваших данных TBATS НЕ требуется, так как:**
            - Текущая лучшая модель ({best_model}) успешно обрабатывает сезонность
            - Сезонность простая (одна периодичность)
            - Использование TBATS увеличит сложность без значительного улучшения качества
            """
        else:
            recommendations["model_selection"]["tbats_usage"] = """
            **TBATS не требуется, так как:**
            - В данных нет сезонности
            - Использование TBATS избыточно для несезонных данных
            """
    
    # Рекомендации по обновлению модели
    update_strategies = {
        "Naive": "Модель обновляется автоматически при каждом новом наблюдении (используется последнее значение)",
        "Seasonal Naive": "Модель обновляется автоматически при каждом новом наблюдении (используются последние значения сезона)",
        "SES": "Переобучайте модель каждый период (например, ежедневно или еженедельно) на расширенном окне данных",
        "AR": "Переобучайте модель каждый период на расширенном окне данных. Минимальный размер окна: 50-100 наблюдений",
        "MA": "Переобучайте модель каждый период на расширенном окне данных. Минимальный размер окна: 50-100 наблюдений",
        "ARMA": "Переобучайте модель каждый период на расширенном окне данных. Минимальный размер окна: 50-100 наблюдений",
        "ARIMA": "Переобучайте модель еженедельно или ежемесячно на расширенном окне данных. Минимальный размер окна: 100-200 наблюдений",
        "SARIMA": "Переобучайте модель ежемесячно на расширенном окне данных. Минимальный размер окна: 2-3 сезона (например, 6-12 месяцев для месячной сезонности)",
        "SARIMAX": "Переобучайте модель ежемесячно на расширенном окне данных. Убедитесь, что экзогенные переменные доступны для прогноза",
        "GARCH": "Переобучайте модель еженедельно или ежемесячно. Модель волатильности требует достаточно данных (минимум 100 наблюдений)",
        "VAR": "Переобучайте модель ежемесячно на расширенном окне данных. Минимальный размер окна: 100-200 наблюдений",
        "VECM": "Переобучайте модель ежемесячно на расширенном окне данных. Минимальный размер окна: 100-200 наблюдений",
        "TBATS": "Переобучайте модель ежемесячно на расширенном окне данных. Модель требует много данных (минимум 2-3 сезона)",
        "Prophet": "Модель может обновляться ежедневно или еженедельно. Prophet хорошо работает с добавлением новых данных без полного переобучения",
        "LinearRegression": "Переобучайте модель каждый период на расширенном окне данных. Минимальный размер окна: 50-100 наблюдений",
        "RandomForestRegressor": "Переобучайте модель еженедельно или ежемесячно на расширенном окне данных. Минимальный размер окна: 100-200 наблюдений",
    }
    
    # Находим стратегию обновления для модели
    update_strategy = "Переобучайте модель периодически на расширенном окне данных"
    
    # Проверяем точное совпадение или частичное совпадение названий моделей
    for model_key in sorted(update_strategies.keys(), key=len, reverse=True):  # Сортируем по длине для более точного совпадения
        if model_key in best_model:
            update_strategy = update_strategies[model_key]
            break
    
    # Если не нашли, проверяем по группе модели
    if update_strategy == "Переобучайте модель периодически на расширенном окне данных":
        if best_group == "Benchmarks":
            if "Naive" in best_model:
                update_strategy = update_strategies.get("Naive", update_strategy)
            elif "Seasonal" in best_model:
                update_strategy = update_strategies.get("Seasonal Naive", update_strategy)
            elif "SES" in best_model:
                update_strategy = update_strategies.get("SES", update_strategy)
        elif best_group == "Basic":
            if "ARIMA" in best_model:
                if "SARIMA" in best_model:
                    update_strategy = update_strategies.get("SARIMA", update_strategy)
                elif "SARIMAX" in best_model:
                    update_strategy = update_strategies.get("SARIMAX", update_strategy)
                else:
                    update_strategy = update_strategies.get("ARIMA", update_strategy)
            elif "ARMA" in best_model:
                update_strategy = update_strategies.get("ARMA", update_strategy)
            elif "AR" in best_model and "MA" not in best_model:
                update_strategy = update_strategies.get("AR", update_strategy)
            elif "MA" in best_model:
                update_strategy = update_strategies.get("MA", update_strategy)
        elif best_group == "Seasonal":
            if "Prophet" in best_model:
                update_strategy = update_strategies.get("Prophet", update_strategy)
            elif "TBATS" in best_model:
                update_strategy = update_strategies.get("TBATS", update_strategy)
        elif best_group == "ML Models":
            if "LinearRegression" in best_model:
                update_strategy = update_strategies.get("LinearRegression", update_strategy)
            elif "RandomForest" in best_model:
                update_strategy = update_strategies.get("RandomForestRegressor", update_strategy)
        elif best_group == "Volatility":
            update_strategy = update_strategies.get("GARCH", update_strategy)
        elif best_group == "Multivariate":
            if "VAR" in best_model:
                update_strategy = update_strategies.get("VAR", update_strategy)
            elif "VECM" in best_model:
                update_strategy = update_strategies.get("VECM", update_strategy)
    
    recommendations["updating_strategy"]["strategy"] = update_strategy
    recommendations["updating_strategy"]["general_guidelines"] = """
    **Общие рекомендации по обновлению модели:**
    
    1. **Частота обновления:**
       - Простые модели (Naive, SES): ежедневно
       - Средние модели (ARIMA, LinearRegression): еженедельно
       - Сложные модели (SARIMA, TBATS, Prophet): ежемесячно
    
    2. **Размер окна данных:**
       - Используйте расширяющееся окно для стабильных данных
       - Используйте скользящее окно для данных с изменениями тренда
       - Минимальный размер: 2-3 сезона для сезонных моделей
    
    3. **Мониторинг качества:**
       - Отслеживайте метрики качества (MAE, RMSE, MAPE)
       - Переобучайте модель при ухудшении качества более чем на 20%
       - Проверяйте остатки на белый шум и нормальность
    
    4. **Автоматизация:**
       - Настройте автоматическое переобучение по расписанию
       - Используйте кросс-валидацию для проверки качества
       - Настройте алерты при ухудшении качества
    """
    
    # Общие заметки
    recommendations["general_notes"] = [
        f"Рекомендуемая модель: **{best_model}** ({best_group})",
        f"Длина данных: {data_length} наблюдений",
        "Учитывайте вычислительную сложность при выборе частоты обновления",
        "Регулярно проверяйте качество модели на новых данных",
        "Используйте доверительные интервалы для оценки неопределенности прогноза",
    ]
    
    return recommendations


def stage9(
    analysis_data: Optional[Dict[str, Any]],
    lab_state: Dict[str, bool],
) -> Dict[str, Any]:
    """
    Этап 9. Выводы и рекомендации
    
    - Сравнение топ-3 моделей
    - Экспорт выводов
    """
    if analysis_data is None:
        analysis_data = {}


    if not lab_state.get("stage5_completed"):
        st.info("Завершите этап 5, чтобы перейти к выводам и рекомендациям.")
        return analysis_data

    # Получаем результаты прогнозирования из этапа 5
    forecast_results: List[Any] = analysis_data.get("forecast_results", [])
    if not forecast_results:
        st.warning("Не найдены результаты прогнозирования. Завершите этап 5.")
        return analysis_data

    # Получаем обучающие данные (нужно раньше для вычисления MASE)
    train_series = analysis_data.get("stage5_train_series")
    source_df = analysis_data.get("source_df", pd.DataFrame())
    target_column = analysis_data.get("target_column", "")
    
    # Получаем результаты диагностики из этапа 6
    diagnostics_results = analysis_data.get("diagnostics_results", [])
    diagnostics_dict = {diag.get("model_name"): diag for diag in diagnostics_results if diag.get("model_name")}

    # Получаем результаты сравнения из этапа 7
    comparison_df = analysis_data.get("comparison_table")
    if comparison_df is None or comparison_df.empty:
        # Создаем простую таблицу сравнения из результатов этапа 5
        comparison_data = []
        for result in forecast_results:
            # Получаем время обучения из details
            train_time = np.nan
            predict_time = np.nan
            if result.details and isinstance(result.details, dict):
                train_time = result.details.get("train_time", np.nan)
                predict_time = result.details.get("predict_time", np.nan)
            
            # Получаем диагностику для p-values
            diagnostic = diagnostics_dict.get(result.name, {})
            ljung_box = diagnostic.get("ljung_box", {})
            shapiro_wilk = diagnostic.get("shapiro_wilk", {})
            
            comparison_data.append({
                "Модель": result.name,
                "Группа": result.group,
                "Горизонт": len(result.forecast),
                "MAE": result.metrics.get("mae", np.nan),
                "RMSE": result.metrics.get("rmse", np.nan),
                "MAPE": result.metrics.get("mape", np.nan),
                "MASE": np.nan,  # Будет вычислено позже
                "p-value (Ljung-Box)": ljung_box.get("pvalue", np.nan),
                "p-value (Shapiro-Wilk)": shapiro_wilk.get("pvalue", np.nan),
                "Время обучения (сек)": train_time,
                "Время прогноза (сек)": predict_time,
            })
        comparison_df = pd.DataFrame(comparison_data)
        
        # Вычисляем MASE, если есть обучающие данные
        if train_series is not None and len(train_series) > 0:
            from model_evaluation_comparison import compute_mase
            
            for idx, row in comparison_df.iterrows():
                model_name = row["Модель"]
                model_result = next((r for r in forecast_results if r.name == model_name), None)
                if model_result:
                    y_true = model_result.actual.values.flatten() if model_result.actual.values.ndim > 1 else model_result.actual.values
                    y_pred = model_result.forecast.values.flatten() if model_result.forecast.values.ndim > 1 else model_result.forecast.values
                    y_train = train_series.values.flatten() if train_series.values.ndim > 1 else train_series.values
                    
                    seasonality = 7 if len(train_series) >= 14 else 1
                    mase_value = compute_mase(y_true, y_pred, y_train, seasonality)
                    comparison_df.at[idx, "MASE"] = mase_value
    
    # Характеристики данных больше не нужны для упрощенной версии этапа 9

    # Создаем таблицу с ранжированием для топ-3 моделей
    ranking_summary = []
    
    # Определяем горизонты
    horizons = comparison_df["Горизонт"].unique() if "Горизонт" in comparison_df.columns else [len(forecast_results[0].forecast)]
    
    for horizon in horizons:
        horizon_df = comparison_df[comparison_df["Горизонт"] == horizon].copy() if "Горизонт" in comparison_df.columns else comparison_df.copy()
        
        if horizon_df.empty:
            continue
        
        # Добавляем оценки для каждой модели
        for idx, row in horizon_df.iterrows():
            model_name = row["Модель"]
            model_group = row["Группа"]
            
            # Находим детали модели
            model_result = next((r for r in forecast_results if r.name == model_name), None)
            details = model_result.details if model_result else {}
            
            # Получаем диагностику
            diagnostic = diagnostics_dict.get(model_name, {})
            ljung_box = diagnostic.get("ljung_box", {})
            shapiro_wilk = diagnostic.get("shapiro_wilk", {})
            
            # Получаем время обучения
            train_time = row.get("Время обучения (сек)", np.nan)
            predict_time = row.get("Время прогноза (сек)", np.nan)
            
            if np.isnan(train_time) or np.isnan(predict_time):
                if model_result and model_result.details:
                    if isinstance(model_result.details, dict):
                        if np.isnan(train_time):
                            train_time = model_result.details.get("train_time", np.nan)
                        if np.isnan(predict_time):
                            predict_time = model_result.details.get("predict_time", np.nan)
            
            # Определяем характеристики модели
            has_model_seasonality = "SARIMA" in model_name or "SARIMAX" in model_name or "TBATS" in model_name or "Prophet" in model_name
            has_model_exogenous = "SARIMAX" in model_name or "VAR" in model_name or "VECM" in model_name
            
            # Вычисляем оценки
            mase = row.get("MASE", np.nan)
            lb_pvalue = ljung_box.get("pvalue", np.nan)
            sw_pvalue = shapiro_wilk.get("pvalue", np.nan)
            
            complexity = compute_model_complexity_score(
                train_time=train_time if not np.isnan(train_time) else 5.0,
                predict_time=predict_time if not np.isnan(predict_time) else 1.0,
                has_seasonality=has_model_seasonality,
                has_exogenous=has_model_exogenous,
                model_type=model_name,
            )
            
            interpretability = compute_interpretability_score(
                model_type=model_name,
                has_details=bool(details),
                details=details if isinstance(details, dict) else None,
            )
            
            comprehensive_score = compute_comprehensive_score(
                mase=mase,
                ljung_box_pvalue=lb_pvalue,
                shapiro_wilk_pvalue=sw_pvalue,
                complexity=complexity,
                interpretability=interpretability,
            )
            
            ranking_summary.append({
                "Горизонт": horizon,
                "Модель": model_name,
                "Группа": model_group,
                "RMSE": row.get("RMSE", np.nan),
                "MAE": row.get("MAE", np.nan),
                "MAPE": row.get("MAPE", np.nan),
                "MASE": mase,
                "Оценка_качества": mase if not np.isnan(mase) else row.get("RMSE", np.nan),
                "Оценка_адекватности": (lb_pvalue + sw_pvalue) / 2.0 if not (np.isnan(lb_pvalue) and np.isnan(sw_pvalue)) else np.nan,
                "Оценка_интерпретируемости": interpretability,
                "Оценка_сложности": complexity,
                "Комплексная_оценка": comprehensive_score,
                "p_value_LB": lb_pvalue,
                "p_value_SW": sw_pvalue,
            })
    
    if not ranking_summary:
        st.warning("Не удалось создать таблицу ранжирования.")
        return analysis_data
    
    ranking_df = pd.DataFrame(ranking_summary)
    
    # Находим лучшую модель по комплексной оценке (нужно для экспорта)
    best_overall = ranking_df.loc[ranking_df["Комплексная_оценка"].idxmin()]
    best_model = best_overall["Модель"]
    best_group = best_overall["Группа"]
    best_horizon = best_overall["Горизонт"]
    best_result = next((r for r in forecast_results if r.name == best_model), None)
    
    # Сравнение топ-3 моделей
    st.markdown("#### 🥇 Сравнение топ-3 моделей")
    
    # Находим топ-3 модели по комплексной оценке
    top3_df = ranking_df.nsmallest(3, "Комплексная_оценка")
    
    comparison_cols = ["Модель", "Группа", "RMSE", "MAE", "MASE", 
                      "Оценка_качества", "Оценка_адекватности", 
                      "Оценка_интерпретируемости", "Оценка_сложности", 
                      "Комплексная_оценка"]
    available_comparison_cols = [col for col in comparison_cols if col in top3_df.columns] if not top3_df.empty else []
    
    if not top3_df.empty:
        st.dataframe(top3_df[available_comparison_cols], use_container_width=True, hide_index=True)
        
        # Визуализация сравнения топ-3
        try:
            fig = go.Figure()
            
            metrics_to_plot = ["Оценка_качества", "Оценка_адекватности", "Оценка_интерпретируемости"]
            available_metrics = [m for m in metrics_to_plot if m in top3_df.columns]
            
            for metric in available_metrics:
                # Инвертируем оценки, где меньше = лучше
                if metric == "Оценка_качества":
                    values = top3_df[metric].values
                elif metric == "Оценка_адекватности":
                    # Для адекватности: больше = лучше, но мы хотим показать наоборот для визуализации
                    values = 1 - top3_df[metric].values
                else:
                    # Для интерпретируемости: больше = лучше, инвертируем
                    values = 10 - top3_df[metric].values
                
                fig.add_trace(
                    go.Bar(
                        x=top3_df["Модель"],
                        y=values,
                        name=metric.replace("_", " "),
                    )
                )
            
            fig.update_layout(
                title="Сравнение топ-3 моделей по критериям (меньше = лучше)",
                xaxis_title="Модель",
                yaxis_title="Нормализованная оценка",
                barmode="group",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Ошибка при создании графика сравнения: {e}")
    
    # Экспорт результатов этапа 9
    st.markdown("#### 💾 Экспорт выводов")
    
    if not ranking_df.empty:
        # Экспорт таблицы ранжирования
        ranking_csv = ranking_df.to_csv(index=False, encoding='utf-8-sig')
        ranking_filename = f"model_ranking_comprehensive_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        st.download_button(
            label="📥 Скачать таблицу ранжирования с комплексными оценками",
            data=ranking_csv,
            file_name=ranking_filename,
            mime="text/csv",
            help="Экспортировать таблицу ранжирования моделей с оценками качества, адекватности, интерпретируемости и сложности",
            key="stage9_download_ranking_csv"
        )
        
        # Экспорт выводов в текстовом формате
        # Получаем время для текста
        train_time_for_text = np.nan
        predict_time_for_text = np.nan
        
        if best_result and best_result.details:
            if isinstance(best_result.details, dict):
                train_time_for_text = best_result.details.get("train_time", np.nan)
                predict_time_for_text = best_result.details.get("predict_time", np.nan)
        
        # Форматируем значения для текста
        rmse_val = best_overall.get('RMSE', np.nan)
        mae_val = best_overall.get('MAE', np.nan)
        mape_val = best_overall.get('MAPE', np.nan)
        mase_val = best_overall.get('MASE', np.nan)
        lb_pval = best_overall.get('p_value_LB', np.nan)
        sw_pval = best_overall.get('p_value_SW', np.nan)
        adequacy_val = best_overall.get('Оценка_адекватности', np.nan)
        interpretability_val = best_overall.get('Оценка_интерпретируемости', np.nan)
        complexity_val = best_overall.get('Оценка_сложности', np.nan)
        comprehensive_val = best_overall.get('Комплексная_оценка', np.nan)
        
        # Форматируем значения с проверкой на NaN
        def format_value(val, fmt='.4f'):
            if np.isnan(val):
                return 'N/A'
            return f"{val:{fmt}}"
        
        # Экспорт топ-3 моделей
        if not top3_df.empty and available_comparison_cols:
            top3_export = top3_df[available_comparison_cols].to_csv(index=False, encoding='utf-8-sig')
            top3_filename = f"top3_models_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            st.download_button(
                label="📥 Скачать сравнение топ-3 моделей (CSV)",
                data=top3_export,
                file_name=top3_filename,
                mime="text/csv",
                help="Экспортировать таблицу сравнения топ-3 моделей",
                key="stage9_download_top3_csv"
            )
            
            # Экспорт выводов в текстовом формате
            conclusions_text = f"""
ВЫВОДЫ ПО АНАЛИЗУ МОДЕЛЕЙ
========================

Лучшая модель: {best_model} ({best_group})
Горизонт: {best_horizon}

МЕТРИКИ КАЧЕСТВА:
-----------------
- RMSE: {format_value(rmse_val, '.4f')}
- MAE: {format_value(mae_val, '.4f')}
- MAPE: {format_value(mape_val, '.2f')}%
- MASE: {format_value(mase_val, '.4f')}

АДЕКВАТНОСТЬ МОДЕЛИ:
--------------------
- Ljung-Box p-value: {format_value(lb_pval, '.4f')}
- Shapiro-Wilk p-value: {format_value(sw_pval, '.4f')}
- Оценка адекватности: {format_value(adequacy_val, '.4f')}

ХАРАКТЕРИСТИКИ МОДЕЛИ:
-----------------------
- Оценка интерпретируемости: {format_value(interpretability_val, '.2f')}/10
- Оценка сложности: {format_value(complexity_val, '.2f')}/10
- Время обучения: {format_value(train_time_for_text, '.4f')} сек
- Время прогноза: {format_value(predict_time_for_text, '.4f')} сек
- Комплексная оценка: {format_value(comprehensive_val, '.4f')}

ТОП-3 МОДЕЛИ:
-------------
"""
            
            # Добавляем топ-3 модели
            for idx, (_, row) in enumerate(top3_df.iterrows(), start=1):
                conclusions_text += f"""
{idx}. {row['Модель']} ({row['Группа']})
   - RMSE: {format_value(row.get('RMSE', np.nan), '.4f')}
   - MAE: {format_value(row.get('MAE', np.nan), '.4f')}
   - MASE: {format_value(row.get('MASE', np.nan), '.4f')}
   - Комплексная оценка: {format_value(row.get('Комплексная_оценка', np.nan), '.4f')}
"""
            
            conclusions_filename = f"conclusions_{best_model.replace(' ', '_').replace('(', '').replace(')', '')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            st.download_button(
                label="📥 Скачать выводы в текстовом формате",
                data=conclusions_text,
                file_name=conclusions_filename,
                mime="text/plain",
                help="Экспортировать выводы по анализу моделей в текстовом формате",
                key="stage9_download_conclusions_txt"
            )
    
    # Сохраняем результаты
    analysis_data["ranking_summary"] = ranking_df
    analysis_data["best_model"] = best_model
    analysis_data["top3_models"] = top3_df.to_dict('records') if not top3_df.empty else []
    lab_state["stage9_completed"] = True
    
    st.success("Выводы и рекомендации завершены.")
    
    return analysis_data


__all__ = [
    "stage9",
    "compute_model_complexity_score",
    "compute_interpretability_score",
    "compute_comprehensive_score",
]

