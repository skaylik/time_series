# model_evaluation_module.py - Этап 7: Оценка качества моделей (ИСПРАВЛЕННАЯ)

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Статистические тесты
from scipy import stats
from scipy.stats import norm

# ============================================================
# ФУНКЦИИ ДЛЯ ВЫЧИСЛЕНИЯ МЕТРИК
# ============================================================

def calculate_mae(y_true, y_pred):
    """Средняя абсолютная ошибка"""
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true, y_pred):
    """Среднеквадратичная ошибка"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mape(y_true, y_pred, epsilon=1e-10):
    """Средняя абсолютная процентная ошибка"""
    # Избегаем деления на ноль
    mask = np.abs(y_true) > epsilon
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_mase(y_true, y_pred, y_train, seasonal_period=1):
    """Масштабированная средняя абсолютная ошибка (MASE)"""
    # Наивный прогноз (сдвиг на seasonal_period)
    if len(y_train) > seasonal_period:
        naive_forecast = y_train.iloc[seasonal_period:].values
        naive_actual = y_train.iloc[:-seasonal_period].values
        naive_error = np.mean(np.abs(naive_forecast - naive_actual))
    else:
        # Если недостаточно данных для сезонного прогноза
        naive_forecast = y_train.iloc[1:].values
        naive_actual = y_train.iloc[:-1].values
        naive_error = np.mean(np.abs(naive_forecast - naive_actual))
    
    if naive_error == 0:
        return np.nan
    
    forecast_error = np.mean(np.abs(y_true - y_pred))
    return forecast_error / naive_error

def calculate_rmsse(y_true, y_pred, y_train, seasonal_period=1):
    """Масштабированная среднеквадратичная ошибка (RMSSE)"""
    # Наивный прогноз (сдвиг на seasonal_period)
    if len(y_train) > seasonal_period:
        naive_forecast = y_train.iloc[seasonal_period:].values
        naive_actual = y_train.iloc[:-seasonal_period].values
        naive_error = np.mean((naive_forecast - naive_actual) ** 2)
    else:
        # Если недостаточно данных для сезонного прогноза
        naive_forecast = y_train.iloc[1:].values
        naive_actual = y_train.iloc[:-1].values
        naive_error = np.mean((naive_forecast - naive_actual) ** 2)
    
    if naive_error == 0:
        return np.nan
    
    forecast_error = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(forecast_error / naive_error)

def calculate_all_metrics(y_true, y_pred, y_train, model_name="", seasonal_period=1):
    """Вычисление всех метрик качества"""
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'model': model_name,
            'MAE': np.nan,
            'RMSE': np.nan,
            'MAPE': np.nan,
            'MASE': np.nan,
            'RMSSE': np.nan
        }
    
    try:
        metrics = {
            'model': model_name,
            'MAE': calculate_mae(y_true, y_pred),
            'RMSE': calculate_rmse(y_true, y_pred),
            'MAPE': calculate_mape(y_true, y_pred),
            'MASE': calculate_mase(y_true, y_pred, y_train, seasonal_period),
            'RMSSE': calculate_rmsse(y_true, y_pred, y_train, seasonal_period)
        }
        return metrics
    except Exception as e:
        st.warning(f"Ошибка при вычислении метрик для {model_name}: {str(e)}")
        return {
            'model': model_name,
            'MAE': np.nan,
            'RMSE': np.nan,
            'MAPE': np.nan,
            'MASE': np.nan,
            'RMSSE': np.nan
        }

# ============================================================
# ТЕСТ ДИБОЛЬДА-МАРИАНО (DIEBOLD-MARIANO TEST)
# ============================================================

def dm_test(forecast_A, forecast_B, actual, h=1, test="two_sided"):
    """
    Тест Дибольда-Мариано для сравнения двух прогнозов
    
    Parameters:
    -----------
    forecast_A : array-like
        Прогнозы модели A
    forecast_B : array-like
        Прогнозы модели B
    actual : array-like
        Фактические значения
    h : int
        Горизонт прогнозирования (по умолчанию 1)
    test : str
        Тип теста: "two_sided", "less", "greater"
        "less": модель A лучше модели B (меньшие ошибки)
        "greater": модель A хуже модели B
        "two_sided": модели различаются
    
    Returns:
    --------
    dm_stat : float
        Статистика DM
    p_value : float
        p-значение
    """
    
    # Проверяем входные данные
    forecast_A = np.asarray(forecast_A)
    forecast_B = np.asarray(forecast_B)
    actual = np.asarray(actual)
    
    if len(forecast_A) != len(forecast_B) or len(forecast_A) != len(actual):
        raise ValueError("Все массивы должны иметь одинаковую длину")
    
    if len(forecast_A) < 2:
        return np.nan, np.nan
    
    # Вычисляем ошибки прогнозов
    error_A = forecast_A - actual
    error_B = forecast_B - actual
    
    # Разность квадратов ошибок (или абсолютных ошибок - зависит от функции потерь)
    # Здесь используем квадратичную функцию потерь
    loss_A = error_A ** 2
    loss_B = error_B ** 2
    d = loss_A - loss_B  # Разность потерь
    
    # Среднее разности потерь
    d_mean = np.mean(d)
    
    # Автоковариационная функция разности потерь
    n = len(d)
    
    # Вычисляем оценку дисперсии с учетом автокорреляции
    gamma = []
    for lag in range(h):
        if lag == 0:
            gamma.append(np.cov(d, d)[0, 0])
        else:
            gamma.append(np.cov(d[lag:], d[:-lag])[0, 0])
    
    # Дисперсия среднего разности
    var_d_mean = (gamma[0] + 2 * sum(gamma[1:])) / n
    
    # Избегаем деления на ноль
    if var_d_mean <= 0:
        var_d_mean = 1e-10
    
    # Статистика DM
    dm_stat = d_mean / np.sqrt(var_d_mean)
    
    # p-значение
    if test == "two_sided":
        p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    elif test == "less":
        p_value = norm.cdf(dm_stat)
    elif test == "greater":
        p_value = 1 - norm.cdf(dm_stat)
    else:
        raise ValueError("test должен быть 'two_sided', 'less' или 'greater'")
    
    return dm_stat, p_value

def pairwise_dm_tests(models_predictions, actual_values, h=1, test="two_sided"):
    """
    Попарное сравнение всех моделей с помощью теста DM
    
    Returns:
    --------
    dm_matrix : pd.DataFrame
        Матрица p-значений
    stats_matrix : pd.DataFrame
        Матрица статистик DM
    """
    
    model_names = list(models_predictions.keys())
    n_models = len(model_names)
    
    # Инициализируем матрицы
    dm_matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
    stats_matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
    
    # Заполняем диагональ
    for i in range(n_models):
        dm_matrix.iloc[i, i] = 1.0
        stats_matrix.iloc[i, i] = 0.0
    
    # Попарное сравнение
    for i in range(n_models):
        for j in range(i+1, n_models):
            model_i = model_names[i]
            model_j = model_names[j]
            
            pred_i = models_predictions[model_i]
            pred_j = models_predictions[model_j]
            
            try:
                dm_stat, p_value = dm_test(pred_i, pred_j, actual_values, h=h, test=test)
                dm_matrix.loc[model_i, model_j] = p_value
                dm_matrix.loc[model_j, model_i] = p_value
                stats_matrix.loc[model_i, model_j] = dm_stat
                stats_matrix.loc[model_j, model_i] = -dm_stat  # Симметрия
            except Exception as e:
                st.warning(f"Ошибка в тесте DM для {model_i} vs {model_j}: {str(e)}")
                dm_matrix.loc[model_i, model_j] = np.nan
                dm_matrix.loc[model_j, model_i] = np.nan
                stats_matrix.loc[model_i, model_j] = np.nan
                stats_matrix.loc[model_j, model_i] = np.nan
    
    return dm_matrix, stats_matrix

# ============================================================
# ФУНКЦИИ ДЛЯ РАНЖИРОВАНИЯ МОДЕЛЕЙ
# ============================================================

def rank_models(metrics_df, primary_metric='MASE', secondary_metric='MAE', ascending=True):
    """
    Ранжирование моделей по метрикам
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame с метриками моделей
    primary_metric : str
        Основная метрика для ранжирования
    secondary_metric : str
        Вторичная метрика для разрешения ничьих
    ascending : bool
        True: чем меньше, тем лучше
        False: чем больше, тем лучше
    
    Returns:
    --------
    ranked_df : pd.DataFrame
        DataFrame с рангами
    """
    
    df = metrics_df.copy()
    
    # Сначала сортируем по основной метрике
    df = df.sort_values(primary_metric, ascending=ascending, na_position='last')
    
    # Затем сортируем по вторичной метрике (в пределах одинаковых значений основной)
    df = df.sort_values([primary_metric, secondary_metric], ascending=[ascending, ascending])
    
    # Назначаем ранги
    df['Rank'] = range(1, len(df) + 1)
    
    # Для моделей с NaN в основной метрике ставим последние ранги
    nan_mask = df[primary_metric].isna()
    if nan_mask.any():
        df.loc[nan_mask, 'Rank'] = range(len(df) - nan_mask.sum() + 1, len(df) + 1)
    
    # Переупорядочиваем столбцы
    cols = ['Rank'] + [col for col in df.columns if col != 'Rank']
    df = df[cols]
    
    return df

def add_dm_significance(ranked_df, dm_matrix, reference_model=None):
    """
    Добавление информации о статистической значимости сравнения с референсной моделью
    
    Parameters:
    -----------
    ranked_df : pd.DataFrame
        DataFrame с рангами моделей
    dm_matrix : pd.DataFrame
        Матрица p-значений теста DM
    reference_model : str, optional
        Референсная модель (по умолчанию лучшая по рангу)
    
    Returns:
    --------
    ranked_df : pd.DataFrame
        DataFrame с добавленными столбцами значимости
    """
    
    if reference_model is None:
        # Используем модель с рангом 1
        reference_model = ranked_df[ranked_df['Rank'] == 1].iloc[0]['model']
    
    df = ranked_df.copy()
    
    # Добавляем столбец с p-value сравнения с референсной моделью
    p_values = []
    for model in df['model']:
        if model == reference_model:
            p_values.append(1.0)  # Сравнение с самой собой
        else:
            try:
                p_value = dm_matrix.loc[model, reference_model]
                p_values.append(p_value)
            except:
                p_values.append(np.nan)
    
    df[f'p(DM vs {reference_model})'] = p_values
    
    # Добавляем столбец со звездочками значимости
    significance = []
    for p in p_values:
        if pd.isna(p):
            significance.append('')
        elif p < 0.01:
            significance.append('***')
        elif p < 0.05:
            significance.append('**')
        elif p < 0.1:
            significance.append('*')
        else:
            significance.append('')
    
    df['Significance'] = significance
    
    return df

# ============================================================
# ФУНКЦИИ ДЛЯ СБОРА ДАННЫХ ИЗ ПРЕДЫДУЩИХ ЭТАПОВ
# ============================================================

def collect_predictions_from_stage3():
    """Сбор прогнозов из Этапа 3 (ML модели)"""
    
    if 'modeling_results' not in st.session_state:
        return {}
    
    modeling_results = st.session_state.modeling_results
    
    try:
        predictions = {}
        
        # Получаем тестовые данные
        X_test = modeling_results.get('X_test')
        y_test = modeling_results.get('y_test')
        
        if X_test is None or y_test is None:
            return {}
        
        # Получаем оптимизатор
        optimizer = modeling_results.get('optimizer')
        
        if optimizer and hasattr(optimizer, 'best_models'):
            for model_name, model in optimizer.best_models.items():
                try:
                    # Получаем прогнозы
                    if hasattr(model, 'predict'):
                        y_pred = model.predict(X_test)
                        predictions[model_name] = y_pred
                except Exception as e:
                    st.warning(f"Не удалось получить прогнозы для модели {model_name}: {str(e)}")
        
        # Также проверяем сохраненные результаты сравнения
        comparison_df = modeling_results.get('comparison_df')
        if isinstance(comparison_df, pd.DataFrame) and not comparison_df.empty:
            # Здесь можно добавить другие модели из сравнения
            pass
        
        return predictions, y_test
        
    except Exception as e:
        st.error(f"Ошибка при сборе прогнозов из Этапа 3: {str(e)}")
        return {}, None

def collect_predictions_from_stage4():
    """Сбор прогнозов из Этапа 4 (стратегии прогнозирования)"""
    
    if 'forecast_results' not in st.session_state:
        return {}
    
    forecast_results = st.session_state.forecast_results
    
    try:
        predictions = {}
        
        # Получаем результаты стратегий
        strategy_results = forecast_results.get('strategy_results', {})
        y_test = forecast_results.get('y_test')
        
        if not strategy_results or y_test is None:
            return {}
        
        for strategy_name, strategy_data in strategy_results.items():
            if isinstance(strategy_data, dict):
                # Извлекаем прогнозы (берем только первый шаг для сравнения)
                forecasts = strategy_data.get('forecasts')
                if forecasts is not None and len(forecasts) > 0:
                    # Если прогнозы многомерные (по горизонту), берем первый шаг
                    if isinstance(forecasts, list) and len(forecasts) > 0:
                        # Пробуем разные форматы
                        if isinstance(forecasts[0], (np.ndarray, list)):
                            # Берем только первый шаг прогноза
                            first_step_preds = []
                            for forecast in forecasts:
                                if len(forecast) > 0:
                                    first_step_preds.append(forecast[0])
                                else:
                                    first_step_preds.append(np.nan)
                            predictions[strategy_name] = np.array(first_step_preds)
                        else:
                            # Уже одномерные прогнозы
                            predictions[strategy_name] = np.array(forecasts)
        
        return predictions, y_test
        
    except Exception as e:
        st.error(f"Ошибка при сборе прогнозов из Этапа 4: {str(e)}")
        return {}, None

def collect_predictions_from_stage5():
    """Сбор прогнозов из Этапа 5 (интегрированные результаты)"""
    
    if 'integrated_results' not in st.session_state:
        return {}
    
    integrated_results = st.session_state.integrated_results
    
    try:
        predictions = {}
        
        # Получаем данные из этапов 3 и 4
        stage3_data = integrated_results.get('stage3_data', {})
        stage4_data = integrated_results.get('stage4_data', {})
        
        # Собираем прогнозы из этапа 3
        if stage3_data:
            X_test = stage3_data.get('X_test')
            y_test = stage3_data.get('y_test')
            
            if X_test is not None and y_test is not None:
                optimizer = stage3_data.get('optimizer')
                if optimizer and hasattr(optimizer, 'best_models'):
                    for model_name, model in optimizer.best_models.items():
                        try:
                            y_pred = model.predict(X_test)
                            predictions[f"Этап 3: {model_name}"] = y_pred
                        except:
                            pass
        
        # Собираем прогнозы из этапа 4
        if stage4_data:
            strategy_results = stage4_data.get('strategy_results', {})
            y_test_stage4 = stage4_data.get('y_test')
            
            for strategy_name, strategy_data in strategy_results.items():
                if isinstance(strategy_data, dict):
                    forecasts = strategy_data.get('forecasts')
                    if forecasts is not None:
                        # Берем первый шаг прогноза
                        first_step_preds = [f[0] if len(f) > 0 else np.nan for f in forecasts]
                        predictions[f"Этап 4: {strategy_name}"] = np.array(first_step_preds)
        
        # Определяем y_test (предпочитаем из этапа 3, если есть)
        y_test = stage3_data.get('y_test') if stage3_data else stage4_data.get('y_test')
        
        return predictions, y_test
        
    except Exception as e:
        st.error(f"Ошибка при сборе прогнозов из Этапа 5: {str(e)}")
        return {}, None

def collect_training_times():
    """Сбор информации о времени обучения моделей"""
    
    training_times = {}
    
    try:
        # Из этапа 3
        if 'modeling_results' in st.session_state:
            modeling_results = st.session_state.modeling_results
            optimizer = modeling_results.get('optimizer')
            if optimizer and hasattr(optimizer, 'training_times'):
                training_times.update(optimizer.training_times)
        
        # Из этапа 4
        if 'forecast_results' in st.session_state:
            forecast_results = st.session_state.forecast_results
            strategy_results = forecast_results.get('strategy_results', {})
            for strategy_name, strategy_data in strategy_results.items():
                if isinstance(strategy_data, dict):
                    training_time = strategy_data.get('training_time', 0)
                    training_times[strategy_name] = training_time
        
        return training_times
        
    except Exception as e:
        st.warning(f"Ошибка при сборе времени обучения: {str(e)}")
        return {}

def collect_autogluon_ranks():
    """Сбор рангов AutoGluon (если использовался)"""
    
    autogluon_ranks = {}
    
    try:
        # Проверяем, использовался ли AutoGluon в этапе 3
        if 'modeling_results' in st.session_state:
            modeling_results = st.session_state.modeling_results
            
            # Ищем модели с AutoGluon в названии
            optimizer = modeling_results.get('optimizer')
            if optimizer and hasattr(optimizer, 'best_models'):
                for model_name in optimizer.best_models.keys():
                    if 'autogluon' in model_name.lower():
                        # Для AutoGluon моделей можно попробовать получить ранги
                        # В реальном приложении здесь нужно извлекать ранги из модели AutoGluon
                        autogluon_ranks[model_name] = 1  # Заглушка
        
        return autogluon_ranks
        
    except Exception as e:
        st.warning(f"Ошибка при сборе рангов AutoGluon: {str(e)}")
        return {}

# ============================================================
# ОСНОВНОЙ ИНТЕРФЕЙС ЭТАПА 7
# ============================================================

def show_model_evaluation_interface():
    """Основной интерфейс Этапа 7: Оценка качества моделей"""
    
    
    # Проверка наличия данных из предыдущих этапов
    required_keys = ['df_features', 'feature_info', 'split_data']
    missing_keys = [key for key in required_keys if key not in st.session_state]
    
    if missing_keys:
        st.error(f"❌ Сначала выполните Этапы 1-2. Отсутствуют: {', '.join(missing_keys)}")
        return
    
    st.info("""
    **Цель Этапа 7:**
    - Оценить качество моделей с помощью расширенного набора метрик
    - Провести статистическое сравнение моделей с помощью теста Дибольда-Мариано
    - Ранжировать модели для выбора наилучшей
    """)
    
    # Собираем данные из предыдущих этапов
    with st.spinner("Собираем данные о моделях..."):
        # Сначала пробуем получить данные из этапа 5 (интегрированные)
        predictions, y_test = collect_predictions_from_stage5()
        
        # Если не получилось, собираем из этапов 3 и 4 отдельно
        if not predictions:
            pred_stage3, y_test_stage3 = collect_predictions_from_stage3()
            pred_stage4, y_test_stage4 = collect_predictions_from_stage4()
            
            predictions.update(pred_stage3)
            predictions.update(pred_stage4)
            
            # Выбираем y_test (предпочитаем этап 3)
            y_test = y_test_stage3 if y_test_stage3 is not None else y_test_stage4
        
        # Собираем дополнительную информацию
        training_times = collect_training_times()
        autogluon_ranks = collect_autogluon_ranks()
    
    # Проверяем, есть ли данные для оценки
    if not predictions or y_test is None:
        st.error("""
        ❌ Нет данных для оценки качества моделей!
        
        **Требуется выполнить:**
        1. Этап 3: Подбор гиперпараметров ML моделей
        2. Этап 4: Сравнение стратегий прогнозирования
        3. Этап 5: Интеграция и сравнение подходов
        
        Без этих данных оценка качества невозможна.
        """)
        return
    
    # Получаем обучающие данные для масштабированных метрик
    try:
        split_data = st.session_state.split_data
        train_data = split_data['train']
        val_data = split_data['val']
        
        # Объединяем train и val для расчета масштабированных метрик
        feature_info = st.session_state.feature_info
        target_col = feature_info['original_features'][1]
        
        y_train_full = pd.concat([train_data[target_col], val_data[target_col]], axis=0)
        
    except Exception as e:
        st.warning(f"Не удалось получить обучающие данные: {str(e)}")
        y_train_full = pd.Series([])
    
    # Настройки оценки
    st.subheader("⚙️ Настройки оценки качества")
    
    col1, col2 = st.columns(2)
    
    with col1:
        seasonal_period = st.number_input(
            "Сезонный период (для MASE/RMSSE):",
            min_value=1,
            max_value=365,
            value=1,
            help="Период для наивного сезонного прогноза"
        )
        
        dm_horizon = st.number_input(
            "Горизонт для теста DM:",
            min_value=1,
            max_value=10,
            value=1,
            help="Горизонт прогнозирования для теста Дибольда-Мариано"
        )
    
    with col2:
        dm_test_type = st.selectbox(
            "Тип теста DM:",
            options=["two_sided", "less", "greater"],
            index=0,
            help="two_sided: модели различаются, less: первая модель лучше, greater: первая модель хуже"
        )
        
        primary_metric = st.selectbox(
            "Основная метрика для ранжирования:",
            options=["MASE", "MAE", "RMSE", "MAPE", "RMSSE"],
            index=0,
            help="Модели ранжируются по этой метрике в первую очередь"
        )
    
    secondary_metric = st.selectbox(
        "Вторичная метрика для разрешения ничьих:",
        options=["MAE", "MASE", "RMSE", "MAPE", "RMSSE"],
        index=0,
        help="Используется если значения основной метрики равны"
    )
    
    st.markdown("---")
    
    if st.button("🚀 Запустить оценку качества моделей", type="primary", use_container_width=True):
        with st.spinner("Выполняется оценка качества моделей..."):
            try:
                # 1. Вычисляем метрики для всех моделей

                
                all_metrics = []
                valid_predictions = {}
                
                for model_name, y_pred in predictions.items():
                    # Проверяем корректность прогнозов
                    if y_pred is None or len(y_pred) == 0:
                        st.warning(f"Пропускаем модель {model_name}: нет прогнозов")
                        continue
                    
                    # Приводим к одинаковой длине
                    min_len = min(len(y_test), len(y_pred))
                    if min_len == 0:
                        st.warning(f"Пропускаем модель {model_name}: нет данных для сравнения")
                        continue
                    
                    y_true_trimmed = y_test[:min_len]
                    y_pred_trimmed = y_pred[:min_len]
                    
                    # Сохраняем обрезанные прогнозы для теста DM
                    valid_predictions[model_name] = y_pred_trimmed
                    
                    # Вычисляем метрики
                    metrics = calculate_all_metrics(
                        y_true=y_true_trimmed,
                        y_pred=y_pred_trimmed,
                        y_train=y_train_full,
                        model_name=model_name,
                        seasonal_period=seasonal_period
                    )
                    
                    # Добавляем время обучения
                    training_time = training_times.get(model_name, training_times.get(model_name.split(": ")[-1] if ": " in model_name else model_name, np.nan))
                    metrics['training_time'] = training_time
                    
                    # Добавляем ранг AutoGluon
                    autogluon_rank = autogluon_ranks.get(model_name, autogluon_ranks.get(model_name.split(": ")[-1] if ": " in model_name else model_name, np.nan))
                    metrics['autogluon_rank'] = autogluon_rank
                    
                    all_metrics.append(metrics)
                
                if not all_metrics:
                    st.error("❌ Не удалось вычислить метрики ни для одной модели")
                    return
                
                # Создаем DataFrame с метриками
                metrics_df = pd.DataFrame(all_metrics)
                
                # 2. Выполняем тест Дибольда-Мариано

                
                # Обрезаем фактические значения
                min_test_len = min([len(pred) for pred in valid_predictions.values()] + [len(y_test)])
                y_test_trimmed = y_test[:min_test_len]
                
                # Попарное сравнение моделей
                dm_matrix, dm_stats = pairwise_dm_tests(
                    valid_predictions,
                    y_test_trimmed,
                    h=dm_horizon,
                    test=dm_test_type
                )
                
                # 3. Ранжируем модели

                
                # Сортируем по основной метрике (чем меньше, тем лучше)
                ranked_df = rank_models(
                    metrics_df,
                    primary_metric=primary_metric,
                    secondary_metric=secondary_metric,
                    ascending=True
                )
                
                # Добавляем статистическую значимость
                if not dm_matrix.empty:
                    reference_model = ranked_df.iloc[0]['model']
                    ranked_df = add_dm_significance(ranked_df, dm_matrix, reference_model)
                
                # 4. Отображаем результаты
                _display_evaluation_results(
                    ranked_df, 
                    dm_matrix, 
                    dm_stats, 
                    valid_predictions, 
                    y_test_trimmed,
                    primary_metric,  # Передаем primary_metric
                    secondary_metric  # Передаем secondary_metric
                )
                
                # Сохраняем результаты
                st.session_state.evaluation_results = {
                    'metrics_df': metrics_df,
                    'ranked_df': ranked_df,
                    'dm_matrix': dm_matrix,
                    'dm_stats': dm_stats,
                    'predictions': valid_predictions,
                    'y_test': y_test_trimmed,
                    'primary_metric': primary_metric,  # Сохраняем primary_metric
                    'secondary_metric': secondary_metric  # Сохраняем secondary_metric
                }
                
                st.success("✅ Оценка качества моделей завершена!")
                
            except Exception as e:
                st.error(f"❌ Ошибка при выполнении оценки качества: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Если уже есть результаты, показываем их
    elif 'evaluation_results' in st.session_state:
        st.success("✅ Оценка качества уже выполнена!")
        
        results = st.session_state.evaluation_results
        _display_evaluation_results(
            results['ranked_df'],
            results['dm_matrix'],
            results['dm_stats'],
            results['predictions'],
            results['y_test'],
            results.get('primary_metric', 'MASE'),  # Получаем из сохраненных результатов
            results.get('secondary_metric', 'MAE')   # Получаем из сохраненных результатов
        )

def _display_evaluation_results(ranked_df, dm_matrix, dm_stats, predictions, y_test, primary_metric='MASE', secondary_metric='MAE'):
    """Отображение результатов оценки качества"""
    
    # 1. Сводная таблица с метриками и рангами
    st.subheader("📊 Сводная таблица моделей")
    
    # Форматируем таблицу для отображения
    display_df = ranked_df.copy()
    
    # Форматируем числовые столбцы
    numeric_cols = ['MAE', 'RMSE', 'MAPE', 'MASE', 'RMSSE', 'training_time']
    for col in numeric_cols:
        if col in display_df.columns:
            if col == 'MAPE':
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
            elif col == 'training_time':
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}s" if pd.notnull(x) else "N/A")
            else:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    
    # Форматируем ранг AutoGluon
    if 'autogluon_rank' in display_df.columns:
        display_df['autogluon_rank'] = display_df['autogluon_rank'].apply(
            lambda x: f"{int(x)}" if pd.notnull(x) else "N/A"
        )
    
    # Добавляем цветовое кодирование для рангов
    def color_rank(val):
        if val == 1:
            return 'background-color: gold'
        elif val == 2:
            return 'background-color: silver'
        elif val == 3:
            return 'background-color: #cd7f32'
        else:
            return ''
    
    st.dataframe(
        display_df.style.applymap(color_rank, subset=['Rank']),
        width='stretch',
        height=min(400, 50 + len(display_df) * 35)
    )
    
    # 2. Визуализация сравнения моделей
    st.subheader("📈 Визуализация сравнения моделей")
    
    # Выбор метрики для визуализации
    metric_options = ['MAE', 'RMSE', 'MAPE', 'MASE', 'RMSSE']
    available_metrics = [m for m in metric_options if m in ranked_df.columns]
    
    if available_metrics:
        selected_metric = st.selectbox(
            "Выберите метрику для визуализации:",
            options=available_metrics,
            index=0
        )
        
        # Сортируем по выбранной метрике
        viz_df = ranked_df.copy()
        viz_df = viz_df.sort_values(selected_metric, ascending=True)
        
        # График сравнения метрик
        fig_metrics = go.Figure()
        
        fig_metrics.add_trace(go.Bar(
            x=viz_df['model'],
            y=viz_df[selected_metric],
            text=viz_df[selected_metric].round(4),
            textposition='auto',
            marker_color='lightblue',
            name=selected_metric
        ))
        
        fig_metrics.update_layout(
            title=f'Сравнение моделей по {selected_metric}',
            xaxis_title='Модель',
            yaxis_title=selected_metric,
            height=500,
            template='plotly_white',
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Тепловая карта метрик
        st.write("#### Тепловая карта метрик")
        
        # Подготавливаем данные для тепловой карты
        heatmap_data = ranked_df.set_index('model')[available_metrics].copy()
        
        # Нормализуем данные для лучшей визуализации (кроме MAPE если в процентах)
        heatmap_data_norm = heatmap_data.copy()
        for col in heatmap_data_norm.columns:
            if col != 'MAPE':
                # Минимакс нормализация (чем меньше значение, тем лучше)
                if heatmap_data_norm[col].max() > heatmap_data_norm[col].min():
                    heatmap_data_norm[col] = 1 - (heatmap_data_norm[col] - heatmap_data_norm[col].min()) / (heatmap_data_norm[col].max() - heatmap_data_norm[col].min())
                else:
                    heatmap_data_norm[col] = 0.5
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data_norm.values,
            x=heatmap_data_norm.columns,
            y=heatmap_data_norm.index,
            colorscale='RdYlGn_r',  # Красный-желтый-зеленый (обратный, т.к. зеленый = лучше)
            text=heatmap_data.round(4).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverinfo='text',
            hovertemplate='Модель: %{y}<br>Метрика: %{x}<br>Значение: %{text}<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            title='Тепловая карта метрик качества (зеленый = лучше)',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # 3. Матрица p-значений теста Дибольда-Мариано
    if not dm_matrix.empty:
        st.subheader("📊 Матрица p-значений теста Дибольда-Мариано")
        
        # Создаем аннотации для тепловой карты
        annotations = []
        for i, row in enumerate(dm_matrix.index):
            for j, col in enumerate(dm_matrix.columns):
                p_value = dm_matrix.iloc[i, j]
                if pd.isna(p_value):
                    text = 'N/A'
                elif p_value < 0.01:
                    text = '***'
                elif p_value < 0.05:
                    text = '**'
                elif p_value < 0.1:
                    text = '*'
                else:
                    text = f'{p_value:.3f}'
                
                annotations.append(
                    dict(
                        x=col,
                        y=row,
                        text=text,
                        showarrow=False,
                        font=dict(size=10, color='white' if p_value < 0.05 else 'black')
                    )
                )
        
        # Тепловая карта p-значений
        fig_dm = go.Figure(data=go.Heatmap(
            z=dm_matrix.values,
            x=dm_matrix.columns,
            y=dm_matrix.index,
            colorscale='RdYlBu_r',
            zmin=0,
            zmax=1,
            colorbar=dict(title='p-value'),
            hoverinfo='text',
            hovertemplate='Модель A: %{y}<br>Модель B: %{x}<br>p-value: %{z:.4f}<extra></extra>'
        ))
        
        fig_dm.update_layout(
            title='Матрица p-значений теста Дибольда-Мариано',
            height=500,
            template='plotly_white',
            annotations=annotations
        )
        
        st.plotly_chart(fig_dm, use_container_width=True)
        
        # Интерпретация
        st.info("""
        **Интерпретация теста Дибольда-Мариано:**
        
        - **p < 0.01 (***)**: очень сильные статистические различия между моделями
        - **p < 0.05 (**)**: сильные статистические различия  
        - **p < 0.10 (*)**: умеренные статистические различия
        - **p ≥ 0.10**: нет статистически значимых различий
        
        **Важно:** Малые p-значения (особенно < 0.05) указывают на статистически значимые различия в качестве прогнозов.
        """)
    
    # 4. Сравнение прогнозов с фактическими значениями
    st.subheader("📈 Сравнение прогнозов с фактическими значениями")
    
    # Выбор моделей для сравнения
    model_options = list(predictions.keys())
    if len(model_options) >= 2:
        selected_models = st.multiselect(
            "Выберите модели для сравнения прогнозов:",
            options=model_options,
            default=model_options[:min(3, len(model_options))]
        )
        
        if selected_models:
            # График прогнозов
            fig_predictions = go.Figure()
            
            # Фактические значения
            fig_predictions.add_trace(go.Scatter(
                x=list(range(len(y_test))),
                y=y_test,
                mode='lines',
                name='Фактические значения',
                line=dict(color='black', width=3)
            ))
            
            # Прогнозы выбранных моделей
            colors = px.colors.qualitative.Set2
            for i, model_name in enumerate(selected_models):
                if model_name in predictions:
                    y_pred = predictions[model_name]
                    color_idx = i % len(colors)
                    
                    fig_predictions.add_trace(go.Scatter(
                        x=list(range(len(y_pred))),
                        y=y_pred,
                        mode='lines',
                        name=f'{model_name}',
                        line=dict(color=colors[color_idx], width=2, dash='dash'),
                        opacity=0.8
                    ))
            
            fig_predictions.update_layout(
                title='Сравнение прогнозов моделей',
                xaxis_title='Временной индекс',
                yaxis_title='Значение',
                height=500,
                template='plotly_white',
                showlegend=True
            )
            
            st.plotly_chart(fig_predictions, use_container_width=True)
    
    # 5. Анализ ошибок
    st.subheader("📊 Анализ ошибок моделей")
    
    if not ranked_df.empty and 'MAE' in ranked_df.columns:
        # График распределения ошибок для топ-3 моделей
        top_models = ranked_df.head(3)['model'].tolist()
        
        if top_models:
            fig_errors = go.Figure()
            
            for i, model_name in enumerate(top_models):
                if model_name in predictions:
                    y_pred = predictions[model_name]
                    errors = y_test[:len(y_pred)] - y_pred
                    
                    fig_errors.add_trace(go.Box(
                        y=errors,
                        name=model_name,
                        boxpoints='outliers',
                        marker_color=px.colors.qualitative.Set1[i],
                        showlegend=True
                    ))
            
            fig_errors.update_layout(
                title='Распределение ошибок для топ-3 моделей',
                yaxis_title='Ошибка прогноза',
                height=400,
                template='plotly_white',
                showlegend=True
            )
            
            st.plotly_chart(fig_errors, use_container_width=True)
    
    # 6. Выводы и рекомендации
    st.subheader("🎯 Выводы и рекомендации")
    
    if not ranked_df.empty:
        # Лучшая модель
        best_model = ranked_df.iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            **🏆 Лучшая модель:**
            
            **{best_model['model']}**
            
            **Ранг:** {best_model['Rank']}
            **{primary_metric}:** {best_model.get(primary_metric, 'N/A'):.4f}
            **Стат. значимость:** {best_model.get('Significance', 'N/A')}
            
            **Рекомендуется для использования в продакшене**
            """)
        
        with col2:
            # Анализ преимуществ
            if 'MASE' in best_model and best_model['MASE'] < 1:
                mase_interpretation = "Модель лучше наивного прогноза"
            elif 'MASE' in best_model and best_model['MASE'] == 1:
                mase_interpretation = "Модель эквивалентна наивному прогнозу"
            else:
                mase_interpretation = "Модель хуже наивного прогноза"
            
            st.info(f"""
            **Анализ качества:**
            
            - **MASE:** {best_model.get('MASE', 'N/A'):.4f} ({mase_interpretation})
            - **Точность (MAPE):** {best_model.get('MAPE', 'N/A'):.2f}%
            - **Время обучения:** {best_model.get('training_time', 'N/A'):.3f}s
            - **Стабильность (RMSSE):** {best_model.get('RMSSE', 'N/A'):.4f}
            """)
    
    # 7. Детальный анализ (расширяемая секция)
    with st.expander("🔍 Детальный анализ метрик"):
        st.write("#### Интерпретация метрик:")
        
        metric_explanations = {
            'MAE': "Средняя абсолютная ошибка. Чем меньше, тем лучше. Устойчива к выбросам.",
            'RMSE': "Среднеквадратичная ошибка. Чем меньше, тем лучше. Учитывает большие ошибки.",
            'MAPE': "Средняя абсолютная процентная ошибка. Хороша для сравнения разных масштабов.",
            'MASE': "Масштабированная средняя абсолютная ошибка. <1 лучше наивного прогноза.",
            'RMSSE': "Масштабированная среднеквадратичная ошибка. <1 лучше наивного прогноза."
        }
        
        for metric, explanation in metric_explanations.items():
            if metric in ranked_df.columns:
                st.write(f"**{metric}:** {explanation}")
        
        st.write("#### Рекомендации по выбору модели:")
        
        st.info("""
        1. **Для бизнес-решений:** Используйте MAPE для интерпретируемости
        2. **Для чувствительных к выбросам систем:** Используйте MAE
        3. **Для сравнения с базовым методами:** Используйте MASE/RMSSE
        4. **Для комплексной оценки:** Рассмотрите все метрики и тест DM
        5. **Для продакшена:** Учитывайте также время прогнозирования и стабильность
        """)
    
    st.markdown("---")
    st.success("""
    **✅ Этап 7 завершен!**
    
    **Что было сделано:**
    1. Вычислены расширенные метрики качества (MAE, RMSE, MAPE, MASE, RMSSE)
    2. Проведено статистическое сравнение моделей с помощью теста Дибольда-Мариано
    3. Выполнено ранжирование моделей по комплексным критериям
    4. Даны рекомендации по выбору наилучшей модели
    
    **🎯 Итог проекта:** Выполнено полное исследование временных рядов с выбором оптимальной модели на основе статистически обоснованных критериев.
    """)

# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def show_model_evaluation():
    """Основная функция для запуска Этапа 7"""
    show_model_evaluation_interface()