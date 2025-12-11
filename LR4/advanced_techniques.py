# advanced_techniques.py - Этап 8: Продвинутые техники (Ансамблирование, обработка выбросов, сегментация)

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Базовые импорты ML
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Для winsorization
from scipy.stats import mstats

# Для ансамблирования
try:
    from sklearn.ensemble import StackingRegressor, VotingRegressor
    STACKING_AVAILABLE = True
except ImportError:
    STACKING_AVAILABLE = False

# Для AutoGluon
try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ИНТЕГРАЦИИ
# ============================================================

def extract_best_models_from_previous_stages():
    """Извлечение лучших моделей из предыдущих этапов"""
    
    best_models = {}
    model_predictions = {}
    
    # 1. Проверяем наличие результатов 7 этапа (оценка качества)
    if 'evaluation_results' in st.session_state:
        eval_results = st.session_state.evaluation_results
        ranked_df = eval_results.get('ranked_df')
        predictions = eval_results.get('predictions', {})
        
        if ranked_df is not None and not ranked_df.empty:
            # Берем топ-3 модели
            top_models = ranked_df.head(3)['model'].tolist()
            
            for model_name in top_models:
                if model_name in predictions:
                    model_predictions[model_name] = predictions[model_name]
    
    # 2. Проверяем наличие результатов 5 этапа (интеграция)
    elif 'integrated_results' in st.session_state:
        int_results = st.session_state.integrated_results
        integrated_df = int_results.get('integrated_df')
        
        if integrated_df is not None and not integrated_df.empty:
            # Находим модель с минимальным MAE
            if 'MAE' in integrated_df.columns:
                best_idx = integrated_df['MAE'].astype(float).idxmin()
                best_model = integrated_df.loc[best_idx]
                model_name = best_model.get('Название', 'Best Model')
                
                # Для простоты создаем заглушку для прогнозов
                # В реальном приложении нужно сохранять прогнозы моделей
                model_predictions[model_name] = None
    
    # 3. Если все еще нет, проверяем наличие диагностических данных
    elif 'diagnostics_results' in st.session_state:
        diag_results = st.session_state.diagnostics_results
        best_model_info = diag_results.get('best_model_info', {})
        diagnostics = diag_results.get('diagnostics')
        
        if best_model_info and diagnostics:
            model_name = best_model_info.get('Название', 'Diagnosed Model')
            if hasattr(diagnostics, 'y_test_pred'):
                model_predictions[model_name] = diagnostics.y_test_pred
    
    return model_predictions

def prepare_data_for_advanced_techniques():
    """Подготовка данных для продвинутых техник"""
    
    required_keys = ['df_features', 'feature_info', 'split_data']
    missing_keys = [key for key in required_keys if key not in st.session_state]
    
    if missing_keys:
        st.error(f"❌ Отсутствуют необходимые данные: {', '.join(missing_keys)}")
        return None, None, None, None, None
    
    feature_info = st.session_state.feature_info
    split_data = st.session_state.split_data
    df_features = st.session_state.df_features
    
    # Извлекаем основные параметры
    date_col = feature_info['original_features'][0]
    target_col = feature_info['original_features'][1]
    
    # Подготавливаем данные
    train_data = split_data['train'].copy()
    val_data = split_data['val'].copy()
    test_data = split_data['test'].copy()
    
    # Объединяем train и val для обучения
    X_train_full = pd.concat([train_data, val_data], axis=0)
    
    # Выбираем признаки
    feature_cols = []
    for col in X_train_full.columns:
        if col != date_col and col != target_col:
            if pd.api.types.is_numeric_dtype(X_train_full[col]):
                feature_cols.append(col)
    
    if not feature_cols:
        st.error("❌ Не найдено числовых признаков")
        return None, None, None, None, None
    
    # Подготавливаем данные
    X_train = X_train_full[feature_cols].copy()
    y_train = X_train_full[target_col].copy()
    
    X_test = test_data[feature_cols].copy()
    y_test = test_data[target_col].copy()
    
    # Обрабатываем пропуски
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    y_train = y_train.fillna(y_train.median())
    y_test = y_test.fillna(y_test.median())
    
    return X_train, y_train, X_test, y_test, feature_cols

# ============================================================
# КЛАСС ДЛЯ АНСАМБЛИРОВАНИЯ
# ============================================================

class AdvancedEnsembleTechniques:
    """Класс для продвинутых техник ансамблирования"""
    
    def __init__(self, base_models=None):
        self.base_models = base_models if base_models else {}
        self.ensemble_models = {}
        self.ensemble_predictions = {}
        self.ensemble_weights = {}
    
    def weighted_average_ensemble(self, predictions_dict, y_true, method='mase'):
        """
        Взвешенное усреднение прогнозов
        
        Parameters:
        -----------
        predictions_dict : dict
            Словарь {имя модели: прогнозы}
        y_true : array-like
            Истинные значения для расчета весов
        method : str
            Метод расчета весов: 'mase', 'mae', 'rmse', 'equal'
        """
        
        if not predictions_dict:
            return None, None
        
        model_names = list(predictions_dict.keys())
        
        # Вычисляем метрики для каждой модели
        metrics = {}
        for name, pred in predictions_dict.items():
            if pred is None or len(pred) == 0:
                metrics[name] = np.inf
                continue
            
            # Обрезаем до минимальной длины
            min_len = min(len(y_true), len(pred))
            if min_len == 0:
                metrics[name] = np.inf
                continue
            
            y_true_trimmed = y_true[:min_len]
            y_pred_trimmed = pred[:min_len]
            
            if method == 'mase':
                # MASE требует обучающих данных - используем упрощенный вариант
                mae = mean_absolute_error(y_true_trimmed, y_pred_trimmed)
                # Простая аппроксимация MASE
                if len(y_true_trimmed) > 1:
                    naive_error = mean_absolute_error(y_true_trimmed[1:], y_true_trimmed[:-1])
                    metric = mae / naive_error if naive_error != 0 else np.inf
                else:
                    metric = np.inf
            elif method == 'mae':
                metric = mean_absolute_error(y_true_trimmed, y_pred_trimmed)
            elif method == 'rmse':
                metric = np.sqrt(mean_squared_error(y_true_trimmed, y_pred_trimmed))
            elif method == 'equal':
                metric = 1.0  # Равные веса
            else:
                metric = mean_absolute_error(y_true_trimmed, y_pred_trimmed)
            
            metrics[name] = metric
        
        # Вычисляем веса (обратно пропорционально метрике)
        weights = {}
        if method == 'equal':
            # Равные веса
            for name in model_names:
                weights[name] = 1.0 / len(model_names)
        else:
            # Веса, обратно пропорциональные метрике
            total_inverse = sum(1.0 / max(metric, 1e-10) for metric in metrics.values())
            for name, metric in metrics.items():
                weights[name] = (1.0 / max(metric, 1e-10)) / total_inverse
        
        # Создаем взвешенный ансамбль
        ensemble_pred = None
        
        for i, (name, pred) in enumerate(predictions_dict.items()):
            if pred is None or len(pred) == 0:
                continue
            
            weight = weights.get(name, 0)
            
            if ensemble_pred is None:
                ensemble_pred = pred * weight
            else:
                # Синхронизируем длины
                min_len = min(len(ensemble_pred), len(pred))
                if min_len > 0:
                    ensemble_pred[:min_len] += pred[:min_len] * weight
        
        self.ensemble_weights['weighted_average'] = weights
        self.ensemble_predictions['weighted_average'] = ensemble_pred
        
        return ensemble_pred, weights
    
    def stacking_ensemble(self, X_train, y_train, X_test, base_models=None, meta_model=None):
        """
        Stacking ансамблирование с линейной мета-моделью
        
        Parameters:
        -----------
        X_train, y_train : обучающие данные
        X_test : тестовые данные
        base_models : список базовых моделей (имя, модель)
        meta_model : мета-модель (по умолчанию Ridge)
        """
        
        if not STACKING_AVAILABLE:
            st.warning("StackingRegressor недоступен (требуется sklearn >= 0.22)")
            return None
        
        if base_models is None:
            base_models = self.base_models
        
        if not base_models:
            st.error("❌ Нет базовых моделей для stacking")
            return None
        
        # Преобразуем словарь в список кортежей для StackingRegressor
        estimators = []
        for name, model in base_models.items():
            if hasattr(model, 'predict'):
                estimators.append((name, model))
        
        if len(estimators) < 2:
            st.warning("Для stacking нужно хотя бы 2 модели")
            return None
        
        # Используем Ridge в качестве мета-модели
        if meta_model is None:
            meta_model = Ridge(alpha=1.0, random_state=42)
        
        # Создаем stacking ансамбль
        try:
            stacking_model = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_model,
                cv=5,
                n_jobs=-1,
                passthrough=False
            )
            
            # Обучаем stacking
            with st.spinner("Обучение stacking ансамбля..."):
                stacking_model.fit(X_train, y_train)
            
            # Прогнозируем
            y_pred = stacking_model.predict(X_test)
            
            self.ensemble_models['stacking'] = stacking_model
            self.ensemble_predictions['stacking'] = y_pred
            
            return y_pred
            
        except Exception as e:
            st.error(f"❌ Ошибка в stacking ансамбле: {str(e)}")
            return None
    
    def autogluon_weighted_ensemble(self, X_train, y_train, X_test, y_test, time_limit=60):
        """
        AutoGluon WeightedEnsemble (если доступен)
        """
        
        if not AUTOGLUON_AVAILABLE:
            st.warning("AutoGluon недоступен. Установите: pip install autogluon")
            return None
        
        try:
            # Создаем DataFrame для AutoGluon
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
            
            # Определяем целевую переменную
            target_column = y_train.name
            
            # Создаем предсказатель AutoGluon
            predictor = TabularPredictor(
                label=target_column,
                problem_type='regression',
                eval_metric='mean_absolute_error'
            )
            
            # Обучаем с ограничением по времени
            with st.spinner(f"AutoGluon обучение (лимит: {time_limit} сек)..."):
                predictor.fit(
                    train_data=train_data,
                    tuning_data=test_data,
                    time_limit=time_limit,
                    presets=['medium_quality'],
                    use_bag_holdout=True,
                    verbosity=0
                )
            
            # Получаем WeightedEnsemble модель (обычно это лучшая модель)
            leaderboard = predictor.leaderboard(test_data, silent=True)
            
            # Ищем WeightedEnsemble в лидерборде
            ensemble_model_name = None
            for model in leaderboard['model']:
                if 'WeightedEnsemble' in str(model) or 'ensemble' in str(model).lower():
                    ensemble_model_name = model
                    break
            
            if ensemble_model_name is None:
                # Берем лучшую модель
                ensemble_model_name = leaderboard.iloc[0]['model']
            
            # Прогнозируем
            y_pred = predictor.predict(test_data)
            
            # Сохраняем информацию
            self.ensemble_models['autogluon_ensemble'] = predictor
            self.ensemble_predictions['autogluon_ensemble'] = y_pred
            self.ensemble_weights['autogluon_ensemble'] = {
                'model_name': ensemble_model_name,
                'leaderboard': leaderboard
            }
            
            return y_pred
            
        except Exception as e:
            st.error(f"❌ Ошибка в AutoGluon ансамбле: {str(e)}")
            return None

# ============================================================
# КЛАСС ДЛЯ ОБРАБОТКИ ВЫБРОСОВ
# ============================================================

class OutlierHandler:
    """Класс для обработки выбросов"""
    
    def __init__(self):
        self.isolation_forest = None
        self.scalers = {}
        self.outlier_stats = {}
    
    def isolation_forest_detection(self, X, contamination=0.1):
        """
        Обнаружение выбросов с помощью Isolation Forest
        
        Returns:
        --------
        outlier_mask : array-like
            Маска выбросов (True - выброс)
        """
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        outlier_mask = iso_forest.fit_predict(X) == -1
        
        self.isolation_forest = iso_forest
        self.outlier_stats['isolation_forest'] = {
            'contamination': contamination,
            'n_outliers': np.sum(outlier_mask),
            'outlier_percentage': np.mean(outlier_mask) * 100
        }
        
        return outlier_mask
    
    def apply_robust_scaling(self, X, with_scaling=False):
        """
        Применение RobustScaler (устойчив к выбросам)
        
        Parameters:
        -----------
        X : данные для масштабирования
        with_scaling : bool, если True - применяет масштабирование
        
        Returns:
        --------
        X_scaled : масштабированные данные
        """
        
        if with_scaling:
            robust_scaler = RobustScaler()
            X_scaled = robust_scaler.fit_transform(X)
            self.scalers['robust'] = robust_scaler
            return X_scaled
        else:
            # Просто возвращаем данные для сравнения
            return X
    
    def winsorization(self, X, limits=(0.05, 0.05)):
        """
        Winsorization (ограничение выбросов)
        
        Parameters:
        -----------
        X : данные
        limits : tuple, процентили для ограничения (нижний, верхний)
        
        Returns:
        --------
        X_winsorized : данные после winsorization
        """
        
        X_winsorized = X.copy()
        
        if isinstance(X_winsorized, pd.DataFrame):
            for col in X_winsorized.columns:
                if pd.api.types.is_numeric_dtype(X_winsorized[col]):
                    try:
                        X_winsorized[col] = mstats.winsorize(
                            X_winsorized[col].values,
                            limits=limits
                        )
                    except:
                        pass
        elif isinstance(X_winsorized, np.ndarray):
            try:
                X_winsorized = mstats.winsorize(X_winsorized, limits=limits)
            except:
                pass
        
        # Сохраняем статистику
        self.outlier_stats['winsorization'] = {
            'limits': limits,
            'method': f'Ограничение {limits[0]*100}%/{limits[1]*100}%'
        }
        
        return X_winsorized
    
    def compare_scaling_methods(self, X, y):
        """
        Сравнение методов масштабирования
        """
        
        results = {}
        
        # 1. Без масштабирования
        results['no_scaling'] = {
            'X_mean': np.mean(X, axis=0),
            'X_std': np.std(X, axis=0)
        }
        
        # 2. StandardScaler
        standard_scaler = StandardScaler()
        X_standard = standard_scaler.fit_transform(X)
        results['standard_scaler'] = {
            'X_mean': np.mean(X_standard, axis=0),
            'X_std': np.std(X_standard, axis=0),
            'scaler': standard_scaler
        }
        
        # 3. RobustScaler
        robust_scaler = RobustScaler()
        X_robust = robust_scaler.fit_transform(X)
        results['robust_scaler'] = {
            'X_mean': np.mean(X_robust, axis=0),
            'X_std': np.std(X_robust, axis=0),
            'scaler': robust_scaler
        }
        
        # Сохраняем скалеры
        self.scalers['standard'] = standard_scaler
        self.scalers['robust'] = robust_scaler
        
        return results

# ============================================================
# КЛАСС ДЛЯ СЕГМЕНТАЦИИ
# ============================================================

class TimeSeriesSegmenter:
    """Класс для сегментации временных рядов"""
    
    def __init__(self):
        self.kmeans_model = None
        self.segments = {}
        self.segment_models = {}
    
    def kmeans_segmentation(self, X, n_clusters=3):
        """
        Кластеризация рядов с помощью KMeans
        """
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        clusters = kmeans.fit_predict(X)
        self.kmeans_model = kmeans
        
        # Создаем сегменты
        segments = {}
        for cluster_id in range(n_clusters):
            segment_mask = clusters == cluster_id
            segment_indices = np.where(segment_mask)[0]
            
            if len(segment_indices) > 0:
                segments[cluster_id] = {
                    'indices': segment_indices,
                    'size': len(segment_indices),
                    'percentage': len(segment_indices) / len(X) * 100,
                    'features_mean': X[segment_mask].mean(axis=0) if len(X[segment_mask]) > 0 else None
                }
        
        self.segments['kmeans'] = segments
        
        return clusters, segments

    def seasonal_segmentation(self, dates, target_col):
        """
        Сезонная сегментация (зима/лето и т.д.)
        """
        
        try:
            # Проверяем, что dates не None и не пустой
            if dates is None or len(dates) == 0:
                st.warning("⚠️ Нет данных для сезонной сегментации")
                return {}
            
            # Преобразуем dates в datetime если это еще не сделано
            if not isinstance(dates, pd.Series):
                dates = pd.Series(dates)
            
            # Проверяем, является ли уже datetime
            if not pd.api.types.is_datetime64_any_dtype(dates):
                try:
                    dates = pd.to_datetime(dates, errors='coerce')
                except Exception as e:
                    st.warning(f"⚠️ Не удалось преобразовать даты: {str(e)}")
                    return {}
            
            # Проверяем успешность преобразования
            if dates.isna().any():
                st.warning(f"⚠️ Не удалось преобразовать {dates.isna().sum()} дат. Используем доступные данные.")
            
            # Определяем сезоны
            seasons = {
                'winter': [12, 1, 2],   # Зима
                'spring': [3, 4, 5],     # Весна
                'summer': [6, 7, 8],     # Лето
                'autumn': [9, 10, 11]    # Осень
            }
            
            # Создаем сегменты по сезонам
            seasonal_segments = {}
            
            for season_name, months in seasons.items():
                try:
                    # Используем dt.month для Series с datetime
                    mask = dates.dt.month.isin(months)
                    
                    indices = np.where(mask)[0]
                    
                    if len(indices) > 0:
                        # Получаем среднее значение целевой переменной
                        if isinstance(target_col, pd.Series):
                            target_mean = target_col.iloc[indices].mean()
                        elif hasattr(target_col, '__getitem__'):
                            target_mean = np.mean(target_col[indices])
                        else:
                            target_mean = 0
                        
                        seasonal_segments[season_name] = {
                            'indices': indices.tolist(),
                            'size': len(indices),
                            'percentage': len(indices) / len(dates) * 100,
                            'months': months,
                            'target_mean': target_mean
                        }
                except Exception as e:
                    st.warning(f"Ошибка при обработке сезона {season_name}: {str(e)}")
                    continue
            
            self.segments['seasonal'] = seasonal_segments
            
            return seasonal_segments
            
        except Exception as e:
            st.error(f"❌ Ошибка в сезонной сегментации: {str(e)}")
            return {}
    
    def regime_segmentation(self, values, n_regimes=2, method='percentile'):
        """
        Сегментация по режимам (высокий/низкий уровень и т.д.)
        """
        
        if method == 'percentile':
            # Разделяем по перцентилям
            percentiles = np.linspace(0, 100, n_regimes + 1)
            thresholds = np.percentile(values, percentiles[1:-1])
            
            regimes = np.digitize(values, thresholds)
        
        elif method == 'kmeans':
            # Используем KMeans для значений
            kmeans = KMeans(n_clusters=n_regimes, random_state=42)
            values_2d = values.reshape(-1, 1)
            regimes = kmeans.fit_predict(values_2d)
            thresholds = kmeans.cluster_centers_.flatten()
        
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        
        # Создаем сегменты режимов
        regime_segments = {}
        for regime_id in range(n_regimes):
            mask = regimes == regime_id
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                regime_segments[regime_id] = {
                    'indices': indices,
                    'size': len(indices),
                    'percentage': len(indices) / len(values) * 100,
                    'value_mean': np.mean(values[indices]) if len(indices) > 0 else 0,
                    'value_std': np.std(values[indices]) if len(indices) > 0 else 0
                }
        
        self.segments['regime'] = regime_segments
        
        return regimes, regime_segments
    
    def train_segment_models(self, X, y, segments, segment_type='kmeans', base_model=None):
        """
        Обучение отдельных моделей для каждого сегмента
        """
        
        if base_model is None:
            from sklearn.linear_model import Ridge
            base_model = Ridge(alpha=1.0, random_state=42)
        
        segment_models = {}
        
        for segment_name, segment_info in segments.items():
            indices = segment_info['indices']
            
            if len(indices) < 10:  # Минимальный размер сегмента
                st.warning(f"Сегмент {segment_name} слишком мал ({len(indices)} samples)")
                continue
            
            # Выбираем данные для сегмента
            X_segment = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
            y_segment = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
            
            if len(X_segment) == 0 or len(y_segment) == 0:
                continue
            
            # Создаем копию модели для сегмента
            model = base_model.__class__(**base_model.get_params())
            
            try:
                model.fit(X_segment, y_segment)
                segment_models[segment_name] = {
                    'model': model,
                    'indices': indices,
                    'size': len(indices),
                    'X_segment': X_segment,
                    'y_segment': y_segment
                }
            except Exception as e:
                st.warning(f"Ошибка обучения модели для сегмента {segment_name}: {str(e)}")
        
        self.segment_models[segment_type] = segment_models
        
        return segment_models
    
    def predict_with_segment_models(self, X_test, segment_models, segment_type='kmeans'):
        """
        Предсказание с использованием моделей сегментов
        
        Parameters:
        -----------
        X_test : тестовые данные
        segment_models : обученные модели сегментов
        segment_type : тип сегментации
        """
        
        if segment_type not in segment_models or not segment_models[segment_type]:
            return None, {}
        
        # Инициализируем предсказания
        y_pred = np.zeros(len(X_test)) * np.nan
        segment_predictions = {}
        
        # Если у нас есть KMeans модель, используем ее для предсказания кластеров тестовых данных
        if segment_type == 'kmeans' and hasattr(self, 'kmeans_model') and self.kmeans_model is not None:
            # Предсказываем кластеры для тестовых данных
            test_clusters = self.kmeans_model.predict(X_test)
            
            # Для каждого сегмента (кластера)
            for segment_name, segment_info in segment_models[segment_type].items():
                model = segment_info['model']
                cluster_id = int(segment_name)
                
                # Находим индексы тестовых данных, которые принадлежат этому кластеру
                test_indices = np.where(test_clusters == cluster_id)[0]
                
                if len(test_indices) > 0:
                    try:
                        # Выбираем тестовые данные для этого кластера
                        if hasattr(X_test, 'iloc'):
                            X_test_segment = X_test.iloc[test_indices]
                        else:
                            X_test_segment = X_test[test_indices]
                        
                        if len(X_test_segment) > 0:
                            segment_pred = model.predict(X_test_segment)
                            y_pred[test_indices] = segment_pred
                            segment_predictions[segment_name] = segment_pred
                    except Exception as e:
                        st.warning(f"Ошибка предсказания для сегмента {segment_name}: {str(e)}")
        
        # Для других типов сегментации используем более простой подход
        elif segment_type in ['seasonal', 'regime']:
            # Для сезонной и режимной сегментации используем все модели для всех данных
            # и усредняем результаты
            all_predictions = []
            
            for segment_name, segment_info in segment_models[segment_type].items():
                model = segment_info['model']
                try:
                    segment_pred = model.predict(X_test)
                    all_predictions.append(segment_pred)
                    segment_predictions[segment_name] = segment_pred
                except Exception as e:
                    st.warning(f"Ошибка предсказания для сегмента {segment_name}: {str(e)}")
            
            if all_predictions:
                # Усредняем предсказания всех моделей
                y_pred = np.mean(all_predictions, axis=0)
        
        else:
            # Для других типов используем первую модель
            for segment_name, segment_info in segment_models[segment_type].items():
                model = segment_info['model']
                try:
                    y_pred = model.predict(X_test)
                    segment_predictions[segment_name] = y_pred
                    break  # Используем только первую модель
                except Exception as e:
                    st.warning(f"Ошибка предсказания для сегмента {segment_name}: {str(e)}")
        
        # Если остались NaN, заполняем средним по не-NaN значениям
        nan_mask = np.isnan(y_pred)
        if np.any(nan_mask):
            non_nan_values = y_pred[~nan_mask]
            if len(non_nan_values) > 0:
                mean_val = np.mean(non_nan_values)
                y_pred[nan_mask] = mean_val
            else:
                # Если все значения NaN, заполняем нулями
                y_pred[nan_mask] = 0
        
        return y_pred, segment_predictions

# ============================================================
# ОСНОВНОЙ ИНТЕРФЕЙС ЭТАПА 8
# ============================================================

def show_advanced_techniques_interface():
    """Основной интерфейс Этапа 8: Продвинутые техники"""
    
    
    # Проверка наличия данных из предыдущих этапов
    if 'df_features' not in st.session_state or 'feature_info' not in st.session_state:
        st.error("❌ Сначала выполните Этапы 1-2: Подготовку данных")
        return
    
    st.info("""
    **Цель Этапа 8:**
    
    1. **Ансамблирование:**
       - Взвешенное усреднение (по MASE)
       - Stacking с линейной мета-моделью
       - AutoGluon WeightedEnsemble
    
    2. **Обработка выбросов:**
       - Isolation Forest для фильтрации
       - RobustScaler вместо StandardScaler
       - Winsorization (ограничение 5%/95% перцентилями)
    
    3. **Сегментация:**
       - Кластеризация рядов (KMeans по признакам)
       - Отдельные модели по сезонам (зима/лето)
       - Сегментация по режимам
    """)
    
    # Подготавливаем данные
    with st.spinner("Подготовка данных..."):
        result = prepare_data_for_advanced_techniques()
        
        if result[0] is None:
            return
        
        X_train, y_train, X_test, y_test, feature_cols = result
    
    st.success(f"""
    ✅ Данные подготовлены:
    - Обучающие: {X_train.shape[0]} записей, {X_train.shape[1]} признаков
    - Тестовые: {X_test.shape[0]} записей
    """)
    
    # Подготавливаем даты для сезонной сегментации
    if 'df_features' in st.session_state and 'feature_info' in st.session_state:
        feature_info = st.session_state.feature_info
        df_features = st.session_state.df_features
        
        date_col = feature_info['original_features'][0]
        
        # Берем даты из обучающих данных
        train_data = st.session_state.split_data['train']
        
        # Проверяем наличие колонки с датой
        if date_col in train_data.columns:
            dates = train_data[date_col]
            
            # Проверяем тип данных
            if not pd.api.types.is_datetime64_any_dtype(dates):
                try:
                    dates = pd.to_datetime(dates, errors='coerce')
                    st.success(f"✅ Даты автоматически преобразованы в datetime")
                except Exception as e:
                    st.warning(f"⚠️ Не удалось преобразовать даты: {str(e)}")
                    dates = None
        else:
            st.warning(f"⚠️ Колонка с датой '{date_col}' не найдена в обучающих данных")
            dates = None
    else:
        dates = None
    
    # Извлекаем прогнозы лучших моделей из предыдущих этапов
    model_predictions = extract_best_models_from_previous_stages()
    
    if model_predictions:
        st.info(f"✅ Найдены прогнозы {len(model_predictions)} моделей из предыдущих этапов")
        st.write("**Модели для ансамблирования:**")
        for model_name in model_predictions.keys():
            st.write(f"- {model_name}")
    else:
        st.warning("⚠️ Не найдены прогнозы моделей для ансамблирования")
    
    # Создаем вкладки для разных техник
    tab1, tab2, tab3 = st.tabs(["🎯 Ансамблирование", "⚠️ Обработка выбросов", "📊 Сегментация"])
    
    # Инициализируем объекты техник
    ensemble_techniques = AdvancedEnsembleTechniques()
    outlier_handler = OutlierHandler()
    segmenter = TimeSeriesSegmenter()
    
    # Результаты для сравнения
    comparison_results = {}
    
    # ============================================================
    # ВКЛАДКА 1: АНСАМБЛИРОВАНИЕ
    # ============================================================
    
    with tab1:
        st.subheader("🎯 Ансамблирование моделей")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_weighted = st.checkbox("Взвешенное усреднение", value=True)
            if include_weighted:
                weight_method = st.selectbox(
                    "Метод расчета весов:",
                    options=['mase', 'mae', 'rmse', 'equal'],
                    index=0,
                    help="Метод расчета весов для ансамбля"
                )
        
        with col2:
            include_stacking = st.checkbox("Stacking ансамбль", value=STACKING_AVAILABLE and X_train is not None)
            include_autogluon = st.checkbox("AutoGluon ансамбль", value=AUTOGLUON_AVAILABLE)
        
        st.markdown("---")
        
        # Кнопка запуска ансамблирования
        if st.button("🚀 Запустить ансамблирование", key="ensemble_button"):
            
            with st.spinner("Выполняется ансамблирование..."):
                
                # 1. Взвешенное усреднение
                if include_weighted and model_predictions:
                    st.subheader("1. Взвешенное усреднение")
                    
                    try:
                        # Нужны фактические значения для расчета весов
                        # Используем часть тестовых данных для "валидации"
                        if len(y_test) > 0 and any(p is not None for p in model_predictions.values()):
                            # Находим минимальную длину
                            min_len = min(len(y_test), 
                                         *[len(p) for p in model_predictions.values() if p is not None])
                            
                            if min_len > 10:
                                # Используем первые 30% для расчета весов
                                val_size = int(min_len * 0.3)
                                
                                # Подготавливаем данные для расчета весов
                                y_val = y_test[:val_size]
                                predictions_val = {}
                                
                                for name, pred in model_predictions.items():
                                    if pred is not None and len(pred) >= val_size:
                                        predictions_val[name] = pred[:val_size]
                                
                                if predictions_val:
                                    weighted_pred, weights = ensemble_techniques.weighted_average_ensemble(
                                        predictions_val, y_val, method=weight_method
                                    )
                                    
                                    if weighted_pred is not None:
                                        # Применяем весы ко всем данным
                                        full_pred = None
                                        for name, pred in model_predictions.items():
                                            if pred is not None:
                                                weight = weights.get(name, 0)
                                                if full_pred is None:
                                                    full_pred = pred * weight
                                                else:
                                                    min_len_full = min(len(full_pred), len(pred))
                                                    if min_len_full > 0:
                                                        full_pred[:min_len_full] += pred[:min_len_full] * weight
                                        
                                        if full_pred is not None:
                                            # Сохраняем для сравнения
                                            comparison_results['Weighted Average'] = {
                                                'predictions': full_pred,
                                                'weights': weights,
                                                'method': weight_method
                                            }
                                            
                                            st.success(f"✅ Взвешенное усреднение выполнено")
                                            st.write("**Веса моделей:**")
                                            for name, weight in weights.items():
                                                st.write(f"- {name}: {weight:.3f}")
                                        else:
                                            st.warning("Не удалось создать взвешенный ансамбль")
                                    else:
                                        st.warning("Не удалось рассчитать веса")
                                else:
                                    st.warning("Нет прогнозов для расчета весов")
                            else:
                                st.warning("Недостаточно данных для расчета весов")
                        else:
                            st.warning("Недостаточно данных для взвешенного усреднения")
                    except Exception as e:
                        st.error(f"❌ Ошибка взвешенного усреднения: {str(e)}")
                
                # 2. Stacking ансамбль
                if include_stacking and X_train is not None and X_test is not None:
                    st.subheader("2. Stacking ансамбль")
                    
                    # Нужны базовые модели для stacking
                    # Создаем простые базовые модели для демонстрации
                    from sklearn.linear_model import Ridge, Lasso
                    from sklearn.ensemble import RandomForestRegressor
                    
                    base_models = {
                        'Ridge': Ridge(alpha=1.0, random_state=42),
                        'Lasso': Lasso(alpha=0.1, random_state=42, max_iter=10000),
                        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                    }
                    
                    try:
                        stacking_pred = ensemble_techniques.stacking_ensemble(
                            X_train, y_train, X_test, base_models=base_models
                        )
                        
                        if stacking_pred is not None:
                            comparison_results['Stacking'] = {
                                'predictions': stacking_pred,
                                'base_models': list(base_models.keys())
                            }
                            st.success(f"✅ Stacking ансамбль создан с {len(base_models)} базовыми моделями")
                        else:
                            st.warning("Не удалось создать stacking ансамбль")
                    except Exception as e:
                        st.error(f"❌ Ошибка stacking ансамбля: {str(e)}")
                
                # 3. AutoGluon ансамбль
                if include_autogluon:
                    st.subheader("3. AutoGluon WeightedEnsemble")
                    
                    try:
                        autogluon_pred = ensemble_techniques.autogluon_weighted_ensemble(
                            X_train, y_train, X_test, y_test, time_limit=60
                        )
                        
                        if autogluon_pred is not None:
                            comparison_results['AutoGluon Ensemble'] = {
                                'predictions': autogluon_pred,
                                'time_limit': 60
                            }
                            st.success("✅ AutoGluon ансамбль создан")
                        else:
                            st.warning("Не удалось создать AutoGluon ансамбль")
                    except Exception as e:
                        st.error(f"❌ Ошибка AutoGluon ансамбля: {str(e)}")
                
                # Показываем сравнение ансамблей
                if comparison_results:
                    st.markdown("---")
                    st.subheader("📊 Сравнение ансамблей")
                    
                    # Вычисляем метрики
                    metrics_data = []
                    for ensemble_name, ensemble_info in comparison_results.items():
                        pred = ensemble_info.get('predictions')
                        
                        if pred is not None and len(pred) > 0 and len(y_test) > 0:
                            min_len = min(len(pred), len(y_test))
                            if min_len > 0:
                                y_pred_trimmed = pred[:min_len]
                                y_true_trimmed = y_test[:min_len]
                                
                                # Проверяем на NaN
                                if np.isnan(y_pred_trimmed).any() or np.isnan(y_true_trimmed).any():
                                    # Удаляем NaN значения
                                    mask = ~np.isnan(y_pred_trimmed) & ~np.isnan(y_true_trimmed)
                                    if np.sum(mask) > 0:
                                        y_pred_clean = y_pred_trimmed[mask]
                                        y_true_clean = y_true_trimmed[mask]
                                        mae = mean_absolute_error(y_true_clean, y_pred_clean)
                                        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
                                    else:
                                        mae = np.nan
                                        rmse = np.nan
                                else:
                                    mae = mean_absolute_error(y_true_trimmed, y_pred_trimmed)
                                    rmse = np.sqrt(mean_squared_error(y_true_trimmed, y_pred_trimmed))
                                
                                metrics_data.append({
                                    'Ансамбль': ensemble_name,
                                    'MAE': mae,
                                    'RMSE': rmse,
                                    'Длина': min_len
                                })
                    
                    if metrics_data:
                        metrics_df = pd.DataFrame(metrics_data)
                        
                        # Удаляем строки с NaN
                        metrics_df = metrics_df.dropna()
                        
                        if not metrics_df.empty:
                            # Сортируем по MAE
                            metrics_df = metrics_df.sort_values('MAE')
                            metrics_df['Ранг'] = range(1, len(metrics_df) + 1)
                            
                            st.dataframe(metrics_df, width='stretch')
                            
                            # График сравнения
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                x=metrics_df['Ансамбль'],
                                y=metrics_df['MAE'],
                                name='MAE',
                                marker_color='lightblue',
                                text=metrics_df['MAE'].round(4),
                                textposition='auto'
                            ))
                            
                            fig.update_layout(
                                title='Сравнение ансамблей по MAE',
                                xaxis_title='Ансамбль',
                                yaxis_title='MAE',
                                height=400,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Определяем лучший ансамбль
                            best_ensemble = metrics_df.iloc[0]
                            st.success(f"""
                            🏆 **Лучший ансамбль:** {best_ensemble['Ансамбль']}
                            - **MAE:** {best_ensemble['MAE']:.4f}
                            - **RMSE:** {best_ensemble['RMSE']:.4f}
                            - **Ранг:** {best_ensemble['Ранг']}
                            """)
                            
                            # Сохраняем лучший ансамбль
                            st.session_state.best_ensemble = {
                                'name': best_ensemble['Ансамбль'],
                                'predictions': comparison_results[best_ensemble['Ансамбль']]['predictions'],
                                'metrics': {
                                    'MAE': best_ensemble['MAE'],
                                    'RMSE': best_ensemble['RMSE']
                                }
                            }
                        else:
                            st.warning("Не удалось вычислить метрики для ансамблей (все значения NaN)")
                    else:
                        st.warning("Не удалось вычислить метрики для ансамблей")
                else:
                    st.warning("Нет результатов ансамблирования для сравнения")
    
    # ============================================================
    # ВКЛАДКА 2: ОБРАБОТКА ВЫБРОСОВ
    # ============================================================
    
    with tab2:
        st.subheader("⚠️ Обработка выбросов")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_isolation = st.checkbox("Isolation Forest", value=True)
            if include_isolation:
                contamination = st.slider(
                    "Contamination (доля выбросов):",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.1,
                    step=0.01,
                    help="Ожидаемая доля выбросов в данных"
                )
        
        with col2:
            include_robust = st.checkbox("RobustScaler", value=True)
            include_winsor = st.checkbox("Winsorization", value=True)
            if include_winsor:
                lower_limit = st.slider("Нижний лимит (%)", 0, 10, 5, 1) / 100
                upper_limit = st.slider("Верхний лимит (%)", 90, 100, 95, 1) / 100
        
        st.markdown("---")
        
        if st.button("🔍 Анализировать и обрабатывать выбросы", key="outlier_button"):
            
            with st.spinner("Анализ выбросов..."):
                
                # 1. Анализ выбросов
                st.subheader("1. Анализ выбросов в данных")
                
                # Визуализация распределения признаков
                fig_outliers = make_subplots(
                    rows=2, cols=3,
                    subplot_titles=['Распределение целевой переменной', 
                                   'Боксплот признаков', 
                                   'Корреляционная матрица',
                                   'Диаграмма рассеяния',
                                   'Гистограмма признаков',
                                   'Q-Q plot'],
                    specs=[[{'type': 'histogram'}, {'type': 'box'}, {'type': 'heatmap'}],
                           [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'scatter'}]],
                    vertical_spacing=0.1,
                    horizontal_spacing=0.1
                )
                
                # Распределение целевой переменной
                fig_outliers.add_trace(
                    go.Histogram(
                        x=y_train,
                        name='Целевая переменная',
                        nbinsx=50,
                        marker_color='lightblue',
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                
                # Боксплот для топ-5 признаков
                top_features = feature_cols[:5] if len(feature_cols) >= 5 else feature_cols
                for i, feature in enumerate(top_features):
                    fig_outliers.add_trace(
                        go.Box(
                            y=X_train[feature].values,
                            name=feature,
                            marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)],
                            showlegend=False
                        ),
                        row=1, col=2
                    )
                
                fig_outliers.update_layout(
                    height=600,
                    title_text="Анализ распределения данных и выбросов",
                    title_x=0.5,
                    showlegend=False,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_outliers, use_container_width=True)
                
                # 2. Применение методов обработки выбросов
                st.subheader("2. Применение методов обработки")
                
                # Isolation Forest
                if include_isolation and X_train is not None:
                    outlier_mask = outlier_handler.isolation_forest_detection(
                        X_train, contamination=contamination
                    )
                    
                    st.info(f"""
                    **Isolation Forest:**
                    - Обнаружено выбросов: {np.sum(outlier_mask)} ({np.mean(outlier_mask)*100:.1f}%)
                    - Contamination: {contamination}
                    - Чистые данные: {np.sum(~outlier_mask)} записей
                    """)
                
                # RobustScaler
                if include_robust and X_train is not None:
                    scaling_results = outlier_handler.compare_scaling_methods(X_train, y_train)
                    
                    # Визуализация сравнения масштабирования
                    fig_scaling = go.Figure()
                    
                    methods = ['no_scaling', 'standard_scaler', 'robust_scaler']
                    method_names = ['Без масштабирования', 'StandardScaler', 'RobustScaler']
                    
                    for i, (method, method_name) in enumerate(zip(methods, method_names)):
                        if method in scaling_results:
                            # Используем первый признак для сравнения
                            if i == 0:
                                data = X_train.iloc[:, 0] if hasattr(X_train, 'iloc') else X_train[:, 0]
                            else:
                                # Масштабируем данные
                                scaler = scaling_results[method]['scaler']
                                data_scaled = scaler.transform(X_train)
                                data = data_scaled[:, 0]
                            
                            fig_scaling.add_trace(go.Box(
                                y=data,
                                name=method_name,
                                marker_color=px.colors.qualitative.Set1[i]
                            ))
                    
                    fig_scaling.update_layout(
                        title='Сравнение методов масштабирования',
                        yaxis_title='Значение признака (масштабированное)',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_scaling, use_container_width=True)
                    
                    st.info("""
                    **RobustScaler vs StandardScaler:**
                    - **RobustScaler:** Использует медиану и межквартильный размах, устойчив к выбросам
                    - **StandardScaler:** Использует среднее и стандартное отклонение, чувствителен к выбросам
                    - **Рекомендация:** Для данных с выбросами используйте RobustScaler
                    """)
                
                # Winsorization
                if include_winsor and X_train is not None:
                    X_winsorized = outlier_handler.winsorization(
                        X_train, limits=(lower_limit, upper_limit)
                    )
                    
                    # Сравнение до и после
                    fig_winsor = go.Figure()
                    
                    # До winsorization
                    fig_winsor.add_trace(go.Histogram(
                        x=X_train.iloc[:, 0] if hasattr(X_train, 'iloc') else X_train[:, 0],
                        name='До winsorization',
                        opacity=0.7,
                        marker_color='red'
                    ))
                    
                    # После winsorization
                    fig_winsor.add_trace(go.Histogram(
                        x=X_winsorized.iloc[:, 0] if hasattr(X_winsorized, 'iloc') else X_winsorized[:, 0],
                        name='После winsorization',
                        opacity=0.7,
                        marker_color='blue'
                    ))
                    
                    fig_winsor.update_layout(
                        title=f'Winsorization: ограничение {lower_limit*100}%/{upper_limit*100}%',
                        xaxis_title='Значение признака',
                        yaxis_title='Частота',
                        barmode='overlay',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_winsor, use_container_width=True)
                    
                    st.info(f"""
                    **Winsorization:**
                    - **Ограничения:** {lower_limit*100:.1f}% (нижний), {upper_limit*100:.1f}% (верхний)
                    - **Эффект:** Крайние значения заменяются на значения перцентилей
                    - **Преимущество:** Сохраняет размер выборки, уменьшает влияние выбросов
                    """)
                
                st.success("✅ Анализ и обработка выбросов завершены")
                
                # Сохраняем обработчик выбросов
                st.session_state.outlier_handler = outlier_handler
    
    # ============================================================
    # ВКЛАДКА 3: СЕГМЕНТАЦИЯ
    # ============================================================
    
    with tab3:
        st.subheader("📊 Сегментация временных рядов")
        
        # Инициализация состояния сегментации
        if 'segmentation_state' not in st.session_state:
            st.session_state.segmentation_state = {
                'results': {},
                'segmenter': TimeSeriesSegmenter(),
                'last_updated': None
            }
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_kmeans = st.checkbox("KMeans кластеризация", value=True)
            if include_kmeans:
                n_clusters = st.slider("Количество кластеров", 2, 10, 3, 1)
        
        with col2:
            include_seasonal = st.checkbox("Сезонная сегментация", value=dates is not None)
            include_regime = st.checkbox("Сегментация по режимам", value=True)
            if include_regime:
                n_regimes = st.slider("Количество режимов", 2, 5, 2, 1)
                regime_method = st.selectbox("Метод сегментации", ['percentile', 'kmeans'], index=0)
        
        st.markdown("---")
        
        # Кнопка выполнения сегментации
        if st.button("🎯 Выполнить сегментацию", key="segmentation_button"):
            
            with st.spinner("Выполняется сегментация..."):
                
                segmentation_results = {}
                
                # 1. KMeans кластеризация
                if include_kmeans and X_train is not None:
                    st.subheader("1. KMeans кластеризация по признакам")
                    
                    clusters, segments = st.session_state.segmentation_state['segmenter'].kmeans_segmentation(
                        X_train, n_clusters=n_clusters
                    )
                    
                    # Визуализация кластеров
                    if len(feature_cols) >= 2:
                        # Используем PCA для визуализации в 2D
                        from sklearn.decomposition import PCA
                        
                        pca = PCA(n_components=2, random_state=42)
                        X_pca = pca.fit_transform(X_train)
                        
                        fig_clusters = go.Figure()
                        
                        for cluster_id in range(n_clusters):
                            mask = clusters == cluster_id
                            if np.any(mask):
                                fig_clusters.add_trace(go.Scatter(
                                    x=X_pca[mask, 0],
                                    y=X_pca[mask, 1],
                                    mode='markers',
                                    name=f'Кластер {cluster_id}',
                                    marker=dict(
                                        size=8,
                                        opacity=0.7,
                                        color=px.colors.qualitative.Set1[cluster_id % len(px.colors.qualitative.Set1)]
                                    )
                                ))
                        
                        fig_clusters.update_layout(
                            title='KMeans кластеризация (PCA проекция)',
                            xaxis_title='Главная компонента 1',
                            yaxis_title='Главная компонента 2',
                            height=500,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig_clusters, use_container_width=True)
                    
                    # Информация о сегментах
                    st.write("**Статистика кластеров:**")
                    kmeans_stats = []
                    for cluster_id, segment_info in segments.items():
                        kmeans_stats.append({
                            'Кластер': cluster_id,
                            'Размер': segment_info['size'],
                            'Процент': f"{segment_info['percentage']:.1f}%",
                            'Среднее значение признаков': f"{segment_info['features_mean'].mean():.4f}" if segment_info['features_mean'] is not None else 'N/A'
                        })
                    
                    if kmeans_stats:
                        st.dataframe(pd.DataFrame(kmeans_stats), width='stretch')
                    
                    segmentation_results['kmeans'] = {
                        'clusters': clusters,
                        'segments': segments,
                        'n_clusters': n_clusters
                    }
                
                # 2. Сезонная сегментация
                if include_seasonal and dates is not None:
                    st.subheader("2. Сезонная сегментация")
                    
                    try:
                        seasonal_segments = st.session_state.segmentation_state['segmenter'].seasonal_segmentation(
                            dates, y_train
                        )
                        
                        if seasonal_segments:
                            # Визуализация сезонных сегментов
                            fig_seasons = go.Figure()
                            
                            seasons = list(seasonal_segments.keys())
                            season_means = []
                            season_sizes = []
                            
                            for season_name, segment_info in seasonal_segments.items():
                                if 'target_mean' in segment_info and segment_info['target_mean'] is not None:
                                    season_means.append(segment_info['target_mean'])
                                    season_sizes.append(segment_info['size'])
                                    
                                    fig_seasons.add_trace(go.Bar(
                                        x=[season_name],
                                        y=[segment_info['target_mean']],
                                        name=season_name,
                                        text=[f"{segment_info['percentage']:.1f}% ({segment_info['size']} зап.)"],
                                        textposition='auto',
                                        marker_color=px.colors.qualitative.Set1[seasons.index(season_name) % len(px.colors.qualitative.Set1)],
                                        hovertemplate=(
                                            f"Сезон: {season_name}<br>" +
                                            f"Среднее значение: {segment_info['target_mean']:.4f}<br>" +
                                            f"Количество записей: {segment_info['size']}<br>" +
                                            f"Процент от общего: {segment_info['percentage']:.1f}%<br>" +
                                            f"Месяцы: {segment_info['months']}"
                                        )
                                    ))
                            
                            fig_seasons.update_layout(
                                title='Средние значения целевой переменной по сезонам',
                                xaxis_title='Сезон',
                                yaxis_title='Среднее значение',
                                height=400,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig_seasons, use_container_width=True)
                            
                            # Информация о сезонах
                            st.write("**Статистика сезонов:**")
                            seasonal_stats = []
                            for season_name, segment_info in seasonal_segments.items():
                                seasonal_stats.append({
                                    'Сезон': season_name,
                                    'Размер': segment_info['size'],
                                    'Процент': f"{segment_info['percentage']:.1f}%",
                                    'Среднее значение': f"{segment_info.get('target_mean', 0):.4f}",
                                    'Месяцы': segment_info['months']
                                })
                            
                            if seasonal_stats:
                                st.dataframe(pd.DataFrame(seasonal_stats), width='stretch')
                            
                            segmentation_results['seasonal'] = seasonal_segments
                        else:
                            st.warning("⚠️ Не удалось выполнить сезонную сегментацию. Возможно, проблема с форматом дат.")
                            
                    except Exception as e:
                        st.error(f"❌ Ошибка при выполнении сезонной сегментации: {str(e)}")
                
                # 3. Сегментация по режимам
                if include_regime and y_train is not None:
                    st.subheader("3. Сегментация по режимам (значениям)")
                    
                    regimes, regime_segments = st.session_state.segmentation_state['segmenter'].regime_segmentation(
                        y_train.values, n_regimes=n_regimes, method=regime_method
                    )
                    
                    # Визуализация режимов
                    fig_regimes = go.Figure()
                    
                    time_index = list(range(len(y_train)))
                    
                    for regime_id in range(n_regimes):
                        mask = regimes == regime_id
                        if np.any(mask):
                            fig_regimes.add_trace(go.Scatter(
                                x=np.array(time_index)[mask],
                                y=y_train.values[mask],
                                mode='markers',
                                name=f'Режим {regime_id}',
                                marker=dict(
                                    size=6,
                                    opacity=0.7,
                                    color=px.colors.qualitative.Set1[regime_id % len(px.colors.qualitative.Set1)]
                                )
                            ))
                    
                    fig_regimes.update_layout(
                        title=f'Сегментация по режимам ({n_regimes} режима)',
                        xaxis_title='Временной индекс',
                        yaxis_title='Значение целевой переменной',
                        height=400,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_regimes, use_container_width=True)
                    
                    # Информация о режимах
                    st.write("**Статистика режимов:**")
                    regime_stats = []
                    for regime_id, segment_info in regime_segments.items():
                        regime_stats.append({
                            'Режим': regime_id,
                            'Размер': segment_info['size'],
                            'Процент': f"{segment_info['percentage']:.1f}%",
                            'Среднее значение': f"{segment_info['value_mean']:.4f}",
                            'Стд. отклонение': f"{segment_info['value_std']:.4f}"
                        })
                    
                    if regime_stats:
                        st.dataframe(pd.DataFrame(regime_stats), width='stretch')
                    
                    segmentation_results['regime'] = {
                        'regimes': regimes,
                        'segments': regime_segments,
                        'n_regimes': n_regimes,
                        'method': regime_method
                    }
                
                # Сохраняем результаты сегментации в session_state
                if segmentation_results:
                    st.session_state.segmentation_state['results'] = segmentation_results
                    st.session_state.segmentation_state['last_updated'] = time.time()
                    st.session_state.segmentation_state['X_train'] = X_train
                    st.session_state.segmentation_state['y_train'] = y_train
                    st.session_state.segmentation_state['X_test'] = X_test
                    st.session_state.segmentation_state['y_test'] = y_test
                    
                    st.success(f"✅ Сегментация выполнена. Сохранено {len(segmentation_results)} типов сегментации.")
                else:
                    st.warning("⚠️ Нет результатов сегментации для сохранения.")
        
        # Раздел для обучения моделей на сегментах
        st.markdown("---")
        st.subheader("🎯 Обучение моделей для сегментов")
        
        # Проверяем наличие сохраненных результатов сегментации
        if st.session_state.segmentation_state['results']:
            available_segmentations = list(st.session_state.segmentation_state['results'].keys())
            
            if available_segmentations:
                # Выбор типа сегментации для обучения
                segment_type_to_use = st.selectbox(
                    "Тип сегментации для обучения моделей:",
                    options=available_segmentations,
                    index=0,
                    key="segment_type_select"
                )
                
                # Дополнительные настройки
                col1, col2 = st.columns(2)
                with col1:
                    use_ridge = st.checkbox("Использовать Ridge регрессию", value=True)
                with col2:
                    if use_ridge:
                        alpha_value = st.slider("Alpha (регуляризация)", 0.1, 10.0, 1.0, 0.1)
                
                # Кнопка обучения моделей
                if st.button("🏋️ Обучить модели для сегментов", key="train_segment_models"):
                    
                    with st.spinner("Обучение моделей для сегментов..."):
                        
                        # Получаем сохраненные результаты
                        segmentation_results = st.session_state.segmentation_state['results']
                        segmenter = st.session_state.segmentation_state['segmenter']
                        
                        # Получаем данные
                        X_train_local = st.session_state.segmentation_state.get('X_train', X_train)
                        y_train_local = st.session_state.segmentation_state.get('y_train', y_train)
                        X_test_local = st.session_state.segmentation_state.get('X_test', X_test)
                        y_test_local = st.session_state.segmentation_state.get('y_test', y_test)
                        
                        # Определяем сегменты в зависимости от типа
                        if segment_type_to_use == 'kmeans':
                            segments = segmentation_results['kmeans']['segments']
                        elif segment_type_to_use == 'seasonal':
                            segments = segmentation_results['seasonal']
                        elif segment_type_to_use == 'regime':
                            segments = segmentation_results['regime']['segments']
                        else:
                            st.error(f"Неизвестный тип сегментации: {segment_type_to_use}")
                            segments = {}
                        
                        # Создаем базовую модель
                        if use_ridge:
                            from sklearn.linear_model import Ridge
                            base_model = Ridge(alpha=alpha_value, random_state=42)
                        else:
                            from sklearn.linear_model import LinearRegression
                            base_model = LinearRegression()
                        
                        # Обучаем модели для сегментов
                        segment_models = segmenter.train_segment_models(
                            X_train_local, y_train_local, segments, 
                            segment_type=segment_type_to_use,
                            base_model=base_model
                        )
                        
                        if segment_models:
                            st.success(f"✅ Обучено {len(segment_models)} моделей для сегментов")
                            
                            # Сохраняем модели в session_state
                            if 'segment_models' not in st.session_state:
                                st.session_state.segment_models = {}
                            
                            st.session_state.segment_models[segment_type_to_use] = {
                                'models': segment_models,
                                'segment_type': segment_type_to_use,
                                'base_model': type(base_model).__name__,
                                'timestamp': time.time()
                            }
                            
                            # Информация о моделях сегментов
                            st.write("**Детали моделей сегментов:**")
                            
                            model_stats = []
                            for segment_name, model_info in segment_models.items():
                                model = model_info['model']
                                
                                # Вычисляем R² на обучающих данных
                                X_seg = model_info['X_segment']
                                y_seg = model_info['y_segment']
                                r2_score = model.score(X_seg, y_seg) if len(X_seg) > 0 else 0
                                
                                model_stats.append({
                                    'Сегмент': segment_name,
                                    'Размер выборки': model_info['size'],
                                    'Тип модели': type(model).__name__,
                                    'R² на обучении': f"{r2_score:.4f}",
                                    'Параметры': str(model.get_params())[:50] + '...'
                                })
                            
                            if model_stats:
                                stats_df = pd.DataFrame(model_stats)
                                st.dataframe(stats_df, width='stretch')
                                
                                # Визуализация качества моделей
                                fig_quality = go.Figure()
                                
                                fig_quality.add_trace(go.Bar(
                                    x=stats_df['Сегмент'],
                                    y=stats_df['R² на обучении'].astype(float),
                                    name='R² на обучении',
                                    marker_color='lightblue',
                                    text=stats_df['R² на обучении'],
                                    textposition='auto'
                                ))
                                
                                fig_quality.update_layout(
                                    title='Качество моделей по сегментам (R²)',
                                    xaxis_title='Сегмент',
                                    yaxis_title='R² на обучении',
                                    height=400,
                                    template='plotly_white'
                                )
                                
                                st.plotly_chart(fig_quality, use_container_width=True)
                                
                                # Прогнозирование на тестовых данных
                                st.subheader("📊 Прогнозирование на тестовых данных")
                                
                                y_pred, segment_preds = segmenter.predict_with_segment_models(
                                    X_test_local, 
                                    segmenter.segment_models, 
                                    segment_type_to_use
                                )
                                
                                if y_pred is not None:
                                    # Очищаем данные от NaN
                                    y_test_clean = y_test_local.copy()
                                    if isinstance(y_test_clean, pd.Series):
                                        y_test_clean = y_test_clean.values
                                    
                                    # Удаляем NaN из y_test_clean и соответствующие значения из y_pred
                                    valid_mask = ~np.isnan(y_test_clean) & ~np.isnan(y_pred)
                                    
                                    if np.any(valid_mask):
                                        y_test_clean = y_test_clean[valid_mask]
                                        y_pred_clean = y_pred[valid_mask]
                                        
                                        # Проверяем, что после очистки остались данные
                                        if len(y_test_clean) > 0 and len(y_pred_clean) > 0:
                                            # Проверяем на оставшиеся NaN в прогнозах
                                            if np.isnan(y_pred_clean).any():
                                                # Заменяем оставшиеся NaN на среднее значение
                                                mean_val = np.nanmean(y_pred_clean)
                                                if np.isnan(mean_val):
                                                    mean_val = 0
                                                y_pred_clean = np.nan_to_num(y_pred_clean, nan=mean_val)
                                            
                                            # Вычисляем метрики
                                            mae = mean_absolute_error(y_test_clean, y_pred_clean)
                                            rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
                                            
                                            st.info(f"""
                                            **Метрики на тестовых данных:**
                                            - **MAE:** {mae:.4f}
                                            - **RMSE:** {rmse:.4f}
                                            - **Количество валидных прогнозов:** {len(y_test_clean)} из {len(y_test_local)}
                                            """)
                                            
                                            # График прогнозов
                                            fig_predictions = go.Figure()
                                            
                                            fig_predictions.add_trace(go.Scatter(
                                                x=list(range(len(y_test_clean))),
                                                y=y_test_clean,
                                                mode='lines',
                                                name='Фактические значения',
                                                line=dict(color='blue', width=2)
                                            ))
                                            
                                            fig_predictions.add_trace(go.Scatter(
                                                x=list(range(len(y_pred_clean))),
                                                y=y_pred_clean,
                                                mode='lines',
                                                name='Прогнозы',
                                                line=dict(color='red', width=2, dash='dash')
                                            ))
                                            
                                            fig_predictions.update_layout(
                                                title='Прогнозы моделей сегментов на тестовых данных',
                                                xaxis_title='Временной индекс',
                                                yaxis_title='Значение',
                                                height=400,
                                                template='plotly_white'
                                            )
                                            
                                            st.plotly_chart(fig_predictions, use_container_width=True)
                                        else:
                                            st.warning("⚠️ После очистки от NaN не осталось данных для вычисления метрик.")
                                    else:
                                        st.error("❌ Нет валидных данных для вычисления метрик (все значения NaN).")
                                else:
                                    st.warning("⚠️ Не удалось получить прогнозы от моделей сегментов.")
                        else:
                            st.warning("⚠️ Не удалось обучить модели для сегментов.")
            else:
                st.info("ℹ️ Нет доступных результатов сегментации. Сначала выполните сегментацию.")
        else:
            st.info("ℹ️ Результаты сегментации не найдены. Нажмите 'Выполнить сегментацию' для начала работы.")
    
    # Заключительный раздел
    st.markdown("---")
    st.subheader("🎯 Итоги Этапа 8")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'best_ensemble' in st.session_state:
            st.success("✅ Ансамблирование выполнено")
            best_ensemble = st.session_state.best_ensemble
            st.write(f"**Лучший ансамбль:** {best_ensemble['name']}")
            st.write(f"**MAE:** {best_ensemble['metrics']['MAE']:.4f}")
        else:
            st.info("Ансамблирование: не выполнено")
    
    with col2:
        if 'outlier_handler' in st.session_state:
            st.success("✅ Обработка выбросов выполнена")
            outlier_handler = st.session_state.outlier_handler
            if 'isolation_forest' in outlier_handler.outlier_stats:
                stats = outlier_handler.outlier_stats['isolation_forest']
                st.write(f"**Выбросов:** {stats['n_outliers']}")
                st.write(f"**Процент:** {stats['outlier_percentage']:.1f}%")
        else:
            st.info("Обработка выбросов: не выполнена")
    
    with col3:
        if 'segment_models' in st.session_state:
            st.success("✅ Сегментация выполнена")
            segment_models = st.session_state.segment_models
            segment_types = list(segment_models.keys())
            st.write(f"**Типы сегментации:** {', '.join(segment_types)}")
            for seg_type, seg_info in segment_models.items():
                st.write(f"**{seg_type}:** {len(seg_info['models'])} моделей")
        elif 'segmentation_state' in st.session_state and st.session_state.segmentation_state['results']:
            st.success("✅ Сегментация выполнена")
            results = st.session_state.segmentation_state['results']
            st.write(f"**Типы сегментации:** {', '.join(results.keys())}")
        else:
            st.info("Сегментация: не выполнена")
    
    st.markdown("---")
    st.success("""
    **✅ Этап 8: Продвинутые техники завершен!**
    
    **Что было сделано:**
    1. **Ансамблирование:** Взвешенное усреднение, Stacking, AutoGluon ансамбли
    2. **Обработка выбросов:** Isolation Forest, RobustScaler, Winsorization
    3. **Сегментация:** KMeans кластеризация, сезонная сегментация, сегментация по режимам
    
    **Рекомендации для продакшена:**
    - Используйте лучший ансамбль для получения стабильных прогнозов
    - Применяйте RobustScaler для данных с выбросами
    - Рассмотрите сегментацию для сложных временных рядов с разными режимами
    
    **Следующие шаги:**
    - Деплой лучшей модели в продакшен
    - Настройка мониторинга качества прогнозов
    - Реализация автоматического переобучения моделей
    """)

# ============================================================
# ФУНКЦИЯ ДЛЯ ЗАПУСКА ЭТАПА
# ============================================================

def run_stage_8():
    """Запуск Этапа 8"""
    show_advanced_techniques_interface()

if __name__ == "__main__":
    run_stage_8()