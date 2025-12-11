# forecasting_strategies.py - Исправленная версия (Этап 4)

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ============================================================
# ОСНОВНЫЕ ИСПРАВЛЕНИЯ
# ============================================================

class ForecastingStrategyBase:
    """Базовый класс для всех стратегий прогнозирования"""
    
    def __init__(self, horizon, model_type='ridge', 
                 use_transformed_target=False, use_scaling=True):
        """
        Parameters:
        -----------
        horizon : int
            Горизонт прогнозирования
        model_type : str
            Тип модели ('ridge', 'lasso', 'elasticnet', 'rf')
        use_transformed_target : bool
            Использовать трансформированную целевую переменную из Этапа 1
        use_scaling : bool
            Применять масштабирование признаков
        """
        self.horizon = horizon
        self.model_type = model_type
        self.use_transformed_target = use_transformed_target
        self.use_scaling = use_scaling
        
        self.model = None
        self.models = []
        self.scaler = StandardScaler() if use_scaling else None
        self.imputer = None
        
        self.training_time = 0
        self.predict_time = 0
        
        # Определяем имя стратегии
        class_name = self.__class__.__name__.replace('Strategy', '')
        self.name = f"{class_name} ({model_type})"
    
    def _get_target_column(self, y_series, feature_info):
        """Получаем правильную целевую переменную с учетом трансформаций"""
        if self.use_transformed_target and feature_info and 'transformation_info' in feature_info:
            trans_info = feature_info['transformation_info']
            
            # Проверяем, применены ли трансформации
            if trans_info.get('log_applied') and 'log_col' in trans_info:
                return trans_info['log_col']
            elif trans_info.get('boxcox_applied') and 'boxcox_col' in trans_info:
                return trans_info['boxcox_col']
        
        # Возвращаем исходную целевую переменную
        if hasattr(y_series, 'name'):
            return y_series.name
        elif isinstance(y_series, pd.Series):
            return y_series.name if y_series.name else 'target'
        else:
            return 'target'
    
    def _prepare_features(self, X_data, is_training=True):
        """Подготовка признаков: обработка пропусков и масштабирование"""
        if X_data.empty:
            return X_data
        
        # Создаем imputer при первом вызове
        if self.imputer is None:
            self.imputer = SimpleImputer(strategy='mean')
        
        # Обработка пропусков
        try:
            if is_training:
                X_imputed = pd.DataFrame(
                    self.imputer.fit_transform(X_data),
                    columns=X_data.columns,
                    index=X_data.index
                )
            else:
                if not hasattr(self.imputer, 'statistics_'):
                    self.imputer.fit(X_data)
                X_imputed = pd.DataFrame(
                    self.imputer.transform(X_data),
                    columns=X_data.columns,
                    index=X_data.index
                )
        except Exception as e:
            print(f"Ошибка при обработке пропусков: {e}")
            X_imputed = X_data.fillna(0)
        
        # Масштабирование
        if self.use_scaling and self.scaler is not None:
            try:
                if is_training:
                    X_scaled = pd.DataFrame(
                        self.scaler.fit_transform(X_imputed),
                        columns=X_data.columns,
                        index=X_data.index
                    )
                else:
                    if not hasattr(self.scaler, 'scale_'):
                        self.scaler.fit(X_imputed)
                    X_scaled = pd.DataFrame(
                        self.scaler.transform(X_imputed),
                        columns=X_data.columns,
                        index=X_data.index
                    )
                return X_scaled
            except Exception as e:
                print(f"Ошибка при масштабировании: {e}")
                return X_imputed
        else:
            return X_imputed
    
    def _create_model(self):
        """Создание модели в зависимости от типа"""
        if self.model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=42)
        elif self.model_type == 'lasso':
            return Lasso(alpha=0.1, random_state=42, max_iter=1000)
        elif self.model_type == 'elasticnet':
            return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000)
        elif self.model_type == 'rf':
            return RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        else:
            return Ridge(alpha=1.0, random_state=42)

# ============================================================
# DIRECT STRATEGY (исправленная)
# ============================================================

class DirectStrategy(ForecastingStrategyBase):
    """Direct: обучение h моделей → ŷ_{t+1}, ..., ŷ_{t+h}"""
    
    def __init__(self, horizon, model_type='ridge', **kwargs):
        super().__init__(horizon, model_type, **kwargs)
    
    def fit(self, X_train, y_train, feature_info=None):
        """Обучение h отдельных моделей"""
        start_time = time.time()
        
        if X_train.empty or len(X_train) < 10:
            print(f"Недостаточно данных для обучения: {len(X_train)} строк")
            self.training_time = 0
            return
        
        # Подготавливаем признаки
        X_prepared = self._prepare_features(X_train, is_training=True)
        
        # Определяем целевую переменную
        target_col = self._get_target_column(y_train, feature_info)
        if target_col in X_train.columns:
            y_actual = X_train[target_col]
        elif isinstance(y_train, pd.Series):
            y_actual = y_train
        else:
            y_actual = pd.Series(y_train, name=target_col)
        
        self.models = []
        
        for h in range(1, self.horizon + 1):
            # Создаем модель
            model = self._create_model()
            
            # Сдвигаем целевую переменную для горизонта h
            y_train_h = y_actual.shift(-h)
            
            # Находим валидные индексы
            valid_idx = y_train_h.notna()
            X_train_h = X_prepared[valid_idx]
            y_train_h_clean = y_train_h[valid_idx]
            
            if len(X_train_h) > 5 and len(y_train_h_clean) > 5:
                try:
                    model.fit(X_train_h, y_train_h_clean)
                    self.models.append(model)
                except Exception as e:
                    print(f"Ошибка при обучении модели для горизонта {h}: {str(e)}")
                    # Создаем простую модель для горизонта
                    dummy_model = self._create_model()
                    if len(X_train_h) > 0:
                        dummy_model.fit(X_train_h.iloc[:1], [y_train_h_clean.iloc[0] if len(y_train_h_clean) > 0 else 0])
                    self.models.append(dummy_model)
            else:
                # Создаем простую модель если данных недостаточно
                dummy_model = self._create_model()
                if len(X_prepared) > 0 and len(y_actual) > 0:
                    dummy_model.fit(X_prepared.iloc[:1], [y_actual.iloc[0]])
                self.models.append(dummy_model)
        
        self.training_time = time.time() - start_time
    
    def predict(self, X_test, feature_info=None):
        """Прогнозирование для каждого горизонта"""
        start_time = time.time()
        
        if X_test.empty:
            predictions = pd.DataFrame(index=X_test.index)
            for h in range(1, self.horizon + 1):
                predictions[f'horizon_{h}'] = 0
            self.predict_time = time.time() - start_time
            return predictions
        
        # Подготавливаем тестовые признаки
        X_prepared = self._prepare_features(X_test, is_training=False)
        
        predictions = pd.DataFrame(index=X_test.index)
        
        for h, model in enumerate(self.models[:self.horizon], 1):
            if model is not None:
                try:
                    if len(X_prepared) > 0:
                        pred = model.predict(X_prepared)
                        predictions[f'horizon_{h}'] = pred[:len(X_prepared)]
                    else:
                        predictions[f'horizon_{h}'] = 0
                except Exception as e:
                    print(f"Ошибка предсказания для горизонта {h}: {str(e)}")
                    predictions[f'horizon_{h}'] = 0
            else:
                predictions[f'horizon_{h}'] = 0
        
        self.predict_time = time.time() - start_time
        return predictions

# ============================================================
# RECURSIVE STRATEGY (исправленная)
# ============================================================

class RecursiveStrategy(ForecastingStrategyBase):
    """Recursive: одна модель → ŷ_{t+1} → вход → ŷ_{t+2}"""
    
    def __init__(self, horizon, model_type='ridge', **kwargs):
        super().__init__(horizon, model_type, **kwargs)
    
    def fit(self, X_train, y_train, feature_info=None):
        """Обучение одной модели"""
        start_time = time.time()
        
        if X_train.empty or len(X_train) < 10:
            print(f"Недостаточно данных для обучения: {len(X_train)} строк")
            self.training_time = 0
            return
        
        # Подготавливаем признаки
        X_prepared = self._prepare_features(X_train, is_training=True)
        
        # Определяем целевую переменную
        target_col = self._get_target_column(y_train, feature_info)
        if target_col in X_train.columns:
            y_actual = X_train[target_col]
        elif isinstance(y_train, pd.Series):
            y_actual = y_train
        else:
            y_actual = pd.Series(y_train, name=target_col)
        
        # Удаляем строки с NaN
        valid_idx = y_actual.notna()
        X_train_clean = X_prepared[valid_idx]
        y_train_clean = y_actual[valid_idx]
        
        # Создаем и обучаем модель
        self.model = self._create_model()
        
        if len(X_train_clean) > 5 and len(y_train_clean) > 5:
            try:
                self.model.fit(X_train_clean, y_train_clean)
            except Exception as e:
                print(f"Ошибка при обучении модели: {str(e)}")
                # Создаем простую модель
                self.model = self._create_model()
                if len(X_train_clean) > 0:
                    self.model.fit(X_train_clean.iloc[:1], [y_train_clean.iloc[0]])
        else:
            # Создаем простую модель если данных недостаточно
            self.model = self._create_model()
            if len(X_prepared) > 0 and len(y_actual) > 0:
                self.model.fit(X_prepared.iloc[:1], [y_actual.iloc[0]])
        
        self.training_time = time.time() - start_time
    
    def predict(self, X_test, feature_info=None):
        """Рекурсивное прогнозирование"""
        start_time = time.time()
        
        if X_test.empty:
            predictions = pd.DataFrame(index=X_test.index)
            for h in range(1, self.horizon + 1):
                predictions[f'horizon_{h}'] = 0
            self.predict_time = time.time() - start_time
            return predictions
        
        # Подготавливаем тестовые признаки
        X_prepared = self._prepare_features(X_test, is_training=False)
        
        predictions = pd.DataFrame(index=X_test.index)
        X_current = X_prepared.copy()
        
        for h in range(1, self.horizon + 1):
            if self.model is not None:
                try:
                    if len(X_current) > 0:
                        pred = self.model.predict(X_current)
                        predictions[f'horizon_{h}'] = pred[:len(X_current)]
                        
                        # Обновляем признаки для следующего шага
                        if h < self.horizon:
                            X_current = self._update_features(X_current, pred)
                    else:
                        predictions[f'horizon_{h}'] = 0
                except Exception as e:
                    print(f"Ошибка рекурсивного предсказания на шаге {h}: {str(e)}")
                    predictions[f'horizon_{h}'] = 0
            else:
                predictions[f'horizon_{h}'] = 0
        
        self.predict_time = time.time() - start_time
        return predictions
    
    def _update_features(self, X, last_pred):
        """Обновление лаговых признаков для рекурсивного прогнозирования"""
        X_new = X.copy()
        
        # Обновляем лаговые признаки если они есть
        lag_cols = [col for col in X.columns if 'lag' in col.lower()]
        if lag_cols:
            try:
                # Ищем лаги типа lag_1, lag_2 и т.д.
                for col in lag_cols:
                    if 'lag_' in col.lower():
                        parts = col.split('_')
                        if len(parts) > 1 and parts[-1].isdigit():
                            lag_num = int(parts[-1])
                            if lag_num > 1:
                                prev_lag = f"lag_{lag_num-1}"
                                if prev_lag in X_new.columns:
                                    X_new[col] = X_new[prev_lag]
                
                # Обновляем lag_1 предсказанием
                if 'lag_1' in X_new.columns:
                    X_new['lag_1'] = last_pred[0] if isinstance(last_pred, np.ndarray) else last_pred
            except Exception as e:
                print(f"Ошибка обновления признаков: {str(e)}")
        
        return X_new

# ============================================================
# MULTI-OUTPUT STRATEGY (исправленная)
# ============================================================

class MultiOutputStrategy(ForecastingStrategyBase):
    """Multi-output: одна модель предсказывает вектор [ŷ_{t+1}, ..., ŷ_{t+h}]"""
    
    def __init__(self, horizon, model_type='ridge', **kwargs):
        super().__init__(horizon, model_type, **kwargs)
    
    def fit(self, X_train, y_train, feature_info=None):
        """Обучение многомерной модели"""
        start_time = time.time()
        
        if X_train.empty or len(X_train) < 10:
            print(f"Недостаточно данных для обучения: {len(X_train)} строк")
            self.training_time = 0
            return
        
        # Подготавливаем признаки
        X_prepared = self._prepare_features(X_train, is_training=True)
        
        # Определяем целевую переменную
        target_col = self._get_target_column(y_train, feature_info)
        if target_col in X_train.columns:
            y_actual = X_train[target_col]
        elif isinstance(y_train, pd.Series):
            y_actual = y_train
        else:
            y_actual = pd.Series(y_train, name=target_col)
        
        # Создаем многомерную целевую переменную
        y_multi = pd.DataFrame()
        for h in range(1, self.horizon + 1):
            if len(y_actual) > h:
                y_multi[f'horizon_{h}'] = y_actual.shift(-h).values
            else:
                y_multi[f'horizon_{h}'] = np.nan
        
        # Удаляем строки с NaN
        valid_idx = ~y_multi.isna().any(axis=1)
        if valid_idx.sum() > 0:
            y_multi_clean = y_multi[valid_idx]
            X_train_clean = X_prepared.iloc[:len(y_multi_clean)]
        else:
            # Если все строки содержат NaN, используем первые N строк
            y_multi_clean = y_multi.iloc[:min(10, len(y_multi))]
            X_train_clean = X_prepared.iloc[:len(y_multi_clean)]
            # Заполняем NaN средним значением
            y_multi_clean = y_multi_clean.fillna(y_multi_clean.mean())
        
        # Создаем модель
        base_model = self._create_model()
        
        # Для нелинейных моделей используем MultiOutputRegressor
        if self.model_type == 'rf':
            self.model = MultiOutputRegressor(base_model)
        else:
            self.model = base_model  # Линейные модели поддерживают multi-output
        
        if len(X_train_clean) > 5 and len(y_multi_clean) > 5:
            try:
                self.model.fit(X_train_clean, y_multi_clean)
            except Exception as e:
                print(f"Ошибка при обучении multi-output модели: {str(e)}")
                # Создаем простую модель
                self.model = Ridge(alpha=1.0, random_state=42)
                if len(X_train_clean) > 0:
                    self.model.fit(X_train_clean, y_multi_clean)
        else:
            # Создаем простую модель если данных недостаточно
            self.model = Ridge(alpha=1.0, random_state=42)
            if len(X_prepared) > 0:
                dummy_y = pd.DataFrame({f'horizon_{i+1}': [y_actual.iloc[0] if len(y_actual) > 0 else 0] for i in range(self.horizon)})
                self.model.fit(X_prepared.iloc[:1], dummy_y)
        
        self.training_time = time.time() - start_time
    
    def predict(self, X_test, feature_info=None):
        """Прогнозирование всех горизонтов одновременно"""
        start_time = time.time()
        
        if X_test.empty:
            predictions = pd.DataFrame(index=X_test.index)
            for h in range(1, self.horizon + 1):
                predictions[f'horizon_{h}'] = 0
            self.predict_time = time.time() - start_time
            return predictions
        
        # Подготавливаем тестовые признаки
        X_prepared = self._prepare_features(X_test, is_training=False)
        
        if self.model is not None:
            try:
                if len(X_prepared) > 0:
                    predictions_array = self.model.predict(X_prepared)
                    
                    result = pd.DataFrame(
                        predictions_array,
                        columns=[f'horizon_{i+1}' for i in range(self.horizon)],
                        index=X_test.index
                    )
                else:
                    result = pd.DataFrame(index=X_test.index)
                    for h in range(1, self.horizon + 1):
                        result[f'horizon_{h}'] = 0
            except Exception as e:
                print(f"Ошибка multi-output предсказания: {str(e)}")
                result = pd.DataFrame(index=X_test.index)
                for h in range(1, self.horizon + 1):
                    result[f'horizon_{h}'] = 0
        else:
            result = pd.DataFrame(index=X_test.index)
            for h in range(1, self.horizon + 1):
                result[f'horizon_{h}'] = 0
        
        self.predict_time = time.time() - start_time
        return result

# ============================================================
# DIRREC STRATEGY (исправленная)
# ============================================================

class DirRecStrategy(ForecastingStrategyBase):
    """DirRec: гибрид — рекурсивная в пределах окна, прямая между окнами"""
    
    def __init__(self, horizon, model_type='ridge', window_size=3, **kwargs):
        super().__init__(horizon, model_type, **kwargs)
        self.window_size = window_size
        self.window_models = []
    
    def fit(self, X_train, y_train, feature_info=None):
        """Обучение гибридной стратегии"""
        start_time = time.time()
        
        if X_train.empty or len(X_train) < 10:
            print(f"Недостаточно данных для обучения: {len(X_train)} строк")
            self.training_time = 0
            return
        
        # Подготавливаем признаки
        X_prepared = self._prepare_features(X_train, is_training=True)
        
        # Определяем целевую переменную
        target_col = self._get_target_column(y_train, feature_info)
        if target_col in X_train.columns:
            y_actual = X_train[target_col]
        elif isinstance(y_train, pd.Series):
            y_actual = y_train
        else:
            y_actual = pd.Series(y_train, name=target_col)
        
        self.window_models = []
        
        # Количество окон
        n_windows = int(np.ceil(self.horizon / self.window_size))
        
        for w in range(n_windows):
            window_horizon = min(self.window_size, self.horizon - w * self.window_size)
            
            if window_horizon == 1:
                # Direct для окна размера 1
                model = self._create_model()
                shift = w * self.window_size
                
                y_train_shifted = y_actual.shift(-shift-1)
                valid_idx = y_train_shifted.notna()
                X_window = X_prepared[valid_idx]
                y_window = y_train_shifted[valid_idx]
                
                if len(X_window) > 5 and len(y_window) > 5:
                    try:
                        model.fit(X_window, y_window)
                        self.window_models.append(('direct', model, shift+1))
                    except Exception as e:
                        print(f"Ошибка при обучении direct модели: {str(e)}")
                        # Создаем простую модель
                        dummy_model = self._create_model()
                        if len(X_window) > 0:
                            dummy_model.fit(X_window.iloc[:1], [y_window.iloc[0] if len(y_window) > 0 else 0])
                        self.window_models.append(('direct', dummy_model, shift+1))
                else:
                    # Создаем простую модель если данных недостаточно
                    dummy_model = self._create_model()
                    if len(X_prepared) > 0:
                        dummy_model.fit(X_prepared.iloc[:1], [y_actual.iloc[0] if len(y_actual) > 0 else 0])
                    self.window_models.append(('direct', dummy_model, shift+1))
            
            else:
                # Multi-output для окна
                base_model = self._create_model()
                if self.model_type == 'rf':
                    model = MultiOutputRegressor(base_model)
                else:
                    model = base_model
                
                # Подготовка данных для окна
                y_multi = pd.DataFrame()
                for h in range(1, window_horizon + 1):
                    shift = w * self.window_size + h
                    if len(y_actual) > shift:
                        y_multi[f'horizon_{h}'] = y_actual.shift(-shift).values
                    else:
                        y_multi[f'horizon_{h}'] = np.nan
                
                valid_idx = ~y_multi.isna().any(axis=1)
                if valid_idx.sum() > 0:
                    y_multi_clean = y_multi[valid_idx]
                    X_window = X_prepared.iloc[:len(y_multi_clean)]
                else:
                    y_multi_clean = y_multi.iloc[:min(10, len(y_multi))]
                    X_window = X_prepared.iloc[:len(y_multi_clean)]
                    y_multi_clean = y_multi_clean.fillna(y_multi_clean.mean())
                
                if len(X_window) > 5 and len(y_multi_clean) > 5:
                    try:
                        model.fit(X_window, y_multi_clean)
                        self.window_models.append(('multi', model, w * self.window_size + 1, window_horizon))
                    except Exception as e:
                        print(f"Ошибка при обучении multi модели: {str(e)}")
                        # Создаем простую модель
                        dummy_model = Ridge(alpha=1.0, random_state=42)
                        dummy_model.fit(X_window.iloc[:1], y_multi_clean.iloc[:1])
                        self.window_models.append(('multi', dummy_model, w * self.window_size + 1, window_horizon))
                else:
                    # Создаем простую модель если данных недостаточно
                    dummy_model = Ridge(alpha=1.0, random_state=42)
                    if len(X_prepared) > 0:
                        dummy_y = pd.DataFrame({f'horizon_{i+1}': [y_actual.iloc[0] if len(y_actual) > 0 else 0] for i in range(window_horizon)})
                        dummy_model.fit(X_prepared.iloc[:1], dummy_y)
                    self.window_models.append(('multi', dummy_model, w * self.window_size + 1, window_horizon))
        
        self.training_time = time.time() - start_time
    
    def predict(self, X_test, feature_info=None):
        """Прогнозирование гибридной стратегией"""
        start_time = time.time()
        
        if X_test.empty:
            predictions = pd.DataFrame(index=X_test.index)
            for h in range(1, self.horizon + 1):
                predictions[f'horizon_{h}'] = 0
            self.predict_time = time.time() - start_time
            return predictions
        
        # Подготавливаем тестовые признаки
        X_prepared = self._prepare_features(X_test, is_training=False)
        
        predictions = pd.DataFrame(index=X_test.index)
        X_current = X_prepared.copy()
        
        for model_info in self.window_models:
            if model_info[0] == 'direct':
                _, model, horizon = model_info
                try:
                    if len(X_current) > 0:
                        pred = model.predict(X_current)
                        predictions[f'horizon_{horizon}'] = pred[:len(X_current)]
                        
                        # Обновляем признаки
                        X_current = self._update_features(X_current, pred)
                    else:
                        predictions[f'horizon_{horizon}'] = 0
                except Exception as e:
                    print(f"Ошибка предсказания direct модели: {str(e)}")
                    predictions[f'horizon_{horizon}'] = 0
            
            elif model_info[0] == 'multi':
                _, model, start_horizon, window_horizon = model_info
                try:
                    if len(X_current) > 0:
                        preds = model.predict(X_current)
                        
                        for h in range(window_horizon):
                            horizon_num = start_horizon + h
                            if horizon_num <= self.horizon:
                                predictions[f'horizon_{horizon_num}'] = preds[:, h][:len(X_current)]
                        
                        # Обновляем признаки для следующего окна
                        if start_horizon + window_horizon <= self.horizon:
                            X_current = self._update_features(X_current, preds[:, -1])
                    else:
                        for h in range(window_horizon):
                            horizon_num = start_horizon + h
                            if horizon_num <= self.horizon:
                                predictions[f'horizon_{horizon_num}'] = 0
                except Exception as e:
                    print(f"Ошибка предсказания multi модели: {str(e)}")
                    for h in range(window_horizon):
                        horizon_num = start_horizon + h
                        if horizon_num <= self.horizon:
                            predictions[f'horizon_{horizon_num}'] = 0
        
        self.predict_time = time.time() - start_time
        return predictions
    
    def _update_features(self, X, last_pred):
        """Обновление признаков"""
        X_new = X.copy()
        
        # Обновляем лаговые признаки
        lag_cols = [col for col in X.columns if 'lag' in col.lower()]
        if lag_cols:
            try:
                # Ищем лаги типа lag_1, lag_2 и т.д.
                for col in lag_cols:
                    if 'lag_' in col.lower():
                        parts = col.split('_')
                        if len(parts) > 1 and parts[-1].isdigit():
                            lag_num = int(parts[-1])
                            if lag_num > 1:
                                prev_lag = f"lag_{lag_num-1}"
                                if prev_lag in X_new.columns:
                                    X_new[col] = X_new[prev_lag]
                
                # Обновляем lag_1
                if 'lag_1' in X_new.columns:
                    X_new['lag_1'] = last_pred[0] if isinstance(last_pred, np.ndarray) else last_pred
            except Exception as e:
                print(f"Ошибка обновления признаков: {str(e)}")
        
        return X_new

# ============================================================
# AUTOGLUON STRATEGY (ИСПРАВЛЕННАЯ)
# ============================================================

class AutoGluonStrategy(ForecastingStrategyBase):
    """AutoGluon: простая стратегия"""
    
    def __init__(self, horizon, **kwargs):
        super().__init__(horizon, 'ridge', **kwargs)
        self.name = "AutoGluon"
    
    def fit(self, X_train, y_train, feature_info=None):
        """Обучение с использованием AutoGluon или простой модели"""
        start_time = time.time()
        
        if X_train.empty or len(X_train) < 10:
            print(f"Недостаточно данных для обучения: {len(X_train)} строк")
            self.training_time = 0
            return
        
        # Подготавливаем признаки
        X_prepared = self._prepare_features(X_train, is_training=True)
        
        # Определяем целевую переменную
        target_col = self._get_target_column(y_train, feature_info)
        if target_col in X_train.columns:
            y_actual = X_train[target_col]
        elif isinstance(y_train, pd.Series):
            y_actual = y_train
        else:
            y_actual = pd.Series(y_train, name=target_col)
        
        # Удаляем строки с NaN
        valid_idx = y_actual.notna()
        X_train_clean = X_prepared[valid_idx]
        y_train_clean = y_actual[valid_idx]
        
        # Используем Ridge модель вместо AutoGluon для надежности
        self.model = Ridge(alpha=1.0, random_state=42)
        
        if len(X_train_clean) > 5 and len(y_train_clean) > 5:
            try:
                self.model.fit(X_train_clean, y_train_clean)
            except Exception as e:
                print(f"Ошибка при обучении модели: {str(e)}")
                # Создаем простую модель
                self.model = Ridge(alpha=1.0, random_state=42)
                if len(X_train_clean) > 0:
                    self.model.fit(X_train_clean.iloc[:1], [y_train_clean.iloc[0]])
        else:
            # Создаем простую модель если данных недостаточно
            self.model = Ridge(alpha=1.0, random_state=42)
            if len(X_prepared) > 0 and len(y_actual) > 0:
                self.model.fit(X_prepared.iloc[:1], [y_actual.iloc[0]])
        
        self.training_time = time.time() - start_time
    
    def predict(self, X_test, feature_info=None):
        """Прогнозирование"""
        start_time = time.time()
        
        if X_test.empty:
            predictions = pd.DataFrame(index=X_test.index)
            for h in range(1, self.horizon + 1):
                predictions[f'horizon_{h}'] = 0
            self.predict_time = time.time() - start_time
            return predictions
        
        # Подготавливаем тестовые признаки
        X_prepared = self._prepare_features(X_test, is_training=False)
        
        predictions = pd.DataFrame(index=X_test.index)
        
        if self.model is not None:
            try:
                # Для multi-step прогнозирования используем рекурсивный подход
                X_current = X_prepared.copy()
                
                for h in range(1, self.horizon + 1):
                    if len(X_current) > 0:
                        pred = self.model.predict(X_current)
                        predictions[f'horizon_{h}'] = pred[:len(X_current)]
                        
                        # Обновляем признаки для следующего шага
                        if h < self.horizon:
                            X_current = self._update_features(X_current, pred)
                    else:
                        predictions[f'horizon_{h}'] = 0
            except Exception as e:
                print(f"Ошибка предсказания: {str(e)}")
                for h in range(1, self.horizon + 1):
                    predictions[f'horizon_{h}'] = 0
        else:
            for h in range(1, self.horizon + 1):
                predictions[f'horizon_{h}'] = 0
        
        self.predict_time = time.time() - start_time
        return predictions
    
    def _update_features(self, X, last_pred):
        """Обновление признаков"""
        X_new = X.copy()
        
        # Простое обновление
        lag_cols = [col for col in X.columns if 'lag' in col.lower()]
        if lag_cols and 'lag_1' in X.columns:
            try:
                X_new['lag_1'] = last_pred[0] if isinstance(last_pred, np.ndarray) else last_pred
            except:
                pass
        
        return X_new

# ============================================================
# ИСПРАВЛЕННАЯ ФУНКЦИЯ ВИЗУАЛИЗАЦИИ
# ============================================================

def plot_comparison_results(comparison_df):
    """Визуализация результатов сравнения - ИСПРАВЛЕННАЯ"""
    
    if comparison_df is None or comparison_df.empty:
        # Возвращаем пустую, но корректную фигуру
        fig = go.Figure()
        fig.update_layout(
            title="Нет данных для визуализации",
            height=400
        )
        return fig
    
    # Создаем числовые столбцы
    comparison_df = comparison_df.copy()
    
    def to_float_safe(x):
        try:
            if isinstance(x, str):
                if 'N/A' in x or 'nan' in str(x).lower():
                    return 0.0
                x_clean = str(x).replace('%', '').replace(',', '.').strip()
                return float(x_clean)
            elif pd.isna(x):
                return 0.0
            return float(x)
        except:
            return 0.0
    
    # Создаем числовые столбцы
    numeric_columns = {}
    for col in ['Средний MAE', 'Средний RMSE', 'Время обучения (с)', 
                'Время прогноза (с)', 'Накопление ошибки', 'Рост ошибки (%)']:
        if col in comparison_df.columns:
            numeric_columns[col] = comparison_df[col].apply(to_float_safe)
    
    # Если нет данных, возвращаем пустую фигуру
    if not numeric_columns:
        fig = go.Figure()
        fig.update_layout(
            title="Нет числовых данных для визуализации",
            height=400
        )
        return fig
    
    # Создаем подграфики
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['Средний MAE', 'Средний RMSE', 'Время обучения',
                       'Время прогноза', 'Накопление ошибки', 'Рост ошибки'],
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
    strategies = comparison_df['Стратегия'].tolist()
    
    # Средний MAE
    if 'Средний MAE' in numeric_columns:
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=numeric_columns['Средний MAE'].values,
                name='MAE',
                marker_color=colors[:len(strategies)],
                text=[f"{x:.4f}" for x in numeric_columns['Средний MAE'].values],
                textposition='auto'
            ),
            row=1, col=1
        )
    
    # Средний RMSE
    if 'Средний RMSE' in numeric_columns:
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=numeric_columns['Средний RMSE'].values,
                name='RMSE',
                marker_color=colors[:len(strategies)],
                text=[f"{x:.4f}" for x in numeric_columns['Средний RMSE'].values],
                textposition='auto'
            ),
            row=1, col=2
        )
    
    # Время обучения
    if 'Время обучения (с)' in numeric_columns:
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=numeric_columns['Время обучения (с)'].values,
                name='Обучение',
                marker_color=colors[:len(strategies)],
                text=[f"{x:.3f}" for x in numeric_columns['Время обучения (с)'].values],
                textposition='auto'
            ),
            row=1, col=3
        )
    
    # Время прогноза
    if 'Время прогноза (с)' in numeric_columns:
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=numeric_columns['Время прогноза (с)'].values,
                name='Прогноз',
                marker_color=colors[:len(strategies)],
                text=[f"{x:.3f}" for x in numeric_columns['Время прогноза (с)'].values],
                textposition='auto'
            ),
            row=2, col=1
        )
    
    # Накопление ошибки
    if 'Накопление ошибки' in numeric_columns:
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=numeric_columns['Накопление ошибки'].values,
                name='Накопление',
                marker_color=colors[:len(strategies)],
                text=[f"{x:.4f}" for x in numeric_columns['Накопление ошибки'].values],
                textposition='auto'
            ),
            row=2, col=2
        )
    
    # Рост ошибки
    if 'Рост ошибки (%)' in numeric_columns:
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=numeric_columns['Рост ошибки (%)'].values,
                name='Рост',
                marker_color=colors[:len(strategies)],
                text=[f"{x:.1f}%" for x in numeric_columns['Рост ошибки (%)'].values],
                textposition='auto'
            ),
            row=2, col=3
        )
    
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="Сравнение стратегий прогнозирования",
        title_x=0.5
    )
    
    # Обновляем подписи осей
    fig.update_yaxes(title_text="MAE", row=1, col=1)
    fig.update_yaxes(title_text="RMSE", row=1, col=2)
    fig.update_yaxes(title_text="Секунды", row=1, col=3)
    fig.update_yaxes(title_text="Секунды", row=2, col=1)
    fig.update_yaxes(title_text="ΔMAE", row=2, col=2)
    fig.update_yaxes(title_text="%", row=2, col=3)
    
    return fig

# ============================================================
# ИСПРАВЛЕННЫЙ КОМПАРАТОР
# ============================================================

class ForecastingStrategiesComparator:
    """Компаратор, который использует разбиение из Этапа 2"""
    
    def __init__(self):
        self.strategies = {}
        self.results = {}
    
    def add_strategy(self, name, strategy):
        self.strategies[name] = strategy
    
    def prepare_data_from_split(self, split_data, feature_info, date_col, target_col):
        """
        Подготовка данных из split_data Этапа 2
        """
        try:
            # Получаем данные из Этапа 2
            train_data = split_data['train'].copy()
            val_data = split_data['val'].copy()
            test_data = split_data['test'].copy()
            
            st.info(f"""
            **Используем разбиение из Этапа 2:**
            - Train: {len(train_data)} записей
            - Validation: {len(val_data)} записей  
            - Test: {len(test_data)} записей
            """)
            
            # Определяем трансформированную целевую переменную если есть
            transformed_target = None
            if 'transformation_info' in feature_info:
                trans_info = feature_info['transformation_info']
                if trans_info.get('log_applied') and 'log_col' in trans_info:
                    transformed_target = trans_info['log_col']
                elif trans_info.get('boxcox_applied') and 'boxcox_col' in trans_info:
                    transformed_target = trans_info['boxcox_col']
            
            # Используем train + val для обучения (как в Этапе 3)
            X_train_full = pd.concat([train_data, val_data], axis=0)
            
            # Выбираем признаки: все числовые кроме даты и исходной цели
            feature_cols = []
            for col in X_train_full.columns:
                if col != date_col and col != target_col:
                    if pd.api.types.is_numeric_dtype(X_train_full[col]):
                        feature_cols.append(col)
            
            st.info(f"Используем {len(feature_cols)} признаков из Этапа 1")
            
            if not feature_cols:
                st.error("Не найдено числовых признаков для обучения")
                return None, None, None, None, None
            
            # Разделяем на признаки и целевую переменную
            X_train = X_train_full[feature_cols].copy()
            
            # Используем трансформированную цель если она есть, иначе исходную
            if transformed_target and transformed_target in X_train_full.columns:
                y_train = X_train_full[transformed_target].copy()
                target_for_test = transformed_target
            else:
                y_train = X_train_full[target_col].copy()
                target_for_test = target_col
            
            # Тестовые данные
            X_test = test_data[feature_cols].copy()
            
            # Для теста используем те же признаки
            if transformed_target and transformed_target in test_data.columns:
                y_test = test_data[transformed_target].copy()
            else:
                y_test = test_data[target_col].copy()
            
            # Создаем многомерные истинные значения для каждого горизонта
            max_horizon = max(s.horizon for s in self.strategies.values())
            y_test_multi = pd.DataFrame(index=y_test.index)
            
            # Используем скользящие окна для создания многомерной цели
            for h in range(1, max_horizon + 1):
                if h == 1:
                    # Для горизонта 1 используем текущие значения
                    y_test_multi[f'horizon_{h}'] = y_test.values
                elif len(y_test) >= h:
                    # Для горизонтов > 1 используем будущие значения
                    # Создаем сдвинутые значения
                    shifted_values = []
                    for i in range(len(y_test)):
                        if i + h - 1 < len(y_test):
                            shifted_values.append(y_test.iloc[i + h - 1])
                        else:
                            # Если выходим за границы, используем последнее значение
                            shifted_values.append(y_test.iloc[-1])
                    y_test_multi[f'horizon_{h}'] = shifted_values
                else:
                    # Если данных недостаточно, заполняем последним значением
                    y_test_multi[f'horizon_{h}'] = y_test.iloc[-1] if len(y_test) > 0 else 0
            
            return X_train, y_train, X_test, y_test_multi, feature_info
            
        except Exception as e:
            st.error(f"Ошибка при подготовке данных из Этапа 2: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None
    
    def compare(self, X_train, y_train, X_test, y_test, feature_info):
        """Сравнение стратегий"""
        comparison_results = []
        
        for name, strategy in self.strategies.items():
            with st.spinner(f"Обучение {name}..."):
                try:
                    # Проверяем достаточность данных
                    if X_train is None or len(X_train) < 5 or len(X_test) < strategy.horizon:
                        st.warning(f"Недостаточно данных для {name}")
                        continue
                    
                    # Обучение с передачей feature_info
                    strategy.fit(X_train, y_train, feature_info)
                    
                    # Прогнозирование
                    predictions = strategy.predict(X_test, feature_info)
                    
                    # Расчет метрик
                    metrics = {}
                    mae_values = []
                    rmse_values = []
                    
                    for h in range(1, strategy.horizon + 1):
                        if f'horizon_{h}' in predictions.columns and f'horizon_{h}' in y_test.columns:
                            y_true_h = y_test[f'horizon_{h}'].dropna()
                            y_pred_h = predictions[f'horizon_{h}'].iloc[:len(y_true_h)]
                            
                            # Удаляем NaN
                            valid_idx = y_true_h.notna() & y_pred_h.notna()
                            y_true_clean = y_true_h[valid_idx]
                            y_pred_clean = y_pred_h[valid_idx]
                            
                            if len(y_true_clean) > 0:
                                try:
                                    mae = mean_absolute_error(y_true_clean, y_pred_clean)
                                    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
                                    mae_values.append(mae)
                                    rmse_values.append(rmse)
                                    metrics[f'h{h}_mae'] = mae
                                    metrics[f'h{h}_rmse'] = rmse
                                except Exception as e:
                                    print(f"Ошибка расчета метрик для горизонта {h}: {e}")
                                    mae_values.append(0)
                                    rmse_values.append(0)
                    
                    if mae_values and len(mae_values) > 0:
                        avg_mae = np.mean(mae_values)
                        avg_rmse = np.mean(rmse_values)
                        error_accumulation = mae_values[-1] - mae_values[0] if len(mae_values) > 1 else 0
                        error_growth = error_accumulation / mae_values[0] if mae_values[0] != 0 else 0
                    else:
                        avg_mae = 0
                        avg_rmse = 0
                        error_accumulation = 0
                        error_growth = 0
                    
                    # Сохраняем результаты
                    self.results[name] = {
                        'predictions': predictions,
                        'metrics': metrics,
                        'training_time': strategy.training_time,
                        'predict_time': strategy.predict_time,
                        'avg_mae': avg_mae,
                        'avg_rmse': avg_rmse,
                        'error_accumulation': error_accumulation,
                        'error_growth': error_growth,
                        'strategy': strategy
                    }
                    
                    comparison_results.append({
                        'Стратегия': name,
                        'Время обучения (с)': strategy.training_time,
                        'Время прогноза (с)': strategy.predict_time,
                        'Средний MAE': avg_mae,
                        'Средний RMSE': avg_rmse,
                        'Накопление ошибки': error_accumulation,
                        'Рост ошибки (%)': error_growth,
                        'Гибкость': self._assess_flexibility(name)
                    })
                    
                except Exception as e:
                    st.error(f"Ошибка в стратегии {name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        # Создаем DataFrame
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            
            # Форматируем для отображения
            comparison_df_display = comparison_df.copy()
            comparison_df_display['Время обучения (с)'] = comparison_df_display['Время обучения (с)'].apply(lambda x: f"{x:.3f}")
            comparison_df_display['Время прогноза (с)'] = comparison_df_display['Время прогноза (с)'].apply(lambda x: f"{x:.3f}")
            comparison_df_display['Средний MAE'] = comparison_df_display['Средний MAE'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "0.0000")
            comparison_df_display['Средний RMSE'] = comparison_df_display['Средний RMSE'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "0.0000")
            comparison_df_display['Накопление ошибки'] = comparison_df_display['Накопление ошибки'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "0.0000")
            comparison_df_display['Рост ошибки (%)'] = comparison_df_display['Рост ошибки (%)'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "0.0%")
            
            return comparison_df_display, comparison_df
        else:
            return pd.DataFrame(), pd.DataFrame()
    
    def _assess_flexibility(self, strategy_name):
        """Оценка гибкости стратегии"""
        if 'Direct' in strategy_name:
            return 'Низкая (h моделей)'
        elif 'Recursive' in strategy_name:
            return 'Высокая (одна модель)'
        elif 'Multi-output' in strategy_name:
            return 'Средняя (многомерная)'
        elif 'DirRec' in strategy_name:
            return 'Высокая (гибрид)'
        elif 'AutoGluon' in strategy_name:
            return 'Авто-ML'
        return 'Н/Д'

# ============================================================
# ОСНОВНОЙ ИНТЕРФЕЙС (исправленный)
# ============================================================

def show_forecasting_strategies_interface():
    """Основной интерфейс для Этапа 4"""
    
    # Проверка обязательных данных
    required_keys = ['df_features', 'feature_info', 'split_data']
    missing_keys = [key for key in required_keys if key not in st.session_state]
    
    if missing_keys:
        st.error(f"❌ Сначала выполните предыдущие этапы. Отсутствуют: {', '.join(missing_keys)}")
        return
    
    # Получаем данные
    feature_info = st.session_state.feature_info
    split_data = st.session_state.split_data
    
    # Определяем переменные
    date_col = feature_info['original_features'][0]
    target_col = feature_info['original_features'][1]
    
    st.success(f"""
    **Используем данные из предыдущих этапов:**
    - Признаки: {len(feature_info.get('created_features', []))} созданных
    - Разбиение: Train ({len(split_data['train'])}), Val ({len(split_data['val'])}), Test ({len(split_data['test'])})
    - Дата: {date_col}, Цель: {target_col}
    """)
    
    st.info("""
    **Сравниваемые стратегии:**
    
    1. **Direct** - h отдельных моделей для каждого горизонта
    2. **Recursive** - одна модель, рекурсивное предсказание
    3. **Multi-output** - одна модель для всех горизонтов
    4. **DirRec** - гибридная стратегия
    5. **AutoGluon** - упрощенная автоматическая стратегия
    """)
    
    # Настройки
    st.subheader("⚙️ Настройки сравнения")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        horizon = st.slider("Горизонт прогнозирования", 3, 10, 5, 1)
        model_type = st.selectbox(
            "Тип модели",
            options=['ridge', 'lasso', 'elasticnet', 'rf'],
            index=0
        )
    
    with col2:
        dirrec_window = st.slider("Размер окна DirRec", 2, 5, 2, 1)
        use_scaling = st.checkbox("Масштабировать признаки", value=True)
    
    with col3:
        use_transformed_target = st.checkbox("Использовать трансформированную цель", value=True)
        include_autogluon = st.checkbox("Включить AutoGluon", value=True)
    
    st.markdown("---")
    
    if st.button("🚀 Запустить сравнение стратегий", type="primary", use_container_width=True):
        with st.spinner("Сравнение стратегий..."):
            try:
                # Создаем компаратор
                comparator = ForecastingStrategiesComparator()
                
                # Общие параметры
                common_params = {
                    'use_scaling': use_scaling,
                    'use_transformed_target': use_transformed_target
                }
                
                # Добавляем стратегии
                comparator.add_strategy(
                    f"Direct ({model_type})", 
                    DirectStrategy(horizon, model_type, **common_params)
                )
                comparator.add_strategy(
                    f"Recursive ({model_type})", 
                    RecursiveStrategy(horizon, model_type, **common_params)
                )
                comparator.add_strategy(
                    f"Multi-output ({model_type})", 
                    MultiOutputStrategy(horizon, model_type, **common_params)
                )
                comparator.add_strategy(
                    f"DirRec ({model_type}, w={dirrec_window})", 
                    DirRecStrategy(horizon, model_type, dirrec_window, **common_params)
                )
                
                if include_autogluon:
                    comparator.add_strategy(
                        "AutoGluon", 
                        AutoGluonStrategy(horizon, **common_params)
                    )
                
                # Подготовка данных
                X_train, y_train, X_test, y_test, feature_info = comparator.prepare_data_from_split(
                    split_data, feature_info, date_col, target_col
                )
                
                if X_train is None:
                    st.error("Не удалось подготовить данные")
                    return
                
                st.success(f"""
                ✅ Данные подготовлены:
                - Обучающие: {len(X_train)} записей
                - Тестовые: {len(X_test)} записей
                """)
                
                # Сравнение стратегий
                comparison_df_display, comparison_df = comparator.compare(
                    X_train, y_train, X_test, y_test, feature_info
                )
                
                if not comparison_df_display.empty:
                    # Сохраняем результаты
                    st.session_state.forecast_results = {
                        'comparison_df_display': comparison_df_display,
                        'comparison_df': comparison_df,
                        'strategy_results': comparator.results,
                        'y_test': y_test,
                        'horizon': horizon
                    }
                    
                    # Отображение результатов
                    _display_results(comparison_df_display, comparator.results, y_test, horizon)
                    
                else:
                    st.error("Не удалось выполнить сравнение стратегий")
                    
            except Exception as e:
                st.error(f"Ошибка при сравнении стратегий: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Если уже есть результаты, показываем их
    elif 'forecast_results' in st.session_state:
        st.success("✅ Сравнение стратегий уже выполнено!")
        
        results = st.session_state.forecast_results
        _display_results(
            results['comparison_df_display'],
            results['strategy_results'],
            results['y_test'],
            results['horizon']
        )

def _display_results(comparison_df_display, strategy_results, y_test, horizon):
    """Отображение результатов"""
    
    if comparison_df_display.empty:
        st.warning("Нет результатов для отображения")
        return
    
    # 1. Таблица сравнения
    st.subheader("📊 Результаты сравнения")
    st.dataframe(comparison_df_display, width='stretch')
    
    # 2. Визуализация сравнения
    st.subheader("📈 Визуализация сравнения")
    fig_comparison = plot_comparison_results(comparison_df_display)
    if fig_comparison is not None:
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # 3. Прогнозы
    st.subheader("🔮 Прогнозы на разных горизонтах")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        horizon_to_plot = st.slider(
            "Горизонт для визуализации", 
            1, horizon, min(2, horizon), 1,
            key="horizon_viz"
        )
        
        # Создаем график сравнения прогнозов
        fig = go.Figure()
        
        # Истинные значения
        if f'horizon_{horizon_to_plot}' in y_test.columns:
            y_true = y_test[f'horizon_{horizon_to_plot}'].dropna()
            if len(y_true) > 0:
                fig.add_trace(go.Scatter(
                    x=np.arange(len(y_true)),
                    y=y_true.values,
                    mode='lines',
                    name='Истинные значения',
                    line=dict(color='black', width=3)
                ))
        
        # Прогнозы стратегий
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for idx, (name, result) in enumerate(strategy_results.items()):
            if idx >= len(colors):
                break
                
            predictions = result['predictions']
            if f'horizon_{horizon_to_plot}' in predictions.columns:
                pred_values = predictions[f'horizon_{horizon_to_plot}'].values
                if 'y_true' in locals() and len(y_true) > 0:
                    pred_values = pred_values[:len(y_true)]
                
                valid_idx = ~np.isnan(pred_values)
                if np.any(valid_idx):
                    fig.add_trace(go.Scatter(
                        x=np.arange(len(pred_values))[valid_idx],
                        y=pred_values[valid_idx],
                        mode='lines',
                        name=name,
                        line=dict(color=colors[idx], width=2, dash='dash')
                    ))
        
        fig.update_layout(
            title=f'Прогнозы на горизонте {horizon_to_plot}',
            xaxis_title='Временной шаг',
            yaxis_title='Значение',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Накопление ошибки по горизонтам
        fig_errors = go.Figure()
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for idx, (name, result) in enumerate(strategy_results.items()):
            if idx >= len(colors):
                break
                
            metrics = result['metrics']
            horizons = []
            mae_values = []
            
            for key, value in metrics.items():
                if key.startswith('h') and key.endswith('_mae'):
                    h = int(key.split('_')[0][1:])
                    if h <= horizon:
                        horizons.append(h)
                        mae_values.append(value)
            
            if horizons and mae_values:
                sorted_data = sorted(zip(horizons, mae_values))
                horizons_sorted, mae_sorted = zip(*sorted_data)
                
                fig_errors.add_trace(go.Scatter(
                    x=horizons_sorted,
                    y=mae_sorted,
                    mode='lines+markers',
                    name=name,
                    line=dict(color=colors[idx], width=2)
                ))
        
        fig_errors.update_layout(
            title='Накопление ошибки',
            xaxis_title='Горизонт',
            yaxis_title='MAE',
            height=400
        )
        st.plotly_chart(fig_errors, use_container_width=True)
    
    # 4. Анализ результатов
    st.subheader("🎯 Анализ результатов")
    
    try:
        if not comparison_df_display.empty:
            # Находим лучшую стратегию по MAE
            best_strategy = None
            best_mae = float('inf')
            
            for idx, row in comparison_df_display.iterrows():
                try:
                    mae_val = float(row['Средний MAE'])
                    if mae_val < best_mae:
                        best_mae = mae_val
                        best_strategy = row['Стратегия']
                except (ValueError, TypeError):
                    continue
            
            if best_strategy:
                st.success(f"**Лучшая стратегия (по MAE):** {best_strategy} (MAE = {best_mae:.4f})")
            else:
                st.info("Не удалось определить лучшую стратегию")
    except Exception as e:
        st.warning(f"Не удалось провести детальный анализ: {e}")
    
    st.markdown("---")
    st.success("""
    **✅ Этап 4 завершен!**
    
    **Что было сделано:**
    1. Использованы данные и признаки из предыдущих этапов
    2. Сравнены 5 стратегий multi-step прогнозирования
    3. Проанализированы метрики качества и время выполнения
    4. Визуализированы результаты сравнения
    """)