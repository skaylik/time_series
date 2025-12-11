# diagnostics_module.py - Этап 6: Диагностика моделей (ИСПРАВЛЕННАЯ ВЕРСИЯ)

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Статистические тесты
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

# Импорт для обработки NaN
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Импорт для SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Импорт для PDP
try:
    from sklearn.inspection import PartialDependenceDisplay
    PDP_AVAILABLE = True
except ImportError:
    PDP_AVAILABLE = False

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ РАБОТЫ С ДАННЫМИ ИЗ 5 ЭТАПА
# ============================================================

def extract_model_from_integrated_results():
    """Извлечение лучшей модели из интегрированных результатов 5 этапа"""
    
    # Проверяем все возможные ключи, под которыми могут быть сохранены результаты 5 этапа
    integrated_results = None
    
    # Сначала проверяем integrated_results (основной ключ из 5 этапа)
    if 'integrated_results' in st.session_state:
        integrated_results = st.session_state.integrated_results
    # Затем проверяем advanced_modeling_data (альтернативный ключ из 5 этапа)
    elif 'advanced_modeling_data' in st.session_state:
        integrated_results = st.session_state.advanced_modeling_data
    # И проверяем model_comparison_results (еще один возможный ключ)
    elif 'model_comparison_results' in st.session_state:
        integrated_results = st.session_state.model_comparison_results
    
    if integrated_results is None:
        st.error("❌ Сначала выполните Этап 5: Интеграцию и сравнение подходов")
        return None, None, None
    
    try:
        # Получаем лучшую модель из интегрированной таблицы
        # Проверяем разные возможные структуры данных
        
        integrated_df = None
        integrated_df_display = None
        
        # Если integrated_results - это словарь с ключами
        if isinstance(integrated_results, dict):
            integrated_df = integrated_results.get('integrated_df')
            integrated_df_display = integrated_results.get('integrated_df_display')
            
            # Если не нашли в этих ключах, ищем в других возможных ключах
            if integrated_df is None and 'comparison_df' in integrated_results:
                integrated_df = integrated_results.get('comparison_df')
            if integrated_df_display is None and 'comparison_df_display' in integrated_results:
                integrated_df_display = integrated_results.get('comparison_df_display')
        
        # Если integrated_results - это DataFrame напрямую
        elif isinstance(integrated_results, pd.DataFrame):
            integrated_df = integrated_results
            integrated_df_display = integrated_results
        
        if integrated_df is None or integrated_df.empty:
            st.error("❌ Нет данных о моделях в интегрированных результатах")
            return None, None, None
        
        # Находим лучшую модель по MAE
        # Проверяем возможные названия столбцов с MAE
        mae_column = None
        for possible_col in ['MAE', 'Val MAE', 'val_mae', 'CV MAE', 'Средний MAE']:
            if possible_col in integrated_df.columns:
                mae_column = possible_col
                break
        
        if mae_column is None:
            # Если не нашли MAE в названиях, ищем столбец, содержащий "MAE" в названии
            for col in integrated_df.columns:
                if 'mae' in col.lower() or 'MAE' in col:
                    mae_column = col
                    break
        
        if mae_column:
            # Преобразуем MAE в числовой формат
            integrated_df['MAE_numeric'] = pd.to_numeric(integrated_df[mae_column], errors='coerce')
            best_idx = integrated_df['MAE_numeric'].idxmin()
            best_row = integrated_df.loc[best_idx]
            
            # Определяем тип модели
            model_type = best_row.get('Тип', 'N/A')
            if pd.isna(model_type) or model_type == 'N/A':
                # Пробуем определить по названию
                model_name = best_row.get('Название', '')
                if 'Этап 3' in str(model_name) or any(x in str(model_name).lower() for x in ['ridge', 'lasso', 'random', 'forest', 'xgboost']):
                    model_type = 'ML модель (Этап 3)'
                elif 'Этап 4' in str(model_name) or any(x in str(model_name).lower() for x in ['recursive', 'direct', 'dirrec', 'multi']):
                    model_type = 'Стратегия (Этап 4)'
            
            best_model_info = {
                'Тип': model_type,
                'Название': best_row.get('Название', best_row.get('Метод', 'Unknown')),
                'MAE': best_row['MAE_numeric'],
                'Подход': best_row.get('Подход', 'N/A')
            }
            
            return best_model_info, integrated_results, integrated_df
            
        else:
            st.error("❌ Не удалось найти метрику MAE в интегрированных результатах")
            # Показываем доступные столбцы для отладки
            st.write("Доступные столбцы:", list(integrated_df.columns))
            return None, None, None
            
    except Exception as e:
        st.error(f"Ошибка при извлечении лучшей модели: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
        return None, None, None

def prepare_data_for_diagnostics(feature_info, split_data):
    """Подготовка данных для диагностики из Этапов 1 и 2"""
    
    try:
        # Получаем данные из Этапов 1 и 2
        date_col = feature_info['original_features'][0]
        target_col = feature_info['original_features'][1]
        
        # Используем split_data из Этапа 2
        train_data = split_data['train'].copy()
        val_data = split_data['val'].copy()
        test_data = split_data['test'].copy()
        
        # Объединяем train и val для обучения (как в Этапе 3)
        X_train_full = pd.concat([train_data, val_data], axis=0)
        
        # Выбираем признаки: все числовые кроме даты и цели
        feature_cols = []
        for col in X_train_full.columns:
            if col != date_col and col != target_col:
                if pd.api.types.is_numeric_dtype(X_train_full[col]):
                    feature_cols.append(col)
        
        if not feature_cols:
            st.warning("Не найдено числовых признаков для диагностики")
            return None, None, None, None, None
        
        # Подготавливаем данные
        X_train = X_train_full[feature_cols].copy()
        y_train = X_train_full[target_col].copy()
        
        # Тестовые данные
        X_test = test_data[feature_cols].copy()
        y_test = test_data[target_col].copy()
        
        # Проверяем и обрабатываем пропущенные значения
        # 1. Удаляем строки с пропущенными целевыми значениями
        train_mask = y_train.notna()
        test_mask = y_test.notna()
        
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        
        # 2. Заполняем пропущенные значения в признаках
        # Для числовых признаков - медианой
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        # Преобразуем обратно в DataFrame
        X_train = pd.DataFrame(X_train_imputed, columns=feature_cols, index=X_train.index)
        X_test = pd.DataFrame(X_test_imputed, columns=feature_cols, index=X_test.index)
        
        # 3. Дополнительная проверка на NaN
        if X_train.isna().any().any():
            st.warning(f"В тренировочных данных остались NaN значения в колонках: {X_train.columns[X_train.isna().any()].tolist()}")
            # Заполняем оставшиеся нулями
            X_train = X_train.fillna(0)
        
        if X_test.isna().any().any():
            st.warning(f"В тестовых данных остались NaN значения в колонках: {X_test.columns[X_test.isna().any()].tolist()}")
            # Заполняем оставшиеся нулями
            X_test = X_test.fillna(0)
        
        # Проверяем наличие данных
        if len(X_train) == 0 or len(X_test) == 0:
            st.error("Недостаточно данных для диагностики")
            return None, None, None, None, None
        
        st.info(f"✅ Данные подготовлены. Размеры: X_train={X_train.shape}, X_test={X_test.shape}")
        
        return X_train, y_train, X_test, y_test, feature_cols
        
    except Exception as e:
        st.error(f"Ошибка при подготовке данных для диагностики: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None, None

def get_model_object(best_model_info, integrated_results):
    """Получение объекта модели на основе информации из 5 этапа"""
    
    if best_model_info is None:
        return None, None
    
    model_type = best_model_info['Тип']
    model_name = best_model_info['Название']
    
    # Для ML моделей из Этапа 3
    if 'Этап 3' in model_type:
        # Проверяем, есть ли данные из Этапа 3
        if 'modeling_results' not in st.session_state:
            st.warning(f"⚠️ Не найдены данные Этапа 3 для модели {model_name}")
            return None, 'standard'
        
        modeling_results = st.session_state.modeling_results
        
        try:
            # Пробуем найти модель в optimizer
            optimizer = modeling_results.get('optimizer')
            if optimizer and hasattr(optimizer, 'best_models'):
                # Ищем модель по имени
                for key, model in optimizer.best_models.items():
                    if key in model_name or model_name in key:
                        # Проверяем, есть ли у модели pipeline
                        if hasattr(model, 'steps') and len(model.steps) > 1:
                            st.info(f"✅ Найдена модель с pipeline: {model_name}")
                        return model, 'standard'
            
            # Если не нашли, создаем простую модель на основе типа
            st.warning(f"⚠️ Модель {model_name} не найдена в данных Этапа 3. Используем простую замену.")
            
            # Определяем тип модели по названию
            model_name_lower = model_name.lower()
            if 'ridge' in model_name_lower:
                from sklearn.linear_model import Ridge
                return Ridge(alpha=1.0), 'standard'
            elif 'lasso' in model_name_lower:
                from sklearn.linear_model import Lasso
                return Lasso(alpha=0.1, max_iter=10000), 'standard'
            elif 'random' in model_name_lower or 'forest' in model_name_lower:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(n_estimators=50, random_state=42), 'standard'
            elif 'xgboost' in model_name_lower or 'lightgbm' in model_name_lower:
                try:
                    from xgboost import XGBRegressor
                    # XGBoost может обрабатывать NaN
                    return XGBRegressor(n_estimators=50, random_state=42, enable_categorical=False), 'standard'
                except:
                    pass
            elif 'autogluon' in model_name_lower:
                return None, 'autogluon'
            
            # По умолчанию - линейная регрессия
            from sklearn.linear_model import LinearRegression
            return LinearRegression(), 'standard'
            
        except Exception as e:
            st.warning(f"Ошибка при получении модели {model_name}: {str(e)}")
            return None, 'standard'
    
    # Для стратегий из Этапа 4 - пока не поддерживаем диагностику
    elif 'Этап 4' in model_type:
        st.info(f"""
        ⚠️ **Диагностика стратегий прогнозирования (Этап 4)**
        
        Модель **{model_name}** - это стратегия multi-step прогнозирования.
        Полная диагностика для таких стратегий требует отдельной реализации.
        
        **Рекомендация:** Выберите для диагностики ML модель из Этапа 3.
        """)
        return None, 'strategy'
    
    else:
        st.warning(f"Неизвестный тип модели: {model_type}")
        return None, 'unknown'

# ============================================================
# КЛАСС ДЛЯ ДИАГНОСТИКИ МОДЕЛЕЙ (ОБНОВЛЕННЫЙ)
# ============================================================

class ModelDiagnosticsEnhanced:
    """Улучшенный класс для диагностики моделей с интеграцией из 5 этапа"""
    
    def __init__(self, model, X_train, y_train, X_test, y_test, 
                 model_name="Модель", model_type="standard"):
        """
        Инициализация класса диагностики
        
        Parameters:
        -----------
        model : object or None
            Объект модели или None если модель не найдена
        X_train : pd.DataFrame
            Обучающие признаки
        y_train : pd.Series
            Обучающая целевая переменная
        X_test : pd.DataFrame
            Тестовые признаки
        y_test : pd.Series
            Тестовая целевая переменная
        model_name : str
            Имя модели для отображения
        model_type : str
            Тип модели: 'standard', 'baseline', 'autogluon', 'strategy'
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.model_type = model_type
        
        # Предсказания и остатки
        self.y_train_pred = None
        self.y_test_pred = None
        self.train_residuals = None
        self.test_residuals = None
        
        # Результаты тестов
        self.adf_results = None
        self.kpss_results = None
        self.shap_values = None
        self.feature_importance = None
        
        # Параметры модели
        self.model_params = {}
        
    def calculate_predictions_and_residuals(self):
        """Расчет предсказаний и остатков"""
        
        try:
            # Если модель не найдена, используем простой прогноз
            if self.model is None:
                st.warning(f"⚠️ Модель {self.model_name} не найдена. Используем простой прогноз.")
                
                # Используем среднее значение для прогноза
                mean_value = self.y_train.mean() if len(self.y_train) > 0 else 0
                self.y_train_pred = np.full(len(self.y_train), mean_value)
                self.y_test_pred = np.full(len(self.y_test), mean_value)
                
            # Для стандартных моделей
            elif self.model_type == 'standard':
                # Проверяем и обрабатываем NaN в данных перед предсказанием
                X_train_clean = self.X_train.copy()
                X_test_clean = self.X_test.copy()
                
                # Если есть NaN, заполняем их
                if X_train_clean.isna().any().any():
                    imputer = SimpleImputer(strategy='median')
                    X_train_clean = imputer.fit_transform(X_train_clean)
                    X_test_clean = imputer.transform(X_test_clean)
                
                if hasattr(self.model, 'predict'):
                    try:
                        self.y_train_pred = self.model.predict(X_train_clean)
                        self.y_test_pred = self.model.predict(X_test_clean)
                    except Exception as predict_error:
                        st.warning(f"Ошибка предсказания: {predict_error}. Создаем простую модель для диагностики.")
                        # Создаем простую модель для диагностики
                        from sklearn.linear_model import LinearRegression
                        simple_model = LinearRegression()
                        simple_model.fit(X_train_clean, self.y_train)
                        self.y_train_pred = simple_model.predict(X_train_clean)
                        self.y_test_pred = simple_model.predict(X_test_clean)
                else:
                    raise ValueError("Модель не имеет метода predict")
            
            # Для AutoGluon моделей
            elif self.model_type == 'autogluon':
                st.info("⚠️ AutoGluon модель - прогнозирование может занять время")
                try:
                    # Если модель - это AutoGluon predictor
                    if hasattr(self.model, 'predict'):
                        # Проверяем и обрабатываем NaN для AutoGluon
                        X_train_clean = self.X_train.fillna(0)
                        X_test_clean = self.X_test.fillna(0)
                        self.y_train_pred = self.model.predict(X_train_clean)
                        self.y_test_pred = self.model.predict(X_test_clean)
                    else:
                        raise ValueError("AutoGluon модель не поддерживает predict")
                except Exception as e:
                    st.warning(f"Ошибка предсказания AutoGluon: {str(e)}")
                    mean_value = self.y_train.mean() if len(self.y_train) > 0 else 0
                    self.y_train_pred = np.full(len(self.y_train), mean_value)
                    self.y_test_pred = np.full(len(self.y_test), mean_value)
            
            # Для стратегий (не поддерживаем)
            elif self.model_type == 'strategy':
                st.warning("⚠️ Диагностика стратегий прогнозирования не поддерживается")
                mean_value = self.y_train.mean() if len(self.y_train) > 0 else 0
                self.y_train_pred = np.full(len(self.y_train), mean_value)
                self.y_test_pred = np.full(len(self.y_test), mean_value)
                return False
            
            else:
                st.error(f"⚠️ Неизвестный тип модели: {self.model_type}")
                return False
            
            # Рассчитываем остатки
            self.train_residuals = self.y_train - self.y_train_pred
            self.test_residuals = self.y_test - self.y_test_pred
            
            # Проверяем, что остатки корректны
            if np.any(np.isnan(self.train_residuals)) or np.any(np.isnan(self.test_residuals)):
                st.warning("⚠️ В остатках обнаружены NaN значения. Заменяем их на 0.")
                self.train_residuals = np.nan_to_num(self.train_residuals, nan=0.0)
                self.test_residuals = np.nan_to_num(self.test_residuals, nan=0.0)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Ошибка при расчете предсказаний: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return False
    
    def get_model_parameters(self):
        """Получение параметров модели"""
        
        if self.model is None:
            return {}
        
        try:
            params = {}
            
            # Для sklearn моделей
            if hasattr(self.model, 'get_params'):
                params = self.model.get_params()
            
            # Для AutoGluon
            elif self.model_type == 'autogluon':
                params = {'type': 'AutoGluon model'}
            
            # Сохраняем параметры
            self.model_params = params
            
            return params
            
        except Exception as e:
            st.warning(f"Не удалось получить параметры модели: {str(e)}")
            return {}
    
    def analyze_residuals(self):
        """Анализ остатков временного ряда"""
        if self.train_residuals is None:
            if not self.calculate_predictions_and_residuals():
                return None
        
        analysis = {
            'train_mean': float(np.mean(self.train_residuals)),
            'train_std': float(np.std(self.train_residuals)),
            'train_skew': float(stats.skew(self.train_residuals)),
            'train_kurtosis': float(stats.kurtosis(self.train_residuals)),
            'test_mean': float(np.mean(self.test_residuals)),
            'test_std': float(np.std(self.test_residuals)),
            'test_skew': float(stats.skew(self.test_residuals)),
            'test_kurtosis': float(stats.kurtosis(self.test_residuals)),
        }
        
        # Тесты на нормальность
        if len(self.train_residuals) <= 5000:
            try:
                analysis['train_shapiro_p'] = float(stats.shapiro(self.train_residuals)[1])
            except:
                analysis['train_shapiro_p'] = np.nan
        else:
            analysis['train_shapiro_p'] = np.nan
            
        if len(self.test_residuals) <= 5000:
            try:
                analysis['test_shapiro_p'] = float(stats.shapiro(self.test_residuals)[1])
            except:
                analysis['test_shapiro_p'] = np.nan
        else:
            analysis['test_shapiro_p'] = np.nan
        
        return analysis
    
    def stationarity_tests(self):
        """Тесты на стационарность остатков"""
        if self.train_residuals is None:
            return None
        
        results = {}
        
        try:
            # ADF тест (тест Дики-Фуллера)
            adf_test = adfuller(self.train_residuals.dropna())
            results['adf'] = {
                'statistic': float(adf_test[0]),
                'p_value': float(adf_test[1]),
                'critical_values': {k: float(v) for k, v in adf_test[4].items()},
                'stationary': adf_test[1] < 0.05
            }
        except Exception as e:
            results['adf'] = {'error': str(e)}
        
        try:
            # KPSS тест
            kpss_test = kpss(self.train_residuals.dropna(), regression='c', nlags='auto')
            results['kpss'] = {
                'statistic': float(kpss_test[0]),
                'p_value': float(kpss_test[1]),
                'critical_values': {k: float(v) for k, v in kpss_test[3].items()},
                'stationary': kpss_test[1] > 0.05
            }
        except Exception as e:
            results['kpss'] = {'error': str(e)}
        
        self.adf_results = results.get('adf')
        self.kpss_results = results.get('kpss')
        
        return results
    
    def calculate_feature_importance(self):
        """Расчет важности признаков"""
        
        if self.model is None or self.model_type == 'strategy':
            self.feature_importance = pd.DataFrame({
                'feature': ['Модель не найдена'],
                'importance': [0],
                'note': ['Не удалось рассчитать важность признаков']
            })
            return self.feature_importance
        
        try:
            # Для моделей с атрибутом feature_importances_
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                self.feature_importance = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
            # Для линейных моделей с коэффициентами
            elif hasattr(self.model, 'coef_'):
                coef = self.model.coef_
                if len(coef.shape) > 1:
                    coef = coef[0]
                
                self.feature_importance = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': np.abs(coef)
                }).sort_values('importance', ascending=False)
                
            # Для SHAP анализа
            elif SHAP_AVAILABLE and self.model_type != 'autogluon':
                try:
                    # Обрабатываем NaN для SHAP
                    X_train_clean = self.X_train.fillna(self.X_train.median())
                    explainer = shap.Explainer(self.model)
                    shap_values = explainer(X_train_clean)
                    
                    shap_values_abs = np.abs(shap_values.values)
                    self.feature_importance = pd.DataFrame({
                        'feature': self.X_train.columns,
                        'importance': shap_values_abs.mean(axis=0)
                    }).sort_values('importance', ascending=False)
                    
                    self.shap_values = shap_values
                    
                except Exception as e:
                    st.warning(f"SHAP анализ не удался: {str(e)}")
                    self._calculate_correlation_importance()
            
            # Запасной вариант: корреляция
            else:
                self._calculate_correlation_importance()
            
            return self.feature_importance
            
        except Exception as e:
            st.warning(f"Не удалось рассчитать важность признаков: {str(e)}")
            self._calculate_correlation_importance()
            return self.feature_importance
    
    def _calculate_correlation_importance(self):
        """Расчет важности через корреляцию"""
        try:
            # Обрабатываем NaN для корреляции
            X_train_clean = self.X_train.fillna(self.X_train.median())
            y_train_clean = self.y_train.fillna(self.y_train.median())
            
            correlations = []
            for col in X_train_clean.columns:
                try:
                    # Используем только строки без NaN
                    mask = X_train_clean[col].notna() & y_train_clean.notna()
                    if mask.sum() > 1:  # Нужно как минимум 2 точки для корреляции
                        corr = np.corrcoef(X_train_clean.loc[mask, col], y_train_clean[mask])[0, 1]
                        correlations.append(abs(corr) if not np.isnan(corr) else 0)
                    else:
                        correlations.append(0)
                except:
                    correlations.append(0)
            
            self.feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': correlations,
                'note': ['Корреляция с целевой переменной']
            }).sort_values('importance', ascending=False)
            
        except Exception as e:
            self.feature_importance = pd.DataFrame({
                'feature': ['Не удалось рассчитать'],
                'importance': [0],
                'note': [f'Ошибка: {str(e)[:50]}']
            })
    
    def calculate_confidence_intervals(self, alpha=0.05):
        """Расчет доверительных интервалов"""
        if self.y_test_pred is None:
            if not self.calculate_predictions_and_residuals():
                return None
        
        try:
            # Используем стандартное отклонение остатков
            residual_std = np.std(self.train_residuals)
            
            # Z-значение для доверительного интервала
            z_value = stats.norm.ppf(1 - alpha/2)
            
            # Доверительные интервалы
            lower_bound = self.y_test_pred - z_value * residual_std
            upper_bound = self.y_test_pred + z_value * residual_std
            
            coverage = np.mean((self.y_test >= lower_bound) & (self.y_test <= upper_bound))
            
            return {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'coverage': float(coverage),
                'expected_coverage': 1 - alpha,
                'residual_std': float(residual_std)
            }
        except Exception as e:
            st.warning(f"Не удалось рассчитать доверительные интервалы: {str(e)}")
            return None
    
    def find_error_patterns(self, window_size=7):
        """Поиск паттернов в ошибках"""
        if self.test_residuals is None:
            if not self.calculate_predictions_and_residuals():
                return None
        
        try:
            # Обрабатываем NaN в остатках
            residuals_series = pd.Series(self.test_residuals).fillna(0)
            
            patterns = {
                'max_error': float(residuals_series.abs().max()),
                'max_error_idx': int(residuals_series.abs().idxmax()),
                'mean_abs_error': float(residuals_series.abs().mean()),
            }
            
            # Автокорреляция ошибок
            if len(residuals_series) > 1:
                patterns['error_autocorrelation'] = float(residuals_series.autocorr())
            else:
                patterns['error_autocorrelation'] = 0.0
            
            # Скользящие статистики ошибок
            if len(residuals_series) >= window_size:
                rolling_mean = residuals_series.abs().rolling(window=window_size).mean()
                patterns['worst_period_start'] = int(rolling_mean.idxmax())
                patterns['worst_period_value'] = float(rolling_mean.max())
            else:
                patterns['worst_period_start'] = 0
                patterns['worst_period_value'] = 0.0
            
            return patterns
        except Exception as e:
            st.warning(f"Не удалось найти паттерны ошибок: {str(e)}")
            return None

# ============================================================
# ВИЗУАЛИЗАЦИИ (ОБНОВЛЕННЫЕ)
# ============================================================

def plot_residuals_analysis_enhanced(diagnostics):
    """Визуализация анализа остатков для улучшенной диагностики"""
    if diagnostics.train_residuals is None:
        return None
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['Гистограмма остатков', 'QQ-plot остатков', 
                       'Автокорреляция остатков', 'Остатки vs Прогноз',
                       'Остатки во времени', 'Накопленные остатки'],
        specs=[[{'type': 'histogram'}, {'type': 'scatter'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Гистограмма остатков
    fig.add_trace(
        go.Histogram(
            x=diagnostics.train_residuals,
            name='Остатки',
            nbinsx=50,
            marker_color='lightblue',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Добавляем нормальное распределение для сравнения
    try:
        mu, sigma = diagnostics.train_residuals.mean(), diagnostics.train_residuals.std()
        x_norm = np.linspace(diagnostics.train_residuals.min(), diagnostics.train_residuals.max(), 100)
        y_norm = stats.norm.pdf(x_norm, mu, sigma) * len(diagnostics.train_residuals) * (diagnostics.train_residuals.max() - diagnostics.train_residuals.min()) / 50
        
        fig.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                mode='lines',
                name='Нормальное распределение',
                line=dict(color='red', width=2),
                showlegend=False
            ),
            row=1, col=1
        )
    except:
        pass
    
    # 2. QQ-plot
    try:
        # Убираем NaN для QQ-plot
        residuals_clean = diagnostics.train_residuals[~np.isnan(diagnostics.train_residuals)]
        qq = stats.probplot(residuals_clean, dist="norm")
        theoretical_q = qq[0][0]
        sample_q = qq[0][1]
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_q,
                y=sample_q,
                mode='markers',
                name='QQ-plot',
                marker=dict(color='red', size=5, opacity=0.6)
            ),
            row=1, col=2
        )
        
        # Линия идеального распределения
        min_val = min(theoretical_q.min(), sample_q.min())
        max_val = max(theoretical_q.max(), sample_q.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Идеальная линия',
                line=dict(color='black', dash='dash', width=2),
                showlegend=False
            ),
            row=1, col=2
        )
    except:
        pass
    
    # 3. Автокорреляция остатков
    try:
        # Убираем NaN для ACF
        residuals_clean = diagnostics.train_residuals[~np.isnan(diagnostics.train_residuals)]
        acf_values = acf(residuals_clean, nlags=20, fft=False)
        fig.add_trace(
            go.Bar(
                x=list(range(len(acf_values))),
                y=acf_values,
                name='ACF',
                marker_color='orange',
                opacity=0.7
            ),
            row=1, col=3
        )
        
        # Доверительные интервалы
        conf_int = 1.96 / np.sqrt(len(residuals_clean))
        fig.add_trace(
            go.Scatter(
                x=[-1, len(acf_values)],
                y=[conf_int, conf_int],
                mode='lines',
                line=dict(color='gray', dash='dash', width=1),
                showlegend=False
            ),
            row=1, col=3
        )
        fig.add_trace(
            go.Scatter(
                x=[-1, len(acf_values)],
                y=[-conf_int, -conf_int],
                mode='lines',
                line=dict(color='gray', dash='dash', width=1),
                showlegend=False
            ),
            row=1, col=3
        )
    except:
        pass
    
    # 4. Остатки vs Прогноз
    fig.add_trace(
        go.Scatter(
            x=diagnostics.y_train_pred,
            y=diagnostics.train_residuals,
            mode='markers',
            name='Остатки vs Прогноз',
            marker=dict(color='green', size=5, opacity=0.6)
        ),
        row=2, col=1
    )
    
    # Нулевая линия
    fig.add_trace(
        go.Scatter(
            x=[diagnostics.y_train_pred.min(), diagnostics.y_train_pred.max()],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', dash='dash', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 5. Остатки во времени
    fig.add_trace(
        go.Scatter(
            x=list(range(len(diagnostics.train_residuals))),
            y=diagnostics.train_residuals,
            mode='lines',
            name='Остатки во времени',
            line=dict(color='blue', width=1)
        ),
        row=2, col=2
    )
    
    # 6. Накопленные остатки
    cumulative_residuals = np.cumsum(diagnostics.train_residuals)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(cumulative_residuals))),
            y=cumulative_residuals,
            mode='lines',
            name='Накопленные остатки',
            line=dict(color='purple', width=2)
        ),
        row=2, col=3
    )
    
    # Нулевая линия для накопленных остатков
    fig.add_trace(
        go.Scatter(
            x=[0, len(cumulative_residuals)],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', dash='dash', width=1),
            showlegend=False
        ),
        row=2, col=3
    )
    
    fig.update_layout(
        height=800,
        title_text=f"Диагностика остатков: {diagnostics.model_name}",
        title_x=0.5,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

# ============================================================
# ОСНОВНОЙ ИНТЕРФЕЙС ДЛЯ STREAMLIT (ИСПРАВЛЕННЫЙ)
# ============================================================

def show_model_diagnostics_interface_enhanced():
    """Основной интерфейс Этапа 6: Диагностика моделей с интеграцией из 5 этапа"""
    
    
    # Проверка наличия данных из предыдущих этапов
    if 'df_features' not in st.session_state or 'feature_info' not in st.session_state or 'split_data' not in st.session_state:
        st.error("❌ Сначала выполните Этапы 1-2: Подготовку данных и разбиение")
        return
    
    # Получаем лучшую модель из 5 этапа
    best_model_info, integrated_results, integrated_df = extract_model_from_integrated_results()
    
    if best_model_info is None:
        st.error("❌ Не удалось определить лучшую модель для диагностики")
        return
    
    # Подготавливаем данные для диагностики
    feature_info = st.session_state.feature_info
    split_data = st.session_state.split_data
    
    X_train, y_train, X_test, y_test, feature_cols = prepare_data_for_diagnostics(feature_info, split_data)
    
    if X_train is None:
        st.error("❌ Не удалось подготовить данные для диагностики")
        return
    
    # === ОТЛАДОЧНАЯ КНОПКА ДЛЯ ПРОВЕРКИ ДАННЫХ ===
    if st.checkbox("🔍 Показать детальную информацию о данных", value=False):
        st.write("### Детальная информация о данных:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**X_train:**")
            st.write(f"Размер: {X_train.shape}")
            st.write(f"Типы данных: {X_train.dtypes.unique()}")
            st.write(f"Количество NaN: {X_train.isna().sum().sum()}")
            st.write("Первые 5 строк:")
            st.dataframe(X_train.head())
        
        with col2:
            st.write("**X_test:**")
            st.write(f"Размер: {X_test.shape}")
            st.write(f"Типы данных: {X_test.dtypes.unique()}")
            st.write(f"Количество NaN: {X_test.isna().sum().sum()}")
            st.write("Первые 5 строк:")
            st.dataframe(X_test.head())
        
        st.write("**y_train:**")
        st.write(f"Размер: {y_train.shape}")
        st.write(f"Тип: {y_train.dtype}")
        st.write(f"Количество NaN: {y_train.isna().sum()}")
        st.write(f"Статистики: min={y_train.min():.4f}, max={y_train.max():.4f}, mean={y_train.mean():.4f}")
        
        st.write("**y_test:**")
        st.write(f"Размер: {y_test.shape}")
        st.write(f"Тип: {y_test.dtype}")
        st.write(f"Количество NaN: {y_test.isna().sum()}")
        st.write(f"Статистики: min={y_test.min():.4f}, max={y_test.max():.4f}, mean={y_test.mean():.4f}")
    
    # Получаем объект модели
    model_object, model_type = get_model_object(best_model_info, integrated_results)
    
    st.info(f"""
    **📋 Диагностика лучшей модели из Этапа 5:**
    
    **Модель:** {best_model_info['Название']}
    **Тип:** {best_model_info['Тип']}
    **MAE:** {best_model_info['MAE']:.4f}
    **Подход:** {best_model_info['Подход']}
    
    **Данные для диагностики:**
    - Обучающая выборка: {len(X_train)} записей
    - Тестовая выборка: {len(X_test)} записей
    - Количество признаков: {len(feature_cols)}
    """)
    
    # Создаем объект диагностики
    diagnostics = ModelDiagnosticsEnhanced(
        model=model_object,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name=best_model_info['Название'],
        model_type=model_type
    )
    
    # Выполняем расчеты
    with st.spinner("Выполняем диагностику модели..."):
        success = diagnostics.calculate_predictions_and_residuals()
        if not success:
            st.error("❌ Не удалось рассчитать предсказания модели")
            return
        
        residuals_analysis = diagnostics.analyze_residuals()
        stationarity_results = diagnostics.stationarity_tests()
        feature_importance = diagnostics.calculate_feature_importance()
        model_params = diagnostics.get_model_parameters()
    
    st.success("✅ Базовая диагностика выполнена успешно!")
    
    # Настройки диагностики
    st.subheader("⚙️ Настройки диагностики")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_residuals = st.checkbox("📈 Анализ остатков", value=True)
        include_stationarity = st.checkbox("📊 Тесты на стационарность", value=True)
        include_ci = st.checkbox("🎯 Доверительные интервалы", value=True)
    
    with col2:
        if model_type != 'strategy':
            include_feature_importance = st.checkbox("🔝 Важность признаков", value=True)
            include_error_patterns = st.checkbox("⚠️ Анализ ошибок", value=True)
        else:
            include_feature_importance = False
            include_error_patterns = False
    
    # Для SHAP и PDP
    advanced_options = st.expander("⚡ Продвинутые опции диагностики")
    
    with advanced_options:
        col1, col2 = st.columns(2)
        with col1:
            include_shap = st.checkbox("SHAP анализ", value=False) and SHAP_AVAILABLE and model_type not in ['strategy', 'autogluon']
        with col2:
            include_pdp = st.checkbox("Частичные зависимости (PDP)", value=False) and PDP_AVAILABLE and model_type not in ['strategy', 'autogluon']
    
    st.markdown("---")
    
    # Создаем вкладки для диагностики
    tab_names = []
    if include_residuals:
        tab_names.append("📈 Анализ остатков")
    if include_stationarity:
        tab_names.append("📊 Стационарность")
    if include_feature_importance:
        tab_names.append("🔝 Важность признаков")
    if include_ci:
        tab_names.append("🎯 Доверительные интервалы")
    if include_error_patterns:
        tab_names.append("⚠️ Паттерны ошибок")
    if include_shap:
        tab_names.append("🎯 SHAP анализ")
    if include_pdp:
        tab_names.append("📐 Частичные зависимости")
    
    if not tab_names:
        st.warning("Выберите хотя бы один тип диагностики")
        return
    
    tabs = st.tabs(tab_names)
    tab_idx = 0
    
    # 1. Анализ остатков
    if include_residuals:
        with tabs[tab_idx]:
            st.subheader("📈 Анализ остатков модели")
            
            fig_residuals = plot_residuals_analysis_enhanced(diagnostics)
            if fig_residuals:
                st.plotly_chart(fig_residuals, use_container_width=True)
            
            # Статистика остатков
            if residuals_analysis:
                st.subheader("Статистика остатков")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Среднее", f"{residuals_analysis['train_mean']:.4f}")
                with col2:
                    st.metric("Стд. отклонение", f"{residuals_analysis['train_std']:.4f}")
                with col3:
                    st.metric("Скошенность", f"{residuals_analysis['train_skew']:.4f}")
                with col4:
                    st.metric("Эксцесс", f"{residuals_analysis['train_kurtosis']:.4f}")
                
                # Интерпретация
                st.info(f"""
                **Интерпретация остатков для {best_model_info['Название']}:**
                
                - **Среднее ≈ 0**: модель не имеет систематической ошибки
                - **Нормальное распределение**: p-значение теста Шапиро-Вилка: {residuals_analysis.get('train_shapiro_p', 'N/A'):.4f}
                - **Скошенность ≈ 0**: остатки симметричны
                - **Эксцесс ≈ 3**: остатки имеют нормальную остроту пика
                """)
            
            tab_idx += 1
    
    # 2. Тесты на стационарность
    if include_stationarity:
        with tabs[tab_idx]:
            st.subheader("📊 Тесты на стационарность остатков")
            
            if stationarity_results:
                # Создаем таблицу с результатами
                results_df = pd.DataFrame([
                    {
                        'Тест': 'ADF',
                        'Статистика': stationarity_results['adf'].get('statistic', np.nan),
                        'P-значение': stationarity_results['adf'].get('p_value', np.nan),
                        'Стационарность': 'Да' if stationarity_results['adf'].get('stationary', False) else 'Нет'
                    },
                    {
                        'Тест': 'KPSS', 
                        'Статистика': stationarity_results['kpss'].get('statistic', np.nan),
                        'P-значение': stationarity_results['kpss'].get('p_value', np.nan),
                        'Стационарность': 'Да' if stationarity_results['kpss'].get('stationary', False) else 'Нет'
                    }
                ])
                
                st.dataframe(results_df, width='stretch')
                
                # Интерпретация
                st.info("""
                **Интерпретация тестов на стационарность:**
                
                - **ADF тест (Augmented Dickey-Fuller):**
                  - Нулевая гипотеза: ряд нестационарен
                  - p < 0.05: отвергаем нулевую гипотезу, ряд стационарен
                
                - **KPSS тест (Kwiatkowski-Phillips-Schmidt-Shin):**
                  - Нулевая гипотеза: ряд стационарен
                  - p > 0.05: не отвергаем нулевую гипотезу, ряд стационарен
                
                **Для хорошей модели:** остатки должны быть стационарны (белый шум)
                """)
            
            tab_idx += 1
    
    # 3. Важность признаков
    if include_feature_importance:
        with tabs[tab_idx]:
            st.subheader("🔝 Важность признаков")
            
            if feature_importance is not None and not feature_importance.empty:
                # График важности признаков
                fig_importance = go.Figure()
                
                # Берем топ-15 признаков
                top_n = min(15, len(feature_importance))
                top_features = feature_importance.head(top_n)
                
                fig_importance.add_trace(go.Bar(
                    x=top_features['importance'],
                    y=top_features['feature'],
                    orientation='h',
                    marker_color='teal',
                    text=top_features['importance'].round(4),
                    textposition='auto'
                ))
                
                fig_importance.update_layout(
                    title=f'Топ-{top_n} важных признаков',
                    xaxis_title='Важность',
                    yaxis_title='Признак',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Таблица с важностью признаков
                st.subheader("Таблица важности признаков")
                st.dataframe(feature_importance, width='stretch')
            
            else:
                st.warning("Не удалось рассчитать важность признаков для данной модели")
            
            tab_idx += 1
    
    # 4. Доверительные интервалы
    if include_ci:
        with tabs[tab_idx]:
            st.subheader("🎯 Доверительные интервалы")
            
            ci_results = diagnostics.calculate_confidence_intervals()
            
            if ci_results:
                # График с доверительными интервалами
                fig_ci = go.Figure()
                
                # Временная ось
                time_index = list(range(len(diagnostics.y_test)))
                
                # Истинные значения
                fig_ci.add_trace(go.Scatter(
                    x=time_index,
                    y=diagnostics.y_test,
                    mode='lines',
                    name='Истинные значения',
                    line=dict(color='blue', width=2)
                ))
                
                # Предсказания
                fig_ci.add_trace(go.Scatter(
                    x=time_index,
                    y=diagnostics.y_test_pred,
                    mode='lines',
                    name='Предсказания',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Доверительные интервалы
                fig_ci.add_trace(go.Scatter(
                    x=time_index + time_index[::-1],
                    y=np.concatenate([ci_results['upper_bound'], ci_results['lower_bound'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(128, 128, 128, 0.2)',
                    line=dict(color='rgba(128, 128, 128, 0)'),
                    name='95% доверительный интервал',
                    showlegend=True
                ))
                
                coverage_text = f"Покрытие: {ci_results['coverage']:.1%} (ожидается: {ci_results['expected_coverage']:.1%})"
                
                fig_ci.update_layout(
                    title=f'Предсказания с доверительными интервалами<br><sup>{coverage_text}</sup>',
                    xaxis_title='Временной индекс',
                    yaxis_title='Значение',
                    height=500,
                    showlegend=True,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_ci, use_container_width=True)
                
                # Статистика покрытия
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Фактическое покрытие", f"{ci_results['coverage']:.1%}")
                with col2:
                    st.metric("Ожидаемое покрытие", f"{ci_results['expected_coverage']:.1%}")
            
            else:
                st.warning("Не удалось рассчитать доверительные интервалы")
            
            tab_idx += 1
    
    # 5. Паттерны ошибок
    if include_error_patterns:
        with tabs[tab_idx]:
            st.subheader("⚠️ Паттерны ошибок")
            
            error_patterns = diagnostics.find_error_patterns()
            
            if error_patterns:
                # Графики ошибок
                fig_errors = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Абсолютные ошибки', 'Распределение ошибок',
                                   'Накопленные ошибки', 'Автокорреляция ошибок'],
                    specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
                           [{'type': 'scatter'}, {'type': 'bar'}]]
                )
                
                # Абсолютные ошибки
                abs_errors = np.abs(diagnostics.test_residuals)
                fig_errors.add_trace(
                    go.Scatter(
                        x=list(range(len(abs_errors))),
                        y=abs_errors,
                        mode='lines',
                        name='Абсолютные ошибки',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
                
                # Распределение ошибок
                fig_errors.add_trace(
                    go.Histogram(
                        x=diagnostics.test_residuals,
                        name='Распределение ошибок',
                        nbinsx=30,
                        marker_color='lightgreen',
                        opacity=0.7
                    ),
                    row=1, col=2
                )
                
                # Накопленные ошибки
                cumulative_errors = np.cumsum(diagnostics.test_residuals)
                fig_errors.add_trace(
                    go.Scatter(
                        x=list(range(len(cumulative_errors))),
                        y=cumulative_errors,
                        mode='lines',
                        name='Накопленные ошибки',
                        line=dict(color='purple', width=2)
                    ),
                    row=2, col=1
                )
                
                # Автокорреляция ошибок
                try:
                    acf_errors = acf(diagnostics.test_residuals, nlags=20, fft=False)
                    fig_errors.add_trace(
                        go.Bar(
                            x=list(range(len(acf_errors))),
                            y=acf_errors,
                            name='Автокорреляция',
                            marker_color='blue',
                            opacity=0.7
                        ),
                        row=2, col=2
                    )
                except:
                    pass
                
                fig_errors.update_layout(
                    height=600,
                    title_text="Анализ паттернов ошибок",
                    title_x=0.5,
                    showlegend=False,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_errors, use_container_width=True)
                
                # Статистика ошибок
                st.subheader("Статистика ошибок")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Максимальная ошибка", f"{error_patterns['max_error']:.4f}")
                with col2:
                    st.metric("Средняя абс. ошибка", f"{error_patterns['mean_abs_error']:.4f}")
                with col3:
                    st.metric("Автокорреляция", f"{error_patterns['error_autocorrelation']:.4f}")
            
            else:
                st.warning("Не удалось найти паттерны ошибок")
            
            tab_idx += 1
    
    # 6. SHAP анализ
    if include_shap and SHAP_AVAILABLE:
        with tabs[tab_idx]:
            st.subheader("🎯 SHAP анализ")
            
            try:
                if diagnostics.shap_values is None:
                    with st.spinner("Выполняется SHAP анализ..."):
                        # Обрабатываем NaN для SHAP
                        X_train_clean = diagnostics.X_train.fillna(diagnostics.X_train.median())
                        explainer = shap.Explainer(diagnostics.model)
                        diagnostics.shap_values = explainer(X_train_clean)
                
                if diagnostics.shap_values is not None:
                    # Summary plot
                    st.info("SHAP summary plot показывает важность признаков и их влияние на прогноз")
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(
                        diagnostics.shap_values,
                        X_train_clean,
                        plot_type="dot",
                        show=False,
                        max_display=15
                    )
                    
                    st.pyplot(fig)
                    
                    # Дополнительная информация
                    st.info("""
                    **Интерпретация SHAP:**
                    - **Высота столбца**: важность признака
                    - **Цвет точек**: значение признака (красный - высокое, синий - низкое)
                    - **Расположение точек**: влияние на прогноз (право - увеличение, лево - уменьшение)
                    """)
                    
            except Exception as e:
                st.warning(f"Не удалось выполнить SHAP анализ: {str(e)}")
            
            tab_idx += 1
    
    # 7. Частичные зависимости (PDP)
    if include_pdp and PDP_AVAILABLE:
        with tabs[tab_idx]:
            st.subheader("📐 Частичные зависимости (PDP)")
            
            # Выбор признаков для PDP
            if feature_importance is not None and not feature_importance.empty:
                top_features = feature_importance['feature'].head(10).tolist()
                
                selected_features = st.multiselect(
                    "Выберите признаки для анализа частичных зависимостей",
                    options=top_features,
                    default=top_features[:3] if len(top_features) >= 3 else top_features
                )
                
                if selected_features and diagnostics.model is not None:
                    try:
                        # Обрабатываем NaN для PDP
                        X_train_clean = diagnostics.X_train.fillna(diagnostics.X_train.median())
                        
                        fig, ax = plt.subplots(len(selected_features), 1, 
                                              figsize=(10, 4 * len(selected_features)))
                        
                        if len(selected_features) == 1:
                            ax = [ax]
                        
                        for i, feature in enumerate(selected_features):
                            PartialDependenceDisplay.from_estimator(
                                diagnostics.model,
                                X_train_clean,
                                [feature],
                                ax=ax[i],
                                grid_resolution=20
                            )
                            ax[i].set_title(f'Частичная зависимость: {feature}')
                            ax[i].set_xlabel(feature)
                            ax[i].set_ylabel('Влияние на прогноз')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        st.info("""
                        **Интерпретация PDP:**
                        - **Форма кривой**: показывает как изменяется прогноз при изменении признака
                        - **Наклон**: сила влияния признака
                        - **Независимость**: кривая показывает влияние признака при фиксированных других признаках
                        """)
                        
                    except Exception as e:
                        st.warning(f"Не удалось построить графики PDP: {str(e)}")
                
                else:
                    st.info("Выберите признаки для анализа частичных зависимостей")
            
            else:
                st.info("Сначала необходимо рассчитать важность признаков")
            
            tab_idx += 1
    
    # Сохраняем результаты диагностики
    st.session_state.diagnostics_results = {
        'diagnostics': diagnostics,
        'best_model_info': best_model_info,
        'model_params': model_params
    }
    
    st.markdown("---")
    
    # Заключительная информация
    st.subheader("🎯 Выводы и рекомендации")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Оценка модели {best_model_info['Название']}:**
        
        - **Тип модели:** {best_model_info['Тип']}
        - **MAE на тесте:** {best_model_info['MAE']:.4f}
        - **Стационарность остатков:** {'Да' if diagnostics.adf_results and diagnostics.adf_results.get('stationary', False) else 'Нет'}
        - **Качество модели:** {'Хорошее' if residuals_analysis and abs(residuals_analysis['train_mean']) < 0.1 else 'Требует улучшения'}
        """)
    
    with col2:
        st.info("""
        **Рекомендации по улучшению:**
        
        1. **Для улучшения точности:** рассмотрите ансамблевые методы
        2. **Для стационарности:** примените дифференцирование к целевой переменной
        3. **Для важности признаков:** сфокусируйтесь на топ-5 наиболее важных признаков
        4. **Для доверительных интервалов:** увеличьте объем обучающих данных
        """)
    
    st.success(f"""
    **✅ Диагностика модели {best_model_info['Название']} завершена!**
    
    **Что было сделано:**
    1. Проведен анализ остатков модели
    2. Выполнены тесты на стационарность
    3. Проанализирована важность признаков
    4. Рассчитаны доверительные интервалы
    5. Выявлены паттерны ошибок
    
    **Теперь у вас есть полное понимание качества и возможностей вашей модели!**
    """)

# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def show_model_diagnostics_interface():
    """Основная функция для запуска интерфейса диагностики"""
    
    
    # ОТЛАДОЧНЫЙ ВЫВОД В КОНСОЛЬ (показывает в терминале)
    import sys
    print("\n" + "="*50, file=sys.stderr)
    print("[DEBUG] НАЧАЛО ЭТАПА 6 - ПРОВЕРКА ДАННЫХ", file=sys.stderr)
    print(f"[DEBUG] Всего ключей в session_state: {len(st.session_state)}", file=sys.stderr)
    
    # Проверяем наличие данных 5 этапа (все возможные ключи)
    stage5_keys = ['integrated_results', 'advanced_modeling_data', 'model_comparison_results']
    has_stage5_data = any(key in st.session_state for key in stage5_keys)
    
    if not has_stage5_data:
        st.error("""
        ❌ **Не выполнены Этапы 3-5!**
        
        **Что найдено в session_state:**
        """)
        
        # Показываем все ключи
        keys = list(st.session_state.keys())
        for key in keys:
            val = st.session_state[key]
            st.write(f"- `{key}`: {type(val).__name__}")
            
            # Если это словарь, показываем его ключи
            if isinstance(val, dict):
                st.write(f"  Ключи в словаре: {list(val.keys())[:5]}{'...' if len(val) > 5 else ''}")
        
        st.write("""
        **Необходимо выполнить:**
        1. **Этап 3:** Подбор ML моделей и гиперпараметров
        2. **Этап 4:** Сравнение стратегий прогнозирования  
        3. **Этап 5:** Интеграция и сравнение подходов
        
        **Особое внимание:** В 5 этапе должны быть сохранены результаты под одним из ключей:
        - `integrated_results`
        - `advanced_modeling_data` 
        - `model_comparison_results`
        
        **Порядок действий:**
        1. Перейдите на Этап 3 → выполните подбор моделей
        2. Перейдите на Этап 4 → протестируйте стратегии
        3. Перейдите на Этап 5 → выполните интегрированное сравнение
        4. Вернитесь на Этап 6 → выполните диагностику лучшей модели
        """)
        return
    
    # Проверяем наличие данных из Этапов 1-2
    required_stage1_2 = ['feature_info', 'split_data']
    missing_stage1_2 = [key for key in required_stage1_2 if key not in st.session_state]
    
    if missing_stage1_2:
        st.error(f"""
        ❌ Не выполнены Этапы 1-2!
        
        Отсутствующие данные: {', '.join(missing_stage1_2)}
        """)
        return
    
    # Запускаем улучшенный интерфейс
    show_model_diagnostics_interface_enhanced()