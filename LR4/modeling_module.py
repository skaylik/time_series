# modeling_module.py - Модуль для подбора гиперпараметров (Этап 3)

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Импорт для Streamlit
import streamlit as st
import plotly.graph_objects as go

# Импорт для моделей и метрик
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Импорт для градиентного бустинга
import lightgbm as lgb
from xgboost import XGBRegressor

# Импорт для Optuna
import optuna

# ============================================================
# ПРОВЕРКА И ИМПОРТ AUTOGLUON
# ============================================================
AUTOGLUON_AVAILABLE = False
try:
    from autogluon.tabular import TabularPredictor, TabularDataset
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    AUTOGLUON_AVAILABLE = True
    print("✅ AutoGluon успешно импортирован")
except ImportError as e:
    print(f"⚠️ AutoGluon не доступен: {e}")

# ============================================================
# КЛАСС ДЛЯ ПОДБОРА ГИПЕРПАРАМЕТРОВ
# ============================================================

class HyperparameterOptimizer:
    """
    Класс для подбора гиперпараметров временных рядов
    """
    
    def __init__(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None, tscv=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.tscv = tscv or TimeSeriesSplit(n_splits=5)
        self.best_models = {}
        self.results = {}
        
    def linear_models_grid_search(self):
        """
        GridSearchCV для линейных моделей с TimeSeriesSplit
        """
        results = []
        
        # Определяем модели и параметры
        models = {
            'Ridge': {
                'model': Ridge(),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    'fit_intercept': [True, False]
                }
            },
            'Lasso': {
                'model': Lasso(),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'max_iter': [1000, 2000]
                }
            },
            'ElasticNet': {
                'model': ElasticNet(),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'max_iter': [1000, 2000]
                }
            }
        }
        
        for name, model_info in models.items():
            with st.spinner(f"Оптимизация {name}..."):
                try:
                    # Создаем GridSearchCV с TimeSeriesSplit
                    grid_search = GridSearchCV(
                        estimator=model_info['model'],
                        param_grid=model_info['params'],
                        cv=self.tscv,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    # Обучаем
                    grid_search.fit(self.X_train, self.y_train)
                    
                    # Предсказания на валидации
                    y_pred = grid_search.predict(self.X_val)
                    
                    # Метрики
                    mae = mean_absolute_error(self.y_val, y_pred)
                    mse = mean_squared_error(self.y_val, y_pred)
                    r2 = r2_score(self.y_val, y_pred)
                    
                    # Сохраняем результаты
                    model_result = {
                        'model': name,
                        'best_params': grid_search.best_params_,
                        'best_score': -grid_search.best_score_,
                        'val_mae': mae,
                        'val_mse': mse,
                        'val_r2': r2,
                        'model_object': grid_search.best_estimator_
                    }
                    
                    results.append(model_result)
                    
                    # Сохраняем лучшую модель
                    self.best_models[name] = grid_search.best_estimator_
                    
                except Exception as e:
                    st.error(f"❌ Ошибка в {name}: {str(e)}")
        
        self.results['linear'] = results
        return results
    
    def gradient_boosting_optuna(self, n_trials=50):
        """
        Optuna для градиентного бустинга (LightGBM)
        """
        # Функция для Optuna
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'num_leaves': trial.suggest_int('num_leaves', 15, 40),
                'random_state': 42,
                'verbosity': -1,
                'n_jobs': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            
            # Простая кросс-валидация
            scores = []
            for train_idx, val_idx in self.tscv.split(self.X_train):
                X_train_fold = self.X_train.iloc[train_idx]
                y_train_fold = self.y_train.iloc[train_idx]
                X_val_fold = self.X_train.iloc[val_idx]
                y_val_fold = self.y_train.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                scores.append(mean_absolute_error(y_val_fold, y_pred))
            
            return np.mean(scores)
        
        # Создаем исследование Optuna
        study = optuna.create_study(direction='minimize')
        
        # Оптимизируем с меньшим числом trials для теста
        n_trials = min(n_trials, 30)
        
        with st.spinner(f"Optuna оптимизация ({n_trials} trials)..."):
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Лучшие параметры
        best_params = study.best_params
        best_value = study.best_value
        
        # Обучаем финальную модель на лучших параметрах
        final_params = best_params.copy()
        final_params['verbosity'] = -1
        final_params['random_state'] = 42
        final_params['n_jobs'] = -1
        
        final_model = lgb.LGBMRegressor(**final_params)
        final_model.fit(self.X_train, self.y_train)
        
        # Предсказания
        y_pred_train = final_model.predict(self.X_train)
        y_pred_val = final_model.predict(self.X_val)
        
        # Метрики
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        val_mae = mean_absolute_error(self.y_val, y_pred_val)
        val_mse = mean_squared_error(self.y_val, y_pred_val)
        val_r2 = r2_score(self.y_val, y_pred_val)
        
        # Важность признаков
        importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Сохраняем результаты
        result = {
            'model': 'LightGBM',
            'best_params': best_params,
            'best_cv_score': best_value,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'val_mse': val_mse,
            'val_r2': val_r2,
            'feature_importance': importance_df,
            'study': study,
            'model_object': final_model
        }
        
        self.results['gradient_boosting'] = [result]
        self.best_models['LightGBM'] = final_model
        
        return result
    
    def autogluon_automation(self, time_limit=120, presets=None):
        """
        Полная автоматизация с AutoGluon с отладочным выводом
        """
        if not AUTOGLUON_AVAILABLE:
            st.error("""
            ❌ AutoGluon не установлен!
            
            Установите AutoGluon для использования этой функции:
            ```
            pip install autogluon
            ```
            """)
            return None
        
        try:
            st.info("🤖 Запуск AutoGluon для автоматического подбора...")
            print("🔍 [DEBUG] Начало autogluon_automation")
            print(f"   - Размер X_train: {self.X_train.shape}")
            print(f"   - Размер y_train: {self.y_train.shape}")
            print(f"   - Размер X_val: {self.X_val.shape}")
            print(f"   - Размер y_val: {self.y_val.shape}")
            
            # Используем train и val данные из второго этапа
            train_data = pd.concat([self.X_train, self.y_train], axis=1)
            val_data = pd.concat([self.X_val, self.y_val], axis=1)
            
            print(f"🔍 [DEBUG] Размер train_data: {train_data.shape}")
            print(f"🔍 [DEBUG] Размер val_data: {val_data.shape}")
            
            # Определяем целевую переменную
            target_column = self.y_train.name
            print(f"🔍 [DEBUG] Целевая переменная: {target_column}")
            
            # Создаем TabularPredictor
            print("🔍 [DEBUG] Создание TabularPredictor...")
            predictor = TabularPredictor(
                label=target_column,
                problem_type='regression',
                eval_metric='mean_absolute_error'
            )
            
            print(f"🔍 [DEBUG] TabularPredictor создан: {predictor}")
            print(f"🔍 [DEBUG] Атрибуты predictor: {dir(predictor)}")
            
            # Используем presets из задания
            if presets is None:
                presets = ['medium_quality', 'high_quality', 'best_quality']
            
            print(f"🔍 [DEBUG] Используемые presets: {presets}")
            
            # Настройки обучения
            hyperparameters = {
                'GBM': [
                    {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
                    {},
                ],
                'CAT': {},
                'RF': [
                    {'criterion': 'mse', 'ag_args': {'name_suffix': 'MSE'}},
                ],
                'XT': [
                    {'criterion': 'mse', 'ag_args': {'name_suffix': 'MSE'}},
                ],
                'KNN': [
                    {'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}},
                    {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}},
                ],
            }
            
            # Обучаем AutoGluon с исправлением ошибки
            print(f"🔍 [DEBUG] Начало обучения AutoGluon (time_limit={time_limit} сек)...")
            with st.spinner(f"AutoGluon обучение (лимит времени: {time_limit} сек)..."):
                predictor.fit(
                    train_data=train_data,
                    tuning_data=val_data,
                    time_limit=time_limit,
                    presets=presets,
                    hyperparameters=hyperparameters,
                    use_bag_holdout=True,  # КРИТИЧЕСКИ ВАЖНО: добавляем этот параметр
                    verbosity=0
                )
            
            print("🔍 [DEBUG] Обучение AutoGluon завершено")
            
            # Проверяем доступные методы predictor
            print(f"🔍 [DEBUG] Доступные методы predictor после обучения:")
            for attr in dir(predictor):
                if not attr.startswith('_'):
                    print(f"   - {attr}")
            
            # Предсказания
            print("🔍 [DEBUG] Выполнение предсказаний...")
            y_pred_train = predictor.predict(train_data)
            y_pred_val = predictor.predict(val_data)
            
            print(f"🔍 [DEBUG] Размер y_pred_train: {len(y_pred_train)}")
            print(f"🔍 [DEBUG] Размер y_pred_val: {len(y_pred_val)}")
            
            # Метрики
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            val_mae = mean_absolute_error(self.y_val, y_pred_val)
            val_mse = mean_squared_error(self.y_val, y_pred_val)
            val_r2 = r2_score(self.y_val, y_pred_val)
            
            print(f"🔍 [DEBUG] Метрики - Train MAE: {train_mae}, Val MAE: {val_mae}")
            
            # Leaderboard
            print("🔍 [DEBUG] Получение leaderboard...")
            leaderboard = predictor.leaderboard(val_data, silent=True)
            print(f"🔍 [DEBUG] Leaderboard получен. Размер: {leaderboard.shape}")
            print(f"🔍 [DEBUG] Колонки leaderboard: {leaderboard.columns.tolist()}")
            
            # Получаем лучшую модель - ИСПРАВЛЕНИЕ ОШИБКИ
            print("🔍 [DEBUG] Попытка получить лучшую модель...")
            best_model_name = None
            
            # Способ 1: Пробуем разные методы для получения лучшей модели
            try:
                # В новых версиях AutoGluon
                best_model_name = predictor.get_model_best()
                print(f"🔍 [DEBUG] Лучшая модель (через get_model_best): {best_model_name}")
            except AttributeError:
                print("🔍 [DEBUG] Метод get_model_best не найден, пробуем другие методы...")
                try:
                    # В некоторых версиях
                    best_model_name = predictor.model_best
                    print(f"🔍 [DEBUG] Лучшая модель (через model_best): {best_model_name}")
                except AttributeError:
                    try:
                        # Или из leaderboard
                        if not leaderboard.empty:
                            best_model_name = leaderboard.iloc[0]['model']
                            print(f"🔍 [DEBUG] Лучшая модель (из leaderboard): {best_model_name}")
                    except Exception as e:
                        print(f"🔍 [DEBUG] Ошибка при получении лучшей модели из leaderboard: {e}")
            
            # Если все методы не сработали, используем первое значение из leaderboard
            if best_model_name is None and not leaderboard.empty:
                best_model_name = leaderboard.iloc[0]['model']
                print(f"🔍 [DEBUG] Установлена лучшая модель из leaderboard: {best_model_name}")
            
            print(f"🔍 [DEBUG] Итоговая лучшая модель: {best_model_name}")
            
            # Сохраняем результаты
            result = {
                'model': 'AutoGluon',
                'train_mae': train_mae,
                'val_mae': val_mae,
                'val_mse': val_mse,
                'val_r2': val_r2,
                'leaderboard': leaderboard,
                'predictor': predictor,
                'best_model': best_model_name
            }
            
            self.results['autogluon'] = [result]
            self.best_models['AutoGluon'] = predictor
            
            print(f"🔍 [DEBUG] Результаты сохранены. Ключи results: {self.results.keys()}")
            
            st.success(f"✅ AutoGluon завершил обучение за {time_limit} секунд")
            print("🔍 [DEBUG] Функция autogluon_automation успешно завершена")
            return result
            
        except Exception as e:
            print(f"🔍 [DEBUG] ❌ КРИТИЧЕСКАЯ ОШИБКА в AutoGluon: {str(e)}")
            print(f"🔍 [DEBUG] Тип ошибки: {type(e).__name__}")
            import traceback
            print(f"🔍 [DEBUG] Трассировка ошибки:\n{traceback.format_exc()}")
            st.error(f"❌ Ошибка в AutoGluon: {str(e)}")
            return None
    
    def compare_all_models(self):
        """
        Сравнение всех моделей
        """
        comparison_data = []
        
        # Собираем результаты из всех методов
        for method, results in self.results.items():
            if method == 'linear':
                for model_result in results:
                    comparison_data.append({
                        'Метод': model_result['model'],
                        'Тип': 'Линейная',
                        'CV MAE': f"{model_result['best_score']:.4f}",
                        'Val MAE': f"{model_result['val_mae']:.4f}",
                        'Val R²': f"{model_result['val_r2']:.4f}"
                    })
            
            elif method == 'gradient_boosting':
                for model_result in results:
                    comparison_data.append({
                        'Метод': model_result['model'],
                        'Тип': 'Градиентный бустинг',
                        'CV MAE': f"{model_result['best_cv_score']:.4f}",
                        'Val MAE': f"{model_result['val_mae']:.4f}",
                        'Val R²': f"{model_result['val_r2']:.4f}"
                    })
            
            elif method == 'autogluon' and results:
                for model_result in results:
                    comparison_data.append({
                        'Метод': 'AutoGluon',
                        'Тип': 'Автоматический ML',
                        'CV MAE': 'N/A',
                        'Val MAE': f"{model_result['val_mae']:.4f}",
                        'Val R²': f"{model_result['val_r2']:.4f}"
                    })
        
        return pd.DataFrame(comparison_data)
    
    def test_best_model(self, best_model_name):
        """
        Тестирование лучшей модели на тестовых данных
        """
        if self.X_test is None or self.y_test is None:
            return None
        
        best_model = self.best_models.get(best_model_name)
        
        if best_model is None:
            return None
        
        try:
            # Предсказания на тесте
            if best_model_name == 'AutoGluon':
                test_data = pd.concat([self.X_test, self.y_test], axis=1)
                y_pred_test = best_model.predict(test_data)
            else:
                y_pred_test = best_model.predict(self.X_test)
            
            # Метрики на тесте
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            test_mse = mean_squared_error(self.y_test, y_pred_test)
            test_rmse = np.sqrt(test_mse)
            test_r2 = r2_score(self.y_test, y_pred_test)
            
            return {
                'y_pred': y_pred_test,
                'mae': test_mae,
                'mse': test_mse,
                'rmse': test_rmse,
                'r2': test_r2
            }
            
        except Exception as e:
            st.error(f"❌ Ошибка при тестировании модели {best_model_name}: {str(e)}")
            return None

# ============================================================
# ВИЗУАЛИЗАЦИИ
# ============================================================

def plot_model_comparison(comparison_df):
    """
    Визуализация сравнения моделей
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Сравнение MAE', 'Сравнение R²'],
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # 1. Сравнение MAE
    fig.add_trace(
        go.Bar(
            x=comparison_df['Метод'],
            y=comparison_df['Val MAE'].astype(float),
            name='Val MAE',
            marker_color='lightcoral'
        ),
        row=1, col=1
    )
    
    # 2. Сравнение R²
    fig.add_trace(
        go.Bar(
            x=comparison_df['Метод'],
            y=comparison_df['Val R²'].astype(float),
            name='Val R²',
            marker_color='lightgreen'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400, 
        showlegend=True, 
        title_text="Сравнение моделей"
    )
    return fig

def plot_optuna_history(study):
    """
    Визуализация истории Optuna
    """
    import plotly.graph_objects as go
    
    trials_df = study.trials_dataframe()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trials_df['number'],
        y=trials_df['value'],
        mode='lines+markers',
        name='MAE',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=trials_df['number'],
        y=trials_df['value'].cummin(),
        mode='lines',
        name='Лучшее MAE',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Оптимизация Optuna: история trials',
        xaxis_title='Номер trial',
        yaxis_title='MAE',
        hovermode='x',
        height=400
    )
    
    return fig

def plot_feature_importance(importance_df, top_n=20):
    """
    Визуализация важности признаков
    """
    import plotly.graph_objects as go
    
    # Берем топ-N признаков
    top_features = importance_df.head(top_n)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker_color='teal',
        text=top_features['importance'].round(4),
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f'Топ-{top_n} важных признаков',
        xaxis_title='Важность',
        yaxis_title='Признак',
        height=500
    )
    
    return fig

def plot_autogluon_leaderboard(leaderboard):
    """
    Визуализация лидерборда AutoGluon
    """
    import plotly.graph_objects as go
    
    # Берем топ-10 моделей
    top_models = leaderboard.head(10)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_models['model'],
        y=top_models['score_val'],
        name='Score',
        marker_color='purple',
        text=top_models['score_val'].round(4),
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Топ-10 моделей AutoGluon',
        xaxis_title='Модель',
        yaxis_title='Score (MAE)',
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig

# ============================================================
# ОСНОВНОЙ ИНТЕРФЕЙС
# ============================================================

def show_modeling_interface():
    """
    Основной интерфейс для этапа 3
    """
    print("🔍 [DEBUG] ========== НАЧАЛО ЭТАПА 3 ==========")
    print("🔍 [DEBUG] Проверка session_state:")
    for key in st.session_state.keys():
        print(f"  - {key}: {type(st.session_state[key])}")
    
    
    # Проверяем, выполнены ли предыдущие этапы
    if 'split_data' not in st.session_state:
        st.error("❌ Сначала выполните Этап 2: Валидация и разбиение данных!")
        return
    
    if 'df_features' not in st.session_state or 'feature_info' not in st.session_state:
        st.error("❌ Сначала выполните Этап 1: Инжиниринг признаков!")
        return
    
    # Получаем данные из второго этапа
    split_data = st.session_state.split_data
    df_features = st.session_state.df_features
    feature_info = st.session_state.feature_info
    
    # Определяем целевую переменную
    target_col = feature_info['original_features'][1]  # Второй элемент - target
    date_col = feature_info['original_features'][0]    # Первый элемент - дата
    
    st.info("""
    ### 📋 Методы оптимизации:
    1. **GridSearchCV с TimeSeriesSplit** - для линейных моделей (Ridge, Lasso, ElasticNet)
    2. **Optuna** - для градиентного бустинга (LightGBM)
    3. **AutoGluon** - полная автоматизация + ансамблирование
    
    Используются данные из Этапа 2: train (60%), val (20%), test (20%)
    """)
    
    # Подготовка данных
    st.subheader("📊 Подготовка данных для моделирования")
    
    # Извлекаем данные из split_data (второй этап)
    train_data = split_data['train']
    val_data = split_data['val']
    test_data = split_data['test']
    
    # Подготовка признаков и целевой переменной
    def prepare_features(df, target_col, date_col):
        # Удаляем столбцы с датой и таргетом из признаков
        feature_cols = [col for col in df.columns 
                       if col not in [date_col, target_col]]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Обработка пропусков
        X = X.fillna(X.mean())
        
        return X, y
    
    X_train, y_train = prepare_features(train_data, target_col, date_col)
    X_val, y_val = prepare_features(val_data, target_col, date_col)
    X_test, y_test = prepare_features(test_data, target_col, date_col)
    
    st.success(f"""
    ✅ Данные подготовлены из Этапа 2:
    - Train: {X_train.shape[0]} samples, {X_train.shape[1]} features
    - Val: {X_val.shape[0]} samples, {X_val.shape[1]} features
    - Test: {X_test.shape[0]} samples (зарезервировано для финальной оценки)
    """)
    
    # Настройки
    st.subheader("⚙️ Настройки оптимизации")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        run_linear = st.checkbox("GridSearchCV для линейных моделей", value=True)
    
    with col2:
        run_gb = st.checkbox("Optuna для градиентного бустинга", value=True)
        n_trials = st.slider("Количество trials", 10, 100, 30)
    
    with col3:
        if AUTOGLUON_AVAILABLE:
            run_autogluon = st.checkbox("AutoGluon авто-ML", value=True)
            autogluon_time = st.slider("Время (сек) для AutoGluon", 30, 300, 120)
        else:
            st.warning("AutoGluon не установлен")
            run_autogluon = False
    
    # Кнопка запуска
    if st.button("🚀 Запустить подбор гиперпараметров", type="primary", use_container_width=True):
        
        # Инициализируем оптимизатор с данными из второго этапа
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5, max_train_size=365)
        optimizer = HyperparameterOptimizer(X_train, y_train, X_val, y_val, X_test, y_test, tscv)
        
        results_container = st.container()
        
        with results_container:
            st.subheader("📈 Результаты оптимизации")
            
            # 1. Линейные модели
            if run_linear:
                st.markdown("---")
                st.subheader("1. GridSearchCV для линейных моделей")
                
                linear_results = optimizer.linear_models_grid_search()
                
                if linear_results:
                    # Таблица результатов
                    linear_df = pd.DataFrame([
                        {
                            'Модель': r['model'],
                            'Лучшие параметры': str(r['best_params'])[:50] + '...',
                            'CV MAE': f"{r['best_score']:.4f}",
                            'Val MAE': f"{r['val_mae']:.4f}",
                            'Val R²': f"{r['val_r2']:.4f}"
                        }
                        for r in linear_results
                    ])
                    
                    st.dataframe(linear_df, width='stretch')
                    
                    st.success(f"✅ Линейные модели оптимизированы!")
            
            # 2. Градиентный бустинг
            if run_gb:
                st.markdown("---")
                st.subheader("2. Optuna для градиентного бустинга")
                
                gb_result = optimizer.gradient_boosting_optuna(n_trials=n_trials)
                
                if gb_result:
                    # Отображаем лучшие параметры
                    st.info(f"**Лучшие параметры:**")
                    st.json(gb_result['best_params'])
                    
                    st.info(f"**Метрики:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Train MAE", f"{gb_result['train_mae']:.4f}")
                    with col2:
                        st.metric("Val MAE", f"{gb_result['val_mae']:.4f}")
                    with col3:
                        st.metric("Val MSE", f"{gb_result['val_mse']:.4f}")
                    with col4:
                        st.metric("Val R²", f"{gb_result['val_r2']:.4f}")
                    
                    # Визуализация Optuna
                    fig_optuna = plot_optuna_history(gb_result['study'])
                    st.plotly_chart(fig_optuna, use_container_width=True)
                    
                    # Важность признаков
                    st.subheader("Важность признаков (LightGBM)")
                    fig_importance = plot_feature_importance(gb_result['feature_importance'])
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    st.dataframe(gb_result['feature_importance'].head(20), width='stretch')
            
            # 3. AutoGluon
            if run_autogluon and AUTOGLUON_AVAILABLE:
                st.markdown("---")
                st.subheader("3. AutoGluon автоматический ML")
                
                # Используем presets из задания
                presets = ["medium_quality", "high_quality", "best_quality"]
                
                autogluon_result = optimizer.autogluon_automation(
                    time_limit=autogluon_time,
                    presets=presets
                )
                
                if autogluon_result:
                    # Leaderboard
                    st.subheader("AutoGluon Leaderboard")
                    
                    # Визуализация лидерборда
                    fig_leaderboard = plot_autogluon_leaderboard(autogluon_result['leaderboard'])
                    st.plotly_chart(fig_leaderboard, use_container_width=True)
                    
                    # Таблица лидерборда
                    st.dataframe(autogluon_result['leaderboard'], width='stretch')
                    
                    # Метрики
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Train MAE", f"{autogluon_result['train_mae']:.4f}")
                    with col2:
                        st.metric("Val MAE", f"{autogluon_result['val_mae']:.4f}")
                    with col3:
                        st.metric("Val MSE", f"{autogluon_result['val_mse']:.4f}")
                    with col4:
                        st.metric("Val R²", f"{autogluon_result['val_r2']:.4f}")
                    
                    # Информация о лучшей модели
                    best_model_name = autogluon_result['best_model']
                    st.info(f"""
                    **Лучшая модель AutoGluon:** {best_model_name}
                    **Val MAE:** {autogluon_result['val_mae']:.4f}
                    **Val R²:** {autogluon_result['val_r2']:.4f}
                    """)
            
            # Сравнение всех моделей
            st.markdown("---")
            st.subheader("🏆 Сравнение всех моделей")
            
            comparison_df = optimizer.compare_all_models()
            
            if not comparison_df.empty:
                # Таблица сравнения
                st.dataframe(comparison_df, width='stretch')
                
                # Визуализация
                fig_comparison = plot_model_comparison(comparison_df)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Определяем лучшую модель
                comparison_df['Val MAE_num'] = comparison_df['Val MAE'].astype(float)
                best_model_row = comparison_df.loc[comparison_df['Val MAE_num'].idxmin()]
                st.success(f"""
                ### 🏆 Лучшая модель: **{best_model_row['Метод']}**
                - **Val MAE:** {best_model_row['Val MAE']}
                - **Val R²:** {best_model_row['Val R²']}
                - **Тип:** {best_model_row['Тип']}
                """)
                
                # Сохраняем результаты в session state
                st.session_state.modeling_results = {
                    'optimizer': optimizer,
                    'comparison_df': comparison_df,
                    'best_model': best_model_row.to_dict(),
                    'X_test': X_test,
                    'y_test': y_test
                }
                
                # Кнопка для финальной оценки на тесте
                st.markdown("---")
                st.subheader("📊 Финальная оценка на тестовой выборке")
                
                if st.button("✅ Выполнить финальную оценку на тесте", type="primary"):
                    with st.spinner("Оценка на тестовой выборке..."):
                        # Получаем лучшую модель
                        best_model_name = best_model_row['Метод']
                        
                        # Тестируем лучшую модель на тесте
                        test_results = optimizer.test_best_model(best_model_name)
                        
                        if test_results:
                            # Визуализация
                            fig_test = go.Figure()
                            fig_test.add_trace(go.Scatter(
                                x=np.arange(len(y_test)),
                                y=y_test.values,
                                mode='lines',
                                name='Фактические значения',
                                line=dict(color='blue', width=2)
                            ))
                            fig_test.add_trace(go.Scatter(
                                x=np.arange(len(y_test)),
                                y=test_results['y_pred'],
                                mode='lines',
                                name='Предсказания',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                            
                            fig_test.update_layout(
                                title=f'Предсказания на тестовой выборке: {best_model_name}',
                                xaxis_title='Индекс',
                                yaxis_title=target_col,
                                height=500,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig_test, use_container_width=True)
                            
                            # Метрики
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Test MAE", f"{test_results['mae']:.4f}")
                            with col2:
                                st.metric("Test RMSE", f"{test_results['rmse']:.4f}")
                            with col3:
                                st.metric("Test MSE", f"{test_results['mse']:.4f}")
                            with col4:
                                st.metric("Test R²", f"{test_results['r2']:.4f}")
                            
                            # Сохраняем финальные результаты
                            st.session_state.final_results = {
                                'best_model': best_model_name,
                                'test_metrics': test_results,
                                'y_test': y_test.values,
                                'y_pred': test_results['y_pred']
                            }
                            
                            st.success("🎉 Финальная оценка завершена!")
                            st.balloons()
                        else:
                            st.error("Не удалось выполнить тестирование лучшей модели!")
            else:
                st.warning("Нет результатов для сравнения")
    
    # Если уже есть результаты, показываем их
    elif 'modeling_results' in st.session_state:
        st.success("✅ Оптимизация уже выполнена!")
        
        results = st.session_state.modeling_results
        comparison_df = results['comparison_df']
        
        # Показываем сравнение
        st.subheader("🏆 Сравнение моделей")
        st.dataframe(comparison_df, width='stretch')
        
        best_model = results['best_model']
        st.success(f"""
        ### 🏆 Лучшая модель: **{best_model['Метод']}**
        - **Val MAE:** {best_model['Val MAE']}
        - **Val R²:** {best_model['Val R²']}
        - **Тип:** {best_model['Тип']}
        """)
        
        # Если есть финальные результаты, показываем их
        if 'final_results' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Финальные результаты на тесте")
            
            final_results = st.session_state.final_results
            test_metrics = final_results['test_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test MAE", f"{test_metrics['mae']:.4f}")
            with col2:
                st.metric("Test RMSE", f"{test_metrics['rmse']:.4f}")
            with col3:
                st.metric("Test MSE", f"{test_metrics['mse']:.4f}")
            with col4:
                st.metric("Test R²", f"{test_metrics['r2']:.4f}")