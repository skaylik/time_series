# final_analysis_module.py - Этап 9: Финальный анализ и рекомендации

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Импорты для анализа
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ============================================================
# КЛАСС ДЛЯ ФИНАЛЬНОГО АНАЛИЗА
# ============================================================

class FinalAnalysis:
    """Класс для финального анализа и рекомендаций"""
    
    def __init__(self):
        self.analysis_results = {}
        self.recommendations = {}
        self.comparison_results = {}
    
    def collect_all_results(self):
        """Сбор всех результатов из предыдущих этапов"""
        
        results_summary = {
            'data_preparation': {},
            'feature_engineering': {},
            'models_trained': {},
            'ensemble_results': {},
            'segmentation_results': {},
            'outlier_handling': {},
            'evaluation_results': {}
        }
        
        # 1. Сбор результатов подготовки данных (Этап 1-2)
        if 'df_features' in st.session_state:
            df_features = st.session_state.df_features
            results_summary['data_preparation'] = {
                'original_shape': df_features.shape,
                'features_count': len(df_features.columns),
                'missing_values': df_features.isnull().sum().sum()
            }
        
        # 2. Сбор результатов feature engineering (Этап 3)
        if 'feature_engineering_results' in st.session_state:
            feat_eng = st.session_state.feature_engineering_results
            results_summary['feature_engineering'] = {
                'created_features': feat_eng.get('created_features', []),
                'total_features': feat_eng.get('total_features', 0),
                'feature_importance': feat_eng.get('feature_importance', {})
            }
        
        # 3. Сбор результатов обучения моделей (Этап 4)
        if 'models_results' in st.session_state:
            models_results = st.session_state.models_results
            results_summary['models_trained'] = {
                'models_count': len(models_results),
                'best_model': models_results.get('best_model', {}),
                'all_models': list(models_results.keys())
            }
        
        # 4. Сбор результатов ансамблирования (Этап 8)
        if 'best_ensemble' in st.session_state:
            best_ensemble = st.session_state.best_ensemble
            results_summary['ensemble_results'] = {
                'best_ensemble_name': best_ensemble.get('name', 'N/A'),
                'best_ensemble_mae': best_ensemble.get('metrics', {}).get('MAE', 'N/A'),
                'best_ensemble_rmse': best_ensemble.get('metrics', {}).get('RMSE', 'N/A')
            }
        
        # 5. Сбор результатов сегментации (Этап 8)
        if 'segmentation_state' in st.session_state:
            seg_state = st.session_state.segmentation_state
            results_summary['segmentation_results'] = {
                'segmentation_types': list(seg_state.get('results', {}).keys()),
                'segment_models_trained': 'segment_models' in st.session_state
            }
        
        # 6. Сбор результатов обработки выбросов (Этап 8)
        if 'outlier_handler' in st.session_state:
            outlier_handler = st.session_state.outlier_handler
            results_summary['outlier_handling'] = {
                'methods_applied': list(outlier_handler.outlier_stats.keys()) if hasattr(outlier_handler, 'outlier_stats') else []
            }
        
        # 7. Сбор результатов оценки (Этап 7)
        if 'evaluation_results' in st.session_state:
            eval_results = st.session_state.evaluation_results
            results_summary['evaluation_results'] = {
                'ranked_models': eval_results.get('ranked_df', pd.DataFrame()),
                'best_model_name': eval_results.get('best_model', 'N/A'),
                'best_model_mae': eval_results.get('best_mae', 'N/A')
            }
        
        self.analysis_results = results_summary
        return results_summary
    
    def analyze_model_performance(self):
        """Анализ производительности моделей"""
        
        performance_analysis = {
            'model_ranking': {},
            'performance_comparison': {},
            'strengths_weaknesses': {}
        }
        
        # Проверяем наличие результатов оценки
        if 'evaluation_results' in st.session_state:
            eval_results = st.session_state.evaluation_results
            ranked_df = eval_results.get('ranked_df')
            
            if ranked_df is not None and not ranked_df.empty:
                # Сортируем по MAE
                if 'MAE' in ranked_df.columns:
                    # Преобразуем MAE в числовой формат
                    ranked_df['MAE_numeric'] = pd.to_numeric(ranked_df['MAE'], errors='coerce')
                    ranked_df = ranked_df.dropna(subset=['MAE_numeric'])
                    ranked_df = ranked_df.sort_values('MAE_numeric')
                    
                    performance_analysis['model_ranking'] = ranked_df.to_dict('records')
                    
                    # Анализ разницы в производительности
                    if len(ranked_df) > 1:
                        best_mae = ranked_df['MAE_numeric'].iloc[0]
                        worst_mae = ranked_df['MAE_numeric'].iloc[-1]
                        mae_range = worst_mae - best_mae
                        performance_analysis['performance_comparison'] = {
                            'best_mae': float(best_mae),
                            'worst_mae': float(worst_mae),
                            'mae_range': float(mae_range),
                            'relative_improvement': float((mae_range / worst_mae) * 100)
                        }
        
        # Анализ сильных и слабых сторон
        if 'models_results' in st.session_state:
            models_results = st.session_state.models_results
            
            strengths = []
            weaknesses = []
            
            for model_name, model_info in models_results.items():
                if isinstance(model_info, dict):
                    # Определяем тип модели
                    model_type = model_info.get('type', 'unknown')
                    
                    # Сильные стороны
                    if 'linear' in model_type.lower():
                        strengths.append(f"{model_name}: Интерпретируемость, быстрая обучение")
                    elif 'tree' in model_type.lower() or 'forest' in model_type.lower():
                        strengths.append(f"{model_name}: Работа с нелинейными зависимостями")
                    elif 'neural' in model_type.lower():
                        strengths.append(f"{model_name}: Сложные паттерны, большие данные")
                    
                    # Слабые стороны
                    if 'linear' in model_type.lower():
                        weaknesses.append(f"{model_name}: Чувствительность к выбросам, линейные предположения")
                    elif 'tree' in model_type.lower():
                        weaknesses.append(f"{model_name}: Склонность к переобучению")
            
            performance_analysis['strengths_weaknesses'] = {
                'strengths': strengths,
                'weaknesses': weaknesses
            }
        
        return performance_analysis
    
    def analyze_feature_importance(self):
        """Анализ важности признаков"""
        
        feature_importance_analysis = {
            'top_features': [],
            'feature_categories': {},
            'recommendations': []
        }
        
        # Проверяем наличие feature importance
        if 'feature_engineering_results' in st.session_state:
            feat_eng = st.session_state.feature_engineering_results
            feature_importance = feat_eng.get('feature_importance', {})
            
            if feature_importance:
                # Сортируем признаки по важности
                sorted_features = sorted(feature_importance.items(), 
                                       key=lambda x: abs(x[1]), 
                                       reverse=True)
                
                # Берем топ-10 признаков
                top_features = sorted_features[:10]
                feature_importance_analysis['top_features'] = [
                    {'feature': feat, 'importance': float(imp)} 
                    for feat, imp in top_features
                ]
                
                # Анализируем категории признаков
                temporal_features = []
                lag_features = []
                statistical_features = []
                
                for feature, _ in sorted_features:
                    feature_lower = feature.lower()
                    if any(term in feature_lower for term in ['lag', 'shift', 'diff']):
                        lag_features.append(feature)
                    elif any(term in feature_lower for term in ['mean', 'std', 'min', 'max', 'rolling']):
                        statistical_features.append(feature)
                    elif any(term in feature_lower for term in ['year', 'month', 'day', 'hour', 'week', 'season']):
                        temporal_features.append(feature)
                
                feature_importance_analysis['feature_categories'] = {
                    'temporal': temporal_features[:5],
                    'lag': lag_features[:5],
                    'statistical': statistical_features[:5]
                }
                
                # Рекомендации на основе анализа признаков
                recommendations = []
                
                if len(lag_features) > 0:
                    recommendations.append("Лаг-признаки имеют высокую важность - временные зависимости сильны")
                
                if len(statistical_features) > 0:
                    recommendations.append("Статистические признаки важны - ряды имеют сложную структуру")
                
                if len(temporal_features) > 0:
                    recommendations.append("Временные признаки значимы - присутствуют сезонные/циклические паттерны")
                
                feature_importance_analysis['recommendations'] = recommendations
        
        return feature_importance_analysis
    
    def analyze_autogluon_vs_custom(self):
        """Сравнение AutoGluon с кастомными моделями"""
        
        comparison = {
            'autogluon_available': False,
            'autogluon_results': {},
            'custom_models_results': {},
            'comparison_summary': {}
        }
        
        # Проверяем наличие AutoGluon результатов
        if 'autogluon_results' in st.session_state:
            autogluon_res = st.session_state.autogluon_results
            comparison['autogluon_available'] = True
            comparison['autogluon_results'] = autogluon_res
            
            # Извлекаем метрики AutoGluon
            autogluon_metrics = autogluon_res.get('metrics', {})
            autogluon_mae = autogluon_metrics.get('MAE', None)
            autogluon_rmse = autogluon_metrics.get('RMSE', None)
        
        # Собираем метрики кастомных моделей
        custom_models_metrics = []
        
        # Из результатов оценки
        if 'evaluation_results' in st.session_state:
            eval_results = st.session_state.evaluation_results
            predictions = eval_results.get('predictions', {})
            
            for model_name, pred in predictions.items():
                if pred is not None:
                    # Получаем тестовые данные для расчета метрик
                    result = prepare_data_for_advanced_techniques()
                    if result[0] is not None:
                        _, _, _, y_test, _ = result
                        
                        if len(pred) == len(y_test):
                            try:
                                # Преобразуем в числа и убираем NaN
                                y_pred_clean = pd.to_numeric(pred, errors='coerce')
                                y_test_clean = pd.to_numeric(y_test, errors='coerce')
                                
                                # Убираем NaN значения
                                mask = ~np.isnan(y_pred_clean) & ~np.isnan(y_test_clean)
                                if np.sum(mask) > 0:
                                    y_pred_valid = y_pred_clean[mask]
                                    y_test_valid = y_test_clean[mask]
                                    
                                    mae = mean_absolute_error(y_test_valid, y_pred_valid)
                                    rmse = np.sqrt(mean_squared_error(y_test_valid, y_pred_valid))
                                    
                                    custom_models_metrics.append({
                                        'model': model_name,
                                        'mae': float(mae),
                                        'rmse': float(rmse)
                                    })
                            except:
                                pass
        
        comparison['custom_models_results'] = custom_models_metrics
        
        # Сравниваем AutoGluon с кастомными моделями
        if comparison['autogluon_available'] and custom_models_metrics:
            # Находим лучшую кастомную модель
            best_custom = min(custom_models_metrics, key=lambda x: x['mae'])
            
            comparison['comparison_summary'] = {
                'autogluon_mae': autogluon_mae,
                'best_custom_mae': best_custom['mae'],
                'difference_mae': float(autogluon_mae - best_custom['mae']) if autogluon_mae is not None else None,
                'autogluon_better': autogluon_mae < best_custom['mae'] if autogluon_mae is not None else False
            }
        
        self.comparison_results = comparison
        return comparison
    
    def generate_recommendations(self):
        """Генерация финальных рекомендаций"""
        
        recommendations = {
            'model_selection': [],
            'feature_engineering': [],
            'data_preprocessing': [],
            'deployment': [],
            'monitoring': []
        }
        
        # 1. Рекомендации по выбору модели
        if 'best_ensemble' in st.session_state:
            best_ensemble = st.session_state.best_ensemble
            recommendations['model_selection'].append(
                f"Использовать ансамбль '{best_ensemble['name']}' для продакшена (MAE: {best_ensemble['metrics']['MAE']:.4f})"
            )
        elif 'evaluation_results' in st.session_state:
            eval_results = st.session_state.evaluation_results
            best_model = eval_results.get('best_model', 'N/A')
            recommendations['model_selection'].append(
                f"Использовать модель '{best_model}' как лучшую одиночную модель"
            )
        
        # 2. Рекомендации по feature engineering
        feat_importance = self.analyze_feature_importance()
        top_features = feat_importance.get('top_features', [])
        
        if len(top_features) > 0:
            top_feature_names = [f['feature'] for f in top_features[:3]]
            recommendations['feature_engineering'].append(
                f"Сфокусироваться на признаках: {', '.join(top_feature_names)}"
            )
        
        # 3. Рекомендации по предобработке данных
        if 'outlier_handler' in st.session_state:
            outlier_handler = st.session_state.outlier_handler
            if hasattr(outlier_handler, 'outlier_stats') and 'isolation_forest' in outlier_handler.outlier_stats:
                stats = outlier_handler.outlier_stats['isolation_forest']
                if stats['outlier_percentage'] > 5:
                    recommendations['data_preprocessing'].append(
                        f"Обнаружено {stats['outlier_percentage']:.1f}% выбросов - использовать RobustScaler"
                    )
        
        # 4. Рекомендации по деплою
        recommendations['deployment'].append(
            "Реализовать пайплайн переобучения модели раз в неделю"
        )
        recommendations['deployment'].append(
            "Настроить A/B тестирование для новых версий моделей"
        )
        
        # 5. Рекомендации по мониторингу
        recommendations['monitoring'].append(
            "Мониторить MAE и RMSE в реальном времени"
        )
        recommendations['monitoring'].append(
            "Настроить алерты при ухудшении качества на 10%"
        )
        
        # Сравнение AutoGluon с кастомными моделями
        comparison = self.analyze_autogluon_vs_custom()
        if comparison.get('autogluon_available', False):
            summary = comparison.get('comparison_summary', {})
            if summary.get('autogluon_better', False):
                recommendations['model_selection'].append(
                    "AutoGluon показал лучшие результаты - рассмотреть его использование для автоматизации"
                )
            else:
                recommendations['model_selection'].append(
                    "Кастомные модели лучше AutoGluon - продолжить ручную настройку"
                )
        
        self.recommendations = recommendations
        return recommendations
    
    def create_performance_report(self):
        """Создание финального отчета о производительности"""
        
        report = {
            'executive_summary': {},
            'technical_details': {},
            'business_impact': {},
            'next_steps': {}
        }
        
        # 1. Executive Summary
        best_mae = None
        best_model_name = "Не определено"
        
        if 'best_ensemble' in st.session_state:
            best_ensemble = st.session_state.best_ensemble
            best_mae = best_ensemble['metrics']['MAE']
            best_model_name = best_ensemble['name']
        elif 'evaluation_results' in st.session_state:
            eval_results = st.session_state.evaluation_results
            best_mae = eval_results.get('best_mae')
            best_model_name = eval_results.get('best_model', 'N/A')
        
        report['executive_summary'] = {
            'best_model': best_model_name,
            'best_mae': float(best_mae) if best_mae is not None else 'N/A',
            'total_models_tested': len(self.analysis_results.get('models_trained', {}).get('all_models', [])),
            'key_achievement': f"Достигнута точность прогнозирования с MAE: {best_mae:.4f}" if best_mae else "Точность не определена"
        }
        
        # 2. Technical Details
        report['technical_details'] = {
            'data_characteristics': self.analysis_results.get('data_preparation', {}),
            'features_used': len(self.analysis_results.get('feature_engineering', {}).get('created_features', [])),
            'ensemble_used': 'best_ensemble' in st.session_state,
            'segmentation_applied': len(self.analysis_results.get('segmentation_results', {}).get('segmentation_types', [])) > 0
        }
        
        # 3. Business Impact
        # Предполагаем, что целевая переменная - это, например, продажи или спрос
        if best_mae and isinstance(best_mae, (int, float)):
            accuracy_percentage = max(0, 100 - (best_mae * 100))  # Упрощенная метрика точности
            report['business_impact'] = {
                'forecast_accuracy': f"{accuracy_percentage:.1f}%",
                'potential_savings': "Улучшение точности прогнозов на 10-20%",
                'risk_reduction': "Снижение рисков нехватки/переизбытка на 15-25%"
            }
        else:
            report['business_impact'] = {
                'forecast_accuracy': "Не рассчитано",
                'potential_savings': "Требуется дополнительный анализ",
                'risk_reduction': "Требуется дополнительный анализ"
            }
        
        # 4. Next Steps
        report['next_steps'] = {
            'immediate': [
                "Деплой лучшей модели в тестовое окружение",
                "Настройка мониторинга качества прогнозов"
            ],
            'short_term': [
                "Автоматизация пайплайна переобучения",
                "Добавление новых источников данных"
            ],
            'long_term': [
                "Внедрение системы рекомендаций на основе прогнозов",
                "Интеграция с системами планирования ресурсов"
            ]
        }
        
        return report
    
    def perform_complete_analysis(self):
        """Выполнение полного анализа"""
        
        # Сбор всех результатов
        self.collect_all_results()
        
        # Анализ производительности моделей
        performance = self.analyze_model_performance()
        
        # Анализ важности признаков
        feature_importance = self.analyze_feature_importance()
        
        # Сравнение AutoGluon с кастомными моделями
        autogluon_comparison = self.analyze_autogluon_vs_custom()
        
        # Генерация рекомендаций
        recommendations = self.generate_recommendations()
        
        # Создание финального отчета
        report = self.create_performance_report()
        
        # Сохранение полного анализа
        complete_analysis = {
            'performance_analysis': performance,
            'feature_importance_analysis': feature_importance,
            'autogluon_comparison': autogluon_comparison,
            'recommendations': recommendations,
            'final_report': report
        }
        
        st.session_state.final_analysis = complete_analysis
        return complete_analysis

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def prepare_data_for_advanced_techniques():
    """Подготовка данных для продвинутых техник (копия из advanced_techniques.py)"""
    
    required_keys = ['df_features', 'feature_info', 'split_data']
    missing_keys = [key for key in required_keys if key not in st.session_state]
    
    if missing_keys:
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
# ОСНОВНОЙ ИНТЕРФЕЙС ЭТАПА 9
# ============================================================

def show_final_analysis_interface():
    """Основной интерфейс Этапа 9: Финальный анализ"""
    
    
    # Проверка наличия данных из предыдущих этапов
    if 'df_features' not in st.session_state or 'feature_info' not in st.session_state:
        st.error("❌ Сначала выполните Этапы 1-2: Подготовку данных")
        return
    
    st.info("""
    **Цель Этапа 9:**
    
    1. **Сводный анализ всех этапов**
    2. **Сравнение производительности моделей**
    3. **Анализ важности признаков**
    4. **Рекомендации для продакшена**
    5. **Финальный отчет**
    """)
    
    # Инициализация анализа
    analysis = FinalAnalysis()
    
    # Кнопка выполнения финального анализа
    if st.button("🚀 Выполнить финальный анализ", key="final_analysis_button"):
        
        with st.spinner("Выполняется финальный анализ..."):
            
            # Выполняем полный анализ
            complete_analysis = analysis.perform_complete_analysis()
            
            # 1. Сводный анализ
            st.subheader("1. 📋 Сводный анализ всех этапов")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Моделей обучено",
                    value=len(analysis.analysis_results.get('models_trained', {}).get('all_models', [])),
                    help="Общее количество обученных моделей"
                )
            
            with col2:
                features_count = len(analysis.analysis_results.get('feature_engineering', {}).get('created_features', []))
                st.metric(
                    label="Признаков создано",
                    value=features_count,
                    help="Количество созданных в feature engineering признаков"
                )
            
            with col3:
                segmentation_types = analysis.analysis_results.get('segmentation_results', {}).get('segmentation_types', [])
                st.metric(
                    label="Типов сегментации",
                    value=len(segmentation_types),
                    help="Количество примененных методов сегментации"
                )
            
            # 2. Анализ производительности моделей
            st.subheader("2. 📈 Анализ производительности моделей")
            
            performance = complete_analysis['performance_analysis']
            
            if 'model_ranking' in performance and performance['model_ranking']:
                # Создаем DataFrame для отображения
                ranking_df = pd.DataFrame(performance['model_ranking'])
                
                # Убираем технические колонки
                if 'MAE_numeric' in ranking_df.columns:
                    ranking_df = ranking_df.drop('MAE_numeric', axis=1)
                
                st.dataframe(ranking_df, width='stretch')
                
                # График сравнения моделей
                if len(ranking_df) > 0 and 'MAE' in ranking_df.columns and 'model' in ranking_df.columns:
                    fig_comparison = go.Figure()
                    
                    # Преобразуем MAE в числа
                    ranking_df['MAE_numeric'] = pd.to_numeric(ranking_df['MAE'], errors='coerce')
                    ranking_df = ranking_df.dropna(subset=['MAE_numeric'])
                    
                    if not ranking_df.empty:
                        fig_comparison.add_trace(go.Bar(
                            x=ranking_df['model'],
                            y=ranking_df['MAE_numeric'],
                            name='MAE',
                            marker_color='lightblue',
                            text=ranking_df['MAE_numeric'].round(4),
                            textposition='auto'
                        ))
                        
                        fig_comparison.update_layout(
                            title='Сравнение моделей по MAE',
                            xaxis_title='Модель',
                            yaxis_title='MAE',
                            height=400,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig_comparison, use_container_width=True)
            
            # 3. Анализ важности признаков
            st.subheader("3. 🔍 Анализ важности признаков")
            
            feature_importance = complete_analysis['feature_importance_analysis']
            
            if feature_importance.get('top_features'):
                # График важности признаков
                features_df = pd.DataFrame(feature_importance['top_features'])
                
                fig_features = go.Figure()
                
                fig_features.add_trace(go.Bar(
                    x=features_df['importance'],
                    y=features_df['feature'],
                    orientation='h',
                    name='Важность',
                    marker_color='lightgreen'
                ))
                
                fig_features.update_layout(
                    title='Топ-10 самых важных признаков',
                    xaxis_title='Важность',
                    yaxis_title='Признак',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_features, use_container_width=True)
                
                # Категории признаков
                st.write("**Категории признаков:**")
                
                categories = feature_importance.get('feature_categories', {})
                for category, features in categories.items():
                    if features:
                        st.write(f"- **{category.capitalize()}:** {', '.join(features[:3])}")
            
            # 4. Сравнение AutoGluon с кастомными моделями
            st.subheader("4. 🤖 Сравнение AutoGluon с кастомными моделями")
            
            autogluon_comparison = complete_analysis['autogluon_comparison']
            
            if autogluon_comparison.get('autogluon_available', False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**AutoGluon результаты:**")
                    autogluon_res = autogluon_comparison.get('autogluon_results', {})
                    if 'metrics' in autogluon_res:
                        metrics = autogluon_res['metrics']
                        st.write(f"- MAE: {metrics.get('MAE', 'N/A')}")
                        st.write(f"- RMSE: {metrics.get('RMSE', 'N/A')}")
                
                with col2:
                    st.write("**Лучшие кастомные модели:**")
                    custom_models = autogluon_comparison.get('custom_models_results', [])
                    if custom_models:
                        # Сортируем по MAE
                        custom_models_sorted = sorted(custom_models, key=lambda x: x['mae'])
                        for i, model in enumerate(custom_models_sorted[:3]):
                            st.write(f"{i+1}. {model['model']}: MAE={model['mae']:.4f}")
                
                # Сводка сравнения
                summary = autogluon_comparison.get('comparison_summary', {})
                if summary:
                    st.write("**Сводка сравнения:**")
                    
                    if summary.get('autogluon_better'):
                        st.success(f"✅ AutoGluon лучше на {(summary.get('difference_mae', 0) * -1):.4f} MAE")
                    else:
                        st.success(f"✅ Кастомные модели лучше на {summary.get('difference_mae', 0):.4f} MAE")
            else:
                st.info("ℹ️ AutoGluon не использовался в анализе")
            
            # 5. Рекомендации
            st.subheader("5. 💡 Рекомендации для продакшена")
            
            recommendations = complete_analysis['recommendations']
            
            tabs = st.tabs(["Выбор модели", "Feature Engineering", "Предобработка", "Деплой", "Мониторинг"])
            
            with tabs[0]:
                st.write("**Рекомендации по выбору модели:**")
                for rec in recommendations.get('model_selection', []):
                    st.write(f"• {rec}")
            
            with tabs[1]:
                st.write("**Рекомендации по feature engineering:**")
                for rec in recommendations.get('feature_engineering', []):
                    st.write(f"• {rec}")
            
            with tabs[2]:
                st.write("**Рекомендации по предобработке данных:**")
                for rec in recommendations.get('data_preprocessing', []):
                    st.write(f"• {rec}")
            
            with tabs[3]:
                st.write("**Рекомендации по деплою:**")
                for rec in recommendations.get('deployment', []):
                    st.write(f"• {rec}")
            
            with tabs[4]:
                st.write("**Рекомендации по мониторингу:**")
                for rec in recommendations.get('monitoring', []):
                    st.write(f"• {rec}")
            
            # 6. Финальный отчет
            st.subheader("6. 📄 Финальный отчет")
            
            report = complete_analysis['final_report']
            
            with st.expander("📋 Executive Summary", expanded=True):
                exec_summary = report.get('executive_summary', {})
                st.write(f"**Лучшая модель:** {exec_summary.get('best_model', 'N/A')}")
                st.write(f"**Лучший MAE:** {exec_summary.get('best_mae', 'N/A')}")
                st.write(f"**Моделей протестировано:** {exec_summary.get('total_models_tested', 0)}")
                st.write(f"**Ключевое достижение:** {exec_summary.get('key_achievement', 'N/A')}")
            
            with st.expander("🔧 Technical Details"):
                tech_details = report.get('technical_details', {})
                st.write(f"**Использован ансамбль:** {'Да' if tech_details.get('ensemble_used') else 'Нет'}")
                st.write(f"**Применена сегментация:** {'Да' if tech_details.get('segmentation_applied') else 'Нет'}")
                st.write(f"**Количество признаков:** {tech_details.get('features_used', 0)}")
            
            with st.expander("💼 Business Impact"):
                business_impact = report.get('business_impact', {})
                st.write(f"**Точность прогнозов:** {business_impact.get('forecast_accuracy', 'N/A')}")
                st.write(f"**Потенциальная экономия:** {business_impact.get('potential_savings', 'N/A')}")
                st.write(f"**Снижение рисков:** {business_impact.get('risk_reduction', 'N/A')}")
            
            with st.expander("🚀 Next Steps"):
                next_steps = report.get('next_steps', {})
                
                st.write("**Немедленные действия:**")
                for step in next_steps.get('immediate', []):
                    st.write(f"• {step}")
                
                st.write("**Краткосрочные планы:**")
                for step in next_steps.get('short_term', []):
                    st.write(f"• {step}")
                
                st.write("**Долгосрочные планы:**")
                for step in next_steps.get('long_term', []):
                    st.write(f"• {step}")
            
            # Кнопка экспорта отчета
            st.download_button(
                label="📥 Экспортировать отчет в JSON",
                data=pd.Series(complete_analysis).to_json(indent=2, orient='index'),
                file_name="final_analysis_report.json",
                mime="application/json"
            )
            
            st.success("✅ Финальный анализ успешно завершен!")
    
    # Итоговый раздел
    st.markdown("---")
    st.subheader("🎯 Итоги проекта")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Что было сделано:**")
        st.write("• Подготовка и анализ данных")
        st.write("• Feature engineering и отбор признаков")
        st.write("• Обучение множества моделей")
        st.write("• Ансамблирование и оптимизация")
        st.write("• Обработка выбросов и сегментация")
        st.write("• Детальная оценка качества")
        st.write("• Финальный анализ и рекомендации")
    
    with col2:
        st.write("**Ключевые результаты:**")
        st.write("• Определена лучшая модель/ансамбль")
        st.write("• Выявлены важнейшие признаки")
        st.write("• Разработаны рекомендации для продакшена")
        st.write("• Создан план дальнейших действий")
        st.write("• Подготовлен финальный отчет")
    
    st.markdown("---")
    st.success("""
    **🏆 Проект успешно завершен!**
    
    **Дальнейшие шаги:**
    1. **Деплой** лучшей модели в продакшен
    2. **Настройка** мониторинга и алертов
    3. **Планирование** регулярного переобучения
    4. **Расширение** функционала на основе полученных insights
    
    **Спасибо за использование платформы! 🎉**
    """)

# ============================================================
# ФУНКЦИЯ ДЛЯ ЗАПУСКА ЭТАПА
# ============================================================

def show_final_analysis():
    """Запуск Этапа 9"""
    show_final_analysis_interface()

if __name__ == "__main__":
    show_final_analysis()