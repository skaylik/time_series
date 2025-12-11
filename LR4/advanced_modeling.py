# advanced_modeling.py - Этап 5

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ИНТЕГРАЦИИ
# ============================================================

def extract_stage3_results():
    """Извлечение результатов из Этапа 3"""
    
    if 'modeling_results' not in st.session_state:
        st.warning("⚠️ Сначала выполните Этап 3: Подбор гиперпараметров")
        return None
    
    modeling_results = st.session_state.modeling_results
    
    try:
        # Получаем оптимизатор и сравнение моделей
        optimizer = modeling_results.get('optimizer')
        comparison_df = modeling_results.get('comparison_df')
        
        if comparison_df is None or comparison_df.empty:
            return None
        
        # Подготавливаем данные для сравнения
        stage3_data = {
            'comparison_df': comparison_df,
            'optimizer': optimizer,
            'best_model': modeling_results.get('best_model', {}),
            'X_test': modeling_results.get('X_test'),
            'y_test': modeling_results.get('y_test')
        }
        
        return stage3_data
        
    except Exception as e:
        st.error(f"Ошибка при извлечении результатов Этапа 3: {str(e)}")
        return None

def extract_stage4_results():
    """Извлечение результатов из Этапа 4"""
    
    if 'forecast_results' not in st.session_state:
        st.warning("⚠️ Сначала выполните Этап 4: Стратегии прогнозирования")
        return None
    
    forecast_results = st.session_state.forecast_results
    
    try:
        # Извлекаем стратегии и метрики
        comparison_df_display = forecast_results.get('comparison_df_display')
        comparison_df = forecast_results.get('comparison_df')
        strategy_results = forecast_results.get('strategy_results', {})
        
        stage4_data = {
            'comparison_df_display': comparison_df_display,
            'comparison_df': comparison_df,
            'strategy_results': strategy_results,
            'y_test': forecast_results.get('y_test'),
            'horizon': forecast_results.get('horizon')
        }
        
        return stage4_data
        
    except Exception as e:
        st.error(f"Ошибка при извлечении результатов Этапа 4: {str(e)}")
        return None

def create_integrated_comparison_table(stage3_data, stage4_data):
    """Создание интегрированной таблицы сравнения"""
    
    integrated_data = []
    
    # Добавляем модели из Этапа 3
    if stage3_data and 'comparison_df' in stage3_data:
        comparison_df = stage3_data['comparison_df']
        
        if not comparison_df.empty and 'Метод' in comparison_df.columns:
            for idx, row in comparison_df.iterrows():
                # Извлекаем метрики
                val_mae = None
                val_r2 = None
                
                # Пробуем разные варианты названий столбцов
                for mae_col in ['Val MAE', 'val_mae', 'CV MAE', 'best_score']:
                    if mae_col in row:
                        try:
                            val_mae = float(str(row[mae_col]).replace(',', '.'))
                            break
                        except:
                            pass
                
                for r2_col in ['Val R²', 'val_r2', 'R2']:
                    if r2_col in row:
                        try:
                            val_r2 = float(str(row[r2_col]).replace(',', '.'))
                            break
                        except:
                            pass
                
                integrated_data.append({
                    'Тип': 'ML модель (Этап 3)',
                    'Название': row['Метод'],
                    'MAE': val_mae if val_mae is not None else 0,
                    'R²': val_r2 if val_r2 is not None else 0,
                    'Время обучения': 0,  # В Этапе 3 нет данных о времени
                    'Подход': 'One-step прогнозирование'
                })
    
    # Добавляем стратегии из Этапа 4
    if stage4_data:
        # Сначала пробуем comparison_df_display
        if 'comparison_df_display' in stage4_data:
            comparison_df = stage4_data['comparison_df_display']
            if isinstance(comparison_df, pd.DataFrame) and not comparison_df.empty:
                for idx, row in comparison_df.iterrows():
                    if 'Стратегия' in row:
                        # Извлекаем метрики
                        try:
                            mae_str = str(row.get('Средний MAE', '0'))
                            mae_val = float(mae_str.replace('%', '').replace(',', '.').strip())
                        except:
                            mae_val = 0
                        
                        try:
                            time_str = str(row.get('Время обучения (с)', '0'))
                            time_val = float(time_str.replace('%', '').replace(',', '.').strip())
                        except:
                            time_val = 0
                        
                        integrated_data.append({
                            'Тип': 'Стратегия (Этап 4)',
                            'Название': row['Стратегия'],
                            'MAE': mae_val,
                            'R²': 0,  # В стратегиях нет R²
                            'Время обучения': time_val,
                            'Подход': 'Multi-step прогнозирование'
                        })
        
        # Затем пробуем strategy_results
        elif 'strategy_results' in stage4_data:
            strategy_results = stage4_data['strategy_results']
            if isinstance(strategy_results, dict):
                for strategy_name, strategy_info in strategy_results.items():
                    if isinstance(strategy_info, dict):
                        mae_val = strategy_info.get('avg_mae', 0)
                        time_val = strategy_info.get('training_time', 0)
                        
                        integrated_data.append({
                            'Тип': 'Стратегия (Этап 4)',
                            'Название': strategy_name,
                            'MAE': mae_val if isinstance(mae_val, (int, float)) else 0,
                            'R²': 0,
                            'Время обучения': time_val if isinstance(time_val, (int, float)) else 0,
                            'Подход': 'Multi-step прогнозирование'
                        })
    
    # Создаем DataFrame
    if integrated_data:
        df = pd.DataFrame(integrated_data)
        
        # Сортируем по MAE
        df = df.sort_values('MAE')
        
        # Форматируем для отображения
        df_display = df.copy()
        df_display['MAE'] = df_display['MAE'].apply(lambda x: f"{x:.4f}")
        df_display['R²'] = df_display['R²'].apply(lambda x: f"{x:.4f}" if x != 0 else "N/A")
        df_display['Время обучения'] = df_display['Время обучения'].apply(lambda x: f"{x:.3f}")
        
        return df, df_display
    
    return pd.DataFrame(), pd.DataFrame()

def plot_integrated_comparison(integrated_df, stage3_data, stage4_data):
    """Визуализация интегрированного сравнения"""
    
    # Создаем фигуру с подграфиками
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Сравнение MAE (чем меньше, тем лучше)', 
            'Сравнение по типам',
            'Распределение MAE по этапам',
            'Время обучения стратегий'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'box'}],
            [{'type': 'bar'}, {'type': 'bar'}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    if integrated_df.empty:
        # Возвращаем пустую фигуру с сообщением
        fig = go.Figure()
        fig.add_annotation(
            text="Нет данных для визуализации",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(height=500)
        return fig
    
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
    
    # 1. Бар-график MAE
    methods = integrated_df['Название'].tolist()
    mae_values = integrated_df['MAE'].astype(float).tolist()
    
    # Определяем цвета для разных типов
    bar_colors = []
    for method_type in integrated_df['Тип']:
        if 'Этап 3' in method_type:
            bar_colors.append('blue')
        else:
            bar_colors.append('green')
    
    fig.add_trace(
        go.Bar(
            x=methods,
            y=mae_values,
            name='MAE',
            marker_color=bar_colors,
            text=[f"{x:.4f}" for x in mae_values],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # 2. Box plot по типам
    if 'Тип' in integrated_df.columns:
        # Разделяем данные по типам
        stage3_mae = integrated_df[integrated_df['Тип'].str.contains('Этап 3')]['MAE'].astype(float)
        stage4_mae = integrated_df[integrated_df['Тип'].str.contains('Этап 4')]['MAE'].astype(float)
        
        fig.add_trace(
            go.Box(
                y=stage3_mae.tolist() if len(stage3_mae) > 0 else [0],
                name='ML модели (Этап 3)',
                marker_color='blue'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Box(
                y=stage4_mae.tolist() if len(stage4_mae) > 0 else [0],
                name='Стратегии (Этап 4)',
                marker_color='green'
            ),
            row=1, col=2
        )
    
    # 3. Сравнение подходов
    if 'Подход' in integrated_df.columns:
        approaches = integrated_df['Подход'].unique()
        approach_mae = []
        
        for approach in approaches:
            approach_data = integrated_df[integrated_df['Подход'] == approach]['MAE'].astype(float)
            if len(approach_data) > 0:
                approach_mae.append(approach_data.mean())
            else:
                approach_mae.append(0)
        
        fig.add_trace(
            go.Bar(
                x=approaches,
                y=approach_mae,
                name='Средний MAE по подходам',
                marker_color=['orange', 'purple'][:len(approaches)],
                text=[f"{x:.4f}" for x in approach_mae],
                textposition='auto'
            ),
            row=2, col=1
        )
    
    # 4. Время обучения стратегий
    if stage4_data and 'comparison_df_display' in stage4_data:
        comparison_df = stage4_data['comparison_df_display']
        if isinstance(comparison_df, pd.DataFrame) and not comparison_df.empty:
            strategies = []
            training_times = []
            
            for idx, row in comparison_df.iterrows():
                if 'Стратегия' in row and 'Время обучения (с)' in row:
                    try:
                        time_str = str(row['Время обучения (с)'])
                        time_val = float(time_str.replace(',', '.').strip())
                        strategies.append(row['Стратегия'])
                        training_times.append(time_val)
                    except:
                        pass
            
            if strategies:
                fig.add_trace(
                    go.Bar(
                        x=strategies,
                        y=training_times,
                        name='Время обучения',
                        marker_color='red',
                        text=[f"{x:.3f}" for x in training_times],
                        textposition='auto'
                    ),
                    row=2, col=2
                )
    
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="Интегрированное сравнение Этапа 3 и Этапа 4",
        title_x=0.5
    )
    
    fig.update_yaxes(title_text="MAE", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=1, col=2)
    fig.update_yaxes(title_text="Средний MAE", row=2, col=1)
    fig.update_yaxes(title_text="Секунды", row=2, col=2)
    
    return fig

def get_best_overall_model(integrated_df):
    """Определение лучшей модели/стратегии"""
    
    if integrated_df.empty:
        return None
    
    try:
        # Преобразуем MAE в числа
        integrated_df['MAE_num'] = integrated_df['MAE'].astype(float)
        
        # Находим лучшую по MAE
        best_row = integrated_df.loc[integrated_df['MAE_num'].idxmin()]
        
        best_model = {
            'Тип': best_row['Тип'],
            'Название': best_row['Название'],
            'MAE': best_row['MAE_num'],
            'Подход': best_row.get('Подход', 'N/A')
        }
        
        return best_model
    
    except Exception as e:
        st.warning(f"Ошибка при определении лучшей модели: {str(e)}")
        return None

# ============================================================
# ОСНОВНОЙ ИНТЕРФЕЙС ЭТАПА 5
# ============================================================

def show_advanced_modeling_interface():
    """Основной интерфейс Этапа 5: Интеграция и сравнение"""
    
    # Проверка наличия данных из предыдущих этапов
    required_keys = ['df_features', 'feature_info', 'split_data']
    missing_keys = [key for key in required_keys if key not in st.session_state]
    
    if missing_keys:
        st.error(f"❌ Сначала выполните Этапы 1-2. Отсутствуют: {', '.join(missing_keys)}")
        return
    
    st.info("""
    **Цель Этапа 5:**
    - Интегрировать результаты ML моделей из Этапа 3
    - Интегрировать результаты стратегий прогнозирования из Этапа 4  
    - Провести сравнительный анализ
    - Определить наилучший подход для вашей задачи
    """)
    
    # Извлекаем данные из предыдущих этапов
    stage3_data = extract_stage3_results()
    stage4_data = extract_stage4_results()
    
    # Проверяем, есть ли данные для сравнения
    if not stage3_data and not stage4_data:
        st.error("""
        ❌ Нет данных для сравнения!
        
        **Требуется выполнить:**
        1. Этап 3: Подбор гиперпараметров ML моделей
        2. Этап 4: Сравнение стратегий прогнозирования
        
        Без этих данных интегрированное сравнение невозможно.
        """)
        return
    
    if not stage3_data:
        st.warning("⚠️ Отсутствуют результаты Этапа 3")
    if not stage4_data:
        st.warning("⚠️ Отсутствуют результаты Этапа 4")
    
    # Информация о доступных данных
    st.subheader("📋 Информация о доступных данных")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if stage3_data:
            st.success("✅ Данные Этапа 3 доступны")
            comparison_df = stage3_data.get('comparison_df', pd.DataFrame())
            st.write(f"- Количество моделей: {len(comparison_df)}")
            
            # Информация о лучшей модели
            best_model = stage3_data.get('best_model', {})
            if best_model and 'Метод' in best_model:
                st.write(f"- Лучшая модель: {best_model['Метод']}")
                st.write(f"- MAE: {best_model.get('Val MAE', 'N/A')}")
        else:
            st.warning("❌ Данные Этапа 3 отсутствуют")
    
    with col2:
        if stage4_data:
            st.success("✅ Данные Этапа 4 доступны")
            
            # Подсчитываем количество стратегий
            strategy_count = 0
            if 'comparison_df_display' in stage4_data:
                df = stage4_data['comparison_df_display']
                if isinstance(df, pd.DataFrame):
                    strategy_count = len(df)
            
            st.write(f"- Количество стратегий: {strategy_count}")
            st.write(f"- Горизонт прогнозирования: {stage4_data.get('horizon', 'N/A')}")
        else:
            st.warning("❌ Данные Этапа 4 отсутствуют")
    
    # Настройки сравнения
    st.subheader("⚙️ Настройки сравнения")
    
    col1, col2 = st.columns(2)
    
    with col1:
        comparison_metric = st.selectbox(
            "Основная метрика для сравнения",
            options=['MAE', 'RMSE', 'Время обучения'],
            index=0
        )
        
        show_detailed_analysis = st.checkbox("Показать детальный анализ", value=True)
    
    with col2:
        include_time_analysis = st.checkbox("Включить анализ времени", value=True)
        normalize_metrics = st.checkbox("Нормализовать метрики", value=False)
      
    st.markdown("---")
    
    if st.button("🔍 Запустить интегрированное сравнение", type="primary", use_container_width=True):
        with st.spinner("Выполняется интегрированное сравнение..."):
            try:
                # Создаем интегрированную таблицу сравнения
                integrated_df, integrated_df_display = create_integrated_comparison_table(stage3_data, stage4_data)
                
                if integrated_df.empty or integrated_df_display.empty:
                    st.error("Не удалось создать интегрированную таблицу сравнения")
                    return
                
                # Сохраняем результаты ПРЯМО ЗДЕСЬ
                st.session_state.integrated_results = {
                    'integrated_df': integrated_df,
                    'integrated_df_display': integrated_df_display,
                    'stage3_data': stage3_data,
                    'stage4_data': stage4_data
                }
                
                # Отображение результатов
                _display_integrated_results(integrated_df_display, integrated_df, stage3_data, stage4_data)
                
            except Exception as e:
                st.error(f"Ошибка при выполнении сравнения: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Если уже есть результаты, показываем их
    elif 'integrated_results' in st.session_state:
        st.success("✅ Интегрированное сравнение уже выполнено!")
        
        results = st.session_state.integrated_results
        _display_integrated_results(
            results['integrated_df_display'],
            results['integrated_df'],
            results['stage3_data'],
            results['stage4_data']
        )

def _display_integrated_results(integrated_df_display, integrated_df, stage3_data, stage4_data):
    """Отображение интегрированных результатов"""
    
    # 1. Интегрированная таблица
    st.subheader("📊 Интегрированная таблица сравнения")
    st.dataframe(integrated_df_display, width='stretch')
    
    # 2. Графическое сравнение
    st.subheader("📈 Графическое сравнение")
    fig_comparison = plot_integrated_comparison(integrated_df, stage3_data, stage4_data)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # 3. Анализ результатов
    st.subheader("🎯 Анализ результатов")
    
    # Находим лучшие модели по типам
    if not integrated_df.empty:
        # Лучшая ML модель
        ml_models = integrated_df[integrated_df['Тип'].str.contains('Этап 3')]
        best_ml = None
        if not ml_models.empty:
            best_ml = ml_models.loc[ml_models['MAE'].astype(float).idxmin()]
        
        # Лучшая стратегия
        strategies = integrated_df[integrated_df['Тип'].str.contains('Этап 4')]
        best_strategy = None
        if not strategies.empty:
            best_strategy = strategies.loc[strategies['MAE'].astype(float).idxmin()]
        
        # Выводим сравнение
        col1, col2 = st.columns(2)
        
        with col1:
            if best_ml is not None:
                st.info(f"""
                **🏆 Лучшая ML модель (Этап 3):**
                - **Модель:** {best_ml['Название']}
                - **MAE:** {float(best_ml['MAE']):.4f}
                - **Подход:** One-step прогнозирование
                """)
            else:
                st.info("Лучшая ML модель не найдена")
        
        with col2:
            if best_strategy is not None:
                st.info(f"""
                **🏆 Лучшая стратегия (Этап 4):**
                - **Стратегия:** {best_strategy['Название']}
                - **MAE:** {float(best_strategy['MAE']):.4f}
                - **Подход:** Multi-step прогнозирование
                """)
            else:
                st.info("Лучшая стратегия не найдена")
        
        # Сравниваем лучшие подходы
        if best_ml is not None and best_strategy is not None:
            ml_mae = float(best_ml['MAE'])
            strategy_mae = float(best_strategy['MAE'])
            
            st.subheader("🤔 Сравнение лучших подходов")
            
            if ml_mae < strategy_mae:
                improvement = ((strategy_mae - ml_mae) / strategy_mae * 100)
                st.success(f"""
                **✅ ML модели показали лучший результат:**
                - ML модель превосходит стратегию на **{improvement:.1f}%** по MAE
                - **Рекомендация:** Использовать ML подход для one-step прогнозирования
                """)
            else:
                improvement = ((ml_mae - strategy_mae) / ml_mae * 100)
                st.success(f"""
                **✅ Стратегии прогнозирования показали лучший результат:**
                - Стратегия превосходит ML модель на **{improvement:.1f}%** по MAE
                - **Рекомендация:** Использовать стратегии multi-step прогнозирования
                """)
    
    # 4. Детальный анализ (если включен)
    st.subheader("📋 Детальный анализ")
    
    # Этап 3: Детали ML моделей
    if stage3_data and 'comparison_df' in stage3_data:
        with st.expander("📊 Детали Этапа 3 (ML модели)"):
            comparison_df = stage3_data['comparison_df']
            if isinstance(comparison_df, pd.DataFrame) and not comparison_df.empty:
                st.dataframe(comparison_df, width='stretch')
                
                # Информация о лучшей модели
                best_model_info = stage3_data.get('best_model', {})
                if best_model_info:
                    st.write("**Лучшая модель Этапа 3:**")
                    for key, value in best_model_info.items():
                        st.write(f"- {key}: {value}")
    
    # Этап 4: Детали стратегий
    if stage4_data:
        with st.expander("📊 Детали Этапа 4 (Стратегии прогнозирования)"):
            if 'comparison_df_display' in stage4_data:
                comparison_df = stage4_data['comparison_df_display']
                if isinstance(comparison_df, pd.DataFrame) and not comparison_df.empty:
                    st.dataframe(comparison_df, width='stretch')
            
            # Информация о стратегиях
            st.write("**Статистика стратегий:**")
            
            if 'strategy_results' in stage4_data:
                strategy_results = stage4_data['strategy_results']
                if isinstance(strategy_results, dict):
                    st.write(f"Количество стратегий: {len(strategy_results)}")
                    
                    # Средние метрики
                    mae_values = []
                    time_values = []
                    
                    for name, info in strategy_results.items():
                        if isinstance(info, dict):
                            mae = info.get('avg_mae', 0)
                            time_val = info.get('training_time', 0)
                            
                            if isinstance(mae, (int, float)):
                                mae_values.append(mae)
                            if isinstance(time_val, (int, float)):
                                time_values.append(time_val)
                    
                    if mae_values:
                        st.write(f"- Средний MAE: {np.mean(mae_values):.4f}")
                        st.write(f"- Минимальный MAE: {np.min(mae_values):.4f}")
                        st.write(f"- Максимальный MAE: {np.max(mae_values):.4f}")
    
    # 5. Рекомендации и выводы
    st.subheader("💡 Рекомендации и выводы")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Для one-step прогнозирования:**
        - Используйте ML модели из Этапа 3
        - Лучшие результаты показывают ансамблевые методы
        - Оптимизируйте гиперпараметры с помощью GridSearchCV или Optuna
        """)
    
    with col2:
        st.info("""
        **Для multi-step прогнозирования:**
        - Используйте стратегии из Этапа 4
        - DirRec стратегия часто показывает лучшие результаты
        - Учитывайте горизонт прогнозирования при выборе стратегии
        """)
    
    # Общие рекомендации
    st.markdown("""
    ### 🎯 Общие рекомендации:
    
    1. **Для коротких горизонтов (1-3 шага):** ML модели могут быть эффективнее
    2. **Для длинных горизонтов (5+ шагов):** Стратегии multi-step прогнозирования
    3. **Для критически важных систем:** Используйте ансамбли подходов
    4. **Для быстрого прототипирования:** AutoGluon или простые стратегии
    5. **Для продакшена:** Учитывайте время обучения и предсказания
    
    ### 📊 Ключевые метрики для мониторинга:
    - **MAE/RMSE:** Качество прогнозов
    - **Время обучения:** Эффективность подбора
    - **Время предсказания:** Производительность в реальном времени
    - **Рост ошибки:** Стабильность на длинных горизонтах
    """)
    
    st.markdown("---")
    st.success("""
    **✅ Этап 5 завершен!**
    
    **Что было сделано:**
    1. Интегрированы результаты ML моделей из Этапа 3
    2. Интегрированы результаты стратегий прогнозирования из Этапа 4
    3. Проведено сравнительное исследование различных подходов
    4. Даны рекомендации по выбору наилучшего подхода
    5. Визуализированы результаты сравнения
    
    **🎯 Итог проекта:** Выполнено полное исследование временных рядов от подготовки данных до сравнения различных подходов к прогнозированию.
    
    **Дальнейшие шаги:**
    - Реализация выбранного подхода в продакшене
    - Мониторинг качества прогнозов в реальном времени
    - Постоянное улучшение моделей по мере поступления новых данных
    """)
    
    # ВАЖНОЕ ИСПРАВЛЕНИЕ: Сохраняем результаты в session_state для 6 этапа
    st.session_state.integrated_results = {
        'integrated_df': integrated_df,
        'integrated_df_display': integrated_df_display,
        'stage3_data': stage3_data,
        'stage4_data': stage4_data
    }
    
    # Для совместимости также сохраняем под другими ключами
    st.session_state.advanced_modeling_data = st.session_state.integrated_results
    st.session_state.model_comparison_results = st.session_state.integrated_results
    
