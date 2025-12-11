# validation_module.py - Модуль для валидации и разбиения данных (Этап 2)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Импорт для Streamlit интерфейса
import streamlit as st
import plotly.graph_objects as go

# Импорт для кросс-валидации
from sklearn.model_selection import TimeSeriesSplit

# ============================================================
# КЛАССЫ ДЛЯ ВАЛИДАЦИИ
# ============================================================

class PurgedWalkForward:
    """
    Класс для реализации Purged Walk-Forward валидации с gap
    """
    
    def __init__(self, n_splits: int = 5, gap: int = 7, max_train_size: int = 365):
        """
        Инициализация Purged Walk-Forward
        
        Parameters:
        -----------
        n_splits : int
            Количество фолдов
        gap : int
            Разрыв между обучающей и тестовой выборками
        max_train_size : int
            Максимальный размер обучающей выборки
        """
        self.n_splits = n_splits
        self.gap = gap
        self.max_train_size = max_train_size
        self.folds_info = []
    
    def split(self, X: pd.DataFrame) -> List[Tuple]:
        """
        Генерация разбиений
        
        Parameters:
        -----------
        X : pd.DataFrame
            Входные данные
            
        Returns:
        --------
        List[Tuple]
            Список кортежей (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Размер тестового окна
        test_size = (n_samples - self.max_train_size) // (self.n_splits + 1)
        
        folds = []
        self.folds_info = []
        
        for i in range(self.n_splits):
            # Вычисляем границы
            train_end = n_samples - test_size * (self.n_splits - i) - self.gap
            test_start = train_end + self.gap
            test_end = test_start + test_size
            
            # Корректируем границы
            train_start = max(0, train_end - self.max_train_size)
            
            # Получаем индексы
            train_idx = indices[train_start:train_end]
            test_idx = indices[test_start:test_end]
            
            # Проверяем, что индексы не пустые
            if len(train_idx) > 0 and len(test_idx) > 0:
                folds.append((train_idx, test_idx))
                
                # Сохраняем информацию о фолде
                fold_info = {
                    'fold': i + 1,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'gap': self.gap
                }
                self.folds_info.append(fold_info)
        
        return folds

class TimeSeriesValidator:
    """
    Класс для валидации временных рядов
    """
    
    def __init__(self):
        self.split_data = None
        self.split_stats = None
        self.tscv_folds = None
        self.purged_folds = None
        
    def chronological_split(self, df: pd.DataFrame, date_col: str, 
                           target_col: str, split_ratios: Dict) -> Dict:
        """
        Хронологическое разбиение данных
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходный DataFrame
        date_col : str
            Колонка с датой
        target_col : str
            Целевая переменная
        split_ratios : Dict
            Словарь с соотношениями разбиения
            
        Returns:
        --------
        Dict
            Словарь с разбитыми данными и статистикой
        """
        # Сортируем по дате
        df_sorted = df.sort_values(date_col).copy()
        
        # Вычисляем границы разбиения
        total_len = len(df_sorted)
        train_end = int(total_len * split_ratios['train'])
        val_end = train_end + int(total_len * split_ratios['val'])
        
        # Разбиваем
        train_data = df_sorted.iloc[:train_end]
        val_data = df_sorted.iloc[train_end:val_end]
        test_data = df_sorted.iloc[val_end:]
        
        # Собираем статистику
        stats = {
            'train': {
                'size': len(train_data),
                'start': train_data[date_col].min(),
                'end': train_data[date_col].max(),
                'target_mean': train_data[target_col].mean()
            },
            'val': {
                'size': len(val_data),
                'start': val_data[date_col].min(),
                'end': val_data[date_col].max(),
                'target_mean': val_data[target_col].mean()
            },
            'test': {
                'size': len(test_data),
                'start': test_data[date_col].min(),
                'end': test_data[date_col].max(),
                'target_mean': test_data[target_col].mean()
            }
        }
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'stats': stats
        }
    
    def time_series_cross_validation(self, df: pd.DataFrame, date_col: str,
                                    n_splits: int = 5, max_train_size: int = 365) -> Dict:
        """
        TimeSeriesSplit кросс-валидация
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходный DataFrame
        date_col : str
            Колонка с датой
        n_splits : int
            Количество фолдов
        max_train_size : int
            Максимальный размер обучающей выборки
            
        Returns:
        --------
        Dict
            Словарь с информацией о фолдах
        """
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        
        # Используем TimeSeriesSplit из sklearn с max_train_size
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=max_train_size
        )
        
        folds = []
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df_sorted)):
            train_data = df_sorted.iloc[train_idx]
            test_data = df_sorted.iloc[test_idx]
            
            folds.append({
                'fold': fold_idx + 1,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'train_start': train_data[date_col].min(),
                'train_end': train_data[date_col].max(),
                'test_start': test_data[date_col].min(),
                'test_end': test_data[date_col].max(),
                'train_indices': train_idx.tolist(),
                'test_indices': test_idx.tolist()
            })
        
        return {
            'folds': folds,
            'n_splits': n_splits,
            'max_train_size': max_train_size,
            'total_samples': len(df_sorted)
        }
    
    def purged_walk_forward_validation(self, df: pd.DataFrame, date_col: str,
                                      n_splits: int = 5, gap: int = 7, 
                                      max_train_size: int = 365) -> Dict:
        """
        Purged Walk-Forward валидация с gap
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходный DataFrame
        date_col : str
            Колонка с датой
        n_splits : int
            Количество фолдов
        gap : int
            Разрыв между обучающей и тестовой выборками
        max_train_size : int
            Максимальный размер обучающей выборки
            
        Returns:
        --------
        Dict
            Словарь с информацией о фолдах
        """
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        
        # Создаем Purged Walk-Forward валидатор
        pwf = PurgedWalkForward(
            n_splits=n_splits,
            gap=gap,
            max_train_size=max_train_size
        )
        
        # Получаем разбиения
        splits = pwf.split(df_sorted)
        
        folds = []
        for i, (train_idx, test_idx) in enumerate(splits):
            train_data = df_sorted.iloc[train_idx]
            test_data = df_sorted.iloc[test_idx]
            
            folds.append({
                'fold': i + 1,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'train_start': train_data[date_col].min(),
                'train_end': train_data[date_col].max(),
                'test_start': test_data[date_col].min(),
                'test_end': test_data[date_col].max(),
                'gap': gap,
                'train_indices': train_idx.tolist(),
                'test_indices': test_idx.tolist()
            })
        
        return {
            'folds': folds,
            'n_splits': n_splits,
            'gap': gap,
            'max_train_size': max_train_size,
            'total_samples': len(df_sorted)
        }

# ============================================================
# ФУНКЦИИ ВИЗУАЛИЗАЦИИ
# ============================================================

def plot_validation_splits(chronological_stats: Dict, 
                          tscv_folds: Optional[Dict] = None,
                          purged_folds: Optional[Dict] = None,
                          date_col: str = 'date'):
    """
    Визуализация всех типов разбиений
    """
    
    # Создаем фигуру с несколькими подграфиками
    fig = go.Figure()
    
    # 1. Хронологическое разбиение
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    y_positions = [2, 1, 0]  # Позиции на оси Y
    
    for idx, (split_name, color) in enumerate(zip(['train', 'val', 'test'], colors)):
        stats = chronological_stats[split_name]
        y_pos = y_positions[idx]
        
        fig.add_trace(go.Scatter(
            x=[stats['start'], stats['end']],
            y=[f"Хронологическое: {split_name}", f"Хронологическое: {split_name}"],
            mode='lines+markers',
            name=f"{split_name} ({stats['size']} записей)",
            line=dict(color=color, width=8),
            marker=dict(size=10),
            legendgroup="chronological",
            showlegend=True
        ))
    
    # 2. TimeSeriesSplit фолды
    if tscv_folds:
        for fold in tscv_folds['folds']:
            y_offset = 0.2 * (fold['fold'] - 1)  # Смещение для разных фолдов
            
            fig.add_trace(go.Scatter(
                x=[fold['train_start'], fold['train_end']],
                y=[f"TimeSeriesSplit: Fold {fold['fold']} train", f"TimeSeriesSplit: Fold {fold['fold']} train"],
                mode='lines',
                name=f"TS Fold {fold['fold']} train",
                line=dict(color='lightblue', width=4, dash='dash'),
                legendgroup="tscv",
                showlegend=True if fold['fold'] == 1 else False
            ))
            
            fig.add_trace(go.Scatter(
                x=[fold['test_start'], fold['test_end']],
                y=[f"TimeSeriesSplit: Fold {fold['fold']} test", f"TimeSeriesSplit: Fold {fold['fold']} test"],
                mode='lines',
                name=f"TS Fold {fold['fold']} test",
                line=dict(color='orange', width=4, dash='dash'),
                legendgroup="tscv",
                showlegend=True if fold['fold'] == 1 else False
            ))
    
    # 3. Purged Walk-Forward фолды
    if purged_folds:
        for fold in purged_folds['folds']:
            y_offset = 0.2 * (fold['fold'] - 1) + 1  # Смещение для разных фолдов
            
            fig.add_trace(go.Scatter(
                x=[fold['train_start'], fold['train_end']],
                y=[f"PurgedWF: Fold {fold['fold']} train", f"PurgedWF: Fold {fold['fold']} train"],
                mode='lines',
                name=f"Purged Fold {fold['fold']} train",
                line=dict(color='green', width=4, dash='dot'),
                legendgroup="purged",
                showlegend=True if fold['fold'] == 1 else False
            ))
            
            fig.add_trace(go.Scatter(
                x=[fold['test_start'], fold['test_end']],
                y=[f"PurgedWF: Fold {fold['fold']} test", f"PurgedWF: Fold {fold['fold']} test"],
                mode='lines',
                name=f"Purged Fold {fold['fold']} test",
                line=dict(color='red', width=4, dash='dot'),
                legendgroup="purged",
                showlegend=True if fold['fold'] == 1 else False
            ))
    
    fig.update_layout(
        title="Визуализация всех типов валидации",
        xaxis_title="Дата",
        yaxis_title="Тип разбиения",
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def show_folds_table(folds_data: Dict, title: str):
    """
    Показать таблицу с информацией о фолдах
    """
    st.subheader(title)
    
    table_data = []
    for fold in folds_data['folds']:
        table_data.append({
            'Фолд': fold['fold'],
            'Train размер': fold['train_size'],
            'Test размер': fold['test_size'],
            'Train начало': fold['train_start'].strftime('%Y-%m-%d'),
            'Train конец': fold['train_end'].strftime('%Y-%m-%d'),
            'Test начало': fold['test_start'].strftime('%Y-%m-%d'),
            'Test конец': fold['test_end'].strftime('%Y-%m-%d'),
            'Gap': fold.get('gap', 0)
        })
    
    if table_data:
        st.dataframe(pd.DataFrame(table_data), width='stretch')

# ============================================================
# ИНТЕРФЕЙС ДЛЯ STREAMLIT
# ============================================================

def show_validation_interface(df: pd.DataFrame, date_col: str, target_col: str):
    """
    Показать интерфейс валидации данных
    """
    print(f"📊 Начало валидации...")
    print(f"   - df_features в session_state: {'df_features' in st.session_state}")
    
    # Проверяем, есть ли созданные признаки
    if 'df_features' not in st.session_state:
        st.error("❌ Сначала выполните Этап 1: Инжиниринг признаков!")
        return
    
    df_features = st.session_state.df_features
    
    st.info("""
    ### 📋 Требования Этапа 2:
    1. **Хронологическое разбиение**: train (60%), val (20%), test (20%)
    2. **TimeSeriesSplit**: n_splits=5, max_train_size=365
    3. **Purged Walk-Forward**: gap между train и test, чтобы избежать утечки
    """)
    
    # Используем форму для предотвращения перезагрузки
    with st.form(key='validation_form'):
        # 1. ХРОНОЛОГИЧЕСКОЕ РАЗБИЕНИЕ
        st.subheader("⚙️ Настройки хронологического разбиения")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            train_size = st.slider("Train размер (%)", 50, 80, 60, 5) / 100
            st.caption(f"Обучающая выборка: {train_size*100:.0f}%")
        
        with col2:
            val_size = st.slider("Validation размер (%)", 10, 40, 20, 5) / 100
            st.caption(f"Валидационная выборка: {val_size*100:.0f}%")
        
        with col3:
            test_size = st.slider("Test размер (%)", 10, 40, 20, 5) / 100
            st.caption(f"Тестовая выборка: {test_size*100:.0f}%")
        
        # Проверка суммы
        total = train_size + val_size + test_size
        if abs(total - 1.0) > 0.01:
            st.warning(f"Сумма долей должна быть равна 100%. Текущая сумма: {total*100:.0f}%")
        
        st.markdown("---")
        
        # 2. TIMESERIESSPLIT НАСТРОЙКИ
        st.subheader("⚙️ Настройки TimeSeriesSplit")
        
        col4, col5 = st.columns(2)
        
        with col4:
            n_splits = st.slider("Количество фолдов (n_splits)", 2, 10, 5, 1)
        
        with col5:
            max_train_size = st.slider("Максимальный размер train (max_train_size)", 100, 1000, 365, 10)
        
        st.markdown("---")
        
        # 3. PURGED WALK-FORWARD НАСТРОЙКИ
        st.subheader("⚙️ Настройки Purged Walk-Forward")
        
        col6, col7 = st.columns(2)
        
        with col6:
            gap_size = st.slider("Размер gap", 1, 30, 7, 1)
            st.caption("Разрыв между train и test для избежания утечки")
        
        with col7:
            pwf_max_train = st.slider("Максимальный размер train (Purged)", 100, 1000, 365, 10)
        
        # Кнопка запуска валидации внутри формы
        submit_button = st.form_submit_button(
            "🚀 Выполнить все типы валидации", 
            type="primary", 
            use_container_width=True
        )
    
    # Если форма отправлена, выполняем валидацию
    if submit_button:
        with st.spinner("Выполнение валидации..."):
            try:
                # Создаем валидатор
                validator = TimeSeriesValidator()
                
                # 1. Хронологическое разбиение
                split_result = validator.chronological_split(
                    df=df_features,
                    date_col=date_col,
                    target_col=target_col,
                    split_ratios={
                        'train': train_size,
                        'val': val_size,
                        'test': test_size
                    }
                )
                
                # 2. TimeSeriesSplit кросс-валидация
                tscv_result = validator.time_series_cross_validation(
                    df=split_result['train'],  # Используем только train данные
                    date_col=date_col,
                    n_splits=n_splits,
                    max_train_size=max_train_size
                )
                
                # 3. Purged Walk-Forward валидация
                purged_result = validator.purged_walk_forward_validation(
                    df=split_result['train'],  # Используем только train данные
                    date_col=date_col,
                    n_splits=n_splits,
                    gap=gap_size,
                    max_train_size=pwf_max_train
                )
                
                # Сохраняем в session state
                st.session_state.split_data = split_result
                st.session_state.tscv_folds = tscv_result
                st.session_state.purged_folds = purged_result
                
                st.success("✅ Все типы валидации выполнены!")
                
            except Exception as e:
                st.error(f"❌ Ошибка при выполнении валидации: {str(e)}")
                print(f"❌ Ошибка: {str(e)}")
    
    # Показываем результаты если они есть
    if 'split_data' in st.session_state:
        split_result = st.session_state.split_data
        
        st.markdown("---")
        st.success("✅ Этап 2: Валидация и разбиение данных завершен!")
        
        # 1. ОБЩАЯ ИНФОРМАЦИЯ
        st.subheader("📊 Общая информация о разбиениях")
        
        info_cols = st.columns(4)
        
        with info_cols[0]:
            st.metric("Всего записей", len(df_features))
        
        with info_cols[1]:
            train_size = split_result['stats']['train']['size']
            st.metric("Train записей", train_size)
        
        with info_cols[2]:
            val_size = split_result['stats']['val']['size']
            st.metric("Val записей", val_size)
        
        with info_cols[3]:
            test_size = split_result['stats']['test']['size']
            st.metric("Test записей", test_size)
        
        st.markdown("---")
        
        # 2. ХРОНОЛОГИЧЕСКОЕ РАЗБИЕНИЕ
        st.subheader("📅 Хронологическое разбиение (60/20/20)")
        
        stats_cols = st.columns(3)
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for idx, (split_name, color) in enumerate(zip(['train', 'val', 'test'], colors)):
            with stats_cols[idx]:
                stats = split_result['stats'][split_name]
                st.metric(
                    label=f"{split_name.upper()} выборка",
                    value=f"{stats['size']:,} записей",
                    delta=f"{stats['size']/len(df_features)*100:.1f}%"
                )
        
        # Таблица с деталями хронологического разбиения
        chronological_table = []
        for split_name in ['train', 'val', 'test']:
            stats = split_result['stats'][split_name]
            chronological_table.append({
                'Выборка': split_name.upper(),
                'Записей': stats['size'],
                'Процент': f"{stats['size']/len(df_features)*100:.1f}%",
                'Начало': stats['start'].strftime('%Y-%m-%d'),
                'Конец': stats['end'].strftime('%Y-%m-%d'),
                'Среднее целевой': f"{stats['target_mean']:.4f}"
            })
        
        st.dataframe(pd.DataFrame(chronological_table), width='stretch')
        
        st.markdown("---")
        
        # 3. TIMESERIESSPLIT РЕЗУЛЬТАТЫ
        if 'tscv_folds' in st.session_state:
            tscv_result = st.session_state.tscv_folds
            
            st.subheader(f"🔄 TimeSeriesSplit (n_splits={tscv_result['n_splits']}, max_train_size={tscv_result['max_train_size']})")
            
            # Таблица с фолдами
            show_folds_table(tscv_result, "Детали фолдов TimeSeriesSplit")
            
            # Статистика по фолдам
            fold_stats = []
            for fold in tscv_result['folds']:
                fold_stats.append({
                    'Фолд': fold['fold'],
                    'Train размер': fold['train_size'],
                    'Test размер': fold['test_size'],
                    'Test/Train': f"{(fold['test_size']/fold['train_size'])*100:.1f}%"
                })
            
            st.dataframe(pd.DataFrame(fold_stats), width='stretch')
        
        st.markdown("---")
        
        # 4. PURGED WALK-FORWARD РЕЗУЛЬТАТЫ
        if 'purged_folds' in st.session_state:
            purged_result = st.session_state.purged_folds
            
            st.subheader(f"🚶 Purged Walk-Forward (n_splits={purged_result['n_splits']}, gap={purged_result['gap']}, max_train_size={purged_result['max_train_size']})")
            
            # Таблица с фолдами
            show_folds_table(purged_result, "Детали фолдов Purged Walk-Forward")
            
            # Информация о gap
            st.info(f"""
            **Purged Walk-Forward с gap={purged_result['gap']}:**
            - Gap предотвращает утечку информации из будущего
            - Между концом train и началом test есть разрыв в {purged_result['gap']} дней
            - Это особенно важно для временных рядов с автокорреляцией
            """)
        
        st.markdown("---")
        
        # 5. ВИЗУАЛИЗАЦИЯ ВСЕХ РАЗБИЕНИЙ
        st.subheader("📊 Визуализация всех типов валидации")
        
        # Получаем данные для визуализации
        tscv_folds = st.session_state.get('tscv_folds')
        purged_folds = st.session_state.get('purged_folds')
        
        # Создаем визуализацию
        fig = plot_validation_splits(
            chronological_stats=split_result['stats'],
            tscv_folds=tscv_folds,
            purged_folds=purged_folds,
            date_col=date_col
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 6. ИНФОРМАЦИЯ О ЗАВЕРШЕНИИ ЭТАПА
        st.markdown("---")
        st.success("""
        ### ✅ Этап 2 успешно завершен!
        
        **Что сделано:**
        1. ✅ Хронологическое разбиение: train/val/test (60/20/20)
        2. ✅ TimeSeriesSplit: 5 фолдов для кросс-валидации с max_train_size=365
        3. ✅ Purged Walk-Forward: валидация с gap для избежания утечки данных
        
        **Данные готовы для Этапа 3: Подбор гиперпараметров и моделирование.**
        """)
        
        # Добавляем информацию о следующих шагах
        st.info("""
        **Следующий шаг:** Перейдите к Этапу 3 для подбора гиперпараметров и моделирования.
        Для градиентного бустинга будет использоваться Optuna, для линейных моделей - GridSearchCV,
        а также полная автоматизация с AutoGluon.
        """)