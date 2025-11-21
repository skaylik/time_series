"""
Модуль для генерации признаков и разбиения данных (Этап 2).
Создает временные признаки (лаги, скользящие статистики, календарные и циклические признаки),
выполняет хронологическое разбиение на train/validation/test выборки.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils import parse_int_list


def engineer_time_features(
    transformed_series: pd.Series,
    datetime_series: pd.Series,
    lags: List[int],
    rolling_windows: List[int],
    exogenous: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    target_name = transformed_series.name or "target"
    datetime_name = datetime_series.name or "datetime"

    # Проверка входных данных
    if transformed_series.empty:
        raise ValueError("transformed_series пустой. Проверьте данные на этапе 1.")
    if datetime_series.empty:
        raise ValueError("datetime_series пустой. Проверьте данные на этапе 1.")
    
    # Проверяем, что есть хотя бы одно не-NaN значение
    if transformed_series.notna().sum() == 0:
        raise ValueError("transformed_series содержит только NaN. Проверьте данные на этапе 1.")

    # Преобразуем datetime_series в datetime, если еще не преобразован
    datetime_converted = pd.to_datetime(datetime_series, errors="coerce")
    
    # Проверяем, что длины совпадают
    if len(transformed_series) != len(datetime_series):
        raise ValueError(
            f"Длины временного ряда ({len(transformed_series)}) и дат ({len(datetime_series)}) не совпадают. "
            f"Проверьте данные на этапе 1."
        )
    
    # Используем позиционное выравнивание для создания DataFrame
    # Выравниваем оба Series по индексу transformed_series перед созданием DataFrame
    if isinstance(transformed_series.index, pd.DatetimeIndex):
        # Если transformed_series имеет DatetimeIndex, используем его как дату
        datetime_for_df = pd.Series(transformed_series.index, index=transformed_series.index, name=datetime_name)
    else:
        # Выравниваем datetime_converted по индексу transformed_series
        if transformed_series.index.equals(datetime_converted.index):
            datetime_for_df = datetime_converted
        else:
            # Выравниваем по индексу
            datetime_for_df = datetime_converted.reindex(transformed_series.index)
            # Если после выравнивания есть NaN, заполняем их используя позиционное выравнивание
            if datetime_for_df.isna().any():
                # Используем позиционное выравнивание через .values
                datetime_for_df = pd.Series(
                    datetime_converted.values, 
                    index=transformed_series.index, 
                    name=datetime_name
                )
    
    # Создаем DataFrame из выровненных Series
    combined = pd.DataFrame({
        target_name: transformed_series,
        datetime_name: datetime_for_df
    })
    
    # Проверка после объединения
    if combined.empty:
        raise ValueError(
            f"После объединения selected_series и datetime_series получился пустой DataFrame. "
            f"selected_series длина={len(transformed_series)}, datetime_series длина={len(datetime_series)}, "
            f"selected_series индекс тип={type(transformed_series.index)}, "
            f"datetime_series индекс тип={type(datetime_series.index)}. "
            f"Проверьте, что индексы совпадают или имеют одинаковую длину."
        )

    # Удаляем только строки, где ОБА (и дата, и целевая переменная) NaN одновременно
    # Это важно, потому что лаги создают NaN в начале, но целевая переменная должна быть
    initial_len = len(combined)
    
    # Проверяем наличие данных перед удалением
    if initial_len == 0:
        raise ValueError(
            f"DataFrame пустой после объединения. "
            f"transformed_series длина={len(transformed_series)}, datetime_series длина={len(datetime_series)}."
        )
    
    # Проверяем, что колонки существуют
    if target_name not in combined.columns:
        raise ValueError(f"Колонка {target_name} не найдена в combined. Доступные колонки: {combined.columns.tolist()}")
    if datetime_name not in combined.columns:
        raise ValueError(f"Колонка {datetime_name} не найдена в combined. Доступные колонки: {combined.columns.tolist()}")
    
    # Проверяем количество не-NaN значений
    target_notna = combined[target_name].notna().sum()
    datetime_notna = combined[datetime_name].notna().sum()
    
    # Если дата NaN, но индекс - это DatetimeIndex, используем индекс как дату
    # Это нужно сделать ДО удаления строк, чтобы сохранить данные
    if isinstance(combined.index, pd.DatetimeIndex):
        # Если в колонке даты есть NaN, заполняем их индексом
        if combined[datetime_name].isna().any():
            combined[datetime_name] = combined.index
            datetime_notna = len(combined)
        # Если все значения NaN, заменяем полностью
        elif combined[datetime_name].isna().all():
            combined[datetime_name] = combined.index
            datetime_notna = len(combined)
    
    # Сохраняем исходные значения для восстановления после dropna
    original_datetime_values = combined[datetime_name].copy()
    original_index = combined.index.copy()
    original_index_is_datetime = isinstance(combined.index, pd.DatetimeIndex)
    
    # Удаляем строки, где нет целевой переменной (это критично)
    before_drop = len(combined)
    combined.dropna(subset=[target_name], inplace=True)
    after_drop_target = len(combined)
    
    # Если после удаления нет данных, значит все значения целевой переменной были NaN
    if combined.empty:
        raise ValueError(
            f"После удаления строк без целевой переменной не осталось данных. "
            f"Исходная длина: {initial_len}, не-NaN значений в {target_name}: {target_notna}, "
            f"не-NaN значений в {datetime_name}: {datetime_notna}. "
            f"Проверьте, что selected_series содержит не-NaN значения."
        )
    
    # После dropna нужно восстановить даты для оставшихся строк
    # Используем позиционное выравнивание: находим позиции, где target не NaN
    datetime_after_drop = combined[datetime_name].notna().sum()
    
    # Если дата NaN в некоторых строках, заполняем их используя позиционное выравнивание
    if combined[datetime_name].isna().any():
        # Находим позиции в исходном transformed_series, где значения не NaN (используем .values для позиционного доступа)
        valid_positions_mask = transformed_series.notna().values
        
        # Используем позиционное выравнивание через .values для избежания проблем с индексами
        # Берем значения из datetime_converted по позициям, где target не NaN
        datetime_values_array = datetime_converted.values
        valid_datetime_values = datetime_values_array[valid_positions_mask]
        
        # Всегда используем позиционное выравнивание, если есть достаточно валидных значений
        if len(valid_datetime_values) >= len(combined):
            # Используем позиционное выравнивание - берем первые len(combined) значений
            combined[datetime_name] = pd.Series(
                valid_datetime_values[:len(combined)],
                index=combined.index,
                name=datetime_name
            )
        else:
            # Меньше валидных значений - заполняем только NaN значения, сохраняя существующие
            # Находим позиции, где дата NaN в combined
            nan_mask = combined[datetime_name].isna()
            nan_count = nan_mask.sum()
            
            if len(valid_datetime_values) >= nan_count:
                # Заполняем NaN значения из valid_datetime_values
                combined.loc[nan_mask, datetime_name] = valid_datetime_values[:nan_count]
            elif original_index_is_datetime:
                # Если исходный индекс был DatetimeIndex, используем его значения для заполнения NaN
                if combined.index.equals(original_index):
                    # Индексы совпадают - используем исходные значения для NaN
                    combined.loc[nan_mask, datetime_name] = original_datetime_values[nan_mask]
                else:
                    # Используем индекс как дату, если это DatetimeIndex
                    if isinstance(combined.index, pd.DatetimeIndex):
                        combined.loc[nan_mask, datetime_name] = combined.index[nan_mask]
                    else:
                        # Заполняем оставшиеся NaN из valid_datetime_values, если есть
                        if len(valid_datetime_values) > 0 and nan_count > 0:
                            fill_count = min(nan_count, len(valid_datetime_values))
                            combined.loc[nan_mask, datetime_name] = valid_datetime_values[:fill_count]
                        # Если все еще есть NaN, удаляем строки без даты
                        if combined[datetime_name].isna().any():
                            before_final_drop = len(combined)
                            combined.dropna(subset=[datetime_name], inplace=True)
                            if combined.empty:
                                raise ValueError(
                                    f"После удаления строк без даты не осталось данных. "
                                    f"Исходная длина: {initial_len}, после удаления без целевой переменной: {after_drop_target}, "
                                    f"перед финальным удалением: {before_final_drop}. "
                                    f"Не удалось выровнять datetime_series."
                                )
            else:
                # Заполняем оставшиеся NaN из valid_datetime_values, если есть
                if len(valid_datetime_values) > 0 and nan_count > 0:
                    fill_count = min(nan_count, len(valid_datetime_values))
                    combined.loc[nan_mask, datetime_name] = valid_datetime_values[:fill_count]
                
                # Если все еще есть NaN, удаляем строки без даты
                if combined[datetime_name].isna().any():
                    before_final_drop = len(combined)
                    combined.dropna(subset=[datetime_name], inplace=True)
                    if combined.empty:
                        raise ValueError(
                            f"После удаления строк без даты не осталось данных. "
                            f"Исходная длина: {initial_len}, после удаления без целевой переменной: {after_drop_target}, "
                            f"перед финальным удалением: {before_final_drop}. "
                            f"Не удалось выровнять datetime_series."
                        )
    
    # Финальная проверка - если дата все еще NaN, пытаемся удалить строки без даты
    datetime_final = combined[datetime_name].notna().sum()
    if combined[datetime_name].isna().any():
        # Последняя попытка: удаляем строки без даты
        before_final_drop = len(combined)
        combined.dropna(subset=[datetime_name], inplace=True)
        if combined.empty:
            index_type = type(combined.index).__name__
            is_datetime_index = isinstance(combined.index, pd.DatetimeIndex)
            raise ValueError(
                f"После удаления строк без даты не осталось данных. "
                f"Исходная длина: {initial_len}, после удаления без целевой переменной: {after_drop_target}, "
                f"не-NaN дат до обновления: {datetime_after_drop}, после обновления: {datetime_final}, "
                f"перед финальным удалением: {before_final_drop}, текущая длина: {len(combined)}, "
                f"тип индекса: {index_type}, является DatetimeIndex: {is_datetime_index}. "
                f"Проверьте данные."
            )
    
    if combined.empty:
        raise ValueError(
            f"После удаления пропусков не осталось данных. "
            f"Исходная длина: {initial_len}, после удаления без целевой переменной: {after_drop_target}. "
            f"Проверьте, что selected_series и datetime_series имеют совпадающие индексы и не содержат только NaN."
        )
    
    # Проверяем, нет ли конфликта между именем колонки и именем индекса
    # Если индекс имеет имя, которое совпадает с datetime_name, сбрасываем имя индекса
    if combined.index.name == datetime_name:
        combined.index.name = None
    
    combined.sort_values(datetime_name, inplace=True)

    target_series = combined[target_name]
    for lag in lags:
        combined[f"lag_{lag}"] = target_series.shift(lag)

    for window in rolling_windows:
        rolling = target_series.rolling(window=window, min_periods=1)  # min_periods=1 чтобы не терять данные
        combined[f"roll_mean_{window}"] = rolling.mean()
        combined[f"roll_std_{window}"] = rolling.std()
        combined[f"roll_min_{window}"] = rolling.min()
        combined[f"roll_max_{window}"] = rolling.max()

    dt = combined[datetime_name]
    combined["dayofweek"] = dt.dt.dayofweek
    combined["month"] = dt.dt.month
    combined["is_holiday"] = dt.dt.dayofweek.isin([5, 6]).astype(int)

    t_week = dt.dt.dayofweek
    t_month = dt.dt.month - 1
    combined["sin_2pi_t_over_7"] = np.sin(2 * np.pi * t_week / 7)
    combined["cos_2pi_t_over_12"] = np.cos(2 * np.pi * t_month / 12)

    if exogenous is not None and not exogenous.empty:
        exog_aligned = exogenous.reindex(combined.index).ffill().bfill()
        combined = combined.join(exog_aligned, how="left")

    # Удаляем только строки, где нет целевой переменной (критично для обучения)
    # NaN в признаках (лагах, скользящих окнах) оставляем - модели могут их обработать
    # или они будут заполнены позже
    initial_len = len(combined)
    combined.dropna(subset=[target_name], inplace=True)
    
    if combined.empty:
        raise ValueError(
            f"После удаления строк без целевой переменной не осталось данных. "
            f"Исходная длина: {initial_len}, проверьте данные."
        )
    
    combined.reset_index(drop=True, inplace=True)
    combined.rename(columns={datetime_name: "datetime"}, inplace=True)
    return combined


def chronological_split(
    features_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(features_df)
    if n == 0:
        return features_df, features_df, features_df

    train_end = int(np.floor(n * train_ratio))
    val_end = train_end + int(np.floor(n * val_ratio))

    train_df = features_df.iloc[:train_end].copy()
    val_df = features_df.iloc[train_end:val_end].copy()
    test_df = features_df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def stage2(
    analysis_data: Optional[Dict[str, Any]],
    lab_state: Dict[str, bool],
    default_lags: List[int],
    default_rolling_windows: List[int],
    default_split: Tuple[int, int, int],
) -> Dict[str, Any]:
    if analysis_data is None:
        analysis_data = {}


    if not lab_state.get("stage1_completed"):
        st.info("Сначала завершите предыдущий блок, чтобы продолжить к feature engineering.")
        return analysis_data

    source_df = analysis_data.get("source_df")
    selected_series: Optional[pd.Series] = analysis_data.get("selected_series")
    datetime_series: Optional[pd.Series] = analysis_data.get("datetime_series")

    if source_df is None or selected_series is None or selected_series.empty or datetime_series is None:
        st.warning("Не удалось получить данные для генерации признаков. Пересоздайте декомпозицию на этапе 1.")
        return analysis_data

    lag_defaults = analysis_data.get("lag_values", default_lags)
    rolling_defaults = analysis_data.get("rolling_values", default_rolling_windows)
    split_defaults = analysis_data.get("split_percentages", default_split)
    if isinstance(split_defaults, tuple):
        split_defaults = list(split_defaults)
    if not isinstance(split_defaults, (list, tuple)) or sum(split_defaults) != 100:
        split_defaults = list(default_split)
    exog_defaults = analysis_data.get("exog_selection", [])
    selected_pipeline_label = analysis_data.get("selected_pipeline_label", "—")

    with st.form("feature_engineering_form"):
        feature_col1, feature_col2 = st.columns(2)
        with feature_col1:
            lag_input = st.text_input(
                "Лаги (через запятую)",
                value=", ".join(str(lag) for lag in lag_defaults) if lag_defaults else "",
                help="Выберите, какие лаги строить. Пример: 1, 2, 7, 30",
            )
            rolling_input = st.text_input(
                "Окна для скользящих (через запятую)",
                value=", ".join(str(window) for window in rolling_defaults) if rolling_defaults else "",
                help="Размеры окон (в шагах) для скользящих mean/std/min/max.",
            )
        with feature_col2:
            split_col1, split_col2, split_col3 = st.columns(3)
            train_ratio_input = split_col1.number_input(
                "Train %",
                min_value=10,
                max_value=90,
                value=int(split_defaults[0]),
                step=5,
            )
            val_ratio_input = split_col2.number_input(
                "Validation %",
                min_value=5,
                max_value=80,
                value=int(split_defaults[1]),
                step=5,
            )
            test_ratio_input = split_col3.number_input(
                "Test %",
                min_value=5,
                max_value=80,
                value=int(split_defaults[2]),
                step=5,
            )

        available_exog_columns = [
            col
            for col in source_df.columns
            if col not in {analysis_data["target_column"], analysis_data["date_column"]}
            and pd.api.types.is_numeric_dtype(source_df[col])
        ]
        exog_selection = st.multiselect(
            "Экзогенные признаки (опционально)",
            available_exog_columns,
            default=exog_defaults,
            help="Дополнительные внешние факторы, которые можно передать в SARIMAX/Prophet.",
        )

        feature_submit = st.form_submit_button("Сформировать признаки")

    if feature_submit:
        has_error = False
        try:
            lag_values = parse_int_list(lag_input) or default_lags
        except ValueError as exc:
            st.error(f"Ошибка в списке лагов: {exc}")
            lag_values = lag_defaults
            has_error = True

        try:
            rolling_values = parse_int_list(rolling_input) or default_rolling_windows
        except ValueError as exc:
            st.error(f"Ошибка в списке скользящих окон: {exc}")
            rolling_values = rolling_defaults
            has_error = True

        ratio_sum = int(train_ratio_input) + int(val_ratio_input) + int(test_ratio_input)
        if ratio_sum != 100:
            st.error("Сумма долей Train/Validation/Test должна равняться 100%.")
            has_error = True

        if not has_error:
            train_ratio = train_ratio_input / 100.0
            val_ratio = val_ratio_input / 100.0
            exogenous_df = source_df[exog_selection] if exog_selection else None

            # Проверка данных перед генерацией признаков
            features_df = None
            train_df = None
            val_df = None
            test_df = None
            
            if selected_series is None or selected_series.empty:
                st.error("❌ Ошибка: временной ряд пустой. Проверьте этап 1.")
                has_error = True
            elif datetime_series is None or datetime_series.empty:
                st.error("❌ Ошибка: ряд с датами пустой. Проверьте этап 1.")
                has_error = True
            else:
                try:
                    # Добавляем диагностическую информацию
                    st.info(f"🔍 Диагностика: selected_series длина={len(selected_series)}, datetime_series длина={len(datetime_series)}")
                    if isinstance(selected_series.index, pd.DatetimeIndex):
                        st.info(f"🔍 selected_series имеет DatetimeIndex: {type(selected_series.index)}")
                    if isinstance(datetime_series.index, pd.DatetimeIndex):
                        st.info(f"🔍 datetime_series имеет DatetimeIndex: {type(datetime_series.index)}")
                    
                    features_df = engineer_time_features(
                        transformed_series=selected_series,
                        datetime_series=datetime_series,
                        lags=lag_values,
                        rolling_windows=rolling_values,
                        exogenous=exogenous_df,
                    )
                    
                    if features_df.empty:
                        st.error("❌ Ошибка: после генерации признаков DataFrame пустой. Проверьте данные и параметры.")
                        has_error = True
                    else:
                        train_df, val_df, test_df = chronological_split(
                            features_df,
                            train_ratio=train_ratio,
                            val_ratio=val_ratio,
                        )
                except Exception as exc:
                    st.error(f"❌ Ошибка при генерации признаков: {exc}")
                    st.exception(exc)
                    has_error = True

            if not has_error and features_df is not None:
                lab_state["stage2_completed"] = True
                lab_state["stage3_completed"] = False
                lab_state["stage4_completed"] = False
                lab_state["stage5_completed"] = False

                analysis_data.update(
                    {
                        "features_df": features_df,
                        "train_df": train_df,
                        "val_df": val_df,
                        "test_df": test_df,
                        "lag_values": lag_values,
                        "rolling_values": rolling_values,
                        "split_percentages": [train_ratio_input, val_ratio_input, test_ratio_input],
                        "exog_selection": exog_selection,
                        "target_feature_name": selected_series.name or analysis_data.get("target_column"),
                        "feature_cols": [
                            col for col in features_df.columns if col not in {"datetime", selected_series.name}
                        ],
                        "selected_pipeline_label": selected_pipeline_label,
                    }
                )

                st.success("Признаки успешно сгенерированы и выборки сформированы.")
        else:
            st.warning("Некорректные параметры. Применены значения по умолчанию.")

    if lab_state.get("stage2_completed"):
        train_df = analysis_data.get("train_df")
        val_df = analysis_data.get("val_df")
        test_df = analysis_data.get("test_df")
        features_df = analysis_data.get("features_df")
        
        # Проверка наличия данных
        if train_df is None or val_df is None or test_df is None or features_df is None:
            st.warning("⚠️ Данные выборок не найдены. Пожалуйста, нажмите кнопку 'Сформировать признаки' для создания выборок.")
        else:
            train_len = len(train_df) if hasattr(train_df, '__len__') else 0
            val_len = len(val_df) if hasattr(val_df, '__len__') else 0
            test_len = len(test_df) if hasattr(test_df, '__len__') else 0
            
            if train_len == 0 and val_len == 0 and test_len == 0:
                st.error("❌ Все выборки пустые. Возможно, после удаления пропусков (dropna) не осталось данных. Проверьте исходные данные и параметры лагов.")
            else:
                st.markdown("#### 📦 Сформированные наборы данных")
                col1, col2, col3 = st.columns(3)
                col1.metric("Train", train_len)
                col2.metric("Validation", val_len)
                col3.metric("Test", test_len)

        st.markdown("#### 🧮 Примеры признаков")
        sample_df = analysis_data.get("features_df", pd.DataFrame())
        if sample_df is not None and not sample_df.empty:
            st.dataframe(sample_df.head(10))
        else:
            st.info("Нет данных для отображения. Сформируйте признаки, нажав кнопку 'Сформировать признаки'.")

        with st.expander("ℹ️ Подробности текущей конфигурации", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔢 Лаги:**")
                lag_values = analysis_data.get("lag_values", [])
                if lag_values:
                    st.markdown(f"`{', '.join(map(str, lag_values))}`")
                else:
                    st.info("Не указаны")
                
                st.markdown("**📊 Скользящие окна:**")
                rolling_values = analysis_data.get("rolling_values", [])
                if rolling_values:
                    st.markdown(f"`{', '.join(map(str, rolling_values))}`")
                else:
                    st.info("Не указаны")
                
                st.markdown("**🔀 Пропорции разбиения:**")
                split_percentages = analysis_data.get("split_percentages", [])
                if split_percentages and len(split_percentages) == 3:
                    split_col1, split_col2, split_col3 = st.columns(3)
                    split_col1.metric("Train", f"{split_percentages[0]}%")
                    split_col2.metric("Val", f"{split_percentages[1]}%")
                    split_col3.metric("Test", f"{split_percentages[2]}%")
                else:
                    st.info("Не указаны")
            
            with col2:
                st.markdown("**🌐 Экзогенные признаки:**")
                exog_selection = analysis_data.get("exog_selection", [])
                if exog_selection:
                    for exog in exog_selection:
                        st.markdown(f"- `{exog}`")
                else:
                    st.info("Не выбраны")
                
                st.markdown("**🔧 Вариант пайплайна:**")
                pipeline_label = analysis_data.get("selected_pipeline_label", "—")
                st.markdown(f"`{pipeline_label}`")

    return analysis_data


__all__ = ["stage2", "engineer_time_features", "chronological_split"]

