"""
Скрипт для генерации "грязных" данных с дубликатами, пропусками и выбросами
для демонстрации возможностей предобработки
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_messy_temperature_data():
    """
    Генерация температурных данных с проблемами качества:
    - Дубликаты по времени
    - Пропущенные значения
    - Выбросы
    - Нарушение монотонности
    """
    np.random.seed(42)
    
    # Базовый ряд
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Тренд + сезонность + шум
    trend = np.linspace(15, 17, n)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise = np.random.normal(0, 2, n)
    temperature = trend + seasonal + noise
    
    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature
    })
    
    # 1. Добавляем дубликаты (5% строк)
    n_duplicates = int(n * 0.05)
    duplicate_indices = np.random.choice(n, n_duplicates, replace=False)
    duplicates = df.iloc[duplicate_indices].copy()
    # Небольшое изменение значений у дубликатов
    duplicates['temperature'] += np.random.normal(0, 0.5, n_duplicates)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # 2. Добавляем пропуски (10% значений)
    n_missing = int(len(df) * 0.10)
    missing_indices = np.random.choice(len(df), n_missing, replace=False)
    df.loc[missing_indices, 'temperature'] = np.nan
    
    # 3. Добавляем выбросы (3% значений)
    n_outliers = int(len(df) * 0.03)
    outlier_indices = np.random.choice(len(df), n_outliers, replace=False)
    # Выбросы - очень высокие или очень низкие значения
    outlier_direction = np.random.choice([-1, 1], n_outliers)
    df.loc[outlier_indices, 'temperature'] = df.loc[outlier_indices, 'temperature'] + outlier_direction * np.random.uniform(20, 40, n_outliers)
    
    # 4. Нарушаем монотонность - несколько записей с нарушенным порядком дат
    n_shuffle = 10
    shuffle_indices = np.random.choice(len(df)-1, n_shuffle, replace=False)
    for idx in shuffle_indices:
        # Меняем местами соседние даты
        df.loc[idx, 'date'], df.loc[idx+1, 'date'] = df.loc[idx+1, 'date'], df.loc[idx, 'date']
    
    # Не сортируем, чтобы сохранить нарушение монотонности
    
    df.to_csv('data/messy_temperature.csv', index=False)
    print("✅ Создан файл: data/messy_temperature.csv")
    print(f"   - Строк: {len(df)}")
    print(f"   - Дубликаты: ~{n_duplicates}")
    print(f"   - Пропуски: ~{n_missing}")
    print(f"   - Выбросы: ~{n_outliers}")
    print(f"   - Нарушений монотонности: {n_shuffle}")
    return df


def generate_messy_sales_data():
    """
    Генерация данных о продажах с проблемами
    """
    np.random.seed(123)
    
    # Базовый ряд
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Тренд + недельная сезонность
    trend = np.linspace(1000, 2000, n)
    day_of_week = np.array([d.dayofweek for d in dates])
    weekly_seasonal = -200 * (day_of_week >= 5).astype(float)  # Меньше продаж в выходные
    noise = np.random.normal(0, 100, n)
    sales = trend + weekly_seasonal + noise
    sales = np.clip(sales, 0, None)
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'customers': (sales / 50 + np.random.normal(0, 10, n)).clip(0).astype(int)
    })
    
    # Добавляем проблемы
    
    # 1. Дубликаты
    n_duplicates = int(n * 0.03)
    duplicate_indices = np.random.choice(n, n_duplicates, replace=False)
    duplicates = df.iloc[duplicate_indices].copy()
    duplicates['sales'] += np.random.normal(0, 50, n_duplicates)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # 2. Пропуски в разных колонках
    n_missing_sales = int(len(df) * 0.08)
    missing_indices_sales = np.random.choice(len(df), n_missing_sales, replace=False)
    df.loc[missing_indices_sales, 'sales'] = np.nan
    
    n_missing_customers = int(len(df) * 0.05)
    missing_indices_customers = np.random.choice(len(df), n_missing_customers, replace=False)
    df.loc[missing_indices_customers, 'customers'] = np.nan
    
    # 3. Выбросы (например, распродажи или ошибки учета)
    n_outliers = int(len(df) * 0.02)
    outlier_indices = np.random.choice(len(df), n_outliers, replace=False)
    # Случайные очень высокие продажи
    df.loc[outlier_indices, 'sales'] = df.loc[outlier_indices, 'sales'] * np.random.uniform(3, 5, n_outliers)
    
    df.to_csv('data/messy_sales.csv', index=False)
    print("✅ Создан файл: data/messy_sales.csv")
    print(f"   - Строк: {len(df)}")
    print(f"   - Дубликаты: ~{n_duplicates}")
    print(f"   - Пропуски (sales): ~{n_missing_sales}")
    print(f"   - Пропуски (customers): ~{n_missing_customers}")
    print(f"   - Выбросы: ~{n_outliers}")
    return df


def generate_sensor_data_with_gaps():
    """
    Генерация данных с датчика с нерегулярной частотой записи
    """
    np.random.seed(789)
    
    # Создаем нерегулярный временной ряд
    start_date = datetime(2023, 1, 1)
    dates = []
    current_date = start_date
    
    # Генерируем даты с нерегулярными интервалами
    for _ in range(1000):
        # Интервал от 30 секунд до 1 часа
        interval = timedelta(seconds=np.random.randint(30, 3600))
        current_date += interval
        dates.append(current_date)
    
    n = len(dates)
    
    # Значения датчика
    values = 100 + np.cumsum(np.random.normal(0, 2, n))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'sensor_value': values
    })
    
    # Добавляем проблемы
    
    # 1. Дубликаты (датчик записал дважды)
    n_duplicates = 20
    duplicate_indices = np.random.choice(n, n_duplicates, replace=False)
    duplicates = df.iloc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # 2. Большие пропуски (датчик не работал)
    gap_starts = np.random.choice(len(df), 5, replace=False)
    for gap_start in gap_starts:
        gap_size = np.random.randint(10, 30)
        gap_end = min(gap_start + gap_size, len(df))
        df.loc[gap_start:gap_end, 'sensor_value'] = np.nan
    
    # 3. Выбросы (сбои датчика)
    n_outliers = 15
    outlier_indices = np.random.choice(len(df), n_outliers, replace=False)
    df.loc[outlier_indices, 'sensor_value'] = np.random.uniform(-1000, 1000, n_outliers)
    
    df.to_csv('data/messy_sensor.csv', index=False)
    print("✅ Создан файл: data/messy_sensor.csv")
    print(f"   - Строк: {len(df)}")
    print(f"   - Нерегулярная частота: от 30 сек до 1 часа")
    print(f"   - Дубликаты: {n_duplicates}")
    print(f"   - Пропуски: множественные")
    print(f"   - Выбросы: {n_outliers}")
    return df


if __name__ == '__main__':
    print("🚀 Генерация 'грязных' данных для тестирования предобработки...")
    print()
    
    generate_messy_temperature_data()
    print()
    generate_messy_sales_data()
    print()
    generate_sensor_data_with_gaps()
    
    print()
    print("✨ Все 'грязные' датасеты созданы в папке 'data/'!")
    print()
    print("Эти файлы идеально подходят для демонстрации:")
    print("  - Удаления дубликатов")
    print("  - Обработки пропусков")
    print("  - Обнаружения и обработки выбросов")
    print("  - Проверки монотонности")
    print("  - Ресемплирования")

