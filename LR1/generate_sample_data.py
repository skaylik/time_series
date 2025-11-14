"""
Скрипт для генерации примеров данных для тестирования
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_temperature_data():
    """Генерация температурных данных (дневные)"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Тренд: постепенное потепление
    trend = np.linspace(15, 17, n)
    
    # Годовая сезонность
    seasonal_yearly = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    
    # Случайный шум
    noise = np.random.normal(0, 2, n)
    
    # Итоговая температура
    temperature = trend + seasonal_yearly + noise
    
    # Дополнительные признаки
    humidity = 50 + 20 * np.sin(2 * np.pi * np.arange(n) / 365.25) + np.random.normal(0, 5, n)
    wind_speed = 10 + 5 * np.random.randn(n)
    wind_speed = np.clip(wind_speed, 0, None)  # Скорость не может быть отрицательной
    
    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed
    })
    
    df.to_csv('data/temperature_daily.csv', index=False)
    print("✅ Создан файл: data/temperature_daily.csv")
    return df


def generate_sales_data():
    """Генерация данных о продажах (месячные)"""
    np.random.seed(123)
    
    dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='M')
    n = len(dates)
    
    # Тренд роста
    trend = np.linspace(1000, 2000, n)
    
    # Годовая сезонность (пик продаж в конце года)
    seasonal = 300 * np.sin(2 * np.pi * np.arange(n) / 12 - np.pi/2)
    
    # Случайный шум
    noise = np.random.normal(0, 100, n)
    
    # Продажи
    sales = trend + seasonal + noise
    sales = np.clip(sales, 0, None)
    
    # Маркетинговые расходы
    marketing_spend = 100 + 50 * np.random.randn(n) + 0.05 * sales
    marketing_spend = np.clip(marketing_spend, 0, None)
    
    # Количество клиентов
    customers = (sales / 50 + np.random.normal(0, 5, n)).astype(int)
    customers = np.clip(customers, 0, None)
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'marketing_spend': marketing_spend,
        'customers': customers
    })
    
    df.to_csv('data/sales_monthly.csv', index=False)
    print("✅ Создан файл: data/sales_monthly.csv")
    return df


def generate_energy_data():
    """Генерация данных об энергопотреблении (часовые)"""
    np.random.seed(456)
    
    dates = pd.date_range(start='2023-01-01', end='2023-12-31 23:00:00', freq='H')
    n = len(dates)
    
    # Общий тренд
    trend = np.linspace(100, 110, n)
    
    # Суточная сезонность (пик утром и вечером)
    hour_of_day = np.array([d.hour for d in dates])
    daily_seasonal = 20 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/2)
    
    # Недельная сезонность (выше в будни)
    day_of_week = np.array([d.dayofweek for d in dates])
    weekly_seasonal = 10 * (day_of_week < 5).astype(float)  # Выше в понедельник-пятницу
    
    # Сезонность по месяцам (выше зимой и летом из-за отопления/кондиционирования)
    month = np.array([d.month for d in dates])
    monthly_seasonal = 15 * np.abs(np.sin(2 * np.pi * month / 12))
    
    # Случайный шум
    noise = np.random.normal(0, 5, n)
    
    # Энергопотребление
    energy = trend + daily_seasonal + weekly_seasonal + monthly_seasonal + noise
    energy = np.clip(energy, 0, None)
    
    # Температура
    temperature = 20 + 10 * np.sin(2 * np.pi * np.arange(n) / (24*365)) + np.random.normal(0, 3, n)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'energy_consumption': energy,
        'temperature': temperature
    })
    
    df.to_csv('data/energy_hourly.csv', index=False)
    print("✅ Создан файл: data/energy_hourly.csv")
    return df


def generate_stock_data():
    """Генерация данных о цене акций (дневные)"""
    np.random.seed(789)
    
    dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
    # Исключаем выходные
    dates = dates[dates.dayofweek < 5]
    n = len(dates)
    
    # Случайное блуждание с трендом
    returns = np.random.normal(0.001, 0.02, n)
    price = 100 * np.exp(np.cumsum(returns))
    
    # Объем торгов
    volume = np.random.lognormal(15, 0.5, n)
    
    # Максимальные и минимальные цены
    high = price * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = price * (1 - np.abs(np.random.normal(0, 0.01, n)))
    
    # Цена открытия и закрытия
    open_price = price * (1 + np.random.normal(0, 0.005, n))
    close_price = price
    
    df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close_price,
        'volume': volume
    })
    
    df.to_csv('data/stock_daily.csv', index=False)
    print("✅ Создан файл: data/stock_daily.csv")
    return df


def generate_website_traffic_data():
    """Генерация данных о трафике сайта (дневные)"""
    np.random.seed(321)
    
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Тренд роста
    trend = np.linspace(1000, 3000, n)
    
    # Недельная сезонность (меньше в выходные)
    day_of_week = np.array([d.dayofweek for d in dates])
    weekly_seasonal = -500 * (day_of_week >= 5).astype(float)
    
    # Случайный шум
    noise = np.random.normal(0, 200, n)
    
    # Посетители
    visitors = trend + weekly_seasonal + noise
    visitors = np.clip(visitors, 100, None).astype(int)
    
    # Просмотры страниц (в среднем 3-5 на посетителя)
    page_views = (visitors * np.random.uniform(3, 5, n)).astype(int)
    
    # Конверсия (2-5%)
    conversions = (visitors * np.random.uniform(0.02, 0.05, n)).astype(int)
    
    df = pd.DataFrame({
        'date': dates,
        'visitors': visitors,
        'page_views': page_views,
        'conversions': conversions
    })
    
    df.to_csv('data/website_traffic_daily.csv', index=False)
    print("✅ Создан файл: data/website_traffic_daily.csv")
    return df


if __name__ == '__main__':
    print("🚀 Генерация примеров данных...")
    print()
    
    generate_temperature_data()
    generate_sales_data()
    generate_energy_data()
    generate_stock_data()
    generate_website_traffic_data()
    
    print()
    print("✨ Все примеры данных успешно созданы в папке 'data/'!")
    print()
    print("Доступные файлы:")
    print("  - data/temperature_daily.csv - Температурные данные (дневные)")
    print("  - data/sales_monthly.csv - Данные о продажах (месячные)")
    print("  - data/energy_hourly.csv - Энергопотребление (часовые)")
    print("  - data/stock_daily.csv - Цены акций (дневные)")
    print("  - data/website_traffic_daily.csv - Трафик сайта (дневные)")

