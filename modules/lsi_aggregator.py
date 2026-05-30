import pandas as pd
import numpy as np
import os
from datetime import datetime

PROCESSED_DIR = "data/processed"

def load_module_data(module_name: str) -> pd.DataFrame:
    """Загружает данные модуля из parquet"""
    path = os.path.join(PROCESSED_DIR, f"{module_name}_output.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"  Загружен {module_name}: {len(df)} строк, {df['date'].min().date()} - {df['date'].max().date()}")
        return df
    else:
        print(f"  Предупреждение: {path} не найден")
        return pd.DataFrame()

def normalize_stress(stress_series: pd.Series) -> pd.Series:
    """Приводит stress (0-10) к 0-1"""
    return stress_series / 10.0

def calculate_lsi_weighted(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Рассчитывает LSI как взвешенную сумму сигналов с сигмоидой
    weights: {'m1': 0.3, 'm2': 0.3, 'm3': 0.15, 'm4': 0.15, 'm5': 0.1}
    """
    df = df.copy()
    
    # Собираем доступные сигналы
    available_signals = []
    total_weight = 0
    
    for module, weight in weights.items():
        col_name = f'signal_{module}'
        if col_name in df.columns:
            available_signals.append(df[col_name] * weight)
            total_weight += weight
            print(f"  Используется {module} с весом {weight}")
    
    if not available_signals:
        df['lsi_raw'] = 0
    else:
        # Суммируем с нормализацией по весу
        df['lsi_raw'] = sum(available_signals) / total_weight
    
    # Применяем сигмоиду для нелинейности
    k = 6.0
    df['lsi'] = 100 / (1 + np.exp(-k * (df['lsi_raw'] - 0.5)))
    
    # Цветовая зона
    df['status'] = 'ЗЕЛЁНЫЙ'
    df.loc[df['lsi'] >= 40, 'status'] = 'ЖЁЛТЫЙ'
    df.loc[df['lsi'] >= 70, 'status'] = 'КРАСНЫЙ'
    
    return df

def run():
    print("=" * 50)
    print("Агрегатор LSI запущен")
    print(f"Время: {datetime.now()}")
    print("=" * 50)
    
    # 1. Загружаем все доступные модули
    print("\n1. Загрузка данных модулей:")
    m1 = load_module_data('m1')
    m2 = load_module_data('m2')
    
    # 2. Склеиваем по дате
    print("\n2. Склеивание данных...")
    dfs = []
    if not m1.empty:
        dfs.append(m1[['date', 'stress_m1']])
    if not m2.empty:
        dfs.append(m2[['date', 'stress_m2']])
    
    if not dfs:
        raise RuntimeError("Нет данных ни от одного модуля")
    
    # Начинаем с первого
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on='date', how='outer')
    
    result = result.sort_values('date').reset_index(drop=True)
    print(f"  Объединено: {len(result)} уникальных дат")
    
    # 3. Заполняем пропуски (ИСПРАВЛЕННАЯ ЧАСТЬ)
    print("\n3. Обработка пропусков...")
    for col in ['stress_m1', 'stress_m2']:
        if col in result.columns:
            # Используем ffill() и bfill() вместо fillna(method=...)
            result[col] = result[col].ffill().bfill().fillna(0)
    
    # 4. Нормализуем сигналы
    print("\n4. Нормализация сигналов...")
    result['signal_m1'] = normalize_stress(result['stress_m1'])
    result['signal_m2'] = normalize_stress(result['stress_m2'])
    
    # Заглушки для будущих модулей (пока 0)
    result['signal_m3'] = 0
    result['signal_m4'] = 0
    result['signal_m5'] = 0
    
    # 5. Рассчитываем LSI (пока только M1 и M2, веса равные)
    print("\n5. Расчёт LSI...")
    weights = {
        'm1': 0.5,
        'm2': 0.5,
        'm3': 0.0,
        'm4': 0.0,
        'm5': 0.0
    }
    result = calculate_lsi_weighted(result, weights)
    
    # 6. Сохраняем результат
    print("\n6. Сохранение...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, "lsi_output.parquet")
    result.to_parquet(output_path, index=False)
    
    # 7. Статистика
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ АГРЕГАЦИИ:")
    print(f"  Период: {result['date'].min().date()} — {result['date'].max().date()}")
    print(f"  Всего дней: {len(result)}")
    print(f"  LSI мин: {result['lsi'].min():.1f}")
    print(f"  LSI макс: {result['lsi'].max():.1f}")
    print(f"  LSI средний: {result['lsi'].mean():.1f}")
    print(f"  Красных дней (LSI >= 70): {(result['lsi'] >= 70).sum()}")
    print(f"  Жёлтых дней (40-70): {((result['lsi'] >= 40) & (result['lsi'] < 70)).sum()}")
    print(f"  Зелёных дней (<40): {(result['lsi'] < 40).sum()}")
    print("=" * 50)
    
    return result

if __name__ == "__main__":
    run()