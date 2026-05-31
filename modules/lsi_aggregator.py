import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess

PROCESSED_DIR = "data/processed"
MODULES_DIR = "modules"

def ensure_module_output(module_name: str) -> bool:
    """Запускает модуль, если его выходной файл отсутствует."""
    output_path = os.path.join(PROCESSED_DIR, f"{module_name}_output.parquet")
    if os.path.exists(output_path):
        return True
    
    print(f"  Выходной файл {module_name}_output.parquet не найден. Запускаю модуль {module_name}...")
    try:
        script_path = os.path.join(MODULES_DIR, f"{module_name}.py")
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode != 0:
            print(f"  Ошибка при запуске {module_name}: {result.stderr}")
            return False
        print(f"  Модуль {module_name} успешно выполнен")
        return os.path.exists(output_path)
    except Exception as e:
        print(f"  Не удалось запустить {module_name}: {e}")
        return False

def load_module_data(module_name: str) -> pd.DataFrame:
    """Загружает данные модуля из parquet, при необходимости сначала запуская модуль."""
    ensure_module_output(module_name)
    path = os.path.join(PROCESSED_DIR, f"{module_name}_output.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"  Загружен {module_name}: {len(df)} строк, {df['date'].min().date()} - {df['date'].max().date()}")
        return df
    else:
        print(f"  Предупреждение: {path} не найден даже после запуска")
        return pd.DataFrame()

def normalize_stress(stress_series: pd.Series) -> pd.Series:
    """Приводит stress (0-10) к 0-1 с усилением низких значений"""
    normalized = stress_series / 10.0
    # Усиливаем сигнал (возводим в степень 0.7 для поднятия низких значений)
    return normalized ** 0.7

def calculate_lsi_weighted(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Рассчитывает LSI как взвешенную сумму сигналов с сигмоидой
    """
    df = df.copy()
    
    # Собираем доступные сигналы
    available_signals = []
    total_weight = 0
    
    for module, weight in weights.items():
        col_name = f'signal_{module}'
        if col_name in df.columns and weight > 0:
            available_signals.append(df[col_name] * weight)
            total_weight += weight
            print(f"  Используется {module} с весом {weight}")
    
    if not available_signals:
        df['lsi_raw'] = 0
    else:
        df['lsi_raw'] = sum(available_signals) / total_weight
    
    # Применяем сезонный множитель M4
    if 'Seasonal_Factor' in df.columns:
        df['lsi_raw'] = df['lsi_raw'] * df['Seasonal_Factor']
        print(f"  Применён Seasonal_Factor (диапазон: {df['Seasonal_Factor'].min():.2f} - {df['Seasonal_Factor'].max():.2f})")
    
    # Агрессивная сигмоида (k=10, порог 0.4)
    k = 10.0
    df['lsi'] = 100 / (1 + np.exp(-k * (df['lsi_raw'] - 0.4)))
    df['lsi'] = df['lsi'].clip(0, 100)
    
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
    m3 = load_module_data('m3')
    m4 = load_module_data('m4')
    m5 = load_module_data('m5')
    
    # 2. Склеиваем по дате
    print("\n2. Склеивание данных...")
    dfs = []
    if not m1.empty:
        dfs.append(m1[['date', 'stress_m1']])
    if not m2.empty:
        dfs.append(m2[['date', 'stress_m2']])
    if not m3.empty:
        dfs.append(m3[['date', 'stress_m3']])
    if not m5.empty:
        dfs.append(m5[['date', 'stress_m5']])
    
    if not dfs:
        raise RuntimeError("Нет данных ни от одного модуля")
    
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on='date', how='outer')
    
    # Добавляем M4 (налоги) с флагами
    if not m4.empty:
        m4_cols = ['date', 'Tax_Week_Flag', 'End_of_Month_Flag', 'End_of_Quarter_Flag', 'Seasonal_Factor']
        result = result.merge(m4[m4_cols], on='date', how='left')
    
    result = result.sort_values('date').reset_index(drop=True)
    print(f"  Объединено: {len(result)} уникальных дат")
    
    # 3. Обработка пропусков
    print("\n3. Обработка пропусков...")
    for col in ['stress_m1', 'stress_m2', 'stress_m3', 'stress_m5']:
        if col in result.columns:
            result[col] = result[col].ffill().bfill().fillna(0)
    
    # Заполняем M4 пропуски
    for col in ['Tax_Week_Flag', 'End_of_Month_Flag', 'End_of_Quarter_Flag', 'Seasonal_Factor']:
        if col in result.columns:
            result[col] = result[col].ffill().bfill().fillna(0)
        else:
            result[col] = 0
    
    # 4. Нормализуем сигналы (с усилением)
    print("\n4. Нормализация сигналов...")
    result['signal_m1'] = normalize_stress(result['stress_m1'])
    result['signal_m2'] = normalize_stress(result['stress_m2'])
    result['signal_m3'] = normalize_stress(result['stress_m3'])
    result['signal_m5'] = normalize_stress(result['stress_m5'])
    
    # 5. Расчёт LSI с обновлёнными весами
    print("\n5. Расчёт LSI...")
    weights = {
        'm1': 0.20,
        'm2': 0.45,   # репо ЦБ — главный индикатор
        'm3': 0.20,
        'm5': 0.15,
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
    
    # Статистика по флагам M4
    if 'Tax_Week_Flag' in result.columns:
        print(f"\n  Налоговых недель: {result['Tax_Week_Flag'].sum()} дней")
        print(f"  Seasonal_Factor диапазон: {result['Seasonal_Factor'].min():.2f} — {result['Seasonal_Factor'].max():.2f}")
    
    print("\n  СТРЕСС-ЭПИЗОДЫ:")
    episodes = {
        'Декабрь 2014': ('2014-12-01', '2014-12-31'),
        'Февраль-март 2022': ('2022-02-01', '2022-03-31'),
        'Август 2023': ('2023-08-01', '2023-08-31'),
    }
    for name, (start, end) in episodes.items():
        mask = (result['date'] >= start) & (result['date'] <= end)
        if mask.any():
            ep_df = result[mask]
            print(f"    {name}: средний LSI={ep_df['lsi'].mean():.1f}, макс={ep_df['lsi'].max():.1f}")
    
    # 8. Генерация LLM-комментария (бонус)
    print("\n8. Генерация аналитического комментария...")
    try:
        from modules.llm_commentator import add_commentary_to_lsi
        
        last_row = result.iloc[-1]
        
        modules_contrib = {
            'm1': result['signal_m1'].iloc[-1] * 100,
            'm2': result['signal_m2'].iloc[-1] * 100,
            'm3': result['signal_m3'].iloc[-1] * 100,
            'm5': result['signal_m5'].iloc[-1] * 100,
        }
        
        active_flags = []
        if 'Tax_Week_Flag' in result.columns and result['Tax_Week_Flag'].iloc[-1]:
            active_flags.append('Налоговая неделя')
        if 'End_of_Month_Flag' in result.columns and result['End_of_Month_Flag'].iloc[-1]:
            active_flags.append('Конец месяца')
        if 'End_of_Quarter_Flag' in result.columns and result['End_of_Quarter_Flag'].iloc[-1]:
            active_flags.append('Конец квартала')
        
        tax_dates = []
        if not m4.empty:
            future_tax = m4[m4['date'] > pd.Timestamp.now()]['date'].head(5)
            tax_dates = future_tax.dt.strftime('%Y-%m-%d').tolist()
        
        ofz_dates = []
        if not m3.empty:
            auction_days = m3.drop_duplicates(subset=['cover_ratio'], keep='first')
            future_ofz = auction_days[auction_days['date'] > pd.Timestamp.now()]['date'].head(5)
            ofz_dates = future_ofz.dt.strftime('%Y-%m-%d').tolist()
        
        commentary = add_commentary_to_lsi(
            lsi_value=last_row['lsi'],
            status=last_row['status'],
            modules_contrib=modules_contrib,
            active_flags=active_flags,
            tax_calendar=tax_dates,
            upcoming_ofz=ofz_dates
        )
        
        print(f"\n💬 АНАЛИТИЧЕСКИЙ КОММЕНТАРИЙ:\n{commentary}")
        
        commentary_path = os.path.join(PROCESSED_DIR, "last_commentary.txt")
        with open(commentary_path, 'w', encoding='utf-8') as f:
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"LSI: {last_row['lsi']:.1f} ({last_row['status']})\n")
            f.write(f"Комментарий:\n{commentary}")
            
    except ImportError:
        print("  Модуль llm_commentator не найден. Пропускаем генерацию комментария.")
    except Exception as e:
        print(f"  Ошибка при генерации комментария: {e}")
    
    print("\n" + "=" * 50)
    return result

if __name__ == "__main__":
    run()
