import pandas as pd
import numpy as np
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.utils import validate_module_output

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TAX_CALENDAR_CSV = os.path.join(RAW_DIR, "tax_calendar.csv")

def ensure_tax_calendar():
    if not os.path.exists(TAX_CALENDAR_CSV):
        print("  Налоговый календарь не найден. Запускаю его загрузку...")
        script_path = Path(__file__).parent.parent / "scripts" / "m4_fetch_tax_calendar.py"
        if not script_path.exists():
            raise FileNotFoundError(f"Скрипт {script_path} не найден.")
        result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("Не удалось загрузить налоговый календарь")
        print("  Загрузка завершена.")

def load_tax_calendar() -> pd.DataFrame:
    ensure_tax_calendar()
    df = pd.read_csv(TAX_CALENDAR_CSV, parse_dates=['date'], encoding='utf-8')
    print(f"  Загружено налоговых событий: {len(df)}")
    return df

def build_calendar_flags(df_tax: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    date_range = pd.date_range(start_date, end_date, freq='D')
    calendar = pd.DataFrame({'date': date_range})
    
    calendar['End_of_Month_Flag'] = (calendar['date'] == calendar['date'].dt.to_period('M').dt.end_time.dt.normalize()).astype(int)
    
    quarter_end_months = [3, 6, 9, 12]
    calendar['End_of_Quarter_Flag'] = (calendar['date'].dt.month.isin(quarter_end_months) & 
                                        (calendar['date'] == calendar['date'].dt.to_period('Q').dt.end_time.dt.normalize())).astype(int)
    
    tax_event_dates = set(df_tax['date'].dt.normalize())
    calendar['Tax_Week_Flag'] = 0
    
    for tax_date in tax_event_dates:
        week_start = tax_date - timedelta(days=7)
        week_end = tax_date + timedelta(days=7)
        mask = (calendar['date'] >= week_start) & (calendar['date'] <= week_end)
        calendar.loc[mask, 'Tax_Week_Flag'] = 1
    
    return calendar

def calculate_seasonal_factor(calendar: pd.DataFrame) -> pd.DataFrame:
    """Рассчитывает Seasonal_Factor с градацией по типу налогового периода"""
    calendar = calendar.copy()
    calendar['Seasonal_Factor'] = 1.0
    
    # Базовая налоговая неделя
    mask_tax = calendar['Tax_Week_Flag'] == 1
    calendar.loc[mask_tax, 'Seasonal_Factor'] = 1.1
    
    # Конец квартала — дополнительное давление
    mask_quarter = calendar['End_of_Quarter_Flag'] == 1
    calendar.loc[mask_quarter & mask_tax, 'Seasonal_Factor'] = 1.25
    
    # Конец года (декабрь) — максимальное давление
    mask_year_end = (calendar['date'].dt.month == 12) & mask_quarter
    calendar.loc[mask_year_end & mask_tax, 'Seasonal_Factor'] = 1.4
    
    return calendar

def process_m4(df_m1: pd.DataFrame = None, df_m2: pd.DataFrame = None) -> pd.DataFrame:
    if df_m1 is None:
        df_m1 = pd.DataFrame()
    if df_m2 is None:
        df_m2 = pd.DataFrame()
    
    start_date = pd.Timestamp('2014-01-01')
    end_date = pd.Timestamp.today()
    
    print("Загрузка налогового календаря...")
    df_tax = load_tax_calendar()
    
    print("Построение календаря флагов...")
    calendar = build_calendar_flags(df_tax, start_date, end_date)
    
    print("Расчёт Seasonal_Factor...")
    calendar = calculate_seasonal_factor(calendar)
    
    result = calendar[['date', 'Tax_Week_Flag', 'End_of_Month_Flag', 'End_of_Quarter_Flag', 'Seasonal_Factor']].copy()
    result['date'] = pd.to_datetime(result['date']).astype('datetime64[ns]')
    
    validate_module_output(result, 'M4')
    return result

def run(df_m1: pd.DataFrame = None, df_m2: pd.DataFrame = None) -> pd.DataFrame:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    if df_m1 is None:
        m1_path = os.path.join(PROCESSED_DIR, "m1_output.parquet")
        if os.path.exists(m1_path):
            df_m1 = pd.read_parquet(m1_path)
            print(f"Загружен M1: {len(df_m1)} строк")
        else:
            df_m1 = pd.DataFrame()
    
    if df_m2 is None:
        m2_path = os.path.join(PROCESSED_DIR, "m2_output.parquet")
        if os.path.exists(m2_path):
            df_m2 = pd.read_parquet(m2_path)
            print(f"Загружен M2: {len(df_m2)} строк")
        else:
            df_m2 = pd.DataFrame()
    
    result = process_m4(df_m1, df_m2)
    
    output_path = os.path.join(PROCESSED_DIR, "m4_output.parquet")
    result.to_parquet(output_path, index=False)
    print(f"\nМ4 готов: {len(result)} строк")
    print(f"Период: {result['date'].min().date()} — {result['date'].max().date()}")
    print(f"Налоговых недель: {result['Tax_Week_Flag'].sum()}")
    print(f"Seasonal_Factor распределение:\n{result['Seasonal_Factor'].value_counts().sort_index()}")
    
    return result

if __name__ == "__main__":
    run()
