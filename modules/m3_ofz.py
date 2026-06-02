import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.utils import mad_score, validate_module_output

# Импортируем функцию обновления данных из scripts
try:
    from scripts.m3_update_ofz_data import main as update_ofz_data
except ImportError:
    print("Ошибка: не найден модуль scripts.m3_update_ofz_data. Убедитесь, что структура папок корректна.")
    raise

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OFZ_CSV = os.path.join(RAW_DIR, "ofz_clean.csv")
DYNAMIC_CSV = os.path.join(RAW_DIR, "dynamic.csv")

def ensure_data_updated():
    """Проверяет наличие необходимых файлов и при их отсутствии запускает процедуру загрузки и подготовки данных."""
    ofz_path = Path(RAW_DIR) / "ofz_clean.csv"
    dynamic_path = Path(RAW_DIR) / "dynamic.csv"
    
    if not ofz_path.exists() or not dynamic_path.exists():
        print("Не найдены данные для M3 (ofz_clean.csv или dynamic.csv). Запускаем их загрузку и подготовку...")
        update_ofz_data()
        print("Подготовка данных завершена.\n")
    else:
        print("Данные для M3 уже существуют. (Для принудительного обновления удалите файлы в data/raw/ и запустите снова.)")

def fetch_ofz_auctions() -> pd.DataFrame:
    if not os.path.exists(OFZ_CSV):
        raise FileNotFoundError(f"Не найден {OFZ_CSV}. Попробуйте удалить файлы в data/raw/ и запустить заново.")

    df = pd.read_csv(OFZ_CSV, encoding='utf-8-sig')
    df.columns = df.columns.str.lower()
    
    col_map = {
        'date': 'date',
        'isin': ['code', 'код  выпуска'],
        'days_to_maturity': ['days', 'дней до погашения'],
        'offer': ['offer_volume', 'объем предложения'],
        'demand': 'demand',
        'placement': 'placement',
        'yield_auction': ['yield_avg', 'доходность по средневзвешенной цене']
    }
    
    selected = {}
    for target, possible in col_map.items():
        if isinstance(possible, list):
            found = None
            for p in possible:
                if p in df.columns:
                    found = p
                    break
            if found is None:
                raise KeyError(f"Колонка для '{target}' не найдена. Доступны: {list(df.columns)}")
            selected[target] = found
        else:
            if possible not in df.columns:
                raise KeyError(f"Колонка '{possible}' не найдена")
            selected[target] = possible
    
    result = pd.DataFrame()
    result['date'] = pd.to_datetime(df[selected['date']], errors='coerce')
    result['isin'] = df[selected['isin']].astype(str).str.strip()
    result['days_to_maturity'] = pd.to_numeric(df[selected['days_to_maturity']], errors='coerce')
    result['offer'] = pd.to_numeric(df[selected['offer']], errors='coerce')
    result['demand'] = pd.to_numeric(df[selected['demand']], errors='coerce')
    result['placement'] = pd.to_numeric(df[selected['placement']], errors='coerce')
    result['yield_auction'] = pd.to_numeric(df[selected['yield_auction']], errors='coerce')
    
    result = result.dropna(subset=['date', 'isin', 'offer', 'demand', 'placement', 'yield_auction', 'days_to_maturity'])
    result = result[~result['isin'].str.match(r'^\d+$', na=False)]
    result = result[result['isin'] != '']
    result = result.sort_values('date').reset_index(drop=True)
    print(f"Загружено аукционов: {len(result)}")
    return result

def fetch_dynamic_curve() -> pd.DataFrame:
    if not os.path.exists(DYNAMIC_CSV):
        raise FileNotFoundError(f"Файл {DYNAMIC_CSV} не найден.")

    with open(DYNAMIC_CSV, 'r', encoding='cp1251') as f:
        lines = f.readlines()

    header_line = None
    for i, line in enumerate(lines):
        if 'tradedate' in line.lower() and 'B1' in line:
            header_line = i
            break
    if header_line is None:
        raise ValueError("Не найдена строка заголовков в dynamic.csv")

    data_lines = lines[header_line:]
    csv_str = "".join(data_lines)
    df = pd.read_csv(pd.io.common.StringIO(csv_str), sep=';', decimal=',', encoding='cp1251')
    required_cols = ['tradedate', 'B1', 'B2', 'B3', 'T1', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Колонка '{col}' отсутствует в dynamic.csv")
    df = df[required_cols].copy()
    df['tradedate'] = pd.to_datetime(df['tradedate'], dayfirst=True, errors='coerce')
    for col in required_cols[1:]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=['tradedate'])
    df = df.sort_values('tradedate').reset_index(drop=True)
    return df

def nelson_siegel_svensson_yield(t_years, B1, B2, B3, T1):
    if t_years == 0:
        return B1
    tau1 = T1 if T1 > 0 else 1.0
    term2 = (1 - np.exp(-t_years/tau1)) / (t_years/tau1)
    term3 = term2 - np.exp(-t_years/tau1)
    return B1 + B2 * term2 + B3 * term3

def get_benchmark_yield(curve_df, auction_date, days_to_maturity):
    curve_df = curve_df[curve_df['tradedate'] <= auction_date]
    if curve_df.empty:
        return np.nan
    params = curve_df.iloc[-1]
    t_years = days_to_maturity / 365.0
    B1 = params['B1'] / 100.0
    B2 = params['B2'] / 100.0
    B3 = params['B3'] / 100.0
    T1 = params['T1'] if params['T1'] > 0 else 1.0
    return nelson_siegel_svensson_yield(t_years, B1, B2, B3, T1)

def process_m3(auctions_df: pd.DataFrame, curve_df: pd.DataFrame) -> pd.DataFrame:
    benchmarks = []
    for _, row in auctions_df.iterrows():
        by = get_benchmark_yield(curve_df, row['date'], row['days_to_maturity'])
        benchmarks.append(by)
    auctions_df = auctions_df.copy()
    auctions_df['benchmark_yield'] = benchmarks
    auctions_df['yield_spread'] = auctions_df['yield_auction'] - auctions_df['benchmark_yield']
    auctions_df['cover_ratio'] = auctions_df['demand'] / auctions_df['offer']

    daily_agg = auctions_df.groupby('date', as_index=False).agg({
        'cover_ratio': 'mean',
        'yield_spread': 'mean'
    })

    date_range = pd.date_range(daily_agg['date'].min(), pd.Timestamp.today(), freq='D')
    daily = pd.DataFrame({'date': date_range})
    daily = daily.merge(daily_agg, on='date', how='left')
    daily['cover_ratio'] = daily['cover_ratio'].ffill().fillna(1.0)
    daily['yield_spread'] = daily['yield_spread'].ffill().fillna(0.0)

    daily['mad_score_cover'] = mad_score(daily['cover_ratio'])
    daily['mad_score_yield'] = mad_score(daily['yield_spread'])

    daily['Flag_Nedospros'] = (daily['cover_ratio'] < 1.2).astype(int)
    daily['Flag_Perespros'] = (daily['cover_ratio'] > 2.0).astype(int)

    pos_cover = daily['mad_score_cover'].clip(lower=0)
    pos_yield = daily['mad_score_yield'].clip(lower=0)
    stress = (pos_cover + pos_yield) / 2 * 2
    daily['stress_m3'] = stress.clip(0, 10)

    result = daily[['date', 'cover_ratio', 'yield_spread', 'mad_score_cover',
                    'mad_score_yield', 'Flag_Nedospros', 'Flag_Perespros', 'stress_m3']].copy()
    result = result.drop_duplicates(subset=['date'])
    validate_module_output(result, 'M3')
    return result

def run() -> pd.DataFrame:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    ensure_data_updated()

    print("Загрузка аукционов ОФЗ...")
    auctions = fetch_ofz_auctions()
    print("Загрузка кривой бескупонной доходности (dynamic.csv)...")
    curve = fetch_dynamic_curve()
    print("Обработка M3...")
    result = process_m3(auctions, curve)

    out_path = os.path.join(PROCESSED_DIR, "m3_output.parquet")
    result.to_parquet(out_path, index=False)
    print(f"M3 сохранён: {len(result)} строк, с {result['date'].min().date()} по {result['date'].max().date()}")
    return result

if __name__ == "__main__":
    df = run()
    print(df.tail())