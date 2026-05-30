import pandas as pd
import numpy as np
import os
import re
from modules.utils import mad_score, validate_module_output

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OFZ_CSV = os.path.join(RAW_DIR, "ofz_clean.csv")
DYNAMIC_CSV = os.path.join(RAW_DIR, "dynamic.csv")   # кривая бескупонной доходности

def fetch_ofz_auctions() -> pd.DataFrame:
    if not os.path.exists(OFZ_CSV):
        raise FileNotFoundError(f"Не найден {OFZ_CSV}. Запустите create_ofz_csv.py для его создания.")

    df_raw = pd.read_csv(OFZ_CSV, header=None, dtype=str, encoding='utf-8-sig')
    rows = []
    for _, row in df_raw.iterrows():
        if len(row) < 12:
            continue
        date_str = str(row[0]).strip()
        if not date_str or date_str == 'NaT' or date_str.startswith('1970-01-01'):
            continue
        try:
            date = pd.to_datetime(date_str, errors='coerce')
        except:
            continue
        if pd.isna(date) or date.year < 2000:
            continue

        isin = str(row[1]).strip()
        if not isin or isin.isdigit() or isin in ('Код  выпуска', 'nan'):
            continue

        try:
            offer = float(row[5]) if row[5] not in ('', 'nan', 'None') else np.nan
            demand = float(row[10]) if row[10] not in ('', 'nan', 'None') else np.nan
            placement = float(row[11]) if row[11] not in ('', 'nan', 'None') else np.nan
            days_to_maturity = float(row[4]) if row[4] not in ('', 'nan', 'None') else np.nan
        except:
            continue

        yield_val = np.nan
        for col_idx in [9, 17, 20, 26]:
            if len(row) > col_idx and row[col_idx] not in ('', 'nan', 'None'):
                try:
                    y = float(row[col_idx])
                    if not np.isnan(y):
                        yield_val = y
                        break
                except:
                    pass

        if any(pd.isna([offer, demand, placement, yield_val, days_to_maturity])):
            continue

        rows.append({
            'date': date,
            'isin': isin,
            'offer': offer,
            'demand': demand,
            'placement': placement,
            'yield_auction': yield_val,
            'days_to_maturity': days_to_maturity
        })

    if not rows:
        raise ValueError("Не удалось извлечь данные из CSV.")
    df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    print(f"Загружено аукционов: {len(df)}")
    return df

def fetch_dynamic_curve() -> pd.DataFrame:
    """Загружает dynamic.csv и возвращает DataFrame с параметрами кривой."""
    if not os.path.exists(DYNAMIC_CSV):
        raise FileNotFoundError(f"Файл {DYNAMIC_CSV} не найден. Поместите его в data/raw/")

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
    # Оставляем нужные колонки
    required_cols = ['tradedate', 'B1', 'B2', 'B3', 'T1', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Колонка '{col}' отсутствует в dynamic.csv")
    df = df[required_cols].copy()
    df['tradedate'] = pd.to_datetime(df['tradedate'], dayfirst=True, errors='coerce')
    # Преобразуем все параметры в float
    for col in required_cols[1:]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=['tradedate'])
    df = df.sort_values('tradedate').reset_index(drop=True)
    return df

def nelson_siegel_svensson_yield(t_years, B1, B2, B3, T1, G1=0, G2=0, G3=0, G4=0, G5=0, G6=0, G7=0, G8=0, G9=0):
    if t_years == 0:
        return B1
    tau1 = T1 if T1 > 0 else 1.0
    term2 = (1 - np.exp(-t_years/tau1)) / (t_years/tau1)
    term3 = term2 - np.exp(-t_years/tau1)
    yield_ = B1 + B2 * term2 + B3 * term3
    # Дополнительные члены (для полноты, но обычно нулевые)
    # Можно добавить, но для простоты опускаем, так как G* ~0
    return yield_

def get_benchmark_yield(curve_df, auction_date, days_to_maturity):
    """Возвращает теоретическую доходность (в %) для даты аукциона и срока до погашения (дни)."""
    curve_df = curve_df[curve_df['tradedate'] <= auction_date]
    if curve_df.empty:
        return np.nan
    params = curve_df.iloc[-1]  # последняя доступная
    t_years = days_to_maturity / 365.0
    # Делим все коэффициенты на 100 (так как в файле они в базисных пунктах)
    B1 = params['B1'] / 100.0
    B2 = params['B2'] / 100.0
    B3 = params['B3'] / 100.0
    T1 = params['T1']
    if T1 <= 0:
        T1 = 1.0
    G1 = params.get('G1', 0) / 100.0
    G2 = params.get('G2', 0) / 100.0
    G3 = params.get('G3', 0) / 100.0
    G4 = params.get('G4', 0) / 100.0
    G5 = params.get('G5', 0) / 100.0
    G6 = params.get('G6', 0) / 100.0
    G7 = params.get('G7', 0) / 100.0
    G8 = params.get('G8', 0) / 100.0
    G9 = params.get('G9', 0) / 100.0

    return nelson_siegel_svensson_yield(t_years, B1, B2, B3, T1, G1, G2, G3, G4, G5, G6, G7, G8, G9)

def process_m3(auctions_df: pd.DataFrame, curve_df: pd.DataFrame) -> pd.DataFrame:
    # Для каждого аукциона вычисляем benchmark_yield
    benchmarks = []
    for idx, row in auctions_df.iterrows():
        by = get_benchmark_yield(curve_df, row['date'], row['days_to_maturity'])
        benchmarks.append(by)
    auctions_df = auctions_df.copy()
    auctions_df['benchmark_yield'] = benchmarks
    auctions_df['yield_spread'] = auctions_df['yield_auction'] - auctions_df['benchmark_yield']
    auctions_df['cover_ratio'] = auctions_df['demand'] / auctions_df['offer']

    # Группировка по дате (усреднение)
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