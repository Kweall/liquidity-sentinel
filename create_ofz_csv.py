from pathlib import Path
import pandas as pd
import numpy as np

MISSING_VALUES = {"-***", "-****", "***", "****", "-", ""}

def clean_value(x):
    if isinstance(x, str):
        x = x.strip()
        if x in MISSING_VALUES:
            return np.nan
        return x
    return x

def safe_to_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def find_header_row(df):
    for i in range(len(df)):
        row = df.iloc[i].astype(str).str.lower()
        row_text = " ".join([x for x in row.values if str(x) != "nan"])
        score = 0
        if "дата" in row_text:
            score += 1
        if "код" in row_text:
            score += 1
        if "тип" in row_text:
            score += 1
        if "аукцион" in row_text or "выпуск" in row_text:
            score += 1
        if score >= 2:
            return i
    raise ValueError("Header row not found")

def is_index_row(row):
    vals = [str(x).strip() for x in row.values if pd.notna(x)]

    if len(vals) < 10:
        return False

    try:
        nums = [int(v) for v in vals]
    except:
        return False

    return nums == list(range(1, len(nums) + 1))

LEGACY_MAP = {
    1: "date",
    2: "code",
    3: "type",
    4: "maturity_date",
    5: "days",
    6: "offer_volume",
    7: "cut_price",
    8: "avg_price",
    9: "yield_cut",
    10: "yield_avg",
    11: "demand",
    12: "placement",
    13: "revenue",
    14: "activity",
    15: "placement_ratio",
}

MODERN_MAP = {
    "date": "Дата",
    "format": "Формат*",
    "code": "Код  выпуска",
    "type": "Тип бумаги**",
    "maturity_date": "Дата погашения",
    "days": "Дней до погашения",
    "offer_volume": "Объем предложения",
    "cut_price": "Цена отсечения",
    "avg_price": "Цена средневзвешенная",
    "yield_cut": "Доходность по цене отсечения***",
    "yield_avg": "Доходность по средневзвешенной цене***",
    "demand": "Совокупный объем спроса по номиналу",
    "placement": "Объем размещения по номиналу",
    "revenue": "Объем выручки",
    "placement_ratio": "Коэффициент удовлетворения спроса на аукционе",
}

def detect_schema(columns):
    cols = [str(c).lower() for c in columns]

    if any("формат" in c for c in cols):
        return "modern"

    return "legacy"

def extract_by_position(df):
    out = pd.DataFrame()

    for i, col in enumerate(df.columns[:15], start=1):
        key = LEGACY_MAP.get(i)
        if key:
            out[key] = df[col]

    return out


def extract_by_name(df):
    out = pd.DataFrame()

    for key, col_name in MODERN_MAP.items():
        if col_name in df.columns:
            out[key] = df[col_name]
        else:
            out[key] = np.nan

    return out

def load_ofz_excel(file_path):
    raw = pd.read_excel(file_path, sheet_name=0, header=None)
    raw = raw.map(clean_value)

    header_idx = find_header_row(raw)

    header = raw.iloc[header_idx].fillna("").astype(str).tolist()
    df = raw.iloc[header_idx + 1:].copy()

    df.columns = header
    df = df.dropna(how="all")

    df = df[~df.apply(is_index_row, axis=1)]

    schema = detect_schema(df.columns)

    if schema == "modern":
        out = extract_by_name(df)
    else:
        out = extract_by_position(df)

    if "date" in out:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    for c in out.columns:
        if c not in ["date", "code", "type", "format"]:
            out[c] = safe_to_numeric(out[c])

    out["source_file"] = Path(file_path).name

    return out.reset_index(drop=True)

def main():
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "ofz_clean.csv"
    
    files = []
    for year in range(2015, 2027):
        xlsx_path = raw_dir / f"ofz_{year}.xlsx"
        if xlsx_path.exists():
            files.append(xlsx_path)
    xls_path = raw_dir / "ofz_2015.xls"
    if xls_path.exists() and xls_path not in files:
        files.append(xls_path)
    
    if not files:
        raise FileNotFoundError("Нет файлов ofz_2015.xls или ofz_201*.xlsx в data/raw/")
    
    all_dfs = []
    for f in sorted(files):
        print(f"Обработка {f.name}...")
        try:
            df = load_ofz_excel(str(f))
            df["source_file"] = f.name
            all_dfs.append(df)
        except Exception as e:
            print(f"Ошибка в {f.name}: {e}")
    
    if not all_dfs:
        raise RuntimeError("Не удалось загрузить ни одного файла")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nСохранено {len(combined)} строк в {output_path}")
    print(f"Первые 3 строки:\n{combined.head(3).to_string()}")

if __name__ == "__main__":
    main()