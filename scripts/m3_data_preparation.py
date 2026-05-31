from pathlib import Path
import pandas as pd
import numpy as np
import re

MISSING_VALUES = {"-***", "-****", "***", "****", "-", ""}

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


def detect_schema(columns):
    cols = [str(c).lower() for c in columns]
    if any("формат" in c for c in cols) or len(columns) >= 16:
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
    """
    Для современных файлов (2024+) колонки имеют фиксированные индексы:
    0 - Дата
    1 - Формат* (пропускаем)
    2 - Код выпуска
    3 - Тип бумаги
    4 - Дата погашения
    5 - Дней до погашения
    6 - Объем предложения
    7 - Цена отсечения
    8 - Цена средневзвешенная
    9 - Доходность по цене отсечения
    10 - Доходность по средневзвешенной цене
    11 - Совокупный объем спроса по номиналу
    12 - Объем размещения по номиналу
    13 - Объем выручки
    14 - Коэффициент удовлетворения спроса
    """
    out = pd.DataFrame()
    
    # Используем индексы для современных файлов
    out['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    out['code'] = df.iloc[:, 2].astype(str).str.strip()
    out['type'] = df.iloc[:, 3]
    out['maturity_date'] = df.iloc[:, 4]
    out['days'] = safe_to_numeric(df.iloc[:, 5])
    out['offer_volume'] = safe_to_numeric(df.iloc[:, 6])
    out['cut_price'] = safe_to_numeric(df.iloc[:, 7])
    out['avg_price'] = safe_to_numeric(df.iloc[:, 8])
    out['yield_cut'] = safe_to_numeric(df.iloc[:, 9])
    out['yield_avg'] = safe_to_numeric(df.iloc[:, 10])
    out['demand'] = safe_to_numeric(df.iloc[:, 11])
    out['placement'] = safe_to_numeric(df.iloc[:, 12])
    out['revenue'] = safe_to_numeric(df.iloc[:, 13])
    out['placement_ratio'] = safe_to_numeric(df.iloc[:, 14])
    
    return out


def extract_year(file_name: str):
    match = re.search(r"(20\d{2})", file_name)
    return int(match.group(1)) if match else None


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
    out["source_year"] = extract_year(Path(file_path).name)

    return out.reset_index(drop=True)


def build_ofz_clean_csv(files, output_path):
    """
    files: список путей к Excel-файлам (.xls, .xlsx)
    output_path: Path объекта для сохранения итогового CSV
    """
    all_dfs = []
    for f in sorted(files, key=lambda x: extract_year(x.name) if extract_year(x.name) is not None else 9999):
        print(f"Обработка {f.name}...")
        try:
            df = load_ofz_excel(f)
            all_dfs.append(df)
        except Exception as e:
            print(f"Ошибка в {f.name}: {e}")

    if not all_dfs:
        raise RuntimeError("Нет данных для объединения")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Сохранено {len(combined)} строк в {output_path}")