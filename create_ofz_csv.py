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

def load_ofz_excel(file_path):
    raw = pd.read_excel(file_path, sheet_name=0, header=None)
    raw = raw.map(clean_value)
    header_idx = find_header_row(raw)
    columns = raw.iloc[header_idx].fillna("").astype(str).tolist()
    df = raw.iloc[header_idx + 1:].copy()
    df.columns = columns
    df = df.dropna(how="all")
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        col_lower = col.lower()
        if "дата" in col_lower:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif any(x in col_lower for x in ["код", "тип", "формат"]):
            continue
        else:
            df[col] = safe_to_numeric(df[col])
    return df.reset_index(drop=True)

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