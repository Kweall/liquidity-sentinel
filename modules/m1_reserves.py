import requests
import pandas as pd
import numpy as np
import os
from modules.utils import mad_score, validate_module_output, download_file

RESERVES_URL = (
    "https://www.cbr.ru/vfs/hd_base/RReserves/"
    "required_reserves_table.xlsx"
)
RUONIA_URL = (
    "https://www.cbr.ru/hd_base/ruonia/dynamics/"
    "?UniDbQuery.Posted=True"
    "&UniDbQuery.From=01.01.2010"
    "&UniDbQuery.To=31.12.2030"
)
RUONIA_XLSX_URL = (
    "https://www.cbr.ru/Queries/UniDbQuery/DownloadExcel/14315"
    "?Posted=True"
    "&From=01.01.2010"
    "&To=31.12.2030"
    "&FromDate=01%2F01%2F2010"
    "&ToDate=12%2F31%2F2030"
)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def fetch_reserves() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "required_reserves.xlsx")
    download_file(RESERVES_URL, path)

    df_raw = pd.read_excel(path, header=None)
    data_rows = []
    for _, row in df_raw.iterrows():
        if isinstance(row[0], pd.Timestamp) or (
            hasattr(row[0], 'year') and not isinstance(row[0], float)
        ):
            try:
                data_rows.append({
                    'period_start':   pd.Timestamp(row[0]),
                    'actual_reserves': pd.to_numeric(row[1], errors='coerce'),
                    'required_avg':    pd.to_numeric(row[2], errors='coerce'),
                    'required_acc':    pd.to_numeric(row[3], errors='coerce'),
                })
            except Exception:
                pass

    df = pd.DataFrame(data_rows).sort_values('period_start').reset_index(drop=True)
    df['period_end'] = df['period_start'].shift(-1) - pd.Timedelta(days=1)
    df.loc[df.index[-1], 'period_end'] = pd.Timestamp.today().normalize()

    return df


def fetch_ruonia() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "ruonia.xlsx")

    try:
        download_file(RUONIA_XLSX_URL, path)
        df = pd.read_excel(path)
    except Exception:
        print("XLSX не сработал, парсим HTML...")
        df = _fetch_ruonia_html()

    df = df.rename(columns={'DT': 'date', 'ruo': 'ruonia_rate'})
    df = df[['date', 'ruonia_rate']]

    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df['ruonia_rate'] = pd.to_numeric(
        df['ruonia_rate'].astype(str)
        .str.replace(',', '.').str.replace(' ', ''),
        errors='coerce'
    )
    df = df.dropna().sort_values('date').reset_index(drop=True)
    return df


def _fetch_ruonia_html() -> pd.DataFrame:
    from bs4 import BeautifulSoup

    url = (
        "https://www.cbr.ru/hd_base/ruonia/dynamics/"
        "?UniDbQuery.Posted=True"
        "&UniDbQuery.From=01.01.2010"
        "&UniDbQuery.To=25.05.2026"
    )
    resp = requests.get(url, timeout=30,
                        headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, 'html.parser')

    table = soup.find('table')
    rows = []
    for tr in table.find_all('tr')[1:]:
        cells = [td.text.strip() for td in tr.find_all('td')]
        if len(cells) >= 2:
            rows.append({'date': cells[0], 'ruonia_rate': cells[1]})

    return pd.DataFrame(rows)

def build_daily_reserve_curve(reserves_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    today = pd.Timestamp.today().normalize()

    for _, row in reserves_df.iterrows():
        if pd.isna(row["actual_reserves"]):
            continue

        dates = pd.bdate_range(row["period_start"], row["period_end"])

        total_days = len(dates)
        spread = (row["actual_reserves"] - row["required_avg"])

        for i, d in enumerate(dates):
            if d > today:
                break

            progress = i / max(total_days - 1, 1)
            dynamic_spread = spread * (0.65 + 0.35 * progress)

            rows.append({"date": d,"spread": dynamic_spread, "days_to_period_end": total_days - i - 1,})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).astype('datetime64[ns]')
    return (df.sort_values("date").reset_index(drop=True))

def process_m1(reserves_df: pd.DataFrame,ruonia_df: pd.DataFrame) -> pd.DataFrame:

    daily = build_daily_reserve_curve(reserves_df)
    daily = daily.merge(ruonia_df,on="date",how="left")
    daily["ruonia_rate"] = (daily["ruonia_rate"].ffill(limit=3))

    daily["mad_score_spread"] = mad_score(daily["spread"])
    daily["mad_score_ruonia"] = mad_score(daily["ruonia_rate"])

    daily["Flag_EndOfPeriod"] = (daily["days_to_period_end"] <= 5).astype(int)

    spread_pos = (daily["mad_score_spread"].clip(lower=0))
    ruonia_pos = (daily["mad_score_ruonia"].clip(lower=0))
    interaction = np.sqrt(spread_pos * ruonia_pos)

    stress = (0.6 * interaction+ 0.2 * spread_pos + 0.2 * ruonia_pos)
    stress *= np.where(daily["Flag_EndOfPeriod"] == 1,1.25, 1.0)

    daily["stress_m1"] = stress.clip(0, 10)

    result = daily[[
        "date",
        "spread",
        "ruonia_rate",
        "mad_score_spread",
        "mad_score_ruonia",
        "stress_m1",
        "Flag_EndOfPeriod",
    ]].copy()

    result = result.dropna(subset=["mad_score_spread","mad_score_ruonia",])
    
    validate_module_output(result, "M1")

    return result


def run() -> pd.DataFrame:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    reserves = fetch_reserves()
    ruonia = fetch_ruonia()
    result = process_m1(reserves, ruonia)

    result = result.dropna(subset=['stress_m1'])

    result.to_parquet(
        os.path.join(PROCESSED_DIR, "m1_output.parquet"), index=False
    )
    print(f"М1 готов: {len(result)} строк, "
          f"с {result['date'].min().date()} по {result['date'].max().date()}")
    return result