import requests
import pandas as pd
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import numpy as np
import time
import os
from modules.utils import mad_score, validate_module_output

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

SECINFO_URL = "https://www.cbr.ru/secinfo/secinfo.asmx"
DAILYINFO_URL = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"

def _soap_request(url: str, action: str, body: str) -> ET.Element | None:
    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "SOAPAction": f'"http://web.cbr.ru/{action}"',
    }
    envelope = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                   xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                   xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
      <soap:Body>{body}</soap:Body>
    </soap:Envelope>"""

    resp = requests.post(url, data=envelope.encode("utf-8"),
                         headers=headers, timeout=30)
    resp.raise_for_status()
    return ET.fromstring(resp.text)


def fetch_keyrate_soap(from_year: int = 2013) -> pd.DataFrame:
    start = f"{from_year}-01-01T00:00:00"
    end   = datetime.today().strftime("%Y-%m-%dT00:00:00")

    body = f"""<KeyRateXML xmlns="http://web.cbr.ru/">
        <fromDate>{start}</fromDate>
        <ToDate>{end}</ToDate>
    </KeyRateXML>"""

    root = _soap_request(DAILYINFO_URL, "KeyRateXML", body)
    result_node = root.find(".//{http://web.cbr.ru/}KeyRateXMLResult")

    records = []
    for kr in result_node.findall(".//KR"):
        dt   = kr.findtext("DT")
        rate = kr.findtext("Rate")
        if dt and rate:
            records.append({"date_from": dt, "key_rate": float(rate)})

    df = pd.DataFrame(records)
    df["date_from"] = df["date_from"].apply(
        lambda x: pd.to_datetime(x, errors="coerce").replace(tzinfo=None)
    )
    df["date_from"] = pd.to_datetime(df["date_from"]).dt.normalize()
    df["key_rate"]  = pd.to_numeric(df["key_rate"], errors="coerce")
    return df.dropna().sort_values("date_from").reset_index(drop=True)


def keyrate_to_daily(keyrate_df: pd.DataFrame,
                     date_range: pd.DatetimeIndex) -> pd.DataFrame:
    daily = pd.DataFrame({'date': date_range})
    daily = daily.sort_values('date')

    keyrate_sorted = keyrate_df.sort_values('date_from')
    daily = pd.merge_asof(
        daily,
        keyrate_sorted.rename(columns={'date_from': 'date'}),
        on='date',
        direction='backward'
    )
    return daily


def fetch_repo_soap(start_date: str, end_date: str) -> pd.DataFrame:
    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "SOAPAction": '"http://web.cbr.ru/REPO"',
    }
    envelope = f"""<?xml version="1.0" encoding="utf-8"?>
    <soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                   xmlns:xsd="http://www.w3.org/2001/XMLSchema"
                   xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
      <soap:Body>
        <REPO xmlns="http://web.cbr.ru/">
          <DateFrom>{start_date}</DateFrom>
          <DateTo>{end_date}</DateTo>
        </REPO>
      </soap:Body>
    </soap:Envelope>"""

    resp = requests.post(SECINFO_URL, data=envelope.encode("utf-8"),
                         headers=headers, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    result_node = root.find(".//{http://web.cbr.ru/}REPOResult")
    if result_node is None:
        return pd.DataFrame()

    records = []
    for rp in result_node.findall(".//RP"):
        records.append({child.tag: child.text for child in rp})

    return pd.DataFrame(records)


def fetch_repo_full_history(from_year: int = 2010) -> pd.DataFrame:
    cache_path = "data/raw/repo_cache.csv"
    os.makedirs("data/raw", exist_ok=True)

    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        last_date = cached["date"].max()
        fetch_from = last_date - timedelta(days=7)
        print(f"Кэш найден до {last_date.date()}, докачиваю с {fetch_from.date()}")
    else:
        cached = pd.DataFrame()
        fetch_from = datetime(from_year, 1, 1)
        print(f"Кэша нет, качаю с {fetch_from.date()}")

    today = datetime.today()
    all_chunks = []

    current = fetch_from
    while current < today:
        chunk_end = min(current + timedelta(days=180), today)
        start_str = current.strftime("%Y-%m-%d")
        end_str   = chunk_end.strftime("%Y-%m-%d")

        try:
            df_chunk = fetch_repo_soap(start_str, end_str)
            if df_chunk is not None and len(df_chunk) > 0:
                all_chunks.append(df_chunk)
                print(f"  {start_str} — {end_str}: {len(df_chunk)} аукционов")
            else:
                print(f"  {start_str} — {end_str}: пусто")
        except Exception as e:
            print(f"  {start_str} — {end_str}: ОШИБКА {e}")

        current = chunk_end + timedelta(days=1)
        time.sleep(0.5)

    if not all_chunks:
        return cached

    new_data = pd.concat(all_chunks, ignore_index=True)
    new_data = _clean_repo_df(new_data)

    result = pd.concat([cached, new_data], ignore_index=True)
    result = result.drop_duplicates(subset=["date", "term_days"]).sort_values("date")
    result.to_csv(cache_path, index=False)
    print(f"Сохранено в кэш: {len(result)} строк")
    return result


def _clean_repo_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "Dt":        "date",
        "day_repo":  "term_days",
        "bid":       "demand_volume",
        "avg_deal":  "placement_volume",
        "cut_off_rate": "cutoff_rate",
        "avg_yield": "avg_rate",
    })

    df["date"] = df["date"].apply(
        lambda x: pd.to_datetime(x, errors="coerce").replace(tzinfo=None)
    )
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["term_days"]        = pd.to_numeric(df["term_days"], errors="coerce")
    df["demand_volume"]    = pd.to_numeric(df["demand_volume"], errors="coerce")
    df["placement_volume"] = pd.to_numeric(df["placement_volume"], errors="coerce")
    df["cutoff_rate"]      = pd.to_numeric(df["cutoff_rate"], errors="coerce")
    df["avg_rate"]         = pd.to_numeric(df["avg_rate"], errors="coerce")

    df["cutoff_rate"] = df["cutoff_rate"].fillna(df["avg_rate"])
    df["placement_volume"] = df["placement_volume"].replace(0, pd.NA)

    df = df[["date", "term_days", "demand_volume",
             "placement_volume", "cutoff_rate", "avg_rate"]]
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def process_m2(repo_df, keyrate_df):
    repo_all = repo_df.copy()
    repo_7d = repo_df[repo_df['term_days'] == 7].copy()

    df = repo_7d.copy()
    df['cover_ratio'] = df['demand_volume'] / df['placement_volume']

    date_range = pd.bdate_range(df['date'].min(), df['date'].max())
    keyrate_daily = keyrate_to_daily(keyrate_df, date_range)
    df = df.merge(keyrate_daily[['date', 'key_rate']], on='date', how='left')
    df['rate_spread'] = df['cutoff_rate'] - df['key_rate']
    df['Flag_Demand'] = (df['cover_ratio'] > 2.0).astype(int)

    all_dates = pd.DataFrame({'date': date_range})
    result = all_dates.merge(df, on='date', how='left')

    result['key_rate'] = result['key_rate'].ffill()
    result['cover_ratio'] = result['cover_ratio'].ffill().fillna(1.0)
    result['rate_spread'] = result['rate_spread'].ffill().fillna(0.0)
    result['Flag_Demand'] = result['Flag_Demand'].fillna(0)

    result['mad_score_cover'] = mad_score(result['cover_ratio'])
    result['mad_score_rate_spread'] = mad_score(result['rate_spread'])

    emergency_dates = set(repo_all[repo_all['term_days'] == 1]['date'].dt.date)
    result['Flag_Emergency_1d'] = (
        result['date'].dt.date.isin(emergency_dates).astype(int)
    )

    result['stress_m2'] = (
        0.5 * result['mad_score_cover'].clip(lower=0) +
        0.5 * result['mad_score_rate_spread'].clip(lower=0)
    )
    result['stress_m2'] *= np.where(result['Flag_Emergency_1d'] == 1, 1.3, 1.0)
    result['stress_m2'] = result['stress_m2'].clip(0, 10)
    result = result.dropna(subset=['key_rate'])
    result['date'] = result['date'].astype('datetime64[ns]')

    result = result[[
        'date', 'cover_ratio', 'rate_spread', 'key_rate',
        'mad_score_cover', 'mad_score_rate_spread',
        'stress_m2', 'Flag_Demand', 'Flag_Emergency_1d'
    ]]

    validate_module_output(result, 'M2')
    return result

def run() -> pd.DataFrame:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    repo_raw = fetch_repo_full_history(from_year=2010)
    repo_7d = repo_raw[repo_raw['term_days'] == 7].copy()
    print(f"Найдено 7-дневных аукционов: {len(repo_7d)}")

    keyrate = fetch_keyrate_soap()

    result = process_m2(repo_7d, keyrate)

    result.to_parquet(
        os.path.join(PROCESSED_DIR, "m2_output.parquet"), index=False
    )
    print(f"М2 готов: {len(result)} строк, "
          f"с {result['date'].min().date()} по {result['date'].max().date()}")
    return result