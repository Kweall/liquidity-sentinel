import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

def mad_score(series: pd.Series, window: int = 756) -> pd.Series:
    shifted = series.shift(1)
    rolling_median = shifted.rolling(window, min_periods=60,).median()
    rolling_mad = shifted.rolling(window, min_periods=60,).apply(
        lambda x: np.median(np.abs(x - np.median(x))),
        raw=True,
    )

    denominator = rolling_mad * 1.4826
    rolling_std = shifted.rolling(window, min_periods=60,).std()
    denominator = denominator.where(denominator > 0.01, rolling_std * 0.5)
    denominator = denominator.where(denominator > 0.01, 1.0)

    raw_score = (series - rolling_median) / denominator
    score = np.tanh(raw_score / 5) * 5

    return score

def validate_module_output(df: pd.DataFrame, module_name: str) -> bool:
    assert 'date' in df.columns, f"{module_name}: нет колонки date"
    assert is_datetime64_any_dtype(df['date']), "{module_name}: date не datetime ({df['date'].dtype})"
    assert df['date'].is_monotonic_increasing, f"{module_name}: даты не отсортированы"
    assert not df['date'].duplicated().any(), f"{module_name}: дубли дат"
    return True

import requests
import os
from datetime import datetime

def download_file(url: str, save_path: str, force_update: bool = False) -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    file_exists = os.path.exists(save_path)
    if file_exists:
        age_hours = (datetime.now().timestamp() - 
                     os.path.getmtime(save_path)) / 3600
        is_fresh = age_hours < 24
    
    if not file_exists or not is_fresh or force_update:
        print(f"Скачиваю {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Сохранено: {save_path}")
    else:
        print(f"Файл актуален, пропускаю: {save_path}")
    
    return save_path