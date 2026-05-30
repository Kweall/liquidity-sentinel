# test_m2.py
from modules.m2_repo import run, fetch_repo_soap, fetch_repo_full_history, fetch_keyrate_soap, process_m2

# ── Тест 1: один аукцион (известная дата) ────────────────────────────────

print("=== Тест fetch_repo_soap (одна дата) ===")
df_single = fetch_repo_soap("2026-05-19", "2026-05-19")
print(df_single)
print(f"Колонки: {df_single.columns.tolist()}")
assert 'bid' in df_single.columns or 'demand_volume' in df_single.columns, "Нет колонки со спросом"
assert len(df_single) > 0, "Пустой датафрейм"
print("OK")

# ── Тест 2: ключевая ставка ───────────────────────────────────────────────

print("\n=== Тест fetch_keyrate_soap ===")
keyrate = fetch_keyrate_soap()
print(keyrate.head())
print(keyrate.tail())
print(f"Строк: {len(keyrate)}")
print(f"Пропуски:\n{keyrate.isna().sum()}")
assert len(keyrate) > 0, "Пустой датафрейм"
assert keyrate['key_rate'].min() > 0, "Ставка не может быть <= 0"
assert keyrate['key_rate'].max() < 50, "Ставка > 50% — что-то не так"
print("OK")

# ── Тест 3: полная история репо ───────────────────────────────────────────

print("\n=== Тест fetch_repo_full_history ===")
repo_raw = fetch_repo_full_history(from_year=2010)
print(repo_raw.head())
print(repo_raw.tail())
print(f"Строк всего: {len(repo_raw)}")
print(f"Типы аукционов (term_days):\n{repo_raw['term_days'].value_counts()}")
print(f"Пропуски:\n{repo_raw.isna().sum()}")

# Фильтр 7-дневных
repo_7d = repo_raw[repo_raw['term_days'] == 7].copy()
print(f"\n7-дневных аукционов: {len(repo_7d)}")
print(f"Период: {repo_7d['date'].min().date()} — {repo_7d['date'].max().date()}")
print("OK")

# ── Тест 4: process_m2 ────────────────────────────────────────────────────

print("\n=== Тест process_m2 ===")
result = process_m2(repo_7d, keyrate)
print(result.head(10))
print(result.tail(10))
print(f"\nСтрок итого: {len(result)}")
print(f"Пропуски:\n{result.isna().sum()}")

print("\n=== Проверка значений ===")
print(f"cover_ratio:           min={result['cover_ratio'].min():.2f}, max={result['cover_ratio'].max():.2f}")
print(f"rate_spread:           min={result['rate_spread'].min():.2f}, max={result['rate_spread'].max():.2f}")
print(f"mad_score_cover:       min={result['mad_score_cover'].min():.2f}, max={result['mad_score_cover'].max():.2f}")
print(f"mad_score_rate_spread: min={result['mad_score_rate_spread'].min():.2f}, max={result['mad_score_rate_spread'].max():.2f}")
print(f"Flag_Demand=1: {result['Flag_Demand'].sum()} дней")

# ── Тест 5: стресс-эпизоды ────────────────────────────────────────────────

print("\n=== Стресс-эпизоды (cover_ratio) ===")
episodes = {
    'Декабрь 2014': ('2014-12-01', '2014-12-31'),
    'Февраль 2022': ('2022-02-01', '2022-03-31'),
    'Август 2023':  ('2023-07-01', '2023-09-30'),
}
overall_mean = result['mad_score_cover'].mean()
print(f"Среднее mad_score_cover по всей истории: {overall_mean:.3f}")

for name, (start, end) in episodes.items():
    ep = result[(result['date'] >= start) & (result['date'] <= end)]
    if len(ep) == 0:
        print(f"{name}: нет данных")
        continue
    ep_cover = ep['cover_ratio'].mean()
    ep_mad   = ep['mad_score_cover'].mean()
    print(f"{name}: cover_ratio mean={ep_cover:.2f}, mad_score mean={ep_mad:.3f}  "
          f"({'↑ ВЫШЕ среднего' if ep_mad > overall_mean else '↓ ниже среднего'})")