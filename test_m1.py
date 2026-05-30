from modules.m1_reserves import fetch_reserves, fetch_ruonia, process_m1

# ── Тест 1: просто запустить и посмотреть на данные ──────────────────────

print("=== Тест fetch_reserves ===")
reserves = fetch_reserves()
print(reserves.head())
print(reserves.tail())
print(f"Строк: {len(reserves)}")
print(f"Колонки: {reserves.columns.tolist()}")
print(f"Пропуски:\n{reserves.isna().sum()}")

print("\n=== Тест fetch_ruonia ===")
ruonia = fetch_ruonia()
print(ruonia.head())
print(ruonia.tail())
print(f"Строк: {len(ruonia)}")

print("\n=== Тест process_m1 ===")
result = process_m1(reserves, ruonia)
print(result.head(10))
print(result.tail(10))
print(f"Строк итого: {len(result)}")
print(f"Пропуски:\n{result.isna().sum()}")

# ── Тест 2: проверить диапазон значений ──────────────────────────────────

print("\n=== Проверка значений ===")
print(f"spread: min={result['spread'].min():.1f}, max={result['spread'].max():.1f}")
print(f"mad_score_spread: min={result['mad_score_spread'].min():.2f}, max={result['mad_score_spread'].max():.2f}")
print(f"mad_score_ruonia: min={result['mad_score_ruonia'].min():.2f}, max={result['mad_score_ruonia'].max():.2f}")
print(f"stress_m1: min={result['stress_m1'].min():.2f}, max={result['stress_m1'].max():.2f}")
print(f"Flag_EndOfPeriod=1: {result['Flag_EndOfPeriod'].sum()} дней")

# ── Тест 3: проверить известные стресс-эпизоды ───────────────────────────
# В эти периоды stress_m1 должен быть заметно выше среднего

print("\n=== Стресс-эпизоды ===")
episodes = {
    'Декабрь 2014': ('2014-12-01', '2014-12-31'),
    'Февраль 2022': ('2022-02-01', '2022-03-31'),
    'Август 2023':  ('2023-07-01', '2023-09-30'),
}
overall_mean = result['stress_m1'].mean()
print(f"Среднее stress_m1 по всей истории: {overall_mean:.3f}")

for name, (start, end) in episodes.items():
    ep = result[(result['date'] >= start) & (result['date'] <= end)]
    if len(ep) == 0:
        print(f"{name}: нет данных")
        continue
    ep_mean = ep['stress_m1'].mean()
    ep_max  = ep['stress_m1'].max()
    print(f"{name}: mean={ep_mean:.3f}, max={ep_max:.3f}  "
          f"({'↑ ВЫШЕ среднего' if ep_mean > overall_mean else '↓ ниже среднего'})")
    
missing_ruonia = result[result['ruonia_rate'].isna()]
print(missing_ruonia[['date']].head(20))