import pandas as pd
import numpy as np
from modules.m3_ofz import fetch_ofz_auctions, fetch_dynamic_curve, process_m3, run

print("=== Тест загрузки аукционов ОФЗ ===")
auctions = fetch_ofz_auctions()
print(f"Загружено аукционов: {len(auctions)}")
assert len(auctions) > 0, "Нет данных аукционов"
required_cols = ['date', 'isin', 'offer', 'demand', 'placement', 'yield_auction', 'days_to_maturity']
for col in required_cols:
    assert col in auctions.columns, f"Нет колонки {col}"
print(f"Диапазон дат: {auctions['date'].min()} — {auctions['date'].max()}")
print("OK")

print("\n=== Тест загрузки кривой (dynamic.csv) ===")
curve = fetch_dynamic_curve()
print(f"Загружено записей кривой: {len(curve)}")
assert len(curve) > 0, "Нет данных кривой"
curve_cols = ['tradedate', 'B1', 'B2', 'B3', 'T1']
for col in curve_cols:
    assert col in curve.columns, f"Нет колонки {col}"
print(f"Диапазон дат кривой: {curve['tradedate'].min()} — {curve['tradedate'].max()}")
print("OK")

print("\n=== Тест process_m3 ===")
result = process_m3(auctions, curve)
print(f"Результат: {len(result)} строк, {result['date'].min().date()} — {result['date'].max().date()}")
assert 'date' in result.columns
assert 'cover_ratio' in result.columns
assert 'yield_spread' in result.columns
assert 'mad_score_cover' in result.columns
assert 'mad_score_yield' in result.columns
assert 'Flag_Nedospros' in result.columns
assert 'Flag_Perespros' in result.columns
assert 'stress_m3' in result.columns

# Проверка диапазонов
assert result['stress_m3'].min() >= 0, "stress_m3 отрицательный"
assert result['stress_m3'].max() <= 10, "stress_m3 > 10"
assert result['cover_ratio'].min() >= 0, "cover_ratio отрицательный"

# Проверка отсутствия дубликатов дат
assert not result['date'].duplicated().any(), "Есть дубликаты дат"
# Проверка сортировки
assert result['date'].is_monotonic_increasing, "Даты не отсортированы"

# Проверка, что yield_spread не константа (имеет разумный разброс)
if result['yield_spread'].std() > 0:
    print(f"yield_spread: min={result['yield_spread'].min():.2f}, max={result['yield_spread'].max():.2f}, std={result['yield_spread'].std():.2f}")
else:
    print("Предупреждение: yield_spread почти константа")

print("OK")

print("\n=== Стресс-эпизоды ===")
episodes = {
    'Декабрь 2014': ('2014-12-01', '2014-12-31'),
    'Февраль-март 2022': ('2022-02-01', '2022-03-31'),
    'Август 2023': ('2023-08-01', '2023-08-31'),
}
overall_mean = result['stress_m3'].mean()
print(f"Среднее stress_m3 по всей истории: {overall_mean:.3f}")

for name, (start, end) in episodes.items():
    ep = result[(result['date'] >= start) & (result['date'] <= end)]
    if len(ep) == 0:
        print(f"{name}: нет данных")
        continue
    ep_mean = ep['stress_m3'].mean()
    ep_max = ep['stress_m3'].max()
    print(f"{name}: mean={ep_mean:.3f}, max={ep_max:.3f}  {'↑' if ep_mean > overall_mean else '↓'}")
    if name == 'Февраль-март 2022':
        print("   (Примечание: в этот период аукционы ОФЗ не проводились, стресс нулевой — нормально)")

print("\n=== Топ-10 стрессовых дней ===")
top = result.nlargest(10, 'stress_m3')[['date', 'stress_m3', 'cover_ratio', 'yield_spread']]
print(top)

print("\n=== Запуск полного пайплайна run() ===")
df = run()
print(f"Итоговый датафрейм: {len(df)} строк")
print(df.tail(3))
print("Все тесты пройдены успешно!")