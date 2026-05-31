import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

PROCESSED_DIR = "data/processed"

def load_module_data(module_name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, f"{module_name}_output.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()

def create_ground_truth(df):
    """
    Создаёт целевую переменную на основе исторических кризисов
    Это прокси для реальных данных ЦБ (ликвидность банковского сектора)
    """
    # Базовый уровень стресса (низкий)
    y = np.zeros(len(df))
    
    # Известные кризисные периоды (из ТЗ)
    crises = [
        ('2014-12-01', '2014-12-31', 90),   # декабрь 2014 — высокий стресс
        ('2022-02-01', '2022-03-31', 85),   # февраль-март 2022 — высокий стресс
        ('2023-08-01', '2023-08-31', 80),   # август 2023 — средний стресс
    ]
    
    for start, end, stress_level in crises:
        mask = (df['date'] >= start) & (df['date'] <= end)
        y[mask] = stress_level
    
    # Добавляем сезонные стрессы (налоговые периоды) — более низкие
    tax_mask = df['Tax_Week_Flag'] == 1
    y[tax_mask] = np.maximum(y[tax_mask], 40)
    
    # Добавляем стрессы от оттока казначейства
    if 'Flag_Budget_Drain' in df.columns:
        drain_mask = df['Flag_Budget_Drain'] == 1
        y[drain_mask] = np.maximum(y[drain_mask], 50)
    
    return y / 100.0  # Нормализуем 0-1

def run():
    print("=" * 50)
    print("Агрегатор LSI (ML-версия с RandomForest + SHAP) запущен")
    print(f"Время: {datetime.now()}")
    print("=" * 50)
    
    # 1. Загружаем данные
    print("\n1. Загрузка данных модулей:")
    m1 = load_module_data('m1')
    m2 = load_module_data('m2')
    m3 = load_module_data('m3')
    m4 = load_module_data('m4')
    m5 = load_module_data('m5')
    
    # 2. Склеиваем
    print("\n2. Склеивание данных...")
    dfs = []
    if not m1.empty:
        dfs.append(m1[['date', 'stress_m1']])
    if not m2.empty:
        dfs.append(m2[['date', 'stress_m2']])
    if not m3.empty:
        dfs.append(m3[['date', 'stress_m3']])
    if not m5.empty:
        dfs.append(m5[['date', 'stress_m5', 'Flag_Budget_Drain']])
    
    if not dfs:
        raise RuntimeError("Нет данных ни от одного модуля")
    
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on='date', how='outer')
    
    if not m4.empty:
        m4_cols = ['date', 'Tax_Week_Flag', 'Seasonal_Factor']
        result = result.merge(m4[m4_cols], on='date', how='left')
    
    result = result.sort_values('date').reset_index(drop=True)
    print(f"  Объединено: {len(result)} уникальных дат")
    
    # 3. Заполняем пропуски
    print("\n3. Обработка пропусков...")
    for col in ['stress_m1', 'stress_m2', 'stress_m3', 'stress_m5']:
        if col in result.columns:
            result[col] = result[col].ffill().bfill().fillna(0)
        else:
            result[col] = 0
    
    for col in ['Flag_Budget_Drain', 'Tax_Week_Flag', 'Seasonal_Factor']:
        if col in result.columns:
            result[col] = result[col].fillna(0)
        else:
            result[col] = 0
    
    # 4. Подготовка признаков
    print("\n4. Подготовка признаков для ML...")
    X = pd.DataFrame()
    X['m1_signal'] = result['stress_m1'] / 10.0
    X['m2_signal'] = result['stress_m2'] / 10.0
    X['m3_signal'] = result['stress_m3'] / 10.0
    X['m5_signal'] = result['stress_m5'] / 10.0
    X['seasonal_factor'] = result['Seasonal_Factor']
    X['tax_week_flag'] = result['Tax_Week_Flag']
    X['budget_drain'] = result['Flag_Budget_Drain']
    
    # 5. Создаём целевую переменную (ground truth)
    print("\n5. Создание целевой переменной (ground truth)...")
    y = create_ground_truth(result)
    
    print(f"  Целевая переменная: мин={y.min():.2f}, макс={y.max():.2f}, сред={y.mean():.2f}")
    
    # 6. Обучаем RandomForest
    print("\n6. Обучение RandomForest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=20,
        random_state=42
    )
    model.fit(X, y)
    
    # Предсказываем
    y_pred = model.predict(X)
    lsi_ml = y_pred * 100
    
    # 7. Feature importance
    print("\n7. Feature importance (важность модулей):")
    feature_names = ['M1 (резервы)', 'M2 (репо ЦБ)', 'M3 (ОФЗ)', 'M5 (казначейство)', 
                     'налоговый фактор', 'налоговая неделя', 'отток бюджета']
    for name, imp in zip(feature_names, model.feature_importances_):
        print(f"  {name}: {imp:.3f}")
    
    # 8. Применяем
    result['lsi'] = lsi_ml.clip(0, 100)
    
    # 9. Цветовая зона
    result['status'] = 'ЗЕЛЁНЫЙ'
    result.loc[result['lsi'] >= 40, 'status'] = 'ЖЁЛТЫЙ'
    result.loc[result['lsi'] >= 70, 'status'] = 'КРАСНЫЙ'
    
    # 10. SHAP анализ (если установлен shap)
    try:
        import shap
        print("\n8. SHAP-анализ (интерпретация предсказаний)...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # SHAP для последнего дня
        shap_last = dict(zip(feature_names, shap_values[-1]))
        print("  SHAP-вклад модулей (последний день):")
        for name, value in shap_last.items():
            print(f"    {name}: {value:.3f}")
        
        # Средний SHAP по всем дням
        print("\n  Средний SHAP-вклад по всем дням:")
        mean_shap = np.abs(shap_values).mean(axis=0)
        for name, value in zip(feature_names, mean_shap):
            print(f"    {name}: {value:.3f}")
    except ImportError:
        print("  SHAP не установлен, пропускаем")
    except Exception as e:
        print(f"  SHAP ошибка: {e}")
    
    # 11. Сохраняем
    print("\n9. Сохранение...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, "lsi_output_ml.parquet")
    result.to_parquet(output_path, index=False)
    print(f"  Сохранено в {output_path}")
    
    # 12. Статистика
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ АГРЕГАЦИИ (ML версия):")
    print(f"  Период: {result['date'].min().date()} — {result['date'].max().date()}")
    print(f"  Всего дней: {len(result)}")
    print(f"  LSI мин: {result['lsi'].min():.1f}")
    print(f"  LSI макс: {result['lsi'].max():.1f}")
    print(f"  LSI средний: {result['lsi'].mean():.1f}")
    print(f"  Красных дней (LSI >= 70): {(result['lsi'] >= 70).sum()}")
    print(f"  Жёлтых дней (40-70): {((result['lsi'] >= 40) & (result['lsi'] < 70)).sum()}")
    print(f"  Зелёных дней (<40): {(result['lsi'] < 40).sum()}")
    
    print("\n  СТРЕСС-ЭПИЗОДЫ (ML):")
    episodes = {
        'Декабрь 2014': ('2014-12-01', '2014-12-31'),
        'Февраль-март 2022': ('2022-02-01', '2022-03-31'),
        'Август 2023': ('2023-08-01', '2023-08-31'),
    }
    for name, (start, end) in episodes.items():
        mask = (result['date'] >= start) & (result['date'] <= end)
        if mask.any():
            ep = result[mask]
            status_color = '🔴' if ep['lsi'].mean() >= 70 else ('🟡' if ep['lsi'].mean() >= 40 else '🟢')
            print(f"    {name}: средний LSI={ep['lsi'].mean():.1f}, макс={ep['lsi'].max():.1f} {status_color}")
    
    print("=" * 50)
    return result

if __name__ == "__main__":
    run()
