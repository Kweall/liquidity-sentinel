import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import shap
import os

class LSI_ML_Aggregator:
    """
    ML-агрегатор для LSI с интерпретацией через SHAP
    Использует RandomForestRegressor (не нейросеть, интерпретируемый)
    """
    
    def __init__(self, model_path="models/lsi_model.pkl"):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )
        self.model_path = model_path
        self.feature_names = ['m1_signal', 'm2_signal', 'm3_signal', 'm5_signal', 'seasonal_factor', 'tax_week_flag']
        self.shap_explainer = None
        
    def prepare_features(self, df):
        """Подготавливает признаки для обучения"""
        features = pd.DataFrame()
        features['m1_signal'] = df['stress_m1'] / 10.0
        features['m2_signal'] = df['stress_m2'] / 10.0
        features['m3_signal'] = df['stress_m3'] / 10.0
        features['m5_signal'] = df['stress_m5'] / 10.0
        features['seasonal_factor'] = df['Seasonal_Factor'].fillna(1.0)
        features['tax_week_flag'] = df['Tax_Week_Flag'].fillna(0)
        return features
    
    def load_ground_truth(self, path=None):
        """
        Загружает ground truth из ЦБ РФ
        Таблица "Ликвидность банковского сектора"
        """
        # TODO: парсинг реальных данных с сайта ЦБ
        # Пока создаём синтетическую целевую переменную на основе LSI
        print("Предупреждение: используется синтетическая целевая переменная")
        print("Рекомендуется загрузить реальные данные с cbr.ru/hd_base/bliquidity/")
        return None
    
    def train(self, X, y):
        """Обучает модель"""
        print(f"\nОбучение модели RandomForestRegressor...")
        print(f"Размер выборки: {len(X)}")
        
        # TimeSeriesSplit для временных рядов
        tscv = TimeSeriesSplit(n_splits=5)
        
        train_scores = []
        val_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            
            train_scores.append(self.model.score(X_train, y_train))
            val_scores.append(r2_score(y_val, y_pred))
            
            print(f"  Fold {fold+1}: Train R2={train_scores[-1]:.3f}, Val R2={val_scores[-1]:.3f}")
        
        self.model.fit(X, y)
        
        # Создаём SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)
        
        print(f"\nИтоговый R2 на всех данных: {self.model.score(X, y):.3f}")
        print(f"Feature importance: {dict(zip(self.feature_names, self.model.feature_importances_))}")
        
        return self.model
    
    def predict(self, df):
        """Предсказывает LSI"""
        X = self.prepare_features(df)
        y_pred = self.model.predict(X)
        return np.clip(y_pred, 0, 100)
    
    def explain(self, df, idx=-1):
        """SHAP-объяснение для конкретного дня"""
        if self.shap_explainer is None:
            self.shap_explainer = shap.TreeExplainer(self.model)
        
        X = self.prepare_features(df)
        shap_values = self.shap_explainer.shap_values(X)
        
        contributions = dict(zip(self.feature_names, shap_values[idx]))
        return contributions
    
    def save(self):
        """Сохраняет модель"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Модель сохранена в {self.model_path}")
    
    @staticmethod
    def load(path):
        """Загружает модель"""
        with open(path, 'rb') as f:
            return pickle.load(f)


def run_ml_aggregator(df=None):
    """Запускает ML-агрегацию"""
    if df is None:
        df = pd.read_parquet('data/processed/lsi_output.parquet')
    
    aggregator = LSI_ML_Aggregator()
    
    X = aggregator.prepare_features(df)
    
    # Создаём целевую переменную (пока на основе взвешенного LSI)
    y = df['lsi'] / 100.0  # Нормализуем 0-1
    
    aggregator.train(X, y)
    lsi_ml = aggregator.predict(df)
    
    print("\nСравнение LSI (взвешенный) vs LSI (ML):")
    print(pd.DataFrame({
        'date': df['date'],
        'LSI_weighted': df['lsi'],
        'LSI_ML': lsi_ml
    }).tail(10))
    
    contributions = aggregator.explain(df, -1)
    print(f"\nSHAP-вклад модулей (последний день):")
    for feature, value in contributions.items():
        print(f"  {feature}: {value:.3f}")
    
    return aggregator, lsi_ml

if __name__ == "__main__":
    aggregator, lsi_ml = run_ml_aggregator()
    aggregator.save()
def fetch_cbr_liquidity_data():
    """
    Скачивает данные по ликвидности банковского сектора с сайта ЦБ
    URL: https://www.cbr.ru/hd_base/bliquidity/
    """
    import requests
    from bs4 import BeautifulSoup
    import re
    
    url = "https://www.cbr.ru/hd_base/bliquidity/"
    
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None
    
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    excel_link = None
    for link in soup.find_all('a'):
        href = link.get('href', '')
        if href.endswith('.xlsx') or href.endswith('.xls'):
            excel_link = href
            break
    
    if not excel_link:
        print("Excel файл не найден на странице")
        return None
    
    if excel_link.startswith('/'):
        excel_link = "https://www.cbr.ru" + excel_link
    
    print(f"Скачиваю файл: {excel_link}")
    
    try:
        df = pd.read_excel(excel_link)
        print(f"Загружено: {df.shape}")
        print(f"Колонки: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Ошибка чтения Excel: {e}")
        return None

def integrate_ground_truth(lsi_df, ground_truth_df):
    """
    Интегрирует ground truth с LSI DataFrame
    Нужно сопоставить даты и создать целевую переменную (0-100)
    """
    # TODO: реализовать сопоставление
    # В ground truth должно быть что-то типа "дефицит ликвидности" или "stress_level"
    pass
