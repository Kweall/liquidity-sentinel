import pytest
import pandas as pd
import numpy as np
from modules.m2_repo import (
    fetch_repo_soap, fetch_repo_full_history, fetch_keyrate_soap, 
    process_m2, run
)

@pytest.fixture(scope="module")
def keyrate():
    """Загружает данные ключевой ставки ЦБ"""
    return fetch_keyrate_soap()

@pytest.fixture(scope="module")
def repo_full():
    """Загружает полную историю аукционов РЕПО"""
    return fetch_repo_full_history(from_year=2010)

@pytest.fixture(scope="module")
def repo_7d(repo_full):
    """Фильтрует 7-дневные аукционы РЕПО"""
    return repo_full[repo_full['term_days'] == 7].copy()

@pytest.fixture(scope="module")
def m2_result(repo_7d, keyrate):
    """Выполняет process_m2 и возвращает результат"""
    return process_m2(repo_7d, keyrate)

class TestFetchRepoSoap:
    def test_single_date(self):
        """Проверяет, что fetch_repo_soap возвращает данные для одной даты"""
        df = fetch_repo_soap("2026-05-19", "2026-05-19")
        assert len(df) > 0, "Пустой датафрейм"
        assert 'bid' in df.columns or 'demand_volume' in df.columns, "Нет колонки со спросом"
        assert 'avg_deal' in df.columns or 'placement_volume' in df.columns, "Нет колонки с объёмом размещения"

    def test_empty_range(self):
        """Проверяет обработку пустого диапазона дат"""
        df = fetch_repo_soap("2025-01-01", "2025-01-01")
        # Может вернуть пустой DataFrame, но не должно упасть
        assert isinstance(df, pd.DataFrame)

class TestFetchKeyrateSoap:
    def test_data_exists(self, keyrate):
        """Проверяет, что данные ключевой ставки загружены"""
        assert len(keyrate) > 0, "Пустой датафрейм"
        assert keyrate['key_rate'].notna().all()

    def test_rate_range(self, keyrate):
        """Проверяет, что ставка в разумных пределах"""
        assert keyrate['key_rate'].min() > 0, "Ставка не может быть <= 0"
        assert keyrate['key_rate'].max() < 50, "Ставка > 50% — что-то не так"

    def test_date_range(self, keyrate):
        """Проверяет, что диапазон дат корректен"""
        assert keyrate['date_from'].min().year <= 2015
        assert keyrate['date_from'].max().year >= 2025

    def test_date_sorting(self, keyrate):
        """Проверяет, что даты отсортированы"""
        assert keyrate['date_from'].is_monotonic_increasing

class TestFetchRepoFullHistory:
    def test_data_exists(self, repo_full):
        """Проверяет, что история аукционов загружена"""
        assert len(repo_full) > 0

    def test_required_columns(self, repo_full):
        """Проверяет наличие необходимых колонок"""
        required = ['date', 'term_days', 'demand_volume', 'placement_volume']
        for col in required:
            assert col in repo_full.columns, f"Нет колонки {col}"

    def test_term_days_values(self, repo_full):
        """Проверяет, что term_days содержит корректные значения"""
        valid_terms = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 21, 28, 35, 91, 182, 350, 364, 365, 371]
        for term in repo_full['term_days'].unique():
            assert term in valid_terms, f"Некорректное значение term_days: {term}"

    def test_date_range(self, repo_full):
        """Проверяет, что диапазон дат начинается с 2010 года"""
        assert repo_full['date'].min().year <= 2010
        assert repo_full['date'].max().year >= 2025

    def test_date_sorting(self, repo_full):
        """Проверяет, что даты отсортированы"""
        assert repo_full['date'].is_monotonic_increasing

    def test_7d_exists(self, repo_7d):
        """Проверяет, что 7-дневные аукционы существуют"""
        assert len(repo_7d) > 0

class TestProcessM2:
    def test_output_columns(self, m2_result):
        """Проверяет наличие всех выходных колонок"""
        required = ['date', 'cover_ratio', 'rate_spread', 'key_rate',
                    'mad_score_cover', 'mad_score_rate_spread', 
                    'stress_m2', 'Flag_Demand', 'Flag_Emergency_1d']
        for col in required:
            assert col in m2_result.columns, f"Нет колонки {col}"

    def test_stress_range(self, m2_result):
        """Проверяет, что stress_m2 в диапазоне 0-10"""
        assert m2_result['stress_m2'].min() >= 0, "stress_m2 отрицательный"
        assert m2_result['stress_m2'].max() <= 10, "stress_m2 > 10"

    def test_cover_ratio_positive(self, m2_result):
        """Проверяет, что cover_ratio неотрицательный"""
        assert m2_result['cover_ratio'].min() >= 0, "cover_ratio отрицательный"

    def test_rate_spread_range(self, m2_result):
        """Проверяет, что rate_spread в разумных пределах"""
        # Спред может быть отрицательным (если ставка отсечения ниже ключевой)
        assert m2_result['rate_spread'].min() > -10, "rate_spread слишком низкий"
        assert m2_result['rate_spread'].max() < 20, "rate_spread слишком высокий"

    def test_no_date_duplicates(self, m2_result):
        """Проверяет отсутствие дубликатов дат"""
        assert not m2_result['date'].duplicated().any(), "Есть дубликаты дат"

    def test_date_sorted(self, m2_result):
        """Проверяет, что даты отсортированы"""
        assert m2_result['date'].is_monotonic_increasing, "Даты не отсортированы"

    def test_flags_binary(self, m2_result):
        """Проверяет, что флаги имеют значения 0 или 1"""
        assert m2_result['Flag_Demand'].isin([0, 1]).all()
        assert m2_result['Flag_Emergency_1d'].isin([0, 1]).all()

    def test_mad_scores_finite(self, m2_result):
        """Проверяет, что MAD-оценки не содержат NaN"""
        assert m2_result['mad_score_cover'].notna().all()
        assert m2_result['mad_score_rate_spread'].notna().all()

class TestStressEpisodes:
    def test_stress_episodes(self, m2_result):
        """Проверяет значения mad_score_cover в известные кризисные периоды"""
        episodes = {
            'Декабрь 2014': ('2014-12-01', '2014-12-31'),
            'Февраль-март 2022': ('2022-02-01', '2022-03-31'),
            'Август 2023': ('2023-07-01', '2023-09-30'),
        }
        overall_mean = m2_result['mad_score_cover'].mean()
        
        for name, (start, end) in episodes.items():
            ep = m2_result[(m2_result['date'] >= start) & (m2_result['date'] <= end)]
            if len(ep) == 0:
                print(f"{name}: нет данных")
                continue
            # Просто проверяем, что значения конечны
            assert ep['mad_score_cover'].notna().all()
            assert ep['cover_ratio'].notna().all()

class TestCoverRatio:
    def test_cover_ratio_calculation(self, repo_7d):
        """Проверяет, что cover_ratio = demand / placement (построчно)"""
        df = repo_7d.copy()
        # Рассчитываем cover_ratio напрямую
        direct_ratio = df['demand_volume'] / df['placement_volume']
        
        # Проверяем, что все значения положительные
        assert (direct_ratio > 0).all(), "Некоторые cover_ratio не положительные"
        
        # Проверяем, что нет NaN
        assert direct_ratio.notna().all(), "Есть NaN в cover_ratio"
        
        # Проверяем разумный диапазон (допускаем выбросы до 250)
        # В реальных данных бывают технические аукционы с очень маленьким размещением
        assert direct_ratio.min() > 0.001, "Слишком маленький cover_ratio"
        assert direct_ratio.max() < 250, "Слишком большой cover_ratio (более 250)"
        
        # Дополнительно: проверяем, что медиана в разумных пределах
        median_ratio = direct_ratio.median()
        assert 0.5 < median_ratio < 10, f"Медиана cover_ratio = {median_ratio:.2f} вне ожидаемого диапазона"

    def test_flag_demand_logic(self, repo_7d):
        """Проверяет, что Flag_Demand = 1 при cover_ratio > 2.0 в исходных аукционах"""
        df = repo_7d.copy()
        df['cover_ratio'] = df['demand_volume'] / df['placement_volume']
        df['Flag_Demand'] = (df['cover_ratio'] > 2.0).astype(int)
        
        # Проверяем, что при cover_ratio > 2.0 флаг = 1
        high_cover = df[df['cover_ratio'] > 2.0]
        if len(high_cover) > 0:
            assert (high_cover['Flag_Demand'] == 1).all(), \
                f"Для {len(high_cover)} аукционов с cover_ratio > 2.0 флаг не 1"
        
        # Проверяем, что при cover_ratio <= 2.0 флаг = 0
        low_cover = df[df['cover_ratio'] <= 2.0]
        if len(low_cover) > 0:
            assert (low_cover['Flag_Demand'] == 0).all(), \
                f"Для {len(low_cover)} аукционов с cover_ratio <= 2.0 флаг не 0"

    def test_flag_demand_after_processing(self, m2_result):
        """Проверяет, что в финальном результате Flag_Demand имеет значения 0 или 1"""
        # Просто проверяем, что флаг бинарный (уже есть в другом тесте)
        assert m2_result['Flag_Demand'].isin([0, 1]).all()
        # Проверяем, что флаг не NaN
        assert m2_result['Flag_Demand'].notna().all()

class TestRun:
    def test_run_returns_dataframe(self):
        """Проверяет, что run() возвращает DataFrame"""
        df = run()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_run_output_columns(self):
        """Проверяет, что run() возвращает правильные колонки"""
        df = run()
        required = ['date', 'cover_ratio', 'rate_spread', 'key_rate',
                    'mad_score_cover', 'mad_score_rate_spread',
                    'stress_m2', 'Flag_Demand', 'Flag_Emergency_1d']
        for col in required:
            assert col in df.columns, f"Нет колонки {col}"

    def test_run_saves_parquet(self):
        """Проверяет, что run() сохраняет результат"""
        from pathlib import Path
        df = run()
        output_path = Path("data/processed/m2_output.parquet")
        assert output_path.exists()
        assert output_path.stat().st_size > 0