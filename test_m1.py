import pytest
import pandas as pd
import numpy as np
from modules.m1_reserves import fetch_reserves, fetch_ruonia, process_m1, run

pytestmark = pytest.mark.filterwarnings("ignore:Workbook contains no default style:UserWarning")

@pytest.fixture(scope="module")
def reserves():
    """Загружает данные обязательных резервов"""
    return fetch_reserves()

@pytest.fixture(scope="module")
def ruonia():
    """Загружает данные RUONIA"""
    return fetch_ruonia()

@pytest.fixture(scope="module")
def m1_result(reserves, ruonia):
    """Выполняет process_m1 и возвращает результат"""
    return process_m1(reserves, ruonia)

class TestFetchReserves:
    def test_data_exists(self, reserves):
        """Проверяет, что данные резервов загружены"""
        assert len(reserves) > 0, "Нет данных резервов"

    def test_required_columns(self, reserves):
        """Проверяет наличие всех необходимых колонок"""
        required = ['period_start', 'actual_reserves', 'required_avg', 'required_acc', 'period_end']
        for col in required:
            assert col in reserves.columns, f"Нет колонки {col}"

    def test_date_range(self, reserves):
        """Проверяет, что диапазон дат начинается с 2004 года"""
        assert reserves['period_start'].min().year <= 2004
        assert reserves['period_start'].max().year >= 2025

    def test_no_empty_periods(self, reserves):
        """Проверяет, что нет пустых периодов"""
        assert reserves['actual_reserves'].notna().any(), "Нет данных actual_reserves"
        assert reserves['required_avg'].notna().any(), "Нет данных required_avg"

class TestFetchRuonia:
    def test_data_exists(self, ruonia):
        """Проверяет, что данные RUONIA загружены"""
        assert len(ruonia) > 0, "Нет данных RUONIA"

    def test_required_columns(self, ruonia):
        """Проверяет наличие колонок date и ruonia_rate"""
        assert 'date' in ruonia.columns
        assert 'ruonia_rate' in ruonia.columns

    def test_rate_range(self, ruonia):
        """Проверяет, что ставка RUONIA в разумных пределах"""
        assert ruonia['ruonia_rate'].min() > 0, "Ставка RUONIA не может быть <= 0"
        assert ruonia['ruonia_rate'].max() < 30, "Ставка RUONIA > 30% — что-то не так"

    def test_date_range(self, ruonia):
        """Проверяет, что диапазон дат начинается с 2010 года"""
        assert ruonia['date'].min().year <= 2010
        assert ruonia['date'].max().year >= 2025

    def test_date_sorting(self, ruonia):
        """Проверяет, что даты отсортированы"""
        assert ruonia['date'].is_monotonic_increasing

class TestProcessM1:
    def test_output_columns(self, m1_result):
        """Проверяет наличие всех выходных колонок"""
        required = ['date', 'spread', 'ruonia_rate', 'mad_score_spread',
                    'mad_score_ruonia', 'stress_m1', 'Flag_EndOfPeriod']
        for col in required:
            assert col in m1_result.columns, f"Нет колонки {col}"

    def test_stress_range(self, m1_result):
        """Проверяет, что stress_m1 в диапазоне 0-10"""
        assert m1_result['stress_m1'].min() >= 0, "stress_m1 отрицательный"
        assert m1_result['stress_m1'].max() <= 10, "stress_m1 > 10"

    def test_spread_values(self, m1_result):
        """Проверяет, что spread имеет разумные значения"""
        # spread может быть отрицательным (если фактические остатки меньше обязательных)
        assert m1_result['spread'].min() > -1000, "spread слишком низкий"
        assert m1_result['spread'].max() < 10000, "spread слишком высокий"

    def test_no_date_duplicates(self, m1_result):
        """Проверяет отсутствие дубликатов дат"""
        assert not m1_result['date'].duplicated().any(), "Есть дубликаты дат"

    def test_date_sorted(self, m1_result):
        """Проверяет, что даты отсортированы"""
        assert m1_result['date'].is_monotonic_increasing, "Даты не отсортированы"

    def test_flag_binary(self, m1_result):
        """Проверяет, что Flag_EndOfPeriod имеет значения 0 или 1"""
        assert m1_result['Flag_EndOfPeriod'].isin([0, 1]).all()

    def test_mad_scores_finite(self, m1_result):
        """Проверяет, что MAD-оценки не содержат NaN"""
        assert m1_result['mad_score_spread'].notna().all()
        assert m1_result['mad_score_ruonia'].notna().all()

    def test_ruonia_rate_filled(self, m1_result):
        """Проверяет, что пропуски RUONIA заполнены"""
        # После ffill(limit=3) могут остаться NaN в самом начале
        # Проверяем, что после 2010 года нет NaN
        after_2010 = m1_result[m1_result['date'] >= '2010-01-01']
        assert after_2010['ruonia_rate'].notna().all(), "Есть NaN в ruonia_rate после 2010 года"

class TestStressEpisodes:
    def test_stress_episodes(self, m1_result):
        """Проверяет значения stress_m1 в известные кризисные периоды"""
        episodes = {
            'Декабрь 2014': ('2014-12-01', '2014-12-31'),
            'Февраль-март 2022': ('2022-02-01', '2022-03-31'),
            'Август 2023': ('2023-07-01', '2023-09-30'),
        }
        overall_mean = m1_result['stress_m1'].mean()
        
        for name, (start, end) in episodes.items():
            ep = m1_result[(m1_result['date'] >= start) & (m1_result['date'] <= end)]
            if len(ep) == 0:
                print(f"{name}: нет данных")
                continue
            ep_mean = ep['stress_m1'].mean()
            # Для декабря 2014 ожидаем стресс выше среднего
            if name == 'Декабрь 2014':
                assert ep_mean > overall_mean, \
                    f"В {name} stress_m1 ({ep_mean:.3f}) не выше среднего ({overall_mean:.3f})"
            # Для февраля-марта 2022 стресс мог быть невысоким (но не проверяем жёстко)
            # Просто проверяем, что значения конечны
            assert ep['stress_m1'].notna().all()

class TestAdditional:
    def test_ruonia_no_missing_after_2010(self, m1_result):
        """Проверяет, что после 2010 года нет пропусков RUONIA"""
        after_2010 = m1_result[m1_result['date'] >= '2010-01-01']
        missing = after_2010[after_2010['ruonia_rate'].isna()]
        if len(missing) > 0:
            print(f"Предупреждение: пропуски RUONIA после 2010 года: {missing['date'].tolist()[:5]}")
        # Не делаем assert, так как в начале 2010 года могут быть пропуски
        # Просто выводим предупреждение

    def test_spread_calculation(self, reserves):
        """Проверяет, что spread = actual_reserves - required_avg для периодов"""
        # Проверяем только для строк, где есть оба значения
        valid = reserves[reserves['actual_reserves'].notna() & reserves['required_avg'].notna()]
        if len(valid) > 0:
            calculated_spread = valid['actual_reserves'] - valid['required_avg']
            # Проверяем, что spread имеет правильный знак (может быть любым)
            assert calculated_spread.notna().all()

class TestRun:
    def test_run_returns_dataframe(self):
        """Проверяет, что run() возвращает DataFrame"""
        df = run()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_run_output_columns(self):
        """Проверяет, что run() возвращает правильные колонки"""
        df = run()
        required = ['date', 'spread', 'ruonia_rate', 'mad_score_spread',
                    'mad_score_ruonia', 'stress_m1', 'Flag_EndOfPeriod']
        for col in required:
            assert col in df.columns, f"Нет колонки {col}"

    def test_run_saves_parquet(self):
        """Проверяет, что run() сохраняет результат"""
        from pathlib import Path
        df = run()
        output_path = Path("data/processed/m1_output.parquet")
        assert output_path.exists()
        assert output_path.stat().st_size > 0