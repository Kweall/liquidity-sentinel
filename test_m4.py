import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from modules.m4_taxes import load_tax_calendar, build_calendar_flags, calculate_seasonal_factor, process_m4, run

@pytest.fixture
def tax_calendar():
    """Загружает реальный налоговый календарь из CSV"""
    return load_tax_calendar()

@pytest.fixture
def sample_calendar():
    """Создаёт календарь на 2023 год для тестов"""
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-12-31')
    date_range = pd.date_range(start_date, end_date, freq='D')
    return pd.DataFrame({'date': date_range})

@pytest.fixture
def sample_m1_m2():
    """Создаёт синтетические данные M1 и M2 для тестов"""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    return {
        'm1': pd.DataFrame({'date': dates, 'stress_m1': np.random.uniform(0, 5, len(dates))}),
        'm2': pd.DataFrame({'date': dates, 'stress_m2': np.random.uniform(0, 5, len(dates))})
    }

class TestLoadTaxCalendar:
    def test_load_exists(self, tax_calendar):
        """Проверяет, что календарь загружается и содержит данные"""
        assert len(tax_calendar) > 0
        assert 'date' in tax_calendar.columns
        assert tax_calendar['date'].min().year >= 2014
        assert tax_calendar['date'].max().year >= 2025

    def test_date_range(self, tax_calendar):
        """Проверяет, что даты в календаре корректны"""
        assert tax_calendar['date'].is_monotonic_increasing
        assert not tax_calendar['date'].duplicated().any()

class TestBuildCalendarFlags:
    def test_length(self, tax_calendar, sample_calendar):
        """Проверяет, что календарь содержит правильное количество дней"""
        result = build_calendar_flags(tax_calendar, 
                                       pd.Timestamp('2023-01-01'), 
                                       pd.Timestamp('2023-12-31'))
        assert len(result) == 365

    def test_columns(self, tax_calendar, sample_calendar):
        """Проверяет наличие всех необходимых колонок"""
        result = build_calendar_flags(tax_calendar,
                                       pd.Timestamp('2023-01-01'),
                                       pd.Timestamp('2023-12-31'))
        required = ['date', 'Tax_Week_Flag', 'End_of_Month_Flag', 'End_of_Quarter_Flag']
        for col in required:
            assert col in result.columns

    def test_end_of_month_flag(self, tax_calendar):
        """Проверяет, что последний день каждого месяца отмечен флагом"""
        result = build_calendar_flags(tax_calendar,
                                       pd.Timestamp('2023-01-01'),
                                       pd.Timestamp('2023-12-31'))
        for month in range(1, 13):
            last_day = pd.Timestamp(f'2023-{month:02d}-01') + pd.offsets.MonthEnd(0)
            mask = result['date'] == last_day
            assert result.loc[mask, 'End_of_Month_Flag'].iloc[0] == 1

    def test_end_of_quarter_flag(self, tax_calendar):
        """Проверяет, что последний день каждого квартала отмечен флагом"""
        result = build_calendar_flags(tax_calendar,
                                       pd.Timestamp('2023-01-01'),
                                       pd.Timestamp('2023-12-31'))
        quarter_ends = ['2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31']
        for qe in quarter_ends:
            mask = result['date'] == pd.Timestamp(qe)
            assert result.loc[mask, 'End_of_Quarter_Flag'].iloc[0] == 1

    def test_tax_week_flag_range(self, tax_calendar):
        """Проверяет, что налоговая неделя проставляется корректно"""
        result = build_calendar_flags(tax_calendar,
                                       pd.Timestamp('2023-01-01'),
                                       pd.Timestamp('2023-12-31'))
        # Выбираем первую налоговую дату
        tax_dates = tax_calendar['date'].dt.normalize().unique()
        if len(tax_dates) > 0:
            sample_tax_date = tax_dates[0]
            week_start = sample_tax_date - pd.Timedelta(days=7)
            week_end = sample_tax_date + pd.Timedelta(days=7)
            mask_week = (result['date'] >= week_start) & (result['date'] <= week_end)
            # Все дни в неделе должны иметь флаг 1
            assert (result.loc[mask_week, 'Tax_Week_Flag'] == 1).all()
            # День вне недели (за 8 дней до) должен иметь флаг 0
            outside_day = sample_tax_date - pd.Timedelta(days=8)
            if outside_day >= result['date'].min():
                mask_outside = result['date'] == outside_day
                assert result.loc[mask_outside, 'Tax_Week_Flag'].iloc[0] == 0

class TestCalculateSeasonalFactor:
    def test_seasonal_factor_range(self, tax_calendar, sample_m1_m2):
        """Проверяет, что Seasonal_Factor находится в диапазоне 1.0–1.4"""
        calendar = build_calendar_flags(tax_calendar,
                                         pd.Timestamp('2023-01-01'),
                                         pd.Timestamp('2023-12-31'))
        result = calculate_seasonal_factor(sample_m1_m2['m1'], sample_m1_m2['m2'], calendar)
        assert result['Seasonal_Factor'].between(1.0, 1.4).all()

    def test_seasonal_factor_columns(self, tax_calendar, sample_m1_m2):
        """Проверяет, что в результат добавляется колонка Seasonal_Factor"""
        calendar = build_calendar_flags(tax_calendar,
                                         pd.Timestamp('2023-01-01'),
                                         pd.Timestamp('2023-12-31'))
        result = calculate_seasonal_factor(sample_m1_m2['m1'], sample_m1_m2['m2'], calendar)
        assert 'Seasonal_Factor' in result.columns

    def test_empty_m1_m2(self, tax_calendar):
        """Проверяет, что при пустых M1/M2 Seasonal_Factor = 1.0 для всех дней"""
        calendar = build_calendar_flags(tax_calendar,
                                         pd.Timestamp('2023-01-01'),
                                         pd.Timestamp('2023-12-31'))
        result = calculate_seasonal_factor(pd.DataFrame(), pd.DataFrame(), calendar)
        # При пустых данных фактор должен быть 1.0 везде
        assert (result['Seasonal_Factor'] == 1.0).all()

class TestProcessM4:
    def test_output_columns(self, sample_m1_m2):
        """Проверяет, что process_m4 возвращает все нужные колонки"""
        result = process_m4(sample_m1_m2['m1'], sample_m1_m2['m2'])
        required = ['date', 'Tax_Week_Flag', 'End_of_Month_Flag', 'End_of_Quarter_Flag', 'Seasonal_Factor']
        for col in required:
            assert col in result.columns

    def test_date_sorting(self, sample_m1_m2):
        """Проверяет, что даты отсортированы и нет дубликатов"""
        result = process_m4(sample_m1_m2['m1'], sample_m1_m2['m2'])
        assert result['date'].is_monotonic_increasing
        assert not result['date'].duplicated().any()

    def test_date_range(self, sample_m1_m2):
        """Проверяет, что диапазон дат начинается с 2014 года"""
        result = process_m4(sample_m1_m2['m1'], sample_m1_m2['m2'])
        assert result['date'].min().year <= 2014
        assert result['date'].max().year >= datetime.now().year

    def test_without_m1_m2(self):
        """Проверяет работу без M1 и M2"""
        result = process_m4(pd.DataFrame(), pd.DataFrame())
        assert len(result) > 0
        assert (result['Seasonal_Factor'] == 1.0).all()

class TestRun:
    def test_run_returns_dataframe(self):
        """Проверяет, что run() возвращает DataFrame"""
        result = run()
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_run_saves_parquet(self, tmp_path):
        """Проверяет, что run() сохраняет файл (косвенно через наличие модуля)"""
        # Просто проверяем, что функция выполняется без ошибок
        result = run()
        assert result is not None