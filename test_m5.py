import pytest
import pandas as pd
import numpy as np
from modules.m5_treasury import merge_and_calculate_deltas, process_m5

@pytest.fixture
def sample_cbr_data():
    dates = pd.date_range('2020-01-01', periods=72, freq='ME')
    return pd.DataFrame({
        'date': dates,
        'total_eks': [1500 + i * 5 for i in range(72)]
    })

@pytest.fixture
def sample_roskazna_data():
    dates = pd.date_range('2020-01-01', periods=72, freq='ME')
    return pd.DataFrame({
        'date': dates,
        'deposits_placed': [400 + i * 2 for i in range(72)],
        'participants': [5 + i % 4 for i in range(72)]
    })

class TestM5MergeAndDeltas:
    def test_basic_merge(self, sample_cbr_data, sample_roskazna_data):
        result = merge_and_calculate_deltas(sample_cbr_data, sample_roskazna_data)
        assert len(result) == 72
        assert 'delta_eks_monthly' in result.columns
        assert 'Flag_Budget_Drain' in result.columns
        assert 'participants' in result.columns
        assert result['deposits_placed'].notna().all()

    def test_delta_calculation(self, sample_cbr_data, sample_roskazna_data):
        result = merge_and_calculate_deltas(sample_cbr_data, sample_roskazna_data)
        assert (result['delta_eks_monthly'].iloc[1:] == 5.0).all()
        assert result['delta_eks_monthly'].iloc[0] != result['delta_eks_monthly'].iloc[0]  # NaN в первой строке

    def test_flag_budget_drain_trigger(self):
        dates = pd.date_range('2020-01-01', periods=70, freq='ME')
        eks = [1000] * 68 + [1000, 600]
        cbr_df = pd.DataFrame({'date': dates, 'total_eks': eks})
        rosk_df = pd.DataFrame({'date': dates, 'deposits_placed': [500]*70, 'participants': [5]*70})
        
        result = merge_and_calculate_deltas(cbr_df, rosk_df)
        assert result.iloc[-1]['Flag_Budget_Drain'] == 1
        assert result.iloc[-1]['delta_eks_monthly'] < -300

class TestM5Process:
    def test_stress_range(self, sample_cbr_data, sample_roskazna_data):
        result = process_m5(sample_cbr_data, sample_roskazna_data)
        assert 'stress_m5' in result.columns
        assert result['stress_m5'].between(0, 10).all()
        assert 'mad_score_cbr' in result.columns
        assert 'participants' in result.columns

    def test_nan_handling(self, sample_cbr_data, sample_roskazna_data):
        result = process_m5(sample_cbr_data, sample_roskazna_data)
        assert result['stress_m5'].notna().all()
        assert result['date'].notna().all()

    def test_empty_input(self):
        assert process_m5(pd.DataFrame(), pd.DataFrame()).empty

    def test_mad_window_compatibility(self, sample_cbr_data, sample_roskazna_data):
        result = process_m5(sample_cbr_data, sample_roskazna_data)
        assert result['mad_score_cbr'].notna().sum() > 10 