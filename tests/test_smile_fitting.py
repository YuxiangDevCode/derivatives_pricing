import numpy as np
import pandas as pd
import pytest
from vol_surface.smile_fitting import fit_smile_spline


@pytest.fixture
def sample_smile():
    """Create a simple synthetic smile for testing."""
    log_moneyness = np.array([-0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2])
    iv = np.array([0.25, 0.22, 0.20, 0.21, 0.23, 0.25, 0.28])
    T = 0.25  # 3 months
    
    df = pd.DataFrame({
        'log_moneyness': log_moneyness,
        'implied_volatility': iv,
        'T': T
    })
    return df


def test_fit_smile_spline_output_structure(sample_smile):
    """Test that fit_smile_spline returns correct output structure."""
    result = fit_smile_spline(sample_smile, smoothing=1e-2)
    
    # Check return type
    assert isinstance(result, dict), "Output should be a dictionary"
    
    # Check all required keys present
    assert 'iv_func' in result, "Missing 'iv_func' key"
    assert 'iv_fitted' in result, "Missing 'iv_fitted' key"
    assert 'residuals' in result, "Missing 'residuals' key"


def test_fit_smile_spline_residuals(sample_smile):
    """Test that residuals = iv - iv_fitted."""
    result = fit_smile_spline(sample_smile, smoothing=1e-2)
    
    iv = sample_smile['implied_volatility'].values
    iv_fitted = result['iv_fitted']
    residuals = result['residuals']
    
    # Check residuals calculation
    expected_residuals = iv - iv_fitted
    np.testing.assert_array_almost_equal(residuals, expected_residuals)


def test_fit_smile_spline_interpolation(sample_smile):
    """Test that the spline can interpolate at new points."""
    result = fit_smile_spline(sample_smile, smoothing=1e-2)
    iv_func = result['iv_func']
    
    # Test at training points
    log_m_train = sample_smile['log_moneyness'].values
    iv_at_train = iv_func(log_m_train)
    
    assert len(iv_at_train) == len(log_m_train), "Interpolation output length mismatch"
    assert not np.any(np.isnan(iv_at_train)), "NaN values in interpolation output"
    
    # Test at new points
    log_m_new = np.array([-0.07, 0.05, 0.12])
    iv_at_new = iv_func(log_m_new)
    
    assert len(iv_at_new) == len(log_m_new), "Interpolation at new points failed"
    assert all(0 < iv < 1 for iv in iv_at_new), "Interpolated IVs out of reasonable range"
