"""
Tests for volatility surface construction functions.

Tests cover:
- Implied volatility computation
- Moneyness calculations
- Smile extraction (call, put, composite)
- Surface building
"""

import pytest
import numpy as np
import pandas as pd

from vol_surface.iv_surface import (
    implied_volatility,
    compute_implied_vols,
    add_moneyness_columns,
    extract_option_smile,
    extract_smile_at_expiry,
    compute_atm_vol,
    build_iv_surface,
)
from pricing.black_scholes import black_scholes


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_option_chain():
    """Create a basic option chain with known data."""
    spot = 100.0
    strikes = np.array([90, 95, 100, 105, 110])
    T = 30 / 365  # 30 days
    r = 0.05
    sigma = 0.20
    
    # Generate prices using Black-Scholes
    call_prices = [black_scholes(spot, K, T, r, sigma, 'call') for K in strikes]
    put_prices = [black_scholes(spot, K, T, r, sigma, 'put') for K in strikes]
    
    df = pd.DataFrame({
        'spot': spot,
        'strike': strikes,
        'T': T,
        'call_price': call_prices,
        'put_price': put_prices,
    })
    
    return df


@pytest.fixture
def chain_with_iv(simple_option_chain):
    """Option chain with implied volatilities computed."""
    df = compute_implied_vols(simple_option_chain, r=0.05)
    df = add_moneyness_columns(df)
    return df


# =============================================================================
# TEST SUITE: implied_volatility()
# =============================================================================

class TestImpliedVolatility:
    """Test IV computation from option prices."""
    
    def test_iv_recovery_call(self):
        """IV computed from BS price should recover original sigma."""
        S, K, T, r, sigma_true = 100.0, 100.0, 0.25, 0.05, 0.20
        
        # Generate price with known sigma
        price = black_scholes(S, K, T, r, sigma_true, 'call')
        
        # Recover IV
        sigma_iv = implied_volatility(S, K, T, r, price, 'call')
        
        # Should recover original (within small numerical error)
        assert np.isclose(sigma_iv, sigma_true, rtol=1e-4)
    
    def test_iv_recovery_put(self):
        """IV computed from BS put price should recover sigma."""
        S, K, T, r, sigma_true = 95.0, 100.0, 0.1, 0.03, 0.15
        
        price = black_scholes(S, K, T, r, sigma_true, 'put')
        sigma_iv = implied_volatility(S, K, T, r, price, 'put')
        
        assert np.isclose(sigma_iv, sigma_true, rtol=1e-4)
    
    def test_iv_invalid_price(self):
        """IV should return NaN for prices violating no-arbitrage bounds."""
        S, K, T, r = 100.0, 100.0, 0.25, 0.05
        
        # Price below intrinsic value = arbitrage
        invalid_call_price = -10.0
        
        sigma_iv = implied_volatility(S, K, T, r, invalid_call_price, 'call')
        
        assert np.isnan(sigma_iv)


# =============================================================================
# TEST SUITE: compute_implied_vols()
# =============================================================================

class TestComputeImpliedVols:
    """Test batch IV computation."""
    
    def test_output_shape(self, simple_option_chain):
        """Should return dataframe with same number of rows."""
        result = compute_implied_vols(simple_option_chain, r=0.05)
        
        assert len(result) == len(simple_option_chain)
        assert 'iv_call' in result.columns
        assert 'iv_put' in result.columns
    
    def test_iv_columns_numeric(self, simple_option_chain):
        """IV columns should contain numeric values (or NaN)."""
        result = compute_implied_vols(simple_option_chain, r=0.05)
        
        # All values should be numeric
        assert result['iv_call'].dtype in [np.float64, float]
        assert result['iv_put'].dtype in [np.float64, float]
    
    def test_empty_chain_raises(self):
        """Should raise error for empty dataframe."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            compute_implied_vols(empty_df, r=0.05)


# =============================================================================
# TEST SUITE: add_moneyness_columns()
# =============================================================================

class TestAddMoneyness:
    """Test moneyness calculations."""
    
    def test_atm_moneyness_zero(self, simple_option_chain):
        """ATM strike should have log_moneyness = 0."""
        df = add_moneyness_columns(simple_option_chain)
        
        # Find ATM row
        atm_row = df[df['is_atm']]
        
        assert not atm_row.empty
        assert np.isclose(atm_row['log_moneyness'].iloc[0], 0.0, atol=1e-10)
    
    def test_moneyness_properties(self, simple_option_chain):
        """Moneyness should satisfy S/K definition."""
        spot = simple_option_chain['spot'].iloc[0]
        
        df = add_moneyness_columns(simple_option_chain)
        
        # Check moneyness = S / K
        expected_moneyness = spot / df['strike']
        assert np.allclose(df['moneyness'], expected_moneyness)
        
        # Check log_moneyness = log(S / K)
        expected_log_moneyness = np.log(expected_moneyness)
        assert np.allclose(df['log_moneyness'], expected_log_moneyness)
    
    def test_single_atm_flag(self, simple_option_chain):
        """Only one strike should be marked as ATM."""
        df = add_moneyness_columns(simple_option_chain)
        
        num_atm = (df['is_atm']).sum()
        
        assert num_atm == 1


# =============================================================================
# TEST SUITE: extract_option_smile()
# =============================================================================

class TestExtractOptionSmile:
    """Test smile extraction for single option type."""
    
    def test_call_smile_extraction(self, chain_with_iv):
        """Should extract call IV smile without NaNs."""
        smile = extract_option_smile(chain_with_iv, option_type='call')
        
        assert not smile.empty
        assert 'log_moneyness' in smile.columns
        assert 'implied_volatility' in smile.columns
        assert not smile['implied_volatility'].isna().any()
    
    def test_put_smile_extraction(self, chain_with_iv):
        """Should extract put IV smile."""
        smile = extract_option_smile(chain_with_iv, option_type='put')
        
        assert not smile.empty
        assert 'implied_volatility' in smile.columns
        assert not smile['implied_volatility'].isna().any()
    
    def test_smile_sorted_by_moneyness(self, chain_with_iv):
        """Smile should be sorted by log_moneyness."""
        smile = extract_option_smile(chain_with_iv, option_type='call')
        
        assert np.all(smile['log_moneyness'].diff().iloc[1:] >= 0)


# =============================================================================
# TEST SUITE: extract_smile_at_expiry()
# =============================================================================

class TestExtractSmileAtExpiry:
    """Test composite smile construction (OTM + ATM)."""
    
    def test_composite_smile_has_all_columns(self, chain_with_iv):
        """Should return smile with required columns."""
        smile = extract_smile_at_expiry(chain_with_iv)
        
        required = ['T', 'strike', 'log_moneyness', 'implied_volatility', 'source']
        for col in required:
            assert col in smile.columns
    
    def test_atm_marked_correctly(self, chain_with_iv):
        """Composite smile should have one ATM row with source='atm'."""
        smile = extract_smile_at_expiry(chain_with_iv)
        
        atm_rows = smile[smile['source'] == 'atm']
        assert len(atm_rows) == 1
        assert np.isclose(atm_rows['log_moneyness'].iloc[0], 0.0, atol=1e-2)
    
    def test_smile_uses_otm_data(self, chain_with_iv):
        """Composite smile should use OTM call and put data."""
        smile = extract_smile_at_expiry(chain_with_iv)
        spot = chain_with_iv['spot'].iloc[0]
        
        # Check left wing uses puts
        left_wing = smile[smile['strike'] < spot]
        if not left_wing.empty:
            assert (left_wing['source'] == 'put').all()
        
        # Check right wing uses calls
        right_wing = smile[smile['strike'] > spot]
        if not right_wing.empty:
            assert (right_wing['source'] == 'call').all()


# =============================================================================
# TEST SUITE: compute_atm_vol()
# =============================================================================

class TestComputeATMVol:
    """Test ATM volatility extraction."""
    
    def test_atm_vol_is_numeric(self, chain_with_iv):
        """ATM vol should be a single numeric value."""
        atm_vol = compute_atm_vol(chain_with_iv)
        
        assert isinstance(atm_vol, (float, np.floating))
        assert 0 < atm_vol < 5  # Reasonable vol range
    
    def test_atm_vol_consistency(self, chain_with_iv):
        """ATM vol should be average of call/put at ATM."""
        atm_row = chain_with_iv[chain_with_iv['is_atm']]
        atm_iv_call = atm_row['iv_call'].iloc[0]
        atm_iv_put = atm_row['iv_put'].iloc[0]
        
        atm_vol = compute_atm_vol(chain_with_iv)
        expected = (atm_iv_call + atm_iv_put) / 2
        
        assert np.isclose(atm_vol, expected, rtol=1e-10)


# =============================================================================
# TEST SUITE: build_iv_surface()
# =============================================================================

class TestBuildIVSurface:
    """Test surface construction from multiple smiles."""
    
    def test_surface_combines_multiple_smiles(self, chain_with_iv):
        """Should combine multiple smiles into single dataframe."""
        # Create 2 smiles with different maturities
        smile1 = extract_smile_at_expiry(chain_with_iv)
        
        # Modify T for second smile (simulate different expiry)
        chain2 = chain_with_iv.copy()
        chain2['T'] = 60 / 365
        smile2 = extract_smile_at_expiry(chain2)
        
        surface = build_iv_surface([smile1, smile2])
        
        assert len(surface) == len(smile1) + len(smile2)
        assert 'T' in surface.columns
        assert 'log_moneyness' in surface.columns
        assert 'iv' in surface.columns
    
    def test_surface_no_nans(self, chain_with_iv):
        """Surface should not contain NaN IVs."""
        smile = extract_smile_at_expiry(chain_with_iv)
        surface = build_iv_surface([smile])
        
        assert not surface['iv'].isna().any()
    
    def test_empty_list_raises(self):
        """Should raise error for empty smile list."""
        with pytest.raises(ValueError):
            build_iv_surface([])


# =============================================================================
# INTEGRATION TEST
# =============================================================================

class TestEndToEnd:
    """Test full pipeline from prices to surface."""
    
    def test_pipeline_call_to_surface(self, simple_option_chain):
        """Full pipeline: raw prices → IV → smile → surface."""
        # Step 1: Compute IVs
        chain = compute_implied_vols(simple_option_chain, r=0.05)
        
        # Step 2: Add moneyness
        chain = add_moneyness_columns(chain)
        
        # Step 3: Build composite smile
        smile = extract_smile_at_expiry(chain)
        
        # Step 4: Build surface
        surface = build_iv_surface([smile])
        
        # Verify outputs are sensible
        assert not surface.empty
        assert len(surface) >= 3  # At least few points
        assert surface['iv'].min() > 0.01  # Reasonable vol
        assert surface['iv'].max() < 1.0   # Not too high


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
