"""
Unit tests for the Black-Scholes option pricing module.
These tests cover basic functionality, moneyness scenarios, put-call parity,
Greek properties, and input validation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from pricing.black_scholes import black_scholes, compute_d1_d2




class TestBlackScholesBasic:
    """Core functionality tests."""
    
    def test_call_option_pricing(self):
        """Test basic call option pricing."""
        call = black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        assert call > 0
        assert np.isclose(call, 10.45, rtol=0.01)
    
    def test_put_option_pricing(self):
        """Test basic put option pricing."""
        put = black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put')
        assert put > 0
        assert np.isclose(put, 5.57, rtol=0.01)
    
    def test_default_option_type_is_call(self):
        """Test that default option type is 'call'."""
        call_default = black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
        call_explicit = black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        assert call_default == call_explicit


class TestMoneyness:
    """Test pricing across different moneyness scenarios."""
    
    def test_itm_call_more_than_atm(self):
        """In-the-money call should be more expensive."""
        atm = black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        itm = black_scholes(S=110, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        assert itm > atm
    
    def test_otm_call_less_than_atm(self):
        """Out-of-the-money call should be less expensive."""
        atm = black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        otm = black_scholes(S=90, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        assert otm < atm
    
    def test_itm_put_more_than_atm(self):
        """In-the-money put should be more expensive."""
        atm = black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put')
        itm = black_scholes(S=90, K=100, T=1, r=0.05, sigma=0.2, option_type='put')
        assert itm > atm
    
    def test_deep_itm_call_near_intrinsic(self):
        """Deep ITM call approaches intrinsic value with minimal time value."""
        call = black_scholes(S=150, K=100, T=0.01, r=0.05, sigma=0.2, option_type='call')
        intrinsic = 150 - 100
        time_value = call - intrinsic
        # Time value should be small (< 5% of intrinsic) for deep ITM with short expiration
        assert time_value > 0  # Always positive time value
        assert time_value / intrinsic < 0.05  # Less than 5% premium over intrinsic
    
    def test_deep_otm_call_near_zero(self):
        """Deep OTM call has minimal value (mostly time value)."""
        call = black_scholes(S=50, K=100, T=0.01, r=0.05, sigma=0.2, option_type='call')
        # Call is worthless at expiration (intrinsic = 0)
        # Should be small enough that it doesn't exceed intrinsic of deep ITM call
        intrinsic_otm = max(50 - 100, 0)  # 0 for OTM call
        assert call > intrinsic_otm  # Has some time value
        assert call < 1  # But still very small


class TestPutCallParity:
    """Test put-call parity: C - P = S - K*e^(-rT)."""
    
    def test_parity_holds(self):
        """Put-call parity should always hold."""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
        
        call = black_scholes(S, K, T, r, sigma, option_type='call')
        put = black_scholes(S, K, T, r, sigma, option_type='put')
        
        lhs = call - put
        rhs = S - K * np.exp(-r * T)
        
        assert np.isclose(lhs, rhs, rtol=1e-10)
    
    def test_parity_different_strikes(self):
        """Parity should hold for different strike prices."""
        for S, K in [(100, 80), (100, 100), (100, 120)]:
            call = black_scholes(S, K, T=1, r=0.05, sigma=0.2, option_type='call')
            put = black_scholes(S, K, T=1, r=0.05, sigma=0.2, option_type='put')
            
            lhs = call - put
            rhs = S - K * np.exp(-0.05 * 1)
            
            assert np.isclose(lhs, rhs, rtol=1e-10)


class TestGreekProperties:
    """Test economic properties (delta, vega, theta)."""
    
    def test_higher_volatility_increases_both_options(self):
        """Both call and put prices increase with volatility."""
        S, K, T, r = 100, 100, 1, 0.05
        
        call_low = black_scholes(S, K, T, r, sigma=0.1, option_type='call')
        call_high = black_scholes(S, K, T, r, sigma=0.5, option_type='call')
        
        put_low = black_scholes(S, K, T, r, sigma=0.1, option_type='put')
        put_high = black_scholes(S, K, T, r, sigma=0.5, option_type='put')
        
        assert call_high > call_low
        assert put_high > put_low
    
    def test_longer_expiration_increases_otm_options(self):
        """OTM options are more valuable with more time."""
        S, K, r, sigma = 100, 110, 0.05, 0.2
        
        short = black_scholes(S, K, T=0.1, r=r, sigma=sigma, option_type='call')
        long = black_scholes(S, K, T=1, r=r, sigma=sigma, option_type='call')
        
        assert long > short


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_invalid_option_type_raises_error(self):
        """Invalid option_type should raise ValueError."""
        with pytest.raises(ValueError, match="must be 'call' or 'put'"):
            black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='invalid')
    
    def test_negative_spot_raises_error(self):
        """Negative spot price should raise error."""
        with pytest.raises(ValueError, match="must be positive"):
            black_scholes(S=-100, K=100, T=1, r=0.05, sigma=0.2)
    
    def test_zero_time_raises_error(self):
        """Zero time to expiration should raise error."""
        with pytest.raises(ValueError, match="must be positive"):
            black_scholes(S=100, K=100, T=0, r=0.05, sigma=0.2)
    
    def test_negative_volatility_raises_error(self):
        """Negative volatility should raise error."""
        with pytest.raises(ValueError, match="must be positive"):
            black_scholes(S=100, K=100, T=1, r=0.05, sigma=-0.2)


class TestGreeks:
    """Test Greeks calculation."""
    
    def test_delta_call_atm(self):
        """ATM call delta should match reference value (0.6368 for S=K=100, T=1, r=0.05, σ=0.2)."""
        from pricing.greeks import delta
        d = delta(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        # Reference value from online BS calculator
        assert np.isclose(d, 0.6368, atol=0.0001)
    
    def test_gamma_positive(self):
        """Gamma should always be positive."""
        from pricing.greeks import gamma
        g = gamma(S=100, K=100, T=1, r=0.05, sigma=0.2)
        assert g > 0
    
    def test_vega_positive(self):
        """Vega should always be positive."""
        from pricing.greeks import vega
        v = vega(S=100, K=100, T=1, r=0.05, sigma=0.2)
        assert v > 0
    
    def test_compute_greeks_call(self):
        """Test compute_greeks for call option."""
        from pricing.greeks import compute_greeks
        greeks = compute_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
        
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'vega' in greeks
        assert 'theta' in greeks
        assert 'rho' in greeks
        
        # Test with precise reference values from implementation
        # S=100, K=100, T=1, r=0.05, σ=0.2 (ATM option, 1 year expiration)
        assert np.isclose(greeks['delta'], 0.636831, atol=0.0001)
        assert np.isclose(greeks['gamma'], 0.018762, atol=0.0001)
        assert np.isclose(greeks['vega'], 0.375240, atol=0.0001)
        assert np.isclose(greeks['theta'], -0.017573, atol=0.0001)
        assert np.isclose(greeks['rho'], 0.005323, atol=0.0001)
    
    def test_compute_greeks_put(self):
        """Test compute_greeks for put option."""
        from pricing.greeks import compute_greeks
        greeks = compute_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put')
        
        # Test with precise reference values from implementation
        # ATM put: delta should be negative of call relationship
        assert np.isclose(greeks['delta'], -0.363169, atol=0.0001)
        # Gamma is the same for calls and puts
        assert np.isclose(greeks['gamma'], 0.018762, atol=0.0001)
        # Vega is the same for calls and puts
        assert np.isclose(greeks['vega'], 0.375240, atol=0.0001)
        # Theta for put (different sign from call)
        assert np.isclose(greeks['theta'], -0.004542, atol=0.0001)
        # Rho for put (negative)
        assert np.isclose(greeks['rho'], -0.004189, atol=0.0001)


class TestHelperFunction:
    """Test the compute_d1_d2 helper function."""
    
    def test_d1_greater_than_d2(self):
        """d1 should always be greater than d2."""
        d1, d2 = compute_d1_d2(S=100, K=100, T=1, r=0.05, sigma=0.2)
        assert d1 > d2
    
    def test_d1_d2_relationship(self):
        """d2 should equal d1 - σ√T."""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
        d1, d2 = compute_d1_d2(S, K, T, r, sigma)
        
        expected_d2 = d1 - sigma * np.sqrt(T)
        assert np.isclose(d2, expected_d2)
    
    def test_helper_invalid_inputs(self):
        """Helper should validate inputs."""
        with pytest.raises(ValueError):
            compute_d1_d2(S=-100, K=100, T=1, r=0.05, sigma=0.2)

