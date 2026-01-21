import numpy as np
import pytest
from pricing.monte_carlo import simulate_paths_gbm, monte_carlo_option_price, monte_carlo_option_price_with_ci, compare_with_black_scholes
from pricing.black_scholes import black_scholes


class TestSimulatePathsGBM:
    """Test stock price path simulation."""
    
    def test_paths_shape(self):
        """Test that output shape is correct."""
        paths = simulate_paths_gbm(S0=100, r=0.05, sigma=0.2, T=1.0,
                                   num_steps=252, num_paths=1000, seed=42)
        assert paths.shape == (1000, 253), "Shape should be (num_paths, num_steps + 1)"
    
    def test_initial_price_correct(self):
        """Test that first column is S0."""
        S0 = 100
        paths = simulate_paths_gbm(S0=S0, r=0.05, sigma=0.2, T=1.0,
                                   num_steps=252, num_paths=1000, seed=42)
        assert np.allclose(paths[:, 0], S0), "All paths should start at S0"
    
    def test_prices_stay_positive(self):
        """Test that all simulated prices are positive."""
        paths = simulate_paths_gbm(S0=100, r=0.05, sigma=0.2, T=1.0,
                                   num_steps=252, num_paths=1000, seed=42)
        assert np.all(paths > 0), "All prices must remain positive (GBM property)"
    
    def test_seed_reproducibility(self):
        """Test that same seed produces identical results."""
        paths1 = simulate_paths_gbm(S0=100, r=0.05, sigma=0.2, T=1.0,
                                    num_steps=252, num_paths=100, seed=42)
        paths2 = simulate_paths_gbm(S0=100, r=0.05, sigma=0.2, T=1.0,
                                    num_steps=252, num_paths=100, seed=42)
        assert np.allclose(paths1, paths2), "Same seed should produce identical paths"


class TestMonteCarloOptionPrice:
    """Test European option pricing."""
    
    def test_atm_call_positive(self):
        """Test ATM call price is positive."""
        price = monte_carlo_option_price(S0=100, K=100, r=0.05, sigma=0.2,
                                         T=1.0, option_type="call", num_paths=10000, seed=42)
        assert price > 0, "ATM call price should be positive"
    
    def test_atm_put_positive(self):
        """Test ATM put price is positive."""
        price = monte_carlo_option_price(S0=100, K=100, r=0.05, sigma=0.2,
                                         T=1.0, option_type="put", num_paths=10000, seed=42)
        assert price > 0, "ATM put price should be positive"
    
    def test_itm_call_greater_than_atm(self):
        """Test ITM call > ATM call."""
        price_atm = monte_carlo_option_price(S0=100, K=100, r=0.05, sigma=0.2,
                                             T=1.0, option_type="call", num_paths=10000, seed=42)
        price_itm = monte_carlo_option_price(S0=110, K=100, r=0.05, sigma=0.2,
                                             T=1.0, option_type="call", num_paths=10000, seed=42)
        assert price_itm > price_atm, "ITM call should be worth more than ATM call"
    
    def test_invalid_option_type_raises_error(self):
        """Test that invalid option type raises ValueError."""
        with pytest.raises(ValueError):
            monte_carlo_option_price(S0=100, K=100, r=0.05, sigma=0.2,
                                    T=1.0, option_type="invalid", num_paths=10000, seed=42)
    
    def test_call_price_gte_intrinsic(self):
        """Test that call price >= intrinsic value."""
        S0, K = 110, 100
        price = monte_carlo_option_price(S0=S0, K=K, r=0.05, sigma=0.2,
                                         T=1.0, option_type="call", num_paths=10000, seed=42)
        intrinsic = max(S0 - K, 0)
        assert price >= intrinsic, "Call price should be at least intrinsic value"


class TestMonteCarloWithCI:
    """Test Monte Carlo pricing with confidence intervals."""
    
    def test_ci_structure(self):
        """Test that CI result has all required keys."""
        result = monte_carlo_option_price_with_ci(S0=100, K=100, r=0.05, sigma=0.2,
                                                   T=1.0, num_paths=10000, seed=42)
        required_keys = {'price', 'std_error', 'ci_lower', 'ci_upper'}
        assert set(result.keys()) == required_keys, "Result should have all required keys"
    
    def test_ci_bounds_valid(self):
        """Test that ci_lower < price < ci_upper."""
        result = monte_carlo_option_price_with_ci(S0=100, K=100, r=0.05, sigma=0.2,
                                                   T=1.0, num_paths=10000, seed=42)
        assert result['ci_lower'] < result['price'] < result['ci_upper'], \
            "Price should be within confidence interval bounds"
    
    def test_std_error_positive(self):
        """Test that standard error is positive."""
        result = monte_carlo_option_price_with_ci(S0=100, K=100, r=0.05, sigma=0.2,
                                                   T=1.0, num_paths=10000, seed=42)
        assert result['std_error'] > 0, "Standard error should be positive"


class TestComparisonWithBS:
    """Test comparison with Black-Scholes."""
    
    def test_comparison_structure(self):
        """Test that comparison result has all required keys."""
        bs_price = black_scholes(100, 100, 1.0, 0.05, 0.2, 'call')
        result = compare_with_black_scholes(S0=100, K=100, r=0.05, sigma=0.2,
                                           T=1.0, bs_price=bs_price, num_paths=100000, seed=42)
        required_keys = {'mc_price', 'bs_price', 'absolute_error', 'relative_error'}
        assert set(result.keys()) == required_keys, "Result should have all required keys"
    
    def test_error_metrics_nonnegative(self):
        """Test that error metrics are non-negative."""
        bs_price = black_scholes(100, 100, 1.0, 0.05, 0.2, 'call')
        result = compare_with_black_scholes(S0=100, K=100, r=0.05, sigma=0.2,
                                           T=1.0, bs_price=bs_price, num_paths=100000, seed=42)
        assert result['absolute_error'] >= 0, "Absolute error should be non-negative"
        assert result['relative_error'] >= 0, "Relative error should be non-negative"
    
    def test_mc_close_to_bs_large_sample(self):
        """Test that MC converges to BS with large sample size."""
        bs_price = black_scholes(100, 100, 1.0, 0.05, 0.2, 'call')
        result = compare_with_black_scholes(S0=100, K=100, r=0.05, sigma=0.2,
                                           T=1.0, bs_price=bs_price, num_paths=100000, seed=42)
        # With 100k paths, relative error should be small (< 2%)
        assert result['relative_error'] < 2.0, \
            f"MC should converge to BS. Error: {result['relative_error']:.2f}%"
