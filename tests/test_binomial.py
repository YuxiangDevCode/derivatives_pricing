"""
Unit tests for the CRR binomial pricing module.

Five tests, one per validation category:
1. trusted benchmark      - European tree converges to Black-Scholes
2. independent benchmark  - vectorized engine agrees with the naive reference
3. financial invariants   - American vs European relationships
4. convergence            - error shrinks with the number of steps
5. edge cases / guards    - invalid inputs are rejected rather than priced
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from pricing.binomial import crr_binomial, crr_binomial_reference
from pricing.black_scholes import black_scholes


def test_european_tree_converges_to_black_scholes():
    """Trusted benchmark: with many steps the European tree must reproduce BS."""
    S, K, T, r, sigma, N = 100.0, 100.0, 1.0, 0.05, 0.2, 5000

    for option_type in ('call', 'put'):
        for q in (0.0, 0.03):
            tree = crr_binomial(S, K, T, r, sigma, N, option_type, 'european', q)
            # black_scholes has no dividend yield: price the q>0 case off the
            # dividend-adjusted spot, which is the standard equivalence.
            analytic = black_scholes(S * np.exp(-q * T), K, T, r, sigma, option_type)
            assert tree == pytest.approx(analytic, abs=1e-3)

    # Moneyness sweep, no dividends
    for spot in (80.0, 95.0, 100.0, 105.0, 120.0):
        for option_type in ('call', 'put'):
            tree = crr_binomial(spot, K, T, r, sigma, N, option_type, 'european')
            assert tree == pytest.approx(black_scholes(spot, K, T, r, sigma, option_type), abs=1e-3)


def test_vectorized_matches_naive_reference():
    """Independent benchmark: two implementations that share no induction logic."""
    rng = np.random.default_rng(20260830)
    checked = 0

    for _ in range(60):
        S = rng.uniform(50, 150)
        K = rng.uniform(50, 150)
        T = rng.uniform(0.05, 3.0)
        r = rng.uniform(-0.01, 0.10)
        sigma = rng.uniform(0.10, 0.80)
        N = int(rng.integers(1, 40))
        option_type = str(rng.choice(['call', 'put']))
        exercise_type = str(rng.choice(['european', 'american']))
        q = rng.uniform(0.0, 0.08)

        fast = crr_binomial(S, K, T, r, sigma, N, option_type, exercise_type, q)
        slow = crr_binomial_reference(S, K, T, r, sigma, N, option_type, exercise_type, q)

        assert fast == pytest.approx(slow, rel=1e-11, abs=1e-11)
        checked += 1

    assert checked == 60, "every sampled parameter set should be priceable"


def test_american_invariants():
    """Financial invariants that must hold regardless of discretization."""
    S, K, T, r, sigma, N = 100.0, 100.0, 1.0, 0.05, 0.2, 500

    # American is never worth less than European
    for option_type in ('call', 'put'):
        american = crr_binomial(S, K, T, r, sigma, N, option_type, 'american')
        european = crr_binomial(S, K, T, r, sigma, N, option_type, 'european')
        assert american >= european - 1e-12

    # Without dividends, early exercise of a call is never optimal: exactly equal
    american_call = crr_binomial(S, K, T, r, sigma, N, 'call', 'american')
    european_call = crr_binomial(S, K, T, r, sigma, N, 'call', 'european')
    assert american_call == pytest.approx(european_call, abs=1e-14)

    # With q > r the call acquires a strictly positive early-exercise premium
    american_call_q = crr_binomial(S, K, T, r, sigma, N, 'call', 'american', q=0.08)
    european_call_q = crr_binomial(S, K, T, r, sigma, N, 'call', 'european', q=0.08)
    assert american_call_q > european_call_q + 1e-6

    # The American put dominates its intrinsic value everywhere...
    american_put = crr_binomial(S, K, T, r, sigma, N, 'put', 'american')
    assert american_put >= max(K - S, 0.0)

    # ...and deep in the money it equals intrinsic exactly (exercise immediately)
    deep_itm = crr_binomial(10.0, K, T, r, sigma, 200, 'put', 'american')
    assert deep_itm == pytest.approx(K - 10.0, abs=1e-10)


def test_convergence_error_shrinks_with_steps():
    """Error must decay ~O(1/N). Compared at fixed parity: CRR error oscillates
    with the parity of N, so mixing even and odd N gives a non-monotone series."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    analytic = black_scholes(S, K, T, r, sigma, 'put')

    errors = [abs(crr_binomial(S, K, T, r, sigma, N, 'put', 'european') - analytic)
              for N in (50, 200, 800)]

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]

    # First-order convergence: 50 -> 800 is 16x the steps, so the error should
    # fall by roughly 16. Generous band to absorb the oscillating component.
    assert 8.0 < errors[0] / errors[2] < 32.0

    # The oscillation itself: adjacent N straddle the true value, so averaging
    # them is markedly more accurate than either alone.
    even = crr_binomial(S, K, T, r, sigma, 100, 'put', 'american')
    odd = crr_binomial(S, K, T, r, sigma, 101, 'put', 'american')
    converged = crr_binomial(S, K, T, r, sigma, 20000, 'put', 'american')
    assert min(even, odd) < converged < max(even, odd)
    assert abs((even + odd) / 2 - converged) < abs(even - converged)


def test_invalid_inputs_are_rejected():
    """Guards: bad inputs must raise, never return a plausible wrong price."""
    S, K, T, r, sigma, N = 100.0, 100.0, 1.0, 0.05, 0.2, 50

    with pytest.raises(ValueError):
        crr_binomial(S, K, T, r, sigma, N, 'straddle', 'european')

    with pytest.raises(ValueError):
        crr_binomial(S, K, T, r, sigma, N, 'call', 'bermudan')

    for bad in ({'S': -100.0}, {'K': 0.0}, {'N': 0}, {'N': 10.5}, {'q': -0.02}):
        kwargs = dict(S=S, K=K, T=T, r=r, sigma=sigma, N=N,
                      option_type='call', exercise_type='european')
        kwargs.update(bad)
        with pytest.raises(ValueError):
            crr_binomial(**kwargs)

    # Time step too coarse for the drift: p leaves (0, 1) and the recursion stops
    # being a discounted expectation. Must raise rather than price it.
    with pytest.raises(ValueError, match="probability"):
        crr_binomial(100.0, 100.0, 1.0, 0.5, 0.1, 2, 'call', 'european')
