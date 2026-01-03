"""
Greeks calculation for Black-Scholes options.

Greeks measure option sensitivity to market parameters:
- Delta: Price sensitivity to spot price changes
- Gamma: Delta sensitivity to spot price changes
- Vega: Price sensitivity to volatility changes
- Theta: Price sensitivity to time decay
- Rho: Price sensitivity to interest rate changes
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple
from pricing.black_scholes import compute_d1_d2


def compute_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call'
) -> dict:
    """
    Compute all Greeks for a Black-Scholes option.
    
    Parameters
    ----------
    S : float
        Current spot price
    K : float
        Strike price
    T : float
        Time to expiration (years)
    r : float
        Risk-free rate (annual)
    sigma : float
        Volatility (annual standard deviation)
    option_type : str
        'call' or 'put'
    
    Returns
    -------
    dict
        Dictionary with keys: delta, gamma, vega, theta, rho
    
    Example
    -------
    >>> greeks = compute_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2)
    >>> print(f"Delta: {greeks['delta']:.4f}")
    """
    # Step 1: Validate option_type - it must be 'call' or 'put'
    if option_type not in ['call', 'put']:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Step 2: Calculate d1 and d2
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)

    # Step 3: Calculate each Greek using the formulas and return as a dictionary
    return {
            'delta': delta(S, K, T, r, sigma, option_type),
            'gamma': gamma(S, K, T, r, sigma),
            'vega': vega(S, K, T, r, sigma),
            'theta': theta(S, K, T, r, sigma, option_type),
            'rho': rho(S, K, T, r, sigma, option_type)
        }


def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """
    Sensitivity to spot price changes.
    
    Call: Δ = N(d1)
    Put: Δ = N(d1) - 1
    """
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

    return delta


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Sensitivity of delta to spot price changes.
    
    Γ = n(d1) / (S * σ * √T)
    """
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    return gamma


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Sensitivity to volatility changes (per 1% change).
    
    ν = S * n(d1) * √T / 100
    """
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    return vega

def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """
    Sensitivity to time decay (per calendar day).
    
    Call: Θ = [-S*n(d1)*σ/(2√T) - r*K*e^(-rT)*N(d2)] / 365
    Put: Θ = [-S*n(d1)*σ/(2√T) + r*K*e^(-rT)*N(-d2)] / 365
    """
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    if option_type == 'call':
        theta_value = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta_value = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    return theta_value

def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """
    Sensitivity to interest rate changes (per 1 bp rate change).
    
    Call: ρ = K*T*e^(-rT)*N(d2) / 100
    Put: ρ = -K*T*e^(-rT)*N(-d2) / 100
    """
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)

    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 10000
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 10000

    return rho
