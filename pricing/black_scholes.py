import numpy as np
from scipy.stats import norm


def compute_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple:
    """
    Helper function to compute d1 and d2 for Black-Scholes formula.
    
    The Black-Scholes formula requires two intermediate calculations:
    d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
    d2 = d1 - σ√T
    
    Parameters
    ----------
    S : float
        Current stock price (spot price)
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annual, continuously compounded)
    sigma : float
        Volatility of the underlying stock (annual standard deviation)
    
    Returns
    -------
    tuple
        (d1, d2) as calculated from the Black-Scholes formula
    
    Raises
    ------
    ValueError
        If any input parameters are invalid (negative values where not allowed, T=0, etc.)
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError(
            f"Invalid parameters: S={S}, K={K}, T={T}, sigma={sigma}. "
            "All must be positive."
        )
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return d1, d2


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call"
) -> float:
    """
    Calculate the Black-Scholes price for a European option.
    
    This function implements the Black-Scholes formula for pricing European-style
    options (call or put) on a non-dividend-paying stock.
    
    Call option: C = S·N(d1) - K·e^(-rT)·N(d2)
    Put option:  P = K·e^(-rT)·N(-d2) - S·N(-d1)
    
    Parameters
    ----------
    S : float
        Current stock price (spot price)
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annual, continuously compounded)
    sigma : float
        Volatility of the underlying stock (annual standard deviation)
    option_type : str, optional
        Type of option: 'call' (default) or 'put'
    
    Returns
    -------
    float
        The option price
    
    Raises
    ------
    ValueError
        If option_type is not 'call' or 'put'
        If any input parameters are invalid
    
    Examples
    --------
    >>> # Price an ATM call option with 1 year to expiration
    >>> call_price = black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2)
    >>> print(f"Call price: ${call_price:.2f}")
    Call price: $10.45
    """
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return option_price
