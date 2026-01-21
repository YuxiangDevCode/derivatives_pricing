def implied_volatility(S, K, T, r, market_price, option_type, initial_guess=0.2, tol=1e-6, max_iterations=100):
    """
    Calculate the implied volatility using BS pricing.
    This function uses a numerical method (e.g., Newton-Raphson) to find the volatility
    that matches the market price of the option.
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
    market_price : float
        Market price of the option
    option_type : str
        'call' or 'put'
    initial_guess : float
        Initial guess for volatility
    tol : float
        Tolerance for convergence
    max_iterations : int
        Maximum number of iterations    
    
    Returns
    -------
    float
        Implied volatility as a decimal (e.g., 0.2 for 20%)
    Raises
    ------
    ValueError
        If the implied volatility cannot be found within the maximum iterations
    Example
    -------
    >>> iv = implied_volatility(S=100, K=100, T=1, r=0.05, market_price=10, option_type='call')
    >>> print(f"Implied Volatility: {iv:.2%}")
    Implied Volatility: 20.00%  
    """
    