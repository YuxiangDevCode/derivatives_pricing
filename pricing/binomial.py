import numpy as np

def crr_binomial(S, K, T, r, sigma, N, option_type, exercise_type, q=0.0):
    """
    Calculate the option price using the Cox-Ross-Rubinstein (CRR) binomial model.

    Parameters
    ----------
    S : float
        Current spot price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate (annual, continuously compounded)
    sigma : float
        Volatility (annual standard deviation)
    N : int
        Number of time steps in the binomial tree
    option_type : str
        Type of option, 'call' or 'put'
    exercise_type : str
        Type of exercise, 'european' or 'american'
    q : float
        Dividend yield (annual, continuously compounded)

    Returns
    -------
    float
        Option price calculated using the CRR binomial model

    Raises
    ------
    ValueError
        If option_type is not 'call' or 'put'
        If exercise_type is not 'european' or 'american'
        If S, K, T, or N are not positive values
    Example
    -------
    >>> price = crr_binomial(S=100, K=100, T=1.0, r=0.05, sigma=0.2,
    ...                       N=100, option_type='call', exercise_type='european', q=0.02)
    """
    option_type = option_type.lower()
    exercise_type = exercise_type.lower()
    _input_validation(S, K, T, r, sigma, N, option_type, exercise_type, q)
    dt, u, d, p = _crr_parameters(T, r, sigma, N, q)
    discount_factor = np.exp(-r * dt)
    # Terminal State: N + 1 nodes, j is the number of up moves
    stock_prices = np.array([S * (u ** j) * (d ** (N - j)) for j in range(N + 1)])
    option_values = _payoff(stock_prices, K, option_type)
    # Backward induction
    for i in range(N - 1, -1, -1):
        # Continuation value
        option_values = discount_factor * (p * option_values[1:i + 2] + (1 - p) * option_values[0:i + 1])
        if exercise_type == "american":
            # Compare the continuation value with the intrinsic value for early exercise
            stock_prices = stock_prices[:i + 1] * u
            option_values = np.maximum(option_values, _payoff(stock_prices, K, option_type))

    return option_values[0]

def crr_binomial_reference(S, K, T, r, sigma, N, option_type, exercise_type, q=0.0, return_tree=False):
    """
    Not recommended for use. Use crr_binomial with exercise_type='european' instead.
    This is for completeness by not using vectorization and is less efficient than crr_binomial with exercise_type='european'.
    """
    option_type = option_type.lower()
    exercise_type = exercise_type.lower()
    _input_validation(S, K, T, r, sigma, N, option_type, exercise_type, q)
    dt, u, d, p = _crr_parameters(T, r, sigma, N, q)
    discount_factor = np.exp(-r * dt)
    # Terminal State: N + 1 nodes, j is the number of up moves
    stock_prices = [[np.nan] * (N + 1) for _ in range(N + 2)]
    option_values = [[np.nan] * (N + 1) for _ in range(N + 2)]
    for j in range(N + 1):
        stock_prices[N][j] = S * (u ** j) * (d ** (N - j))
        option_values[N][j] = _payoff(stock_prices[N][j], K, option_type)
    # Backward induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            stock_prices[i][j] = S * (u ** j) * (d ** (i - j))
            continuation_value = discount_factor * (p * option_values[i + 1][j + 1] + (1 - p) * option_values[i + 1][j])
            if exercise_type == "american":
                option_values[i][j] = max(continuation_value, _payoff(stock_prices[i][j], K, option_type))
            else:
                option_values[i][j] = continuation_value

    if return_tree:
        # Rows 0..N, columns 0..i are filled; the rest stay NaN by construction.
        return option_values[0][0], np.array(stock_prices), np.array(option_values)
    return option_values[0][0]

def _crr_parameters(T, r, sigma, N, q):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not (0 <= p <= 1):
        raise ValueError(f"Calculated risk-neutral probability p {p} is out of bounds [0, 1]. Check input parameters. " \
        "time step too coarse relative to volatility — increase N, or check for an unrealistically small sigma.")
    return dt, u, d, p

def _payoff(stock_prices, K, option_type):
    if option_type == "call":
        return np.maximum(stock_prices - K, 0)
    else:  # put option
        return np.maximum(K - stock_prices, 0)

def _input_validation(S, K, T, r, sigma, N, option_type, exercise_type, q):
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")
    if exercise_type not in ["european", "american"]:
        raise ValueError("exercise_type must be 'european' or 'american'")
    if S <= 0 or K <= 0 or T < 0 or N <= 0 or q < 0:
        raise ValueError("S, K, T, N, and q must be positive values")
    if int(N) != N:
        raise ValueError(f"N must be an integer number of steps, got {N}")