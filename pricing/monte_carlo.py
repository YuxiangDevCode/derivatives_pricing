"""
Monte Carlo simulation for option pricing.

This module implements Monte Carlo methods for pricing European and American options.
Useful for path-dependent options, higher-dimensional problems, and numerical validation.
"""

import numpy as np
import scipy.stats as stats
from typing import Tuple

def simulate_paths_gbm(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    num_steps: int,
    num_paths: int,
    seed: int | None = None,
    shocks: np.ndarray | None = None,
    q: float = 0.0,
) -> np.ndarray:
    """
    Simulate stock price paths using geometric Brownian motion (GBM).
    
    Uses the discretized GBM equation:
    S(t+dt) = S(t) * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    
    More efficiently computed via log-returns:
    log-return = ln(S(t + dt)/S(t)) = (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
    
    Then: S(T) = S(0) * exp(cumsum of log-returns)
    
    This vectorized approach is ~100x faster than iterative computation.
    
    Parameters
    ----------
    S0 : float
        Initial stock price
    r : float
        Risk-free interest rate (annual, continuously compounded)
    sigma : float
        Volatility of the underlying stock (annual standard deviation)
    T : float
        Time to expiration (years)
    num_steps : int
        Number of time steps
    num_paths : int
        Number of simulation paths
    seed : int, optional
        Random seed for reproducibility. Ignored when `shocks` is supplied.
        `seed=0` is a valid seed; `seed=None` draws fresh entropy.
    shocks : np.ndarray, optional
        Pre-drawn standard normal shocks of shape (num_paths, num_steps).
        Supplying them makes the simulation deterministic given the array and
        is how common random numbers are threaded through bumped revaluations.
    q : float, optional
        Continuous dividend yield (annual). Enters the drift as (r - q);
        it does not affect discounting.
    
    Returns
    -------
    np.ndarray
        Array of shape (num_paths, num_steps + 1) containing simulated paths.
        Each row is one complete path from S0 to maturity.
    
    Example
    -------
    >>> paths = simulate_paths_gbm(S0=100, r=0.05, sigma=0.2, T=1.0, 
                               num_steps=252, num_paths=10000)
    >>> # paths.shape = (10000, 253)
    
    Notes
    -----
    Alternative iterative approach (for completeness, not recommended):
    
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = S0
        Z = np.random.normal(size=(num_paths, num_steps))
        for t in range(1, num_steps + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(
                (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
            )
    
    This iterative method shows the update rule explicitly but is much slower
    due to repeated Python loop overhead. The vectorized version below is preferred
    for production use.
    """
    if shocks is None:
        # default_rng(None) draws fresh entropy; default_rng(0) is a valid,
        # reproducible seed. Note np.random.seed() is deliberately not used:
        # it mutates global RNG state shared with every other module.
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((num_paths, num_steps))
    else:
        # Caller-supplied shocks enable common random numbers (CRN): pass the
        # same array to bumped revaluations so finite-difference Greeks
        # difference out the simulation noise instead of compounding it.
        Z = np.asarray(shocks, dtype=float)
        if Z.shape != (num_paths, num_steps):
            raise ValueError(
                f"shocks must have shape {(num_paths, num_steps)}, got {Z.shape}"
            )

    dt = T / num_steps

    # Vectorized implementation using log-returns.
    # Drift is the cost of carry (r - q)
    log_returns = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_prices = np.cumsum(log_returns, axis=1)
    log_prices = np.hstack((np.zeros((num_paths, 1)), log_prices))
    paths = S0 * np.exp(log_prices)
    return paths

def monte_carlo_option_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    num_paths: int = 10000,
    num_steps: int = 252,
    seed: int = None,
) -> float:
    """
    Price a European option using Monte Carlo simulation.
    
    Simulates multiple stock price paths under the risk-neutral measure,
    calculates payoffs at maturity, and discounts back to present value.
    
    Parameters
    ----------
    S0 : float
        Current spot price
    K : float
        Strike price
    r : float
        Risk-free rate (annual, continuously compounded)
    sigma : float
        Volatility (annual standard deviation)
    T : float
        Time to expiration (years)
    option_type : str, optional
        Type of option, 'call' or 'put' (default is 'call')
    num_paths : int, optional
        Number of simulation paths (default is 10,000)
    num_steps : int, optional
        Number of time steps per path (default is 252)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    float
        Estimated option price
    
    Raises
    ------
    ValueError
        If option_type is not 'call' or 'put'
    
    Example
    -------
    >>> price = monte_carlo_option_price(S0=100, K=100, r=0.05, sigma=0.2,
    ...                                   T=1.0, option_type='call')
    """

    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")
    paths = simulate_paths_gbm(S0, r, sigma, T, num_steps, num_paths, seed)
    S_T = paths[:, -1]
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:  # put option
        payoffs = np.maximum(K - S_T, 0)
    discounted_payoff = np.exp(-r * T) * np.mean(payoffs)
    
    return discounted_payoff

def monte_carlo_option_price_with_ci(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    num_paths: int = 10000,
    num_steps: int = 252,
    confidence: float = 0.95,
    seed: int = None,
) -> dict:
    """
    Price a European option using Monte Carlo with confidence interval.
    
    Returns the estimated price along with a confidence interval reflecting
    the uncertainty in the Monte Carlo estimate. Useful for understanding
    estimation precision.
    
    Parameters
    ----------
    S0 : float
        Current spot price
    K : float
        Strike price
    r : float
        Risk-free rate (annual, continuously compounded)
    sigma : float
        Volatility (annual standard deviation)
    T : float
        Time to expiration (years)
    option_type : str, optional
        Type of option, 'call' or 'put' (default is 'call')
    num_paths : int, optional
        Number of simulation paths (default is 10,000)
    num_steps : int, optional
        Number of time steps per path (default is 252)
    confidence : float, optional
        Confidence level for interval (default is 0.95 for 95%)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary with keys:
        
        - 'price' : float
            Estimated option price
        - 'std_error' : float
            Standard error of the estimate
        - 'ci_lower' : float
            Lower bound of confidence interval
        - 'ci_upper' : float
            Upper bound of confidence interval
    
    Example
    -------
    >>> result = monte_carlo_option_price_with_ci(S0=100, K=100, r=0.05,
    ...                                            sigma=0.2, T=1.0)
    >>> print(f"Price: ${result['price']:.2f} +/- ${result['std_error']:.2f}")
    """
    paths = simulate_paths_gbm(S0, r, sigma, T, num_steps, num_paths, seed)
    S_T = paths[:, -1]
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:  # put option
        payoffs = np.maximum(K - S_T, 0)
    discounted_payoffs = np.exp(-r * T) * payoffs
    price_estimate = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(num_paths)
    # Confidence interval calculation
    z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
    ci_lower = price_estimate - z_score * std_error
    ci_upper = price_estimate + z_score * std_error

    return {
        "price": price_estimate,
        "std_error": std_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def compare_with_black_scholes(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    bs_price: float,
    option_type: str = "call",
    num_paths: int = 10000,
    seed: int = None,
) -> dict:
    """
    Compare Monte Carlo estimate with Black-Scholes analytical price.
    
    Calculates the Monte Carlo price and compares it to the analytical
    Black-Scholes price. Useful for validating implementation, understanding
    convergence behavior, and benchmarking accuracy.
    
    Parameters
    ----------
    S0 : float
        Current spot price
    K : float
        Strike price
    r : float
        Risk-free rate (annual, continuously compounded)
    sigma : float
        Volatility (annual standard deviation)
    T : float
        Time to expiration (years)
    bs_price : float
        Black-Scholes analytical price (pre-calculated)
    option_type : str, optional
        Type of option, 'call' or 'put' (default is 'call')
    num_paths : int, optional
        Number of simulation paths (default is 10,000)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary with keys:
        
        - 'mc_price' : float
            Monte Carlo estimate
        - 'bs_price' : float
            Black-Scholes analytical price
        - 'absolute_error' : float
            Absolute difference: |mc_price - bs_price|
        - 'relative_error' : float
            Relative error as percentage: (absolute_error / bs_price) * 100
    
    Example
    -------
    >>> from pricing.black_scholes import black_scholes
    >>> bs_price = black_scholes(100, 100, 1.0, 0.05, 0.2, 'call')
    >>> comparison = compare_with_black_scholes(100, 100, 0.05, 0.2, 1.0,
    ...                                         bs_price, num_paths=100000)
    >>> print(f"MC Price: {comparison['mc_price']:.4f}")
    >>> print(f"Relative Error: {comparison['relative_error']:.2f}%")
    """

    mc_price = monte_carlo_option_price(S0, K, r, sigma, T, option_type, num_paths, seed)
    absolute_error = abs(mc_price - bs_price)
    relative_error = (absolute_error / bs_price) * 100 if bs_price != 0 else np.inf
    
    return {
        "mc_price": mc_price,
        "bs_price": bs_price,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
    }

def _barrier_survival(paths, barrier):
    """
    Discrete-monitoring survival indicator for a DOWN barrier.

    Monitoring is on the simulation grid only, including t=0 (column 0 is S0,
    so a spot already at or below the barrier is dead at inception) and T.
    The touch convention is strict: min > barrier survives, so touching the
    barrier exactly knocks out. This matches the closed form, which returns
    exactly 0 at S0 == barrier.

    This is NOT 1{min_t S_t > H}. It is 1{min_i S_{t_i} > H}, which misses
    intra-step crossings and therefore OVERPRICES a knock-out. The bias is
    O(sqrt(dt)) -- see _brownian_bridge_survival_weight for the correction.

    Returns a boolean array of shape (n_paths,).
    """
    return np.min(paths, axis=1) > barrier


def _brownian_bridge_survival_weight(paths, barrier, sigma, dt):
    """
    Continuous-monitoring survival weight for a DOWN barrier, per path.

    Conditional on the two endpoints of a step, the log-price in between is a
    Brownian bridge, so the probability it crossed the barrier is known in
    closed form. Taking that expectation instead of a 0/1 sample removes the
    O(sqrt(dt)) discretisation bias AND makes the weight a smooth function of
    S0, which is what stabilises bump-and-revalue Greeks.

    Should return a float array of shape (n_paths,) with values in [0, 1],
    interchangeable with _barrier_survival(...).astype(float).

    TODO: implement.
      d      = log(paths / barrier), clipped at 0 from below
      p_hit  = exp(-2 * d[:, :-1] * d[:, 1:] / (sigma**2 * dt))   per step
      return prod(1 - p_hit, axis=1)
    Clipping BEFORE multiplying handles a breached endpoint (distance 0 ->
    p_hit 1 -> weight 0) and guarantees the exponent is never positive, so
    exp() cannot overflow.
    """
    raise NotImplementedError(
        "Brownian bridge survival weight is not implemented yet; "
        "call with use_brownian_bridge=False."
    )


def _price_barrier_call_mc(
    S: float,
    K: float,
    barrier: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    knock_out: bool,
    seed: int | None = None,
    use_brownian_bridge: bool = False,
    shocks: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Shared engine for down-and-out / down-and-in European calls.

    The knock-in leg is simulated independently (weight = 1 - survival), not
    derived from the knock-out leg by parity. That keeps
    `DO + DI == Black-Scholes vanilla` a genuine numerical test of the whole
    pipeline rather than an algebraic identity.

    Returns (price, standard_error).
    """
    _validate_barrier_mc_inputs(S, K, barrier, T, sigma, n_paths, n_steps)

    paths = simulate_paths_gbm(S, r, sigma, T, n_steps, n_paths, seed, shocks, q)
    terminal_payoff = np.maximum(paths[:, -1] - K, 0.0)

    if use_brownian_bridge:
        survival = _brownian_bridge_survival_weight(paths, barrier, sigma, T / n_steps)
    else:
        survival = _barrier_survival(paths, barrier).astype(float)

    weight = survival if knock_out else 1.0 - survival

    discounted_payoffs = np.exp(-r * T) * terminal_payoff * weight
    price = float(np.mean(discounted_payoffs))
    std_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths))
    return price, std_error


def price_down_and_out_call_mc(
    S: float,
    K: float,
    barrier: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    use_brownian_bridge: bool = False,
    shocks: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Down-and-out European call, discrete monitoring on the simulation grid.

    Returns:
        price, standard_error
    """
    return _price_barrier_call_mc(
        S, K, barrier, T, r, q, sigma, n_paths, n_steps,
        knock_out=True, seed=seed,
        use_brownian_bridge=use_brownian_bridge, shocks=shocks,
    )


def price_down_and_in_call_mc(
    S: float,
    K: float,
    barrier: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    use_brownian_bridge: bool = False,
    shocks: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Down-and-in European call, discrete monitoring on the simulation grid.

    Simulated independently of the knock-out leg so that
    `DO + DI == vanilla` can be checked against the analytic Black-Scholes
    price as a real validation of drift, discounting and payoff.

    Returns:
        price, standard_error
    """
    return _price_barrier_call_mc(
        S, K, barrier, T, r, q, sigma, n_paths, n_steps,
        knock_out=False, seed=seed,
        use_brownian_bridge=use_brownian_bridge, shocks=shocks,
    )

def _validate_barrier_mc_inputs(
    S: float,
    K: float,
    barrier: float,
    T: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
) -> None:
    if S <= 0:
        raise ValueError("Spot must be positive.")

    if K <= 0:
        raise ValueError("Strike must be positive.")

    if barrier <= 0:
        raise ValueError("Barrier must be positive.")

    if T <= 0:
        raise ValueError("Maturity must be positive.")

    if sigma < 0:
        raise ValueError("Volatility cannot be negative.")

    if n_paths < 2:
        raise ValueError("n_paths must be at least 2.")

    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")