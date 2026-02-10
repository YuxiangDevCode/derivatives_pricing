import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from pricing.black_scholes import black_scholes
from pricing.greeks import vega
from scipy.optimize import brentq

# =============================================================================
# 0. IMPLIED VOLATILITY SOLVER
# =============================================================================

def implied_volatility(S, K, T, r, market_price, option_type, sigma_min=1e-4, sigma_max=5.0, tol=1e-8, max_iterations=100):
    """
    Compute the Black–Scholes implied volatility using a robust
    root-finding algorithm (Brent's method).

    The function enforces no-arbitrage price bounds and returns NaN
    when no arbitrage-free implied volatility exists.

    Parameters
    ----------
    S : float
        Spot price of the underlying
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free interest rate (continuously compounded)
    market_price : float
        Observed market option price
    option_type : str
        Option type: 'call' or 'put'
    sigma_min : float, optional
        Lower bound for volatility search
    sigma_max : float, optional
        Upper bound for volatility search
    tol : float, optional
        Root-finding tolerance
    max_iterations : int, optional
        Maximum number of iterations for the solver

    Returns
    -------
    float
        Implied volatility, or np.nan if no arbitrage-free solution exists

    Notes
    -----
    - Uses Brent's method, which is robust and guarantees convergence
      when a solution is bracketed.
    - Returns np.nan instead of raising an exception when implied
      volatility cannot be determined.
    """
    # Basic sanity checks (S, K, T > 0)
    if S <= 0 or K <= 0 or T <= 0:
        raise ValueError("S, K, and T must be positive values.")
    if option_type not in ['call', 'put']:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # no-arbitrage bounds check
    discount_factor = np.exp(-r * T)
    if option_type == 'call':
        intrinsic_value = max(0, S - K * discount_factor)
        upper_bound = S
    else:  # put
        intrinsic_value = max(0, K * discount_factor - S)
        upper_bound = K * discount_factor
    if not (intrinsic_value <= market_price <= upper_bound):
        return np.nan  # Price violates no-arbitrage bounds
    
    # define objective function for root finding
    def objective(sigma):
        price = black_scholes(S, K, T, r, sigma, option_type)
        return price - market_price
    # Use Brent's method for root finding
    try:
        iv = brentq(objective, sigma_min, sigma_max, xtol=tol, maxiter=max_iterations)
        return iv
    except ValueError:
        return np.nan  # Solver failed to converge

# =============================================================================
# 1. BATCH IV COMPUTATION FOR ONE OPTION CHAIN
# =============================================================================

def compute_implied_vols(
    option_chain: pd.DataFrame,
    r: float,
    sigma_min: float = 1e-4,
    sigma_max: float = 5.0
) -> pd.DataFrame:
    """
    Compute call, put, and mid implied volatilities for a cleaned option chain.

    Parameters
    ----------
    option_chain : pd.DataFrame
        Must contain columns:
        ['spot', 'strike', 'T', 'call_price', 'put_price']
    r : float
        Continuously compounded risk-free rate
    sigma_min : float
        Lower bound for implied volatility search
    sigma_max : float
        Upper bound for implied volatility search

    Returns
    -------
    pd.DataFrame
        Original dataframe with added columns:
        ['iv_call', 'iv_put']
"""
    if option_chain.empty:
        raise ValueError("Empty option chain dataframe")
    
    df= option_chain.copy()
    iv_calls = []
    iv_puts = []
    for _, row in df.iterrows():
        S = row['spot']
        K = row['strike']
        T = row['T']
        call_price = row['call_price']
        put_price = row['put_price']
        
        iv_call = implied_volatility(S, K, T, r, call_price, 'call', sigma_min, sigma_max)
        iv_put = implied_volatility(S, K, T, r, put_price, 'put', sigma_min, sigma_max)
        
        iv_calls.append(iv_call)
        iv_puts.append(iv_put)
    df['iv_call'] = iv_calls
    df['iv_put'] = iv_puts
    return df

# =============================================================================
# 2. Moneyness & ATM Tagging
# =============================================================================

def add_moneyness_columns(
    option_chain: pd.DataFrame,
    method: str = "spot"
) -> pd.DataFrame:
    """
    Add moneyness-related columns to an option chain.

    Parameters
    ----------
    option_chain : pd.DataFrame
        Must contain columns:
        ['spot', 'strike', 'T', 'call_price', 'put_price']
    method : str, optional
        Method used to define moneyness.
        Currently supported:
        - 'spot' : moneyness = K / S
        (forward-based moneyness may be added later)

    Returns
    -------
    pd.DataFrame
        Copy of `option_chain` with additional columns:
        - 'moneyness' : strike / spot
        - 'log_moneyness' : log(strike / spot)
        - 'is_atm' : True for strike closest to spot
    """
    if option_chain.empty:
        raise ValueError("Empty option chain dataframe")
    
    df = option_chain.copy()
    
    # Compute moneyness using CALL MONEYNESS convention
    # moneyness = S / K (increases as spot rises relative to strike)
    # This is the standard used for volatility surface construction
    df['moneyness'] = df['spot'] / df['strike']
    
    # Log-moneyness: log(S / K)
    # ATM corresponds to log_moneyness = 0 (when S = K)
    df['log_moneyness'] = np.log(df['moneyness'])
    
    # Mark ATM: strike closest to spot (moneyness closest to 1.0, log_moneyness closest to 0)
    distance_to_atm = np.abs(df['log_moneyness'])
    atm_idx = distance_to_atm.idxmin()
    df['is_atm'] = False
    df.loc[atm_idx, 'is_atm'] = True
    
    return df

def filter_otm_options(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    Select out-of-the-money (OTM) options only.

    Parameters
    ----------
    df : pd.DataFrame
        Option chain containing:
        - 'strike'
        - 'option_type' ('call' or 'put')
    spot : float
        Current spot price of the underlying.

    Returns
    -------
    pd.DataFrame
        Subset of `df` containing only OTM options:
        - Calls: strike >= spot
        - Puts:  strike <= spot

    Notes
    -----
    - Intended for market-quoted option selection (e.g. calibration, skew).
    - Not intended for smile visualization or surface construction.
    """
    if df.empty:
        raise ValueError("Empty dataframe")
    
    if 'option_type' not in df.columns:
        raise ValueError("DataFrame must contain 'option_type' column")
    
    # OTM calls: strike >= spot
    otm_calls = df[(df['option_type'] == 'call') & (df['strike'] >= spot)]
    
    # OTM puts: strike <= spot
    otm_puts = df[(df['option_type'] == 'put') & (df['strike'] <= spot)]
    
    # Combine and sort by strike
    otm_df = pd.concat([otm_calls, otm_puts], ignore_index=True).sort_values('strike').reset_index(drop=True)
    
    return otm_df



# =============================================================================
# 3. VOLATILITY SMILE EXTRACTION
# =============================================================================

def extract_option_smile(
    option_chain: pd.DataFrame,
    option_type: str = 'call'
) -> pd.DataFrame:
    """
    Extract implied volatility smile for a single option type (call or put)
    at one expiry.

    Parameters
    ----------
    option_chain : pd.DataFrame
        Must contain columns:
        ['spot', 'strike', 'T', 'call_price', 'put_price']  

    option_type : str
        Option type to extract:
        - 'call' → use iv_call
        - 'put'  → use iv_put

    Returns
    -------
    pd.DataFrame
        Smile dataframe with columns:
        - 'T'
        - 'strike'
        - 'log_moneyness'
        - 'iv'

    Notes
    -----
    - Intended for diagnostics and visualization
    - Not suitable for volatility surface fitting
    - Call and put smiles are intentionally kept separate

    """
    if option_chain.empty:
        raise ValueError("Empty option chain dataframe")
    if option_type not in ['call', 'put']:
        raise ValueError("option_type must be 'call' or 'put'")
    df = option_chain.copy()
    iv_column = 'iv_call' if option_type == 'call' else 'iv_put'
    if iv_column not in df.columns:
        raise ValueError(f"DataFrame must contain '{iv_column}' column")
    smile_df = df[['T', 'strike', 'log_moneyness', iv_column]]
    smile_df = smile_df.dropna(subset=[iv_column])
    smile_df = smile_df.rename(columns={iv_column: 'implied_volatility'})
    smile_df = smile_df.sort_values('log_moneyness').reset_index(drop=True)

    return smile_df

def extract_smile_at_expiry(
    option_chain: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct a tradable volatility smile for a single expiry.

    The smile is built using:
    - OTM puts for strikes below ATM
    - ATM option at-the-money
    - OTM calls for strikes above ATM

    This reflects standard equity volatility surface construction practice.

    Parameters
    ----------
    option_chain : pd.DataFrame
        Option chain for a single expiry with columns:
        - 'strike'
        - 'spot'
        - 'T'
        - 'iv_call'
        - 'iv_put'
        - 'log_moneyness'
        - 'is_atm'

    Returns
    -------
    pd.DataFrame
        Smile dataframe sorted by log-moneyness, containing:
        - 'T'             : time to maturity
        - 'strike'
        - 'log_moneyness'
        - 'iv'            : stitched implied volatility
        - 'source'        : {'put', 'call', 'atm'}

    Notes
    -----
    - A single implied volatility is assigned per strike
    - ITM options are excluded because they are redundant via put-call parity
    - ATM IV can be chosen from call, put, or their average (document choice)
    - This output is suitable for surface construction and fitting

    TODO
    ----
    - Validate required columns exist
    - Identify ATM row (is_atm == True)
    - Split:
        * left wing  : strikes < ATM → use iv_put
        * right wing : strikes > ATM → use iv_call
    - Decide ATM iv:
        * option A: iv_call
        * option B: iv_put
        * option C: average (document choice in code comment)
    - Construct unified 'iv' column
    - Add 'source' column indicating origin
    - Drop rows with NaN iv
    - Sort by log_moneyness
    - Return clean smile dataframe
    """
    if option_chain.empty:
        raise ValueError("Empty option chain dataframe")
    df = option_chain.copy()
    required_columns = ['strike', 'spot', 'T', 'iv_call', 'iv_put', 'log_moneyness', 'is_atm']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")
    # Identify ATM row
    atm_row = df[df['is_atm'] == True]
    if atm_row.empty:
        raise ValueError("No ATM row found (is_atm == True)")
    if len(atm_row) > 1:
        raise ValueError("Multiple ATM rows found (is_atm == True)")
    atm_index = atm_row.index[0]
    atm_price = df.loc[atm_index, 'spot']
    # Left wing (OTM puts): strikes < ATM (spot)
    left_wing = df[df['strike'] < atm_price].copy()
    left_wing['implied_volatility'] = left_wing['iv_put']
    left_wing['source'] = 'put'
    # Right wing (OTM calls): strikes > ATM (spot)
    right_wing = df[df['strike'] > atm_price].copy()
    right_wing['implied_volatility'] = right_wing['iv_call']
    right_wing['source'] = 'call'
    # ATM row: choose iv_call, iv_put, or average (document choice)
    # Here we choose the average of call and put IVs for ATM
    atm_iv_call = df.loc[atm_index, 'iv_call']
    atm_iv_put = df.loc[atm_index, 'iv_put']
    if np.isnan(atm_iv_call) and np.isnan(atm_iv_put):
        atm_iv = np.nan
    elif np.isnan(atm_iv_call):
        atm_iv = atm_iv_put
    elif np.isnan(atm_iv_put):
        atm_iv = atm_iv_call
    else:
        atm_iv = (atm_iv_call + atm_iv_put) / 2
    atm_row = df[df['is_atm'] == True].copy()
    atm_row['implied_volatility'] = atm_iv
    atm_row['source'] = 'atm'
    # Combine and clean
    smile_df = pd.concat([left_wing, atm_row, right_wing], ignore_index=True)
    smile_df = smile_df.dropna(subset=['implied_volatility'])
    smile_df = smile_df[['T', 'strike', 'log_moneyness', 'implied_volatility', 'source']]
    smile_df = smile_df.sort_values('log_moneyness').reset_index(drop=True)

    return smile_df


def compute_atm_vol(option_chain: pd.DataFrame) -> float:
    """
    Compute ATM implied volatility for a single expiry.

    ATM is defined as the option with strike closest to spot.

    Parameters
    ----------
    option_chain : pd.DataFrame
        Option chain containing:
        - 'is_atm'
        - 'iv_call'
        - 'iv_put'

    Returns
    -------
    float
        ATM implied volatility

    Notes
    -----
    - Used for term-structure plots and summary diagnostics
    - Choice of call/put/average must be consistent with smile construction

    TODO
    ----
    - Validate ATM row exists and is unique
    - Select ATM IV consistently (same rule as extract_smile_at_expiry)
    - Return NaN if ATM IV is invalid
    """
    required_columns = ['is_atm', 'iv_call', 'iv_put']
    missing_columns = [col for col in required_columns if col not in option_chain.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")
    atm_row = option_chain[option_chain['is_atm'] == True]
    if atm_row.empty:
        raise ValueError("No ATM row found (is_atm == True)")
    if len(atm_row) > 1:
        raise ValueError("Multiple ATM rows found (is_atm == True)")
    atm_index = atm_row.index[0]
    atm_iv_call = option_chain.loc[atm_index, 'iv_call']
    atm_iv_put = option_chain.loc[atm_index, 'iv_put']
    if np.isnan(atm_iv_call) and np.isnan(atm_iv_put):
        return np.nan
    elif np.isnan(atm_iv_call):
        return atm_iv_put
    elif np.isnan(atm_iv_put):
        return atm_iv_call
    else:
        return (atm_iv_call + atm_iv_put) / 2

def build_iv_surface(
    chains: list[pd.DataFrame]
) -> pd.DataFrame:
    """
    Combine volatility smiles across expiries into a surface-ready dataframe.

    Parameters
    ----------
    chains : list of pd.DataFrame
        Each dataframe must be output of extract_smile_at_expiry()
        and contain:
        - 'T'
        - 'log_moneyness'
        - 'iv'

    Returns
    -------
    pd.DataFrame
        Volatility surface dataframe with columns:
        - 'T'
        - 'log_moneyness'
        - 'iv'

    Notes
    -----
    - No filtering or smoothing is performed here
    - Surface fitting/interpolation happens in a separate step
    """
    if not chains:
        raise ValueError("Input list of smiles is empty")
    required_columns = ['T', 'log_moneyness', 'implied_volatility']
    for i, smile in enumerate(chains):
        missing_columns = [col for col in required_columns if col not in smile.columns]
        if missing_columns:
            raise ValueError(f"Smile at index {i} is missing required columns: {missing_columns}")
    surface_df = pd.concat(chains, ignore_index=True)
    surface_df = surface_df.dropna(subset=['implied_volatility'])
    surface_df = surface_df.rename(columns={'implied_volatility': 'iv'})
    surface_df = surface_df[['T', 'log_moneyness', 'iv', 'source']]
    
    return surface_df


# =============================================================================
# 4. SKEW METRICS
# =============================================================================

def compute_atm_skew(smile_df: pd.DataFrame) -> dict:
    """
    Compute skew metrics from a volatility smile.

    Parameters
    ----------
    smile_df : pd.DataFrame
        Smile dataframe containing:
        - 'log_moneyness'
        - 'iv'

    Returns
    -------
    dict
        Dictionary of skew metrics, e.g.:
        - 'put_call_skew'
        - 'slope_near_atm'

    Notes
    -----
    - Skew metrics are typically computed using OTM options.
    - The caller is responsible for any OTM filtering.
    """
    pass


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def forward_price(spot, rate, time_to_expiry):
    """
    Compute forward price: F = S * exp(r*T).
    
    Parameters
    ----------
    spot : float
        Current spot price
    rate : float
        Risk-free rate (annual, continuously compounded)
    time_to_expiry : float
        Time to expiration (years)
    
    Returns
    -------
    float
        Forward price
    """
    return spot * np.exp(rate * time_to_expiry)