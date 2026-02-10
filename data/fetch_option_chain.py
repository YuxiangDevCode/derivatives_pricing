"""
Fetch option chain data from Yahoo Finance with local caching support.

Supports three modes:
1. 'cache': Read from local CSV only (for reproducible demos)
2. 'live': Fetch fresh from Yahoo Finance only
3. 'cache_or_live': Try cache first, fall back to live fetch (recommended)
"""

import yfinance as yf
import pandas as pd
from pathlib import Path


def fetch_option_chain(
    ticker: str,
    expiration_date: str = None,
    mode: str = 'cache_or_live',
    cache_dir: Path = None
) -> pd.DataFrame:
    """
    Fetch option chain data from Yahoo Finance with optional local caching.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'SPY', 'AAPL')
    expiration_date : str, optional
        Expiration date in format 'yyyy-mm-dd', 'mm/dd/yyyy', or 'mm-dd-yyyy'
        If None, uses the nearest (first) expiration date
    mode : str, default 'cache_or_live'
        Data source priority:
        - 'cache': Read from local CSV only (raises error if not found)
        - 'live': Fetch fresh from Yahoo Finance only
        - 'cache_or_live': Try cache first, fall back to live (recommended for demos)
    cache_dir : Path, optional
        Directory to store/read cached CSV files. Defaults to 'data/raw/'
    
    Returns
    -------
    pd.DataFrame
        Option chain data with columns: strike, call_price, put_price, spot, expiration
    
    Example
    -------
    >>> # Demo mode (cache-first, reproducible)
    >>> df = fetch_option_chain('SPY', '2026-08-21', mode='cache_or_live')
    
    >>> # Production mode (always fresh)
    >>> df = fetch_option_chain('SPY', '2026-08-21', mode='live')
    
    >>> # Force cache-only (research notebooks)
    >>> df = fetch_option_chain('SPY', '2026-08-21', mode='cache')
    """
    # Default cache directory: use absolute path from this module's location
    if cache_dir is None:
        # Path(__file__) is data/fetch_option_chain.py, parent is data/, parent.parent is project root
        project_root = Path(__file__).parent.parent
        cache_dir = project_root / 'data' / 'raw'
    cache_dir = Path(cache_dir)
    
    # Standardize the expiration_date input to 'yyyy-mm-dd' format
    if expiration_date is None:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            raise ValueError(f"No options available for {ticker}")
        expiration_date = expirations[0]
    
    try:
        standardized_date = pd.to_datetime(expiration_date).strftime('%Y-%m-%d')
    except Exception:
        raise ValueError(f"Invalid date format: {expiration_date}. Use 'yyyy-mm-dd', 'mm/dd/yyyy', or 'mm-dd-yyyy'")
    
    # Construct cache file path
    cache_file = cache_dir / f"{ticker}_options_{standardized_date}.csv"
    
    # =========================================================================
    # TRY CACHE FIRST (if mode allows)
    # =========================================================================
    if mode in ['cache', 'cache_or_live']:
        if cache_file.exists():
            df = pd.read_csv(cache_file)
            source = "cache"
            print(f"[{source.upper()}] Loaded: {ticker} {standardized_date} from {cache_file}")
            return df
        
        # If mode is 'cache' only, error out
        if mode == 'cache':
            raise FileNotFoundError(
                f"Cache file not found: {cache_file}\n"
                f"Available cache files: {list(cache_dir.glob(f'{ticker}_options_*.csv')) if cache_dir.exists() else 'None'}"
            )
    
    # =========================================================================
    # FETCH FROM YAHOO FINANCE (live mode or cache fallback)
    # =========================================================================
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        
        if not expirations:
            raise ValueError(f"No options available for {ticker}")
        
        if standardized_date not in expirations:
            available_dates = ', '.join(expirations[:5])
            raise ValueError(
                f"No options available for {ticker} on {standardized_date}. "
                f"Available dates: {available_dates}..."
            )
        
        # Fetch option chain
        chain = stock.option_chain(standardized_date)
        
        # Get the current spot price
        try:
            spot = stock.info.get('currentPrice', stock.history(period='1d')['Close'].iloc[-1])
        except Exception as e:
            raise RuntimeError(f"Failed to fetch spot price for {ticker}: {str(e)}")
        
        # Extract call and put data
        calls = chain.calls[['strike', 'lastPrice']].rename(columns={'lastPrice': 'call_price'})
        puts = chain.puts[['strike', 'lastPrice']].rename(columns={'lastPrice': 'put_price'})
        
        # Merge on strike (inner join)
        df = calls.merge(puts, on='strike', how='inner')
        
        # Add metadata
        df['spot'] = spot
        df['expiration'] = standardized_date
        df['ticker'] = ticker
        
        # Cache the result (if cache_dir is specified)
        if mode in ['live', 'cache_or_live']:
            cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_file, index=False)
            print(f"[LIVE] Fetched & cached: {ticker} {standardized_date} to {cache_file}")
        else:
            print(f"[LIVE] Fetched: {ticker} {standardized_date} (not cached)")
        
        return df
    
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {ticker} {standardized_date}: {str(e)}")


# =============================================================================
# MULTI-EXPIRY HELPER
# =============================================================================

def fetch_multi_expiry_chains(
    ticker: str = 'SPY',
    target_expirations: list[int] = None,
    mode: str = 'cache_or_live',
    cache_dir: Path = None
) -> dict[str, pd.DataFrame]:
    """
    Fetch option chains for multiple expirations in one call.
    
    Useful for building volatility surfaces. Automatically selects the closest
    available expiration for each target duration.
    
    Parameters
    ----------
    ticker : str, default 'SPY'
        Stock ticker symbol
    target_expirations : list of int, optional
        Target expiration days (e.g., [30, 60, 90, 180, 270] for 1M, 2M, 3M, 6M, 9M)
        If None, uses [30, 60, 90, 180, 270]
    mode : str, default 'cache_or_live'
        Data source priority ('cache', 'live', 'cache_or_live')
    cache_dir : Path, optional
        Cache directory. Defaults to 'data/raw/'
    
    Returns
    -------
    dict of {expiration_date: DataFrame}
        Option chains for each expiration
    
    Example
    -------
    >>> chains = fetch_multi_expiry_chains('SPY', mode='cache_or_live')
    >>> for exp_date, df in chains.items():
    ...     print(f"{exp_date}: {len(df)} strikes")
    """
    from datetime import datetime, timedelta
    
    if target_expirations is None:
        target_expirations = [30, 60, 90, 180, 270]
    
    if cache_dir is None:
        # Use absolute path from this module's location
        project_root = Path(__file__).parent.parent
        cache_dir = project_root / 'data' / 'raw'
    
    # Get available expirations from Yahoo Finance
    stock = yf.Ticker(ticker)
    available_expirations = stock.options
    
    if not available_expirations:
        raise ValueError(f"No options available for {ticker}")
    
    # Convert to dates
    available_dates = [pd.to_datetime(d) for d in available_expirations]
    today = datetime.now()
    
    # Select closest expiration to each target
    selected_dates = []
    for target_days in target_expirations:
        target_date = today + timedelta(days=target_days)
        closest = min(available_dates, key=lambda x: abs((x - today).days - target_days))
        if closest.strftime('%Y-%m-%d') not in [d.strftime('%Y-%m-%d') for d in selected_dates]:
            selected_dates.append(closest)
    
    # Fetch all chains
    chains = {}
    for exp_date in selected_dates:
        exp_str = exp_date.strftime('%Y-%m-%d')
        try:
            df = fetch_option_chain(ticker, exp_str, mode=mode, cache_dir=cache_dir)
            chains[exp_str] = df
        except Exception as e:
            print(f"Warning: Failed to fetch {ticker} {exp_str}: {e}")
    
    return chains
