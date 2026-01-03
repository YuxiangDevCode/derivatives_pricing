"""
Fetch option chain data from Yahoo Finance.

Simple one-snapshot fetch for demo purposes.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path


def fetch_option_chain(ticker: str, expiration_date: str = None) -> pd.DataFrame:
    """
    Fetch option chain data from Yahoo Finance.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL')
    expiration_date : str, optional
        Expiration date to fetch in format 'yyyy-mm-dd' or 'mm/dd/yyyy' or 'mm-dd-yyyy'
        If None, uses the nearest (first) expiration date
    
    Returns
    -------
    pd.DataFrame
        Option chain data with columns: strike, call_price, put_price, spot, expiration
    
    Example
    -------
    >>> df = fetch_option_chain('AAPL', '2025-01-17')
    >>> df.head()
    """
    try:
        # Create a yfinance Ticker object for the given ticker
        stock = yf.Ticker(ticker)
        
        # Get available option expirations using stock.options
        expirations = stock.options
        if not expirations:
            raise ValueError(f"No options available for {ticker}")
        
        # Standardize the expiration_date input to 'yyyy-mm-dd' format
        if not expiration_date:
            expiration_date = expirations[0]
        try:
            standardized_date = pd.to_datetime(expiration_date).strftime('%Y-%m-%d')
        except Exception:
            raise ValueError(f"Invalid date format: {expiration_date}. Use 'yyyy-mm-dd', 'mm/dd/yyyy', or 'mm-dd-yyyy'")
        
        # Check if the standardized date is available
        if standardized_date not in expirations:
            available_dates = ', '.join(expirations[:5])
            raise ValueError(
                f"No options available for {ticker} on {standardized_date}. "
                f"Available dates: {available_dates}..."
            )
        
        # Fetch option chain for that expiration using stock.option_chain()
        chain = stock.option_chain(standardized_date)
        
        # Get the current spot price
        try:
            spot = stock.info.get('currentPrice', stock.history(period='1d')['Close'].iloc[-1])
        except Exception as e:
            raise RuntimeError(f"Failed to fetch spot price for {ticker}: {str(e)}")
        
        # Extract call and put data
        calls = chain.calls[['strike', 'lastPrice']].rename(columns={'lastPrice': 'call_price'})
        puts = chain.puts[['strike', 'lastPrice']].rename(columns={'lastPrice': 'put_price'})
        
        # Merge calls and puts on 'strike' (inner join)
        # Reasons for inner join:
        # - Focus on strikes with both call and put prices available
        # - Avoid NaNs in either call_price or put_price
        # - Validate put-call parity
        df = calls.merge(puts, on='strike', how='inner')
        
        # Add metadata columns: spot, expiration, ticker
        df['spot'] = spot
        df['expiration'] = standardized_date
        df['ticker'] = ticker
        
        return df
    
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {ticker}: {str(e)}")
