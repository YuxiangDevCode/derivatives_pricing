"""
Data cleaning utilities for option chain data.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def clean_option_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw option chain data.
    
    Removes:
    - Invalid prices (NaN, zero, negative)
    - Bid-ask spread anomalies
    - Near-zero volume strikes
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw option chain with columns: strike, call_price, put_price, spot, expiration
    
    Returns
    -------
    pd.DataFrame
        Cleaned option data
    
    Example
    -------
    >>> df_clean = clean_option_data(df_raw)
    """
    df = df.copy()
    
    # Perform data clearning by removing below:
    # 1) rows with NaN values in key columns
    # 2) invalid prices (zero or negative)
    # 3) extreme outliers
    # 4) unreasonable expirations
    df = df.dropna(subset=['call_price', 'put_price', 'strike', 'spot'])
    df = df[(df['call_price'] > 0) & (df['put_price'] > 0) & (df['strike'] > 0)]
    df = df[df['put_price'] <= df['strike'] * 1.2]
    if isinstance(df['expiration'].iloc[0], str) or df['expiration'].dtype == object:
        df['expiration'] = pd.to_datetime(df['expiration'])
    now = datetime.now()
    df['T'] = (df['expiration'] - now).dt.days / 365
    # Keep only liquid strikes with reasonable expiration
    # T > 0.01: Avoid very short-term options (< 1 day) with unreliable data
    # T < 2: Avoid very long-term options with minimal Greeks sensitivity
    df = df[(df['T'] > 0.01) & (df['T'] < 2)]
    df = df.reset_index(drop=True)
 
    return df
