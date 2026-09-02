import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares
from typing import Callable

def fit_smile_spline(
    smile_df: pd.DataFrame,
    smoothing: float = 0.0
) -> dict:
    """
    Fit a non-parametric spline to a single volatility smile.

    Parameters
    ----------
    smile_df : pd.DataFrame
        Output of `extract_smile_at_expiry`, containing:
        - 'log_moneyness'
        - 'implied_volatility'
        - 'T'

    smoothing : float, optional
        Spline smoothing parameter.

    Returns
    -------
    dict
        Dictionary with:
        - 'iv_func': callable(log_moneyness) -> iv
        - 'iv_fitted': np.ndarray
        - 'residuals': np.ndarray

    Notes
    -----
    - Used as a baseline benchmark.
    - No arbitrage constraints enforced.
    """
    required_columns = ['log_moneyness', 'implied_volatility', 'T']
    missing_columns = [col for col in required_columns if col not in smile_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in smile_df: {missing_columns}")
    # Clean NaNs first
    smile_df = smile_df.dropna(subset=['implied_volatility']).copy()
    if smile_df.empty:
        raise ValueError("No valid implied volatility data after removing NaNs")
    # Ensure data is sorted by log_moneyness for spline fitting
    smile_df = smile_df.sort_values('log_moneyness').reset_index(drop=True)
    log_moneyness = smile_df['log_moneyness'].values
    iv = smile_df['implied_volatility'].values
    strike = smile_df['strike'].values if 'strike' in smile_df.columns else None
    
    # Fit a cubic spline (requires at least 4 points); use lower degree if needed
    n_points = len(log_moneyness)
    spline_degree = min(3, max(1, n_points - 1))  # Cubic if possible, else linear
    
    iv_spline = UnivariateSpline(log_moneyness, iv, k=spline_degree, s=smoothing)
    iv_fitted = iv_spline(log_moneyness)
    residuals = iv - iv_fitted

    return {
        'iv_func': iv_spline,
        'iv_fitted': iv_fitted,
        'residuals': residuals,
        'log_moneyness': log_moneyness,
        'strike': strike
    }

def svi_total_variance(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float
) -> np.ndarray:
    """
    Raw SVI total variance parameterization.

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

    Parameters
    ----------
    k : np.ndarray
        Log-forward moneyness: log(K/F) where F = S*exp(r*T).
        - k = 0 at ATM forward
        - k > 0 for OTM calls (K > F)
        - k < 0 for OTM puts (K < F)

    a, b, rho, m, sigma : float
        SVI parameters.
        - a: baseline variance level
        - b: scale parameter
        - rho: correlation parameter (typically -1 < rho < 1)
        - m: median parameter (controls skew position)
        - sigma: volatility-of-variance parameter

    Returns
    -------
    np.ndarray
        Total implied variance w(k) = sigma^2 * T (where sigma is implied vol).
    """
    w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))
    return w
    

def fit_smile_svi(
    smile_df: pd.DataFrame,
    param_bounds: dict = None
) -> dict:
    """
    Fit an SVI-style parametric smile to a single expiry.

    Uses least-squares optimization to calibrate SVI parameters to market data.

    Parameters
    ----------
    smile_df : pd.DataFrame
        Smile dataframe containing:
        - 'log_forward_moneyness' (k = log(K/F))
        - 'implied_volatility'
        - 'T' (time to expiry)

    param_bounds : dict, optional
        Parameter bounds for least-squares. If None, uses academic defaults
        from Gatheral & Jacquier (2013):
        {
            'a': (-np.inf, np.inf),        # a ∈ ℝ, constrained by non-negativity
            'b': (0.0, np.inf),            # b ≥ 0 (positive)
            'rho': (-0.9999, 0.9999),      # |ρ| < 1 (strictly bounded)
            'm': (-np.inf, np.inf),        # m ∈ ℝ (any real)
            'sigma': (1e-6, np.inf)        # σ > 0 (positive)
        }
        
        Note: The critical constraint is non-negativity of total variance,
        w(k) ≥ 0 for all k. The minimum of w is a + b*σ*√(1-ρ²), so this is
        enforced via penalty on that quantity in the objective function.

    Returns
    -------
    dict
        Dictionary with:
        - 'params': dict of fitted SVI parameters {a, b, rho, m, sigma}
        - 'iv_fitted': np.ndarray of fitted IVs
        - 'residuals': np.ndarray of IV residuals
        - 'rmse': float, root mean squared error in IV space

    Notes
    -----
    - Fit is performed on total variance w = iv^2 * T
    - Parameters are then converted back to implied vol
    - No static arbitrage constraints enforced (checked separately)
    - Calibration is per-expiry
    - Bounds based on: Gatheral & Jacquier (2013) "Arbitrage-free SVI surfaces"
    
    References
    ----------
    Gatheral, J. & Jacquier, A. (2013): "Arbitrage-free SVI volatility surfaces"
    arXiv:1204.0646 https://arxiv.org/abs/1204.0646
    
    Gatheral, J. (2004): "A parsimonious arbitrage-free implied volatility 
    parameterization with application to the valuation of volatility derivatives"
    """
    required_columns = ['log_forward_moneyness', 'implied_volatility', 'T']
    missing_columns = [col for col in required_columns if col not in smile_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in smile_df: {missing_columns}")
    
    df = smile_df.dropna(subset=['implied_volatility']).copy()
    if df.empty:
        raise ValueError("No valid implied volatility data after removing NaNs")
    
    # Sort by moneyness to ensure consistent ordering
    df = df.sort_values('log_forward_moneyness').reset_index(drop=True)
    
    k = df['log_forward_moneyness'].values
    iv = df['implied_volatility'].values
    T = df['T'].iloc[0]  # Assume single expiry
    
    # Convert IV to total variance: w = iv^2 * T
    w_market = (iv ** 2) * T
    
    if param_bounds is None:
        param_bounds = {
            'a': (-1e6, 1e6),           # a ∈ ℝ (wide but practical bounds)
            'b': (1e-6, 1e6),           # b ≥ 0 (strictly positive, avoid 0)
            'rho': (-0.9999, 0.9999),   # |ρ| < 1 (avoid singularity at ±1)
            'm': (-1e6, 1e6),           # m ∈ ℝ (wide but practical bounds)
            'sigma': (1e-6, 1e6)        # σ > 0 (strictly positive, avoid 0)
        }
    
    # Initial parameter guesses (ATM calibration)
    a0 = np.mean(w_market)  # Use mean total variance as starting point
    b0 = 0.1 * a0
    rho0 = -0.3  # Typical negative skew for equities
    m0 = 0.0  # ATM centered
    sigma0 = 0.3
    
    x0 = np.array([a0, b0, rho0, m0, sigma0])
    
    # Build bounds for least-squares (lower, upper)
    lower_bounds = np.array([
        param_bounds['a'][0],
        param_bounds['b'][0],
        param_bounds['rho'][0],
        param_bounds['m'][0],
        param_bounds['sigma'][0]
    ])
    upper_bounds = np.array([
        param_bounds['a'][1],
        param_bounds['b'][1],
        param_bounds['rho'][1],
        param_bounds['m'][1],
        param_bounds['sigma'][1]
    ])
    
    # Define objective function: residuals in total variance space
    def residuals_fn(params):
        a, b, rho, m, sigma = params
        
        # CHECK CONSTRAINT: min_k w(k) = a + b*σ*√(1-ρ²) ≥ 0
        # The minimum of w sits at k* = m - ρσ/√(1-ρ²); substituting back gives
        # w(k*) = a + b*σ*√(1-ρ²). Negative here means imaginary IV at strikes
        # near k*, which may lie outside the quoted range and go unnoticed.
        constraint_val = a + b * sigma * np.sqrt(1 - rho**2)
        
        if constraint_val < 0:
            # Penalize violation, scaled by depth so the solver keeps a gradient
            # pointing back toward the admissible region.
            penalty = 1e10 * abs(constraint_val)
            return np.full_like(w_market, penalty) * (1.0 + np.abs(k))
        
        w_fitted = svi_total_variance(k, a, b, rho, m, sigma)
        return w_fitted - w_market
    
    # Least-squares optimization
    result = least_squares(
        residuals_fn,
        x0,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=10000
    )
    
    if not result.success:
        print(f"Warning: SVI calibration did not converge (message: {result.message})")
    
    a_fit, b_fit, rho_fit, m_fit, sigma_fit = result.x
    
    # Fitted total variance
    w_fitted = svi_total_variance(k, a_fit, b_fit, rho_fit, m_fit, sigma_fit)
    
    # Convert back to implied vol: IV = sqrt(w / T)
    iv_fitted = np.sqrt(w_fitted / T)
    
    # Compute residuals in IV space
    residuals = iv - iv_fitted
    rmse = np.sqrt(np.mean(residuals ** 2))
    strike = df['strike'].values if 'strike' in df.columns else None
    
    return {
        'params': {
            'a': a_fit,
            'b': b_fit,
            'rho': rho_fit,
            'm': m_fit,
            'sigma': sigma_fit
        },
        'iv_fitted': iv_fitted,
        'residuals': residuals,
        'rmse': rmse,
        'log_forward_moneyness': k,
        'strike': strike
    }
    

def compare_smile_fits(
    smile_df: pd.DataFrame,
    spline_fit: dict,
    svi_fit: dict
) -> dict:
    """
    Compare non-parametric (spline) and parametric (SVI) smile fits against market.
    
    Industry-standard benchmarking for equity derivatives desk. Analyzes:
    - Absolute fit quality (RMSE, MAE, R²)
    - Behavior by moneyness region (ATM vs wings)
    - Worst-case error (risk assessment)
    - Model stability vs market data
    
    Parameters
    ----------
    smile_df : pd.DataFrame
        Market data with columns: 
        - 'log_forward_moneyness' (k = log(K/F))
        - 'implied_volatility' (market IV)
        - 'T' (time to expiry)
    
    spline_fit : dict
        Output from fit_smile_spline() containing:
        - 'iv_fitted': spline-fitted IVs
        - 'residuals': spline residuals
    
    svi_fit : dict
        Output from fit_smile_svi() containing:
        - 'iv_fitted': SVI-fitted IVs
        - 'residuals': SVI residuals
        - 'params': SVI parameters
    
    Returns
    -------
    dict
        Comprehensive comparison with keys:
        
        'comparison_df' : pd.DataFrame
            Per-strike analysis with columns:
            - 'k': log-forward moneyness
            - 'market_iv': market IV
            - 'spline_iv': spline fit
            - 'svi_iv': SVI fit
            - 'spline_error': market - spline (in IV basis points)
            - 'svi_error': market - SVI (in IV basis points)
            - 'error_diff': spline_error - svi_error (pos→SVI better)
            - 'region': 'ATM', 'inner_wing', or 'outer_wing'
            
        'summary' : dict
            Overall metrics by method:
            {
                'spline': {'rmse_bps': ..., 'mae_bps': ..., 'max_error_bps': ..., 'r2': ...},
                'svi': {'rmse_bps': ..., 'mae_bps': ..., 'max_error_bps': ..., 'r2': ...}
            }
            All errors in basis points (100 bps = 1% IV)
            
        'regional_analysis' : dict
            Metrics by moneyness region:
            {
                'ATM': {'k_range': ..., 'rmse_spline_bps': ..., 'rmse_svi_bps': ...},
                'inner_wing': {'k_range': ..., ...},
                'outer_wing': {'k_range': ..., ...}
            }
            
        'best_model' : str
            'spline', 'svi', or 'tied' based on overall RMSE
            
        'recommendation' : str
            Industry-style comment on model choice
    
    Notes
    -----
    Region definitions (by log-forward moneyness k):
    - ATM: |k| ≤ 0.05 (roughly ±5% from forward)
    - Inner wing: 0.05 < |k| ≤ 0.15 (5-15% away)
    - Outer wing: |k| > 0.15 (>15% away)
    
    Equity desk priorities:
    1. ATM fit quality (used for delta hedging)
    2. Inner wing stability (risk management)
    3. Outer wing extrapolation (exotic pricing)
    
    Examples
    --------
    >>> result = compare_smile_fits(smile_df, spline_fit, svi_fit)
    >>> result['summary']['spline']['rmse_bps']
    0.42
    >>> result['best_model']
    'svi'
    >>> result['recommendation']
    "SVI fits 0.08 bps better overall. Prefer SVI for extrapolation; use spline for ATM Greeks."
    
    References
    ----------
    - Equity derivatives desk standard: Compare by region, not just global RMSE
    - ATM fit is critical for delta/gamma accuracy
    - Wings are used for vanna/volga hedging
    - Compare R² (explains variance) not just RMSE (absolute error)
    """
    required_cols = ['log_forward_moneyness', 'implied_volatility', 'T']
    missing = [c for c in required_cols if c not in smile_df.columns]
    if missing:
        raise ValueError(f"Missing columns in smile_df: {missing}")
    
    # Prepare clean data
    df = smile_df.dropna(subset=['implied_volatility']).copy()
    if df.empty:
        raise ValueError("No valid IV data after removing NaNs")
    
    df = df.sort_values('log_forward_moneyness').reset_index(drop=True)
    T = df['T'].iloc[0]
    
    # ================================================================
    # ALIGNMENT: Join on strike
    # ================================================================
    # The two fitters sort their inputs on DIFFERENT columns:
    #   fit_smile_spline -> log_moneyness       = log(S/K), ascending
    #   fit_smile_svi    -> log_forward_moneyness = log(K/F), ascending
    # Since k_fwd = -k_spot - r*T, ascending in one is DESCENDING in the other,
    # so the two 'iv_fitted' arrays run in opposite directions. Joining by row
    # position pairs each spline point with the opposite wing's SVI point.
    # Strike is convention-independent and survives differing NaN handling.
    if 'strike' not in df.columns:
        raise ValueError(
            "smile_df must contain a 'strike' column to align market data with fits."
        )
    if spline_fit.get('strike') is None or svi_fit.get('strike') is None:
        raise ValueError(
            "Both fits must carry a 'strike' array for alignment. Refit with a "
            "smile_df that includes 'strike'."
        )
    
    market_df = df[['strike', 'log_moneyness', 'log_forward_moneyness',
                    'implied_volatility']].dropna(subset=['implied_volatility'])
    
    # Strike must identify a row uniquely, or the merge multiplies rows and
    # silently reweights every metric below.
    if market_df['strike'].duplicated().any():
        dupes = market_df.loc[market_df['strike'].duplicated(), 'strike'].tolist()
        raise ValueError(
            f"smile_df contains duplicate strikes {dupes}; a smile must carry one "
            "implied volatility per strike."
        )
    spline_df = pd.DataFrame({'strike': spline_fit['strike'],
                              'spline_iv': spline_fit['iv_fitted']})
    svi_df = pd.DataFrame({'strike': svi_fit['strike'],
                           'svi_iv': svi_fit['iv_fitted']})
    
    merged = (
        market_df
        .merge(spline_df, on='strike', how='inner')
        .merge(svi_df, on='strike', how='inner')
        .sort_values('log_forward_moneyness')
        .reset_index(drop=True)
    )
    
    if merged.empty:
        raise ValueError(
            "No common strikes where all data (market, spline, SVI) align. "
            "Cannot perform meaningful comparison."
        )
    
    n_expected = min(len(spline_df), len(svi_df))
    if len(merged) < n_expected:
        raise ValueError(
            f"Only {len(merged)} of {n_expected} fitted strikes matched across "
            "market data, spline fit and SVI fit. The fits were likely produced "
            "from a different smile than smile_df."
        )
    
    # Extract aligned data (all sorted by log_forward_moneyness)
    k = merged['log_forward_moneyness'].values
    market_iv = merged['implied_volatility'].values
    spline_iv = merged['spline_iv'].values
    svi_iv = merged['svi_iv'].values
    
    # Calculate errors in basis points (1 bp = 0.01%)
    spline_error_bps = (market_iv - spline_iv) * 10000  # Convert to bp
    svi_error_bps = (market_iv - svi_iv) * 10000
    
    # Define moneyness regions (equity desk standard)
    def assign_region(k_val):
        """Assign moneyness region based on distance from forward."""
        abs_k = np.abs(k_val)
        if abs_k <= 0.05:
            return 'ATM'
        elif abs_k <= 0.15:
            return 'inner_wing'
        else:
            return 'outer_wing'
    
    regions = np.array([assign_region(k_i) for k_i in k])
    
    # Build comparison DataFrame (industry-standard format)
    comparison_df = pd.DataFrame({
        'k': k,
        'market_iv': market_iv,
        'spline_iv': spline_iv,
        'svi_iv': svi_iv,
        'spline_error_bps': spline_error_bps,
        'svi_error_bps': svi_error_bps,
        'abs_error_diff_bps': np.abs(spline_error_bps) - np.abs(svi_error_bps),  # Pos → SVI better
        'region': regions
    })
    
    # ================================================================
    # CALCULATE GLOBAL METRICS (for all strikes)
    # ================================================================
    def calc_metrics(fitted_iv, errors_bps):
        """Calculate standard error metrics."""
        rmse = np.sqrt(np.mean(errors_bps ** 2))
        mae = np.mean(np.abs(errors_bps))
        max_error = np.max(np.abs(errors_bps))
        
        # R² = 1 - (SS_res / SS_tot), both in basis points for consistent scaling.
        # SS_tot is measured against the MEAN of the market IVs: it is the error
        # a constant-vol model would make, which is what the fit is judged against.
        market_iv_bps = market_iv * 10000
        ss_res = np.sum(errors_bps ** 2)
        ss_tot = np.sum((market_iv_bps - market_iv_bps.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        
        return {
            'rmse_bps': rmse,
            'mae_bps': mae,
            'max_error_bps': max_error,
            'r2': r2
        }
    
    spline_metrics = calc_metrics(spline_iv, spline_error_bps)
    svi_metrics = calc_metrics(svi_iv, svi_error_bps)
    
    summary = {
        'spline': spline_metrics,
        'svi': svi_metrics
    }
    
    # ================================================================
    # REGIONAL ANALYSIS (equity desk priorities)
    # ================================================================
    regional_analysis = {}
    
    for region in ['ATM', 'inner_wing', 'outer_wing']:
        mask = regions == region
        
        if not np.any(mask):
            regional_analysis[region] = {
                'k_range': 'N/A',
                'n_strikes': 0,
                'rmse_spline_bps': np.nan,
                'rmse_svi_bps': np.nan,
                'mae_spline_bps': np.nan,
                'mae_svi_bps': np.nan,
                'max_error_spline_bps': np.nan,
                'max_error_svi_bps': np.nan,
            }
        else:
            spline_err_region = spline_error_bps[mask]
            svi_err_region = svi_error_bps[mask]
            k_region = k[mask]
            
            regional_analysis[region] = {
                'k_range': f"[{k_region.min():.3f}, {k_region.max():.3f}]",
                'n_strikes': np.sum(mask),
                'rmse_spline_bps': np.sqrt(np.mean(spline_err_region ** 2)),
                'rmse_svi_bps': np.sqrt(np.mean(svi_err_region ** 2)),
                'mae_spline_bps': np.mean(np.abs(spline_err_region)),
                'mae_svi_bps': np.mean(np.abs(svi_err_region)),
                'max_error_spline_bps': np.max(np.abs(spline_err_region)),
                'max_error_svi_bps': np.max(np.abs(svi_err_region)),
            }
    
    # ================================================================
    # DETERMINE BEST MODEL
    # ================================================================
    spline_rmse = spline_metrics['rmse_bps']
    svi_rmse = svi_metrics['rmse_bps']
    rmse_diff = abs(spline_rmse - svi_rmse)
    
    if rmse_diff < 0.01:  # Tied if difference < 0.01 bp
        best_model = 'tied'
    elif svi_rmse < spline_rmse:
        best_model = 'svi'
    else:
        best_model = 'spline'
    
    # ================================================================
    # GENERATE RECOMMENDATION (equity desk style)
    # ================================================================
    rmse_improvement_bps = spline_rmse - svi_rmse
    
    if best_model == 'tied':
        rec = (f"Models equivalent (both RMSE ~{svi_rmse:.2f} bps). "
               f"Use spline for ATM (smoother), SVI for extrapolation (parametric).")
    elif best_model == 'svi':
        rec = (f"SVI fits {rmse_improvement_bps:.2f} bps better overall (RMSE: {svi_rmse:.2f} vs {spline_rmse:.2f} bps). "
               f"Use SVI for: Greeks inference, risk models, exotic pricing. "
               f"ATM RMSE spline={regional_analysis['ATM']['rmse_spline_bps']:.2f} vs SVI={regional_analysis['ATM']['rmse_svi_bps']:.2f} bps.")
    else:  # spline better
        rec = (f"Spline fits {-rmse_improvement_bps:.2f} bps better overall (RMSE: {spline_rmse:.2f} vs {svi_rmse:.2f} bps). "
               f"Use spline for local fitting, ATM focus. "
               f"Note: SVI extrapolates better at extreme moneyness (k>0.2).")
    
    recommendation = rec
    
    # ================================================================
    # RETURN COMPREHENSIVE COMPARISON
    # ================================================================
    return {
        'comparison_df': comparison_df,
        'summary': summary,
        'regional_analysis': regional_analysis,
        'best_model': best_model,
        'recommendation': recommendation
    }

def fit_surface_from_smiles(
    all_smiles: dict,
    spline_smoothing: float = 0.0,
    svi_param_bounds: dict = None,
    verbose: bool = True
) -> dict:
    """
    Fit a volatility surface by fitting smiles across all maturities.
    
    Calibrates both non-parametric (spline) and parametric (SVI) models to
    implied volatility data at each maturity, building a 2D surface in the
    (moneyness, maturity) plane.
    
    Parameters
    ----------
    all_smiles : dict
        Dictionary mapping maturity labels (e.g., expiry dates as strings) to
        smile DataFrames. Each DataFrame should contain:
        - 'log_moneyness': log(S/K) for spline fitting
        - 'log_forward_moneyness': log(K/F) for SVI fitting
        - 'implied_volatility': market implied vols
        - 'T': time to expiry (years)
    
    spline_smoothing : float, optional
        Smoothing parameter for spline fitting. Default 0.0 (interpolation).
        Increase to reduce overfitting on noisy data.
    
    svi_param_bounds : dict, optional
        Parameter bounds for SVI calibration. If None, uses defaults from
        fit_smile_svi(). See that function for details.
    
    verbose : bool, optional
        If True, print calibration status and summary statistics for each maturity.
        Default True.
    
    Returns
    -------
    dict
        Dictionary with keys:
        
        'spline_fits' : dict
            Spline fits keyed by maturity. Each value is output from fit_smile_spline():
            {'iv_func': callable, 'iv_fitted': ndarray, 'residuals': ndarray, ...}
        
        'svi_fits' : dict
            SVI fits keyed by maturity. Each value is output from fit_smile_svi():
            {'params': dict, 'iv_fitted': ndarray, 'residuals': ndarray, ...}
        
        'maturities' : list
            Sorted list of maturity labels (in same order as input).
        
        'n_maturities' : int
            Number of unique maturities in surface.
        
        'calibration_summary' : dict
            Summary statistics per maturity:
            {
                expiry_label: {
                    'spline_rmse_bps': float,
                    'svi_rmse_bps': float,
                    'n_strikes': int,
                    'winner': 'spline' or 'svi'
                }, ...
            }
    
    Notes
    -----
    **Surface Architecture:**
    - Each slice (fixed maturity T) is a smile: IV(K) at fixed T
    - X-axis: moneyness k (log scale)
    - Y-axis: maturity T (years)
    - Z-axis: implied volatility IV
    
    **Calibration Strategy:**
    - Independent smile fits per maturity (no cross-maturity constraints)
    - Spline: local smoothness, good for ATM
    - SVI: parametric, good for extrapolation and Greeks
    
    **Future Enhancements:**
    TODO: Add term-structure constraints (no calendar arbitrage)
    TODO: Implement sticky strike / sticky delta dynamics
    TODO: Add surface smoothing across maturities (Tikhonov regularization)
    TODO: Detect and handle volatility term-structure anomalies
    TODO: Export calibrated surface as 2D interpolator for pricing
    
    Examples
    --------
    >>> # Assuming all_market_smiles is dict of DataFrames by expiry
    >>> surface = fit_surface_from_smiles(
    ...     all_market_smiles,
    ...     spline_smoothing=0.01,
    ...     verbose=True
    ... )
    >>> 
    >>> # Access spline fit for a specific maturity
    >>> spline_fit_mar = surface['spline_fits']['2026-03-20']
    >>> iv_at_atm = spline_fit_mar['iv_func'](0.0)
    >>>
    >>> # View summary across all maturities
    >>> print(surface['calibration_summary'])
    
    References
    ----------
    - Volatility surface theory: Derman & Kani (1994), Dupire (1994)
    - SVI calibration: Gatheral & Jacquier (2013)
    """
    
    # TODO: Input validation - check all smiles have required columns
    # TODO: Add try-except handling for individual smile failures
    # TODO: Add logging instead of prints for production use
    
    if not all_smiles:
        raise ValueError("all_smiles dictionary is empty")
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"CALIBRATING VOLATILITY SURFACE")
        print(f"{'='*80}")
        print(f"Number of maturities: {len(all_smiles)}\n")
    
    spline_fits = {}
    svi_fits = {}
    calibration_summary = {}
    
    maturities = sorted(all_smiles.keys())
    
    # ================================================================
    # LOOP OVER ALL MATURITIES
    # ================================================================
    for idx, maturity_label in enumerate(maturities, 1):
        smile_df = all_smiles[maturity_label]
        
        if verbose:
            print(f"[{idx}/{len(maturities)}] {maturity_label}: ", end="", flush=True)
        
        # TODO: Add graceful handling when smile has <4 points (can't fit cubic spline)
        # TODO: Add diagnostics for poor fits (high RMSE, NaN params)
        
        try:
            # Fit spline model
            spline_fit = fit_smile_spline(smile_df, smoothing=spline_smoothing)
            spline_fits[maturity_label] = spline_fit
            spline_rmse = np.sqrt(np.mean(spline_fit['residuals'] ** 2))
            
            # Fit SVI model
            svi_fit = fit_smile_svi(smile_df, param_bounds=svi_param_bounds)
            svi_fits[maturity_label] = svi_fit
            svi_rmse = svi_fit['rmse']
            
            # Determine winner
            winner = 'svi' if svi_rmse < spline_rmse else 'spline'
            
            # Store summary
            calibration_summary[maturity_label] = {
                'spline_rmse_bps': spline_rmse * 10000,
                'svi_rmse_bps': svi_rmse * 10000,
                'n_strikes': len(smile_df),
                'winner': winner
            }
            
            if verbose:
                print(f"✓ Spline {spline_rmse*10000:.1f} bps | SVI {svi_rmse*10000:.1f} bps [{winner}]")
        
        except Exception as e:
            if verbose:
                print(f"✗ Failed: {str(e)}")
            raise ValueError(f"Failed to fit smile at {maturity_label}: {str(e)}")
    
    # ================================================================
    # BUILD SURFACE SUMMARY
    # ================================================================
    if verbose:
        print(f"\n{'='*80}")
        print(f"SURFACE CALIBRATION COMPLETE")
        print(f"{'='*80}")
        
        # TODO: Add surface statistics (avg term structure, skew evolution)
        # TODO: Add visualization diagnostics (surface topology, arbitrage checks)
        
        avg_spline_rmse = np.mean([
            s['spline_rmse_bps'] for s in calibration_summary.values()
        ])
        avg_svi_rmse = np.mean([
            s['svi_rmse_bps'] for s in calibration_summary.values()
        ])
        spline_wins = sum(1 for s in calibration_summary.values() if s['winner'] == 'spline')
        svi_wins = len(calibration_summary) - spline_wins
        
        print(f"\nAverage RMSE: Spline {avg_spline_rmse:.1f} bps | SVI {avg_svi_rmse:.1f} bps")
        print(f"Model wins: Spline {spline_wins} | SVI {svi_wins}")
        print(f"{'='*80}\n")
    
    return {
        'spline_fits': spline_fits,
        'svi_fits': svi_fits,
        'maturities': maturities,
        'n_maturities': len(maturities),
        'calibration_summary': calibration_summary
    }

