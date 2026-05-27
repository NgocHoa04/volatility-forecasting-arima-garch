"""
Volatility proxy estimators from OHLC data.

Implements:
- Parkinson (1980)
- Garman-Klass (1980)
- Yang-Zhang (2000) - 5-day estimator
"""

import numpy as np
import pandas as pd
from typing import Union
from ..logger import logger


def parkinson_volatility(high: Union[pd.Series, np.ndarray],
                        low: Union[pd.Series, np.ndarray],
                        periods: int = 1) -> pd.Series:
    """
    Parkinson (1980) volatility estimator.
    
    σ_p = sqrt(1/(4*ln(2)) * ln(H/L)^2)
    
    Parameters
    ----------
    high : pd.Series or np.ndarray
        High prices
    low : pd.Series or np.ndarray
        Low prices
    periods : int
        Rolling window period (default 1 = daily)
        
    Returns
    -------
    pd.Series
        Parkinson volatility
    """
    if isinstance(high, pd.Series):
        h = high.values.flatten()
        l = low.values.flatten()
        idx = high.index
    else:
        h = np.asarray(high).flatten()
        l = np.asarray(low).flatten()
        idx = None
    
    # Avoid log(0)
    ratio = np.maximum(h / np.maximum(l, 1e-8), 1.0)
    
    if periods == 1:
        pk_vol = np.sqrt(np.log(ratio)**2 / (4 * np.log(2)))
    else:
        # Rolling version
        pk_vol = np.full_like(h, np.nan, dtype=float)
        for i in range(periods - 1, len(h)):
            window_h = h[i - periods + 1:i + 1]
            window_l = l[i - periods + 1:i + 1]
            window_ratio = np.maximum(window_h / np.maximum(window_l, 1e-8), 1.0)
            pk_vol[i] = np.sqrt(np.mean(np.log(window_ratio)**2) / (4 * np.log(2)))
    
    result = pd.Series(pk_vol, index=idx) if idx is not None else pd.Series(pk_vol)
    return result


def garman_klass_volatility(high: Union[pd.Series, np.ndarray],
                           low: Union[pd.Series, np.ndarray],
                           open_p: Union[pd.Series, np.ndarray],
                           close: Union[pd.Series, np.ndarray],
                           periods: int = 1) -> pd.Series:
    """
    Garman-Klass (1980) volatility estimator.
    
    σ_gk = sqrt(0.5*ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2)
    
    Parameters
    ----------
    high : pd.Series or np.ndarray
        High prices
    low : pd.Series or np.ndarray
        Low prices
    open_p : pd.Series or np.ndarray
        Open prices
    close : pd.Series or np.ndarray
        Close prices
    periods : int
        Rolling window period
        
    Returns
    -------
    pd.Series
        Garman-Klass volatility
    """
    if isinstance(high, pd.Series):
        h = high.values.flatten()
        l = low.values.flatten()
        o = open_p.values.flatten()
        c = close.values.flatten()
        idx = high.index
    else:
        h = np.asarray(high).flatten()
        l = np.asarray(low).flatten()
        o = np.asarray(open_p).flatten()
        c = np.asarray(close).flatten()
        idx = None
    
    # Avoid log(0)
    hl_ratio = np.maximum(h / np.maximum(l, 1e-8), 1.0)
    co_ratio = np.maximum(c / np.maximum(o, 1e-8), 1.0)
    
    term1 = 0.5 * np.log(hl_ratio)**2
    term2 = (2 * np.log(2) - 1) * np.log(co_ratio)**2
    
    if periods == 1:
        gk_vol = np.sqrt(np.maximum(term1 - term2, 0))
    else:
        # Rolling version
        gk_vol = np.full_like(h, np.nan, dtype=float)
        for i in range(periods - 1, len(h)):
            w_h = h[i - periods + 1:i + 1]
            w_l = l[i - periods + 1:i + 1]
            w_o = o[i - periods + 1:i + 1]
            w_c = c[i - periods + 1:i + 1]
            
            w_hl = np.maximum(w_h / np.maximum(w_l, 1e-8), 1.0)
            w_co = np.maximum(w_c / np.maximum(w_o, 1e-8), 1.0)
            
            w_t1 = 0.5 * np.log(w_hl)**2
            w_t2 = (2 * np.log(2) - 1) * np.log(w_co)**2
            gk_vol[i] = np.sqrt(np.maximum(np.mean(w_t1 - w_t2), 0))
    
    result = pd.Series(gk_vol, index=idx) if idx is not None else pd.Series(gk_vol)
    return result


def yang_zhang_volatility(open_p: Union[pd.Series, np.ndarray],
                         high: Union[pd.Series, np.ndarray],
                         low: Union[pd.Series, np.ndarray],
                         close: Union[pd.Series, np.ndarray],
                         periods: int = 5) -> pd.Series:
    """
    Yang-Zhang (2000) volatility estimator (5-day version).
    
    Combines overnight jump and intraday range.
    
    Parameters
    ----------
    open_p : pd.Series or np.ndarray
        Open prices
    high : pd.Series or np.ndarray
        High prices
    low : pd.Series or np.ndarray
        Low prices
    close : pd.Series or np.ndarray
        Close prices
    periods : int
        Rolling window (typically 5 for weekly)
        
    Returns
    -------
    pd.Series
        Yang-Zhang volatility
    """
    if isinstance(open_p, pd.Series):
        o = open_p.values.flatten()
        h = high.values.flatten()
        l = low.values.flatten()
        c = close.values.flatten()
        idx = open_p.index
    else:
        o = np.asarray(open_p).flatten()
        h = np.asarray(high).flatten()
        l = np.asarray(low).flatten()
        c = np.asarray(close).flatten()
        idx = None
    
    # Overnight return (gap from previous close to current open)
    c_prev = np.roll(c, 1)
    c_prev[0] = c[0]  # First day has no previous close
    overnight = np.log(o / np.maximum(c_prev, 1e-8))
    
    # Intraday range
    intraday = np.log(h / np.maximum(l, 1e-8))
    
    # Garman-Klass component
    co_ratio = np.maximum(c / np.maximum(o, 1e-8), 1.0)
    gk_term = 0.5 * np.log(h / np.maximum(l, 1e-8))**2 - (2 * np.log(2) - 1) * np.log(co_ratio)**2
    
    yz_vol = np.full_like(o, np.nan, dtype=float)
    for i in range(periods - 1, len(o)):
        w_o = overnight[i - periods + 1:i + 1]
        w_gk = gk_term[i - periods + 1:i + 1]
        
        var_o = np.var(w_o, ddof=1) if len(w_o) > 1 else 0
        var_gk = np.mean(w_gk)
        
        yz_vol[i] = np.sqrt(var_o + var_gk)
    
    result = pd.Series(yz_vol, index=idx) if idx is not None else pd.Series(yz_vol)
    return result


def compute_multiple_proxies(ohlc_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all three volatility proxies from OHLC data.
    
    Parameters
    ----------
    ohlc_data : pd.DataFrame
        DataFrame with columns: Open, High, Low, Close (or Adj Close)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: parkinson, garman_klass, yang_zhang
    """
    # Normalize column names (handle both 'Close' and 'Adj Close')
    ohlc = ohlc_data.copy()
    if 'Adj Close' in ohlc.columns and 'Close' not in ohlc.columns:
        ohlc['Close'] = ohlc['Adj Close']
    
    result = pd.DataFrame(index=ohlc.index)
    
    result['parkinson'] = parkinson_volatility(
        ohlc['High'],
        ohlc['Low'],
        periods=1
    ) * 100  # Convert to percentage
    
    result['garman_klass'] = garman_klass_volatility(
        ohlc['High'],
        ohlc['Low'],
        ohlc['Open'],
        ohlc['Close'],
        periods=1
    ) * 100  # Convert to percentage
    
    result['yang_zhang'] = yang_zhang_volatility(
        ohlc['Open'],
        ohlc['High'],
        ohlc['Low'],
        ohlc['Close'],
        periods=5
    ) * 100  # Convert to percentage
    
    logger.info(f"Computed volatility proxies for {len(result)} observations")
    
    return result
