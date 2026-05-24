"""Data validation utilities."""

import numpy as np
import pandas as pd
from typing import Tuple

from ..logger import logger


def validate_returns(returns: pd.Series) -> bool:
    """
    Validate return series.
    
    Parameters
    ----------
    returns : pd.Series
        Return series to validate
        
    Returns
    -------
    bool
        True if valid, raises ValueError otherwise
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError("returns must be pandas Series or DataFrame")
    
    if len(returns) == 0:
        raise ValueError("returns series is empty")
    
    if returns.isnull().sum() > 0:
        logger.warning(f"Found {returns.isnull().sum()} NaN values in returns")
    
    if not np.isfinite(returns.values).all():
        raise ValueError("returns contain infinite values")
    
    logger.debug(f"Returns validation passed: {len(returns)} observations")
    return True


def validate_data(data: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate dataframe has required columns.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to validate
    required_columns : list
        List of required column names
        
    Returns
    -------
    bool
        True if valid, raises ValueError otherwise
    """
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if len(data) == 0:
        raise ValueError("DataFrame is empty")
    
    logger.debug(f"Data validation passed: {data.shape[0]} rows, {data.shape[1]} columns")
    return True


def train_test_split(
    data: pd.Series,
    test_size: float = 0.2,
) -> Tuple[pd.Series, pd.Series]:
    """
    Split time series into train and test sets.
    
    Parameters
    ----------
    data : pd.Series
        Time series to split
    test_size : float
        Proportion of data to use for testing (0-1)
        
    Returns
    -------
    tuple
        (train_data, test_data)
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    split_point = int(len(data) * (1 - test_size))
    train = data.iloc[:split_point]
    test = data.iloc[split_point:]
    
    logger.info(f"Train/test split: {len(train)} train, {len(test)} test")
    return train, test
