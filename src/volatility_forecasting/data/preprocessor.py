"""Data preprocessing utilities."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

from ..config import PANDAS_FLOAT_FORMAT, PANDAS_MAX_ROWS
from ..logger import logger


class DataPreprocessor:
    """Preprocess price data into returns."""
    
    def __init__(self, price_data: pd.DataFrame):
        """
        Initialize DataPreprocessor.
        
        Parameters
        ----------
        price_data : pd.DataFrame
            Raw price data with columns like 'Close', 'Open', etc.
        """
        self.price_data = price_data.copy()
        self.returns = None
        self.log_returns = None
    
    def compute_returns(self, column: str = "Adj Close") -> pd.Series:
        """
        Compute simple returns.
        
        Parameters
        ----------
        column : str
            Price column to use
            
        Returns
        -------
        pd.Series
            Simple returns
        """
        if column not in self.price_data.columns:
            raise ValueError(f"Column '{column}' not found. Available: {self.price_data.columns.tolist()}")
        
        self.returns = self.price_data[column].pct_change()
        logger.info(f"Computed returns from {column}")
        
        return self.returns
    
    def compute_log_returns(self, column: str = "Adj Close") -> pd.Series:
        """
        Compute log returns (more suitable for volatility models).
        
        Parameters
        ----------
        column : str
            Price column to use
            
        Returns
        -------
        pd.Series
            Log returns
        """
        if column not in self.price_data.columns:
            raise ValueError(f"Column '{column}' not found. Available: {self.price_data.columns.tolist()}")
        
        prices = self.price_data[column]
        self.log_returns = np.log(prices / prices.shift(1))
        logger.info(f"Computed log returns from {column}")
        
        return self.log_returns
    
    def remove_na(self) -> pd.Series:
        """
        Remove NaN values from returns.
        
        Returns
        -------
        pd.Series
            Returns without NaN
        """
        if self.log_returns is None:
            raise ValueError("No returns computed yet. Call compute_log_returns() first.")
        
        original_len = len(self.log_returns)
        cleaned = self.log_returns.dropna()
        removed = original_len - len(cleaned)
        
        logger.info(f"Removed {removed} NaN values ({removed/original_len*100:.2f}%)")
        return cleaned
    
    def describe_returns(self) -> pd.DataFrame:
        """
        Get descriptive statistics of returns.
        
        Returns
        -------
        pd.DataFrame
            Descriptive statistics
        """
        if self.log_returns is None:
            raise ValueError("No returns computed yet. Call compute_log_returns() first.")
        
        returns_clean = self.remove_na()
        stats = returns_clean.describe()
        
        logger.debug(f"Returns statistics:\n{stats}")
        return stats
    
    def get_returns(self, remove_na: bool = True) -> pd.Series:
        """
        Get processed returns.
        
        Parameters
        ----------
        remove_na : bool
            Remove NaN values
            
        Returns
        -------
        pd.Series
            Returns series
        """
        if self.log_returns is None:
            raise ValueError("No returns computed yet. Call compute_log_returns() first.")
        
        if remove_na:
            return self.remove_na()
        return self.log_returns
