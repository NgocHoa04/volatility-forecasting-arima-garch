"""Data loading utilities."""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import Optional

from ..config import (
    DATA_TICKER,
    DATA_START_DATE,
    DATA_END_DATE,
    DATA_RAW_FILE,
)
from ..logger import logger


def download_data(ticker: str = DATA_TICKER, start: str = DATA_START_DATE, end: str = DATA_END_DATE) -> pd.DataFrame:
    """
    Download historical price data from Yahoo Finance.
    Computes daily log returns (in percent).
    Saves the data to dataset/ folder.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (default: ^GSPC for S&P 500)
    start : str
        Start date in YYYY-MM-DD format
    end : str
        End date in YYYY-MM-DD format

    Returns
    -------
    pd.DataFrame with columns: Close, Log_Return
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = pd.DataFrame()
    df['Close'] = raw['Close'].squeeze()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1)) * 100
    df.dropna(inplace=True)
    
    # Save data to dataset folder
    os.makedirs('dataset', exist_ok=True)
    df.to_csv('dataset/price_data.csv')
    
    logger.info(f'Data downloaded: {ticker} from {start} to {end}')
    logger.info(f'Observations: {len(df):,} trading days')
    logger.info(f'Saved to: dataset/price_data.csv')
    
    return df


class DataLoader:
    """Load financial data from Yahoo Finance or CSV."""
    
    def __init__(
        self,
        ticker: str = DATA_TICKER,
        start_date: str = DATA_START_DATE,
        end_date: str = DATA_END_DATE,
    ):
        """
        Initialize DataLoader.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
    
    def download(self, force: bool = False) -> pd.DataFrame:
        """
        Download data from Yahoo Finance using download_data() function.
        
        Parameters
        ----------
        force : bool
            Force download even if file exists
            
        Returns
        -------
        pd.DataFrame
            Downloaded price data with columns: Close, Log_Return
        """
        if Path('dataset/price_data.csv').exists() and not force:
            logger.info("Loading existing data from dataset/price_data.csv")
            return self.load_csv('dataset/price_data.csv')
        
        self.data = download_data(self.ticker, self.start_date, self.end_date)
        return self.data
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to CSV file
            
        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        logger.info(f"Loading data from {filepath}")
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.data = data
        return data
    
    def get_data(self) -> pd.DataFrame:
        """
        Get loaded data.
        
        Returns
        -------
        pd.DataFrame
            Price data with Close and Log_Return columns
        """
        if self.data is None:
            raise ValueError("No data loaded. Call download() or load_csv() first.")
        return self.data
