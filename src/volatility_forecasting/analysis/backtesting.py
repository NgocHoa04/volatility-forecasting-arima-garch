"""Backtesting utilities for VaR and volatility models."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.stats import chi2

from ..logger import logger


class Backtesting:
    """Backtest VaR and volatility models using Kupiec and Christoffersen tests."""
    
    def __init__(
        self,
        returns: pd.Series,
        var_forecast: pd.Series,
        confidence_level: float = 0.95,
        alpha: float = 0.05,
    ):
        """
        Initialize backtesting.
        
        Parameters
        ----------
        returns : pd.Series
            Actual returns
        var_forecast : pd.Series
            VaR forecast (in dollars, typically negative for losses)
        confidence_level : float
            Confidence level used for VaR
        alpha : float
            Significance level for hypothesis tests
        """
        self.returns = returns
        self.var_forecast = var_forecast
        self.confidence_level = confidence_level
        self.alpha = alpha
        
        # Align series
        valid_idx = returns.index.intersection(var_forecast.index)
        self.returns = returns[valid_idx]
        self.var_forecast = var_forecast[valid_idx]
        
        self.exceptions = None
        self.num_exceptions = None
    
    def _compute_exceptions(self) -> pd.Series:
        """Compute VaR breaches (exceptions)."""
        # Exception occurs when loss > VaR (i.e., actual loss is worse than forecast)
        exceptions = self.returns.values < -self.var_forecast.values
        return pd.Series(exceptions, index=self.returns.index)
    
    def kupiec_test(self) -> Dict[str, float]:
        """
        Kupiec's Proportion of Failures (POF) test.
        
        Tests if the number of VaR exceptions matches the expected frequency.
        
        Returns
        -------
        dict
            Test statistics and p-value
        """
        exceptions = self._compute_exceptions()
        n_exceptions = exceptions.sum()
        n_obs = len(exceptions)
        p_expected = 1 - self.confidence_level
        p_observed = n_exceptions / n_obs
        
        # Likelihood ratio test
        if p_observed > 0 and p_observed < 1:
            lr = 2 * (
                n_exceptions * np.log(p_observed / p_expected) +
                (n_obs - n_exceptions) * np.log((1 - p_observed) / (1 - p_expected))
            )
        else:
            lr = 0
        
        p_value = 1 - chi2.cdf(lr, df=1)
        
        result = {
            "Test": "Kupiec POF",
            "Exceptions": n_exceptions,
            "Expected": n_obs * p_expected,
            "Frequency": p_observed,
            "Expected Frequency": p_expected,
            "LR Statistic": lr,
            "p-value": p_value,
            "Pass": p_value > self.alpha,
        }
        
        logger.info(f"Kupiec test: {n_exceptions} exceptions (expected: {n_obs*p_expected:.1f})")
        return result
    
    def christoffersen_test(self) -> Dict[str, float]:
        """
        Christoffersen's Independence test.
        
        Tests if VaR exceptions are independent (not clustered).
        
        Returns
        -------
        dict
            Test statistics and p-value
        """
        exceptions = self._compute_exceptions().astype(int).values
        n_obs = len(exceptions)
        n_exceptions = exceptions.sum()
        
        # Compute transitions
        n_00 = ((exceptions[:-1] == 0) & (exceptions[1:] == 0)).sum()  # No exception -> No exception
        n_01 = ((exceptions[:-1] == 0) & (exceptions[1:] == 1)).sum()  # No exception -> Exception
        n_10 = ((exceptions[:-1] == 1) & (exceptions[1:] == 0)).sum()  # Exception -> No exception
        n_11 = ((exceptions[:-1] == 1) & (exceptions[1:] == 1)).sum()  # Exception -> Exception
        
        # Probabilities
        p_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0
        p_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0
        p = n_exceptions / n_obs
        
        # Likelihood ratio
        if p_01 > 0 and p_11 > 0 and (1-p_01) > 0 and (1-p_11) > 0 and p > 0 and (1-p) > 0:
            lr = (
                n_00 * np.log((1 - p_01) / (1 - p)) +
                n_01 * np.log(p_01 / p) +
                n_10 * np.log((1 - p_11) / (1 - p)) +
                n_11 * np.log(p_11 / p)
            )
            lr *= 2
        else:
            lr = 0
        
        p_value = 1 - chi2.cdf(lr, df=1)
        
        result = {
            "Test": "Christoffersen",
            "LR Statistic": lr,
            "p-value": p_value,
            "Pass": p_value > self.alpha,
        }
        
        logger.info(f"Christoffersen test: LR={lr:.2f}, p-value={p_value:.4f}")
        return result
    
    def full_backtest(self) -> pd.DataFrame:
        """
        Perform full backtesting (both Kupiec and Christoffersen tests).
        
        Returns
        -------
        pd.DataFrame
            Combined test results
        """
        kupiec = self.kupiec_test()
        christoffersen = self.christoffersen_test()
        
        results_df = pd.DataFrame([kupiec, christoffersen])
        return results_df
