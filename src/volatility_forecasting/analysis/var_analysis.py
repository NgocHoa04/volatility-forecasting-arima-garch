"""Value-at-Risk (VaR) and Expected Shortfall (ES) analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from ..config import VaR_CONFIDENCE_LEVELS
from ..logger import logger


def compute_var(
    returns: pd.Series,
    volatility: pd.Series,
    confidence_level: float = 0.95,
    portfolio_value: float = 1_000_000.0,
) -> pd.Series:
    """
    Compute Value-at-Risk using conditional normal distribution.
    
    Parameters
    ----------
    returns : pd.Series
        Historical returns
    volatility : pd.Series
        Forecasted volatility
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% VaR)
    portfolio_value : float
        Portfolio value
        
    Returns
    -------
    pd.Series
        VaR in dollars
    """
    # Get mean return
    mean_return = returns.mean()
    
    # Z-score for confidence level
    from scipy.stats import norm
    z_score = norm.ppf(1 - confidence_level)
    
    # VaR = mean_return + z_score * volatility
    var_return = mean_return + z_score * volatility
    var_dollars = -portfolio_value * var_return
    
    return var_dollars


def compute_es(
    returns: pd.Series,
    volatility: pd.Series,
    confidence_level: float = 0.95,
    portfolio_value: float = 1_000_000.0,
) -> pd.Series:
    """
    Compute Expected Shortfall (Conditional VaR).
    
    Parameters
    ----------
    returns : pd.Series
        Historical returns
    volatility : pd.Series
        Forecasted volatility
    confidence_level : float
        Confidence level
    portfolio_value : float
        Portfolio value
        
    Returns
    -------
    pd.Series
        ES in dollars
    """
    from scipy.stats import norm
    
    mean_return = returns.mean()
    z_score = norm.ppf(1 - confidence_level)
    
    # ES is larger than VaR
    pdf_z = norm.pdf(z_score)
    es_return = mean_return + (pdf_z / (1 - confidence_level)) * volatility
    es_dollars = -portfolio_value * es_return
    
    return es_dollars


class VaRAnalysis:
    """Compute and analyze Value-at-Risk metrics."""
    
    def __init__(
        self,
        returns: pd.Series,
        volatility: pd.Series,
        confidence_levels: List[float] = VaR_CONFIDENCE_LEVELS,
        portfolio_value: float = 1_000_000.0,
    ):
        """
        Initialize VaR analysis.
        
        Parameters
        ----------
        returns : pd.Series
            Historical returns
        volatility : pd.Series
            Forecasted volatility
        confidence_levels : list
            Confidence levels for VaR
        portfolio_value : float
            Portfolio value
        """
        self.returns = returns
        self.volatility = volatility
        self.confidence_levels = confidence_levels
        self.portfolio_value = portfolio_value
        
        self.var_results = {}
        self.es_results = {}
    
    def compute(self) -> Dict[float, Dict[str, pd.Series]]:
        """
        Compute VaR and ES for all confidence levels.
        
        Returns
        -------
        dict
            Results organized by confidence level
        """
        results = {}
        
        for cl in self.confidence_levels:
            var = compute_var(self.returns, self.volatility, cl, self.portfolio_value)
            es = compute_es(self.returns, self.volatility, cl, self.portfolio_value)
            
            results[cl] = {"VaR": var, "ES": es}
            self.var_results[cl] = var
            self.es_results[cl] = es
            
            logger.info(f"Computed VaR and ES at {cl*100:.0f}% confidence")
        
        return results
    
    def summary(self) -> pd.DataFrame:
        """
        Get summary statistics of VaR and ES.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        if not self.var_results:
            raise ValueError("No results computed yet. Call compute() first.")
        
        summary_data = []
        
        for cl in self.confidence_levels:
            var = self.var_results[cl]
            es = self.es_results[cl]
            
            summary_data.append({
                "Confidence": f"{cl*100:.0f}%",
                "Mean VaR": var.mean(),
                "Std VaR": var.std(),
                "Mean ES": es.mean(),
                "Std ES": es.std(),
            })
        
        return pd.DataFrame(summary_data)
