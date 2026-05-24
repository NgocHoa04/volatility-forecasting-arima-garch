"""GARCH family models for volatility forecasting."""

import numpy as np
import pandas as pd
from arch import arch_model
from typing import Optional, Dict

from ..config import (
    GARCH_P,
    GARCH_Q,
    GARCH_O,
    GARCH_POWER,
)
from ..logger import logger


class BaseGARCHModel:
    """Base class for GARCH models."""
    
    def __init__(self, p: int = GARCH_P, q: int = GARCH_Q):
        """
        Initialize GARCH model.
        
        Parameters
        ----------
        p : int
            ARCH lag order
        q : int
            GARCH lag order
        """
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
        self.fitted = False
        self.aic = None
        self.bic = None
    
    def _build_model(self, returns: pd.Series) -> None:
        """Build GARCH model (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def fit(self, returns: pd.Series, disp: str = "off") -> None:
        """
        Fit GARCH model.
        
        Parameters
        ----------
        returns : pd.Series
            Return series (in percentage, typically)
        disp : str
            Display output ('off', 'final', 'all')
        """
        logger.info(f"Fitting {self.__class__.__name__} model...")
        
        self._build_model(returns)
        
        self.fitted_model = self.model.fit(disp=disp)
        self.aic = self.fitted_model.aic
        self.bic = self.fitted_model.bic
        self.fitted = True
        
        logger.info(
            f"{self.__class__.__name__} fitted. "
            f"AIC: {self.aic:.2f}, BIC: {self.bic:.2f}"
        )
    
    def forecast(self, horizon: int = 1) -> Dict[str, np.ndarray]:
        """
        Forecast conditional variance.
        
        Parameters
        ----------
        horizon : int
            Forecast horizon
            
        Returns
        -------
        dict
            Forecasted variance (and optionally other metrics)
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        forecast = self.fitted_model.forecast(horizon=horizon)
        variance = forecast.variance.values[-1, :]
        
        return {
            "variance": variance,
            "volatility": np.sqrt(variance),
        }
    
    def get_summary(self) -> str:
        """Get model summary."""
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return str(self.fitted_model.summary())
    
    def get_conditional_volatility(self) -> pd.Series:
        """Get in-sample conditional volatility."""
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.fitted_model.conditional_volatility


class GARCHModel(BaseGARCHModel):
    """Standard GARCH(p, q) model."""
    
    def _build_model(self, returns: pd.Series) -> None:
        """Build GARCH model."""
        self.model = arch_model(
            returns,
            vol="Garch",
            p=self.p,
            q=self.q,
        )


class GJRGARCHModel(BaseGARCHModel):
    """GJR-GARCH model (asymmetric GARCH)."""
    
    def __init__(self, p: int = GARCH_P, q: int = GARCH_Q, o: int = GARCH_O):
        """
        Initialize GJR-GARCH model.
        
        Parameters
        ----------
        p, q, o : int
            Lag orders
        """
        super().__init__(p, q)
        self.o = o
    
    def _build_model(self, returns: pd.Series) -> None:
        """Build GJR-GARCH model."""
        self.model = arch_model(
            returns,
            vol="Garch",
            p=self.p,
            o=self.o,
            q=self.q,
        )


class EGARCHModel(BaseGARCHModel):
    """EGARCH model (exponential GARCH)."""
    
    def __init__(self, p: int = GARCH_P, q: int = GARCH_Q, power: float = GARCH_POWER):
        """
        Initialize EGARCH model.
        
        Parameters
        ----------
        p, q : int
            Lag orders
        power : float
            Power for exponential term
        """
        super().__init__(p, q)
        self.power = power
    
    def _build_model(self, returns: pd.Series) -> None:
        """Build EGARCH model."""
        self.model = arch_model(
            returns,
            vol="EGarch",
            p=self.p,
            q=self.q,
            power=self.power,
        )
