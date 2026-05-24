"""ARIMA model for mean forecasting."""

import pandas as pd
import numpy as np
from pmdarima import auto_arima
from typing import Tuple, Optional

from ..config import (
    ARIMA_MAX_P,
    ARIMA_MAX_D,
    ARIMA_MAX_Q,
    ARIMA_STEPWISE,
    ARIMA_INFORMATION_CRITERION,
)
from ..logger import logger


class ARIMAModel:
    """ARIMA model for univariate time series forecasting."""
    
    def __init__(
        self,
        max_p: int = ARIMA_MAX_P,
        max_d: int = ARIMA_MAX_D,
        max_q: int = ARIMA_MAX_Q,
        stepwise: bool = ARIMA_STEPWISE,
        information_criterion: str = ARIMA_INFORMATION_CRITERION,
    ):
        """
        Initialize ARIMA model.
        
        Parameters
        ----------
        max_p, max_d, max_q : int
            Maximum AR, differencing, and MA orders
        stepwise : bool
            Use stepwise algorithm for faster estimation
        information_criterion : str
            Information criterion for model selection ('aic', 'bic', etc.)
        """
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.stepwise = stepwise
        self.information_criterion = information_criterion
        
        self.model = None
        self.order = None
        self.aic = None
        self.bic = None
        self.fitted = False
    
    def fit(self, data: pd.Series, seasonal: bool = False, verbose: bool = True) -> None:
        """
        Fit ARIMA model using auto_arima.
        
        Parameters
        ----------
        data : pd.Series
            Time series data
        seasonal : bool
            Include seasonal component
        verbose : bool
            Print fitting progress
        """
        logger.info("Fitting ARIMA model...")
        
        self.model = auto_arima(
            data,
            max_p=self.max_p,
            max_d=self.max_d,
            max_q=self.max_q,
            seasonal=seasonal,
            stepwise=self.stepwise,
            information_criterion=self.information_criterion,
            trace=verbose,
            error_action="ignore",
            suppress_warnings=True,
            random_state=42,
        )
        
        self.order = self.model.order
        self.aic = self.model.aic()
        self.bic = self.model.bic()
        self.fitted = True
        
        logger.info(f"ARIMA{self.order} fitted. AIC: {self.aic:.2f}, BIC: {self.bic:.2f}")
    
    def forecast(self, steps: int = 1) -> np.ndarray:
        """
        Forecast future values.
        
        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast
            
        Returns
        -------
        np.ndarray
            Forecasted values
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        forecast, _ = self.model.predict(n_periods=steps)
        return forecast
    
    def get_summary(self) -> str:
        """Get model summary."""
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return str(self.model.summary())
