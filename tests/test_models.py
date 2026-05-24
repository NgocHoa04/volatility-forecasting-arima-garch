"""
Unit tests for volatility forecasting models.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from volatility_forecasting.models import GARCHModel, HARModel
from volatility_forecasting.data import DataPreprocessor
from volatility_forecasting.utils import train_test_split


class TestDataPreprocessor:
    """Test data preprocessing."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample price data."""
        dates = pd.date_range("2020-01-01", periods=252)
        prices = pd.DataFrame({
            "Adj Close": 100 + np.cumsum(np.random.randn(252) * 2)
        }, index=dates)
        return prices
    
    def test_compute_log_returns(self, sample_data):
        """Test log return computation."""
        preprocessor = DataPreprocessor(sample_data)
        returns = preprocessor.compute_log_returns("Adj Close")
        
        assert len(returns) == len(sample_data)
        assert returns.isnull().sum() == 1  # First return is NaN
    
    def test_remove_na(self, sample_data):
        """Test NaN removal."""
        preprocessor = DataPreprocessor(sample_data)
        returns = preprocessor.compute_log_returns("Adj Close")
        clean_returns = preprocessor.remove_na()
        
        assert clean_returns.isnull().sum() == 0
        assert len(clean_returns) == len(returns) - 1


class TestGARCHModel:
    """Test GARCH model."""
    
    @pytest.fixture
    def returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        return pd.Series(np.random.randn(252) * 0.01)
    
    def test_garch_fit(self, returns):
        """Test GARCH fitting."""
        model = GARCHModel()
        model.fit(returns, disp="off")
        
        assert model.fitted
        assert model.aic is not None
        assert model.bic is not None
    
    def test_garch_forecast(self, returns):
        """Test GARCH forecast."""
        model = GARCHModel()
        model.fit(returns, disp="off")
        
        forecast = model.forecast(horizon=1)
        
        assert "variance" in forecast
        assert "volatility" in forecast
        assert len(forecast["volatility"]) == 1


class TestHARModel:
    """Test HAR model."""
    
    @pytest.fixture
    def returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        return pd.Series(np.random.randn(252) * 0.01)
    
    def test_har_fit(self, returns):
        """Test HAR fitting."""
        model = HARModel()
        model.fit(returns)
        
        assert model.fitted
        assert model.coefficients is not None
    
    def test_har_forecast(self, returns):
        """Test HAR forecast."""
        model = HARModel()
        model.fit(returns)
        
        forecast = model.forecast(returns, steps=1)
        
        assert isinstance(forecast, (float, np.floating))
        assert forecast >= 0


def test_train_test_split():
    """Test train/test split utility."""
    data = pd.Series(np.arange(100))
    train, test = train_test_split(data, test_size=0.2)
    
    assert len(train) == 80
    assert len(test) == 20
    assert len(train) + len(test) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
