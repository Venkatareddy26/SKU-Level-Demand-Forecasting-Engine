"""Unit tests for evaluation metrics."""
import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from metrics import calculate_mae, calculate_rmse, calculate_mape, evaluate_forecast

class TestMetrics(unittest.TestCase):
    """Test cases for evaluation metrics."""
    
    def setUp(self):
        """Set up test data."""
        self.y_true = np.array([100, 150, 120, 180, 200])
        self.y_pred = np.array([95, 155, 115, 175, 210])
    
    def test_mae(self):
        """Test Mean Absolute Error."""
        mae = calculate_mae(self.y_true, self.y_pred)
        expected_mae = np.mean(np.abs(self.y_true - self.y_pred))
        self.assertAlmostEqual(mae, expected_mae, places=2)
    
    def test_rmse(self):
        """Test Root Mean Squared Error."""
        rmse = calculate_rmse(self.y_true, self.y_pred)
        expected_rmse = np.sqrt(np.mean((self.y_true - self.y_pred) ** 2))
        self.assertAlmostEqual(rmse, expected_rmse, places=2)
    
    def test_mape(self):
        """Test Mean Absolute Percentage Error."""
        mape = calculate_mape(self.y_true, self.y_pred)
        expected_mape = np.mean(np.abs((self.y_true - self.y_pred) / self.y_true)) * 100
        self.assertAlmostEqual(mape, expected_mape, places=2)
    
    def test_evaluate_forecast(self):
        """Test evaluate_forecast function."""
        results = evaluate_forecast(self.y_true, self.y_pred, metric='all')
        
        # Check all metrics are present
        self.assertIn('MAE', results)
        self.assertIn('RMSE', results)
        self.assertIn('MAPE', results)
        
        # Check values are positive
        for metric, value in results.items():
            self.assertGreater(value, 0)

if __name__ == '__main__':
    unittest.main()
