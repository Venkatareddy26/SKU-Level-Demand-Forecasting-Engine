"""Unit tests for evaluation metrics."""
import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from metrics import (calculate_mae, calculate_rmse, calculate_mape, 
                     calculate_rmsse, calculate_wrmsse, evaluate_forecast)

class TestMetrics(unittest.TestCase):
    """Test cases for evaluation metrics."""
    
    def setUp(self):
        """Set up test data."""
        self.y_true = np.array([100, 150, 120, 180, 200])
        self.y_pred = np.array([95, 155, 115, 175, 210])
        self.y_train = np.array([80, 90, 100, 110, 120, 130, 140, 150, 160])
    
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
    
    def test_mape_with_zeros(self):
        """Test MAPE gracefully handles zero actuals."""
        y_true_with_zero = np.array([0, 100, 200])
        y_pred_with_zero = np.array([10, 110, 190])
        mape = calculate_mape(y_true_with_zero, y_pred_with_zero)
        
        # Should only compute on non-zero entries
        expected = np.mean(np.abs(np.array([10/100, 10/200]))) * 100
        self.assertAlmostEqual(mape, expected, places=2)
    
    def test_mape_all_zeros(self):
        """Test MAPE returns NaN when all actuals are zero."""
        y_true_zeros = np.array([0, 0, 0])
        y_pred = np.array([10, 20, 30])
        mape = calculate_mape(y_true_zeros, y_pred)
        self.assertTrue(np.isnan(mape))
    
    def test_rmsse(self):
        """Test RMSSE calculation."""
        rmsse = calculate_rmsse(self.y_true, self.y_pred, self.y_train)
        
        # Must be positive
        self.assertGreater(rmsse, 0)
        
        # Perfect predictions should give RMSSE = 0
        rmsse_perfect = calculate_rmsse(self.y_true, self.y_true, self.y_train)
        self.assertAlmostEqual(rmsse_perfect, 0.0, places=5)
    
    def test_rmsse_short_training(self):
        """Test RMSSE returns NaN for insufficient training data."""
        rmsse = calculate_rmsse(self.y_true, self.y_pred, np.array([100]))
        self.assertTrue(np.isnan(rmsse))
    
    def test_evaluate_forecast(self):
        """Test evaluate_forecast function."""
        results = evaluate_forecast(self.y_true, self.y_pred, metric='all')
        
        # Check all metrics are present
        self.assertIn('MAE', results)
        self.assertIn('RMSE', results)
        self.assertIn('MAPE', results)
        
        # WRMSSE should NOT be present without y_train
        self.assertNotIn('WRMSSE', results)
        
        # Check values are positive
        for metric, value in results.items():
            self.assertGreater(value, 0)
    
    def test_evaluate_forecast_with_wrmsse(self):
        """Test evaluate_forecast includes WRMSSE when y_train provided."""
        results = evaluate_forecast(
            self.y_true, self.y_pred, y_train=self.y_train, metric='all'
        )
        self.assertIn('WRMSSE', results)
        self.assertGreater(results['WRMSSE'], 0)

if __name__ == '__main__':
    unittest.main()
