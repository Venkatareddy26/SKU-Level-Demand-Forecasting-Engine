"""Unit tests for feature engineering."""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from features import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        self.df = pd.DataFrame({
            'id': ['SKU_001'] * len(dates),
            'date': dates,
            'sales': np.random.randint(50, 200, len(dates))
        })
        
        self.fe = FeatureEngineer()
    
    def test_lag_features(self):
        """Test lag feature creation."""
        df_lag = self.fe.create_lag_features(self.df, target_col='sales', lags=[7, 14])
        
        # Check columns exist
        self.assertIn('lag_7', df_lag.columns)
        self.assertIn('lag_14', df_lag.columns)
        
        # Check lag values
        self.assertEqual(df_lag['lag_7'].iloc[7], df_lag['sales'].iloc[0])
    
    def test_rolling_features(self):
        """Test rolling feature creation."""
        df_rolling = self.fe.create_rolling_features(self.df, target_col='sales', windows=[7])
        
        # Check columns exist
        self.assertIn('rolling_mean_7', df_rolling.columns)
        self.assertIn('rolling_std_7', df_rolling.columns)
        
        # Check rolling mean calculation
        manual_mean = self.df['sales'].iloc[:7].mean()
        self.assertAlmostEqual(df_rolling['rolling_mean_7'].iloc[6], manual_mean, places=2)
    
    def test_festival_features(self):
        """Test festival feature creation."""
        df_festival = self.fe.add_festival_features(self.df, date_col='date')
        
        # Check columns exist
        self.assertIn('is_festival', df_festival.columns)
        self.assertIn('days_to_festival', df_festival.columns)
        self.assertIn('is_festival_week', df_festival.columns)
        
        # Check festival flag
        self.assertTrue(df_festival['is_festival'].max() >= 0)
    
    def test_calendar_features(self):
        """Test calendar feature creation."""
        df_calendar = self.fe.add_calendar_features(self.df, date_col='date')
        
        # Check columns exist
        expected_cols = ['day_of_week', 'day_of_month', 'week_of_year', 
                        'month', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end']
        for col in expected_cols:
            self.assertIn(col, df_calendar.columns)
        
        # Check day of week range
        self.assertTrue(df_calendar['day_of_week'].min() >= 0)
        self.assertTrue(df_calendar['day_of_week'].max() <= 6)
        
        # Check month range
        self.assertTrue(df_calendar['month'].min() >= 1)
        self.assertTrue(df_calendar['month'].max() <= 12)
    
    def test_build_features(self):
        """Test complete feature building pipeline."""
        df_features = self.fe.build_features(self.df, target_col='sales', date_col='date')
        
        # Check shape
        self.assertEqual(len(df_features), len(self.df))
        
        # Check all feature types exist
        self.assertIn('lag_7', df_features.columns)
        self.assertIn('rolling_mean_7', df_features.columns)
        self.assertIn('is_festival', df_features.columns)
        self.assertIn('day_of_week', df_features.columns)

if __name__ == '__main__':
    unittest.main()
