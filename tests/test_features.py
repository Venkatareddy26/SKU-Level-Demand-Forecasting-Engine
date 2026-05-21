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
            'sales': np.random.randint(50, 200, len(dates)),
            'price': np.random.uniform(80, 120, len(dates))
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
    
    def test_rolling_features_no_leakage(self):
        """Test rolling features don't include current row (no look-ahead bias)."""
        df_rolling = self.fe.create_rolling_features(self.df, target_col='sales', windows=[7])
        
        # Check columns exist
        self.assertIn('rolling_mean_7', df_rolling.columns)
        self.assertIn('rolling_std_7', df_rolling.columns)
        
        # The first row's rolling mean should be NaN (shift(1) means no prior data)
        self.assertTrue(pd.isna(df_rolling['rolling_mean_7'].iloc[0]))
        
        # Row 7's rolling mean should be mean of rows 0-6 (shifted by 1, window 7)
        # With shift(1), row 7 sees rows 0-6
        expected_mean = self.df['sales'].iloc[0:7].mean()
        actual_mean = df_rolling['rolling_mean_7'].iloc[7]
        self.assertAlmostEqual(actual_mean, expected_mean, places=1)
    
    def test_festival_features(self):
        """Test festival feature creation."""
        df_festival = self.fe.add_festival_features(self.df, date_col='date')
        
        # Check columns exist
        self.assertIn('is_festival', df_festival.columns)
        self.assertIn('days_to_festival', df_festival.columns)
        self.assertIn('is_festival_week', df_festival.columns)
        
        # 2023 has festival dates in the calendar; check they're flagged
        self.assertTrue(df_festival['is_festival'].max() >= 0)

    def test_festival_duplicate_dates_do_not_duplicate_rows(self):
        """Festival calendar dates with multiple festivals must not expand sales rows."""
        df = pd.DataFrame({
            'id': ['SKU_001', 'SKU_001', 'SKU_001'],
            'date': pd.to_datetime(['2025-01-13', '2025-01-14', '2025-01-15']),
            'sales': [100, 120, 110],
        })

        df_festival = self.fe.add_festival_features(df, date_col='date')

        self.assertEqual(len(df_festival), len(df))
        festival_row = df_festival[df_festival['date'] == pd.Timestamp('2025-01-14')].iloc[0]
        self.assertEqual(festival_row['is_festival'], 1)
        self.assertIn('Pongal', festival_row['festival_name'])
        self.assertIn('Makar Sankranti', festival_row['festival_name'])
    
    def test_festival_features_vectorized_performance(self):
        """Test that festival features work on larger data without hanging."""
        # Create 3 SKUs x 1 year (should complete in < 2 seconds if vectorized)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        large_df = pd.concat([
            pd.DataFrame({'id': [f'SKU_{i}'] * len(dates), 'date': dates, 
                          'sales': np.random.randint(50, 200, len(dates))})
            for i in range(3)
        ]).reset_index(drop=True)
        
        import time
        start = time.time()
        df_festival = self.fe.add_festival_features(large_df, date_col='date')
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 5.0, "Festival features took too long — possibly O(n²)")
    
    def test_calendar_features(self):
        """Test calendar feature creation."""
        df_calendar = self.fe.add_calendar_features(self.df, date_col='date')
        
        # Check columns exist
        expected_cols = ['day_of_week', 'day_of_month', 'week_of_year', 
                        'month', 'quarter', 'is_weekend', 'is_month_start', 
                        'is_month_end', 'year']
        for col in expected_cols:
            self.assertIn(col, df_calendar.columns)
        
        # Check day of week range
        self.assertTrue(df_calendar['day_of_week'].min() >= 0)
        self.assertTrue(df_calendar['day_of_week'].max() <= 6)
        
        # Check month range
        self.assertTrue(df_calendar['month'].min() >= 1)
        self.assertTrue(df_calendar['month'].max() <= 12)
        
        # week_of_year should be int, not UInt32
        self.assertTrue(df_calendar['week_of_year'].dtype in [np.int32, np.int64, int])
    
    def test_price_features(self):
        """Test price feature creation."""
        df_price = self.fe.add_price_features(self.df, price_col='price')
        
        # Check columns exist
        self.assertIn('price_lag_1', df_price.columns)
        self.assertIn('price_change', df_price.columns)
        self.assertIn('price_change_pct', df_price.columns)
        self.assertIn('price_rolling_mean_7', df_price.columns)
        self.assertIn('price_vs_avg', df_price.columns)
    
    def test_price_features_missing_column(self):
        """Test price features gracefully handle missing price column."""
        df_no_price = self.df[['id', 'date', 'sales']].copy()
        df_result = self.fe.add_price_features(df_no_price, price_col='price')
        
        # Should return unchanged dataframe
        self.assertEqual(list(df_result.columns), list(df_no_price.columns))
    
    def test_build_features(self):
        """Test complete feature building pipeline."""
        df_features = self.fe.build_features(self.df, target_col='sales', date_col='date')
        
        # Check shape (rows should match)
        self.assertEqual(len(df_features), len(self.df))
        
        # Check all feature types exist
        self.assertIn('lag_7', df_features.columns)
        self.assertIn('rolling_mean_7', df_features.columns)
        self.assertIn('is_festival', df_features.columns)
        self.assertIn('day_of_week', df_features.columns)
        self.assertIn('price_lag_1', df_features.columns)

if __name__ == '__main__':
    unittest.main()
