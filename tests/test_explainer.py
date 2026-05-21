"""Unit tests for DemandExplainer class."""
import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import Mock, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from explainer import DemandExplainer
from models import LightGBMForecaster


class TestDemandExplainer(unittest.TestCase):
    """Test cases for DemandExplainer class.
    
    Tests validate Requirements 3.1, 3.2, 3.3, 3.4, 3.5:
    - SHAP value computation with correct shape
    - Top drivers extraction with required keys
    - Top N filtering and sorting
    - Feature importance summary generation
    - Festival feature explanation content
    """
    
    def setUp(self):
        """Set up test fixtures with trained model and explainer."""
        # Create sample training data
        np.random.seed(42)
        n_samples = 200
        
        # Define feature names
        self.feature_names = [
            'lag_7', 'lag_14', 'lag_28', 'lag_364',
            'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
            'rolling_std_7', 'rolling_std_14', 'rolling_std_28',
            'is_festival', 'is_festival_week', 'days_to_festival',
            'is_weekend', 'day_of_week', 'month', 'quarter',
            'price_change', 'price_vs_avg'
        ]
        
        # Create feature matrix
        X_train = pd.DataFrame({
            'lag_7': np.random.randint(50, 150, n_samples),
            'lag_14': np.random.randint(50, 150, n_samples),
            'lag_28': np.random.randint(50, 150, n_samples),
            'lag_364': np.random.randint(50, 150, n_samples),
            'rolling_mean_7': np.random.uniform(60, 140, n_samples),
            'rolling_mean_14': np.random.uniform(60, 140, n_samples),
            'rolling_mean_28': np.random.uniform(60, 140, n_samples),
            'rolling_std_7': np.random.uniform(10, 30, n_samples),
            'rolling_std_14': np.random.uniform(10, 30, n_samples),
            'rolling_std_28': np.random.uniform(10, 30, n_samples),
            'is_festival': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'is_festival_week': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'days_to_festival': np.random.randint(0, 30, n_samples),
            'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'quarter': np.random.randint(1, 5, n_samples),
            'price_change': np.random.uniform(-10, 10, n_samples),
            'price_vs_avg': np.random.uniform(0.8, 1.2, n_samples),
        })
        
        # Create target variable (sales)
        y_train = (
            X_train['lag_7'] * 0.5 +
            X_train['rolling_mean_7'] * 0.3 +
            X_train['is_festival'] * 50 +
            X_train['is_festival_week'] * 30 +
            np.random.normal(0, 10, n_samples)
        )
        
        # Train a simple LightGBM model
        self.model = LightGBMForecaster(params={
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 15,
            'learning_rate': 0.1,
            'verbose': -1
        })
        self.model.train(X_train, y_train, num_boost_round=50)
        
        # Create explainer
        self.explainer = DemandExplainer(self.model, self.feature_names)
        
        # Create test data
        self.X_test = pd.DataFrame({
            'lag_7': [100, 120, 90],
            'lag_14': [95, 115, 85],
            'lag_28': [100, 110, 95],
            'lag_364': [105, 125, 88],
            'rolling_mean_7': [98.5, 118.2, 92.1],
            'rolling_mean_14': [99.0, 117.5, 91.8],
            'rolling_mean_28': [100.2, 116.8, 93.5],
            'rolling_std_7': [15.2, 18.5, 12.3],
            'rolling_std_14': [16.1, 19.2, 13.1],
            'rolling_std_28': [17.5, 20.1, 14.2],
            'is_festival': [0, 1, 0],
            'is_festival_week': [0, 1, 0],
            'days_to_festival': [15, 0, 25],
            'is_weekend': [0, 1, 0],
            'day_of_week': [2, 5, 1],
            'month': [3, 6, 9],
            'quarter': [1, 2, 3],
            'price_change': [-2.5, 5.0, 0.0],
            'price_vs_avg': [0.95, 1.05, 1.0],
        })
    
    def test_compute_shap_values_returns_correct_shape(self):
        """Test compute_shap_values() returns correct shape (samples, features).
        
        Validates: Requirement 3.1
        """
        # Compute SHAP values
        shap_values = self.explainer.compute_shap_values(self.X_test)
        
        # Check shape matches (samples, features)
        expected_shape = (len(self.X_test), len(self.feature_names))
        self.assertEqual(shap_values.shape, expected_shape,
                        f"SHAP values shape {shap_values.shape} does not match expected {expected_shape}")
        
        # Verify SHAP values are numeric
        self.assertTrue(np.isfinite(shap_values).all(),
                       "SHAP values contain non-finite values")
    
    def test_get_top_drivers_returns_required_keys(self):
        """Test get_top_drivers() returns list with required keys.
        
        Validates: Requirement 3.2
        """
        # Compute SHAP values first
        self.explainer.compute_shap_values(self.X_test)
        
        # Get top drivers for first row
        drivers = self.explainer.get_top_drivers(self.X_test.iloc[0])
        
        # Check it returns a list
        self.assertIsInstance(drivers, list,
                            "get_top_drivers should return a list")
        
        # Check each driver has required keys
        required_keys = {'feature', 'value', 'impact', 'direction', 'explanation'}
        for driver in drivers:
            self.assertIsInstance(driver, dict,
                                "Each driver should be a dictionary")
            self.assertEqual(set(driver.keys()), required_keys,
                           f"Driver keys {set(driver.keys())} do not match required {required_keys}")
            
            # Validate direction values
            self.assertIn(driver['direction'], ['increase', 'decrease'],
                         f"Direction '{driver['direction']}' must be 'increase' or 'decrease'")
            
            # Validate types
            self.assertIsInstance(driver['feature'], str)
            self.assertIsInstance(driver['explanation'], str)
            self.assertTrue(isinstance(driver['value'], (int, float, np.number)))
            self.assertTrue(isinstance(driver['impact'], (int, float, np.number)))
    
    def test_get_top_drivers_returns_exactly_n_drivers(self):
        """Test get_top_drivers(top_n=3) returns exactly 3 drivers sorted by absolute SHAP value.
        
        Validates: Requirement 3.3
        """
        # Compute SHAP values first
        self.explainer.compute_shap_values(self.X_test)
        
        # Test different top_n values
        for top_n in [1, 3, 5, 10]:
            with self.subTest(top_n=top_n):
                drivers = self.explainer.get_top_drivers(self.X_test.iloc[0], top_n=top_n)
                
                # Check exact count
                self.assertEqual(len(drivers), top_n,
                               f"Expected exactly {top_n} drivers, got {len(drivers)}")
                
                # Check sorted by absolute SHAP value (descending)
                abs_impacts = [abs(d['impact']) for d in drivers]
                self.assertEqual(abs_impacts, sorted(abs_impacts, reverse=True),
                               "Drivers should be sorted by absolute impact descending")
    
    def test_get_top_drivers_with_series_input(self):
        """Test get_top_drivers() works with pandas Series input."""
        # Compute SHAP values first
        self.explainer.compute_shap_values(self.X_test)
        
        # Get top drivers with Series input
        row_series = self.X_test.iloc[0]
        drivers = self.explainer.get_top_drivers(row_series, top_n=3)
        
        # Should return 3 drivers
        self.assertEqual(len(drivers), 3)
        
        # Should have required keys
        required_keys = {'feature', 'value', 'impact', 'direction', 'explanation'}
        for driver in drivers:
            self.assertEqual(set(driver.keys()), required_keys)

    def test_get_top_drivers_initializes_explainer_if_needed(self):
        """Test get_top_drivers() works before compute_shap_values() is called."""
        drivers = self.explainer.get_top_drivers(self.X_test.iloc[0], top_n=3)

        self.assertEqual(len(drivers), 3)
        self.assertIsNotNone(self.explainer.explainer)
    
    def test_get_top_drivers_with_dataframe_input(self):
        """Test get_top_drivers() works with single-row DataFrame input."""
        # Compute SHAP values first
        self.explainer.compute_shap_values(self.X_test)
        
        # Get top drivers with DataFrame input
        row_df = self.X_test.iloc[[0]]
        drivers = self.explainer.get_top_drivers(row_df, top_n=3)
        
        # Should return 3 drivers
        self.assertEqual(len(drivers), 3)
    
    def test_get_feature_importance_summary_returns_dataframe(self):
        """Test get_feature_importance_summary() returns DataFrame with feature and importance columns.
        
        Validates: Requirement 3.4
        """
        # Compute SHAP values first
        self.explainer.compute_shap_values(self.X_test)
        
        # Get feature importance summary
        importance_df = self.explainer.get_feature_importance_summary()
        
        # Check it's a DataFrame
        self.assertIsInstance(importance_df, pd.DataFrame,
                            "get_feature_importance_summary should return a DataFrame")
        
        # Check required columns
        self.assertIn('feature', importance_df.columns,
                     "DataFrame should have 'feature' column")
        self.assertIn('importance', importance_df.columns,
                     "DataFrame should have 'importance' column")
        
        # Check number of features matches
        self.assertEqual(len(importance_df), len(self.feature_names),
                        f"Expected {len(self.feature_names)} features, got {len(importance_df)}")
        
        # Check sorted by importance descending
        self.assertTrue((importance_df['importance'].diff().dropna() <= 0).all(),
                       "Feature importance should be sorted descending")
        
        # Check importance values are non-negative
        self.assertTrue((importance_df['importance'] >= 0).all(),
                       "Importance values should be non-negative")
    
    def test_get_feature_importance_summary_without_shap_values_raises_error(self):
        """Test get_feature_importance_summary() raises error when SHAP values not computed."""
        # Create new explainer without computing SHAP values
        explainer = DemandExplainer(self.model, self.feature_names)
        
        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            explainer.get_feature_importance_summary()
        
        self.assertIn("SHAP values not computed", str(context.exception))
    
    def test_generate_explanation_for_festival_features(self):
        """Test _generate_explanation() for festival features contains 'festival' keyword.
        
        Validates: Requirement 3.5
        """
        # Test is_festival feature
        explanation = self.explainer._generate_explanation('is_festival', 1, 50.0)
        self.assertIn('festival', explanation.lower(),
                     "Festival feature explanation should contain 'festival' keyword")
        self.assertIn('50', explanation,
                     "Explanation should include impact value")
        
        # Test is_festival_week feature
        explanation = self.explainer._generate_explanation('is_festival_week', 1, 30.0)
        self.assertIn('festival', explanation.lower(),
                     "Festival week explanation should contain 'festival' keyword")
        self.assertIn('30', explanation,
                     "Explanation should include impact value")
        
        # Test days_to_festival feature
        explanation = self.explainer._generate_explanation('days_to_festival', 5, 20.0)
        self.assertIn('festival', explanation.lower(),
                     "Days to festival explanation should contain 'festival' keyword")
        self.assertIn('5', explanation,
                     "Explanation should include the value")

    def test_generate_explanation_for_no_upcoming_festival(self):
        """Test sentinel days_to_festival values are explained clearly."""
        explanation = self.explainer._generate_explanation('days_to_festival', 999, -20.0)

        self.assertIn('No festival in next 30 days', explanation)
        self.assertNotIn('999 days', explanation)
    
    def test_generate_explanation_direction_consistency(self):
        """Test _generate_explanation() uses correct direction based on impact sign."""
        # Positive impact should say "increasing"
        explanation_pos = self.explainer._generate_explanation('lag_7', 100, 25.5)
        self.assertIn('increasing', explanation_pos,
                     "Positive impact should use 'increasing'")
        
        # Negative impact should say "decreasing"
        explanation_neg = self.explainer._generate_explanation('lag_7', 100, -25.5)
        self.assertIn('decreasing', explanation_neg,
                     "Negative impact should use 'decreasing'")
    
    def test_generate_explanation_for_price_features(self):
        """Test _generate_explanation() handles price features correctly."""
        # Price change
        explanation = self.explainer._generate_explanation('price_change', 5.0, 15.0)
        self.assertIn('price', explanation.lower(),
                     "Price feature explanation should mention price")
        
        # Price vs average
        explanation = self.explainer._generate_explanation('price_vs_avg', 1.1, -10.0)
        self.assertIn('price', explanation.lower(),
                     "Price vs avg explanation should mention price")
        self.assertIn('above', explanation.lower(),
                     "Price 10% above average should say 'above'")
    
    def test_generate_explanation_for_lag_features(self):
        """Test _generate_explanation() handles lag features correctly."""
        explanation = self.explainer._generate_explanation('lag_7', 100, 20.0)
        self.assertIn('lag', explanation.lower(),
                     "Lag feature explanation should mention lag")
        self.assertIn('7', explanation,
                     "Lag explanation should include lag period")
    
    def test_generate_explanation_for_rolling_features(self):
        """Test _generate_explanation() handles rolling features correctly."""
        # Rolling mean
        explanation = self.explainer._generate_explanation('rolling_mean_7', 95.5, 18.0)
        self.assertIn('7', explanation,
                     "Rolling feature explanation should include window size")
        self.assertIn('trend', explanation.lower(),
                     "Rolling mean should mention trend")
        
        # Rolling std
        explanation = self.explainer._generate_explanation('rolling_std_14', 15.2, 8.5)
        self.assertIn('14', explanation,
                     "Rolling std explanation should include window size")
        self.assertIn('volatility', explanation.lower(),
                     "Rolling std should mention volatility")
    
    def test_format_feature_name(self):
        """Test _format_feature_name() converts feature names to readable format."""
        test_cases = {
            'is_festival': 'Festival',
            'is_festival_week': 'Festival Week',
            'lag_7': '1-Week Lag',
            'rolling_mean_7': '7-Day Average',
            'price_change': 'Price Change',
            'is_weekend': 'Weekend',
        }
        
        for feature, expected in test_cases.items():
            with self.subTest(feature=feature):
                formatted = self.explainer._format_feature_name(feature)
                self.assertEqual(formatted, expected,
                               f"Feature '{feature}' should format to '{expected}'")
    
    def test_compute_shap_values_with_large_dataset_samples(self):
        """Test compute_shap_values() samples large datasets correctly."""
        # Create large dataset
        large_X = pd.DataFrame({
            feature: np.random.randn(2000)
            for feature in self.feature_names
        })
        
        # Compute SHAP values with sample_size=500
        shap_values = self.explainer.compute_shap_values(large_X, sample_size=500)
        
        # Should sample to 500 rows
        self.assertEqual(shap_values.shape[0], 500,
                        "Large dataset should be sampled to sample_size")
        self.assertEqual(shap_values.shape[1], len(self.feature_names),
                        "Feature dimension should remain unchanged")
    
    def test_compute_shap_values_with_small_dataset_no_sampling(self):
        """Test compute_shap_values() doesn't sample small datasets."""
        # Create small dataset
        small_X = self.X_test.copy()  # 3 rows
        
        # Compute SHAP values with sample_size=1000
        shap_values = self.explainer.compute_shap_values(small_X, sample_size=1000)
        
        # Should not sample (use all 3 rows)
        self.assertEqual(shap_values.shape[0], len(small_X),
                        "Small dataset should not be sampled")
    
    def test_explainer_stores_shap_values(self):
        """Test that compute_shap_values() stores SHAP values in explainer."""
        # Initially should be None
        self.assertIsNone(self.explainer.shap_values,
                         "SHAP values should be None before computation")
        
        # Compute SHAP values
        self.explainer.compute_shap_values(self.X_test)
        
        # Should now be stored
        self.assertIsNotNone(self.explainer.shap_values,
                           "SHAP values should be stored after computation")
        self.assertEqual(self.explainer.shap_values.shape,
                        (len(self.X_test), len(self.feature_names)),
                        "Stored SHAP values should have correct shape")
    
    def test_explainer_stores_explainer_object(self):
        """Test that compute_shap_values() creates and stores SHAP explainer."""
        # Initially should be None
        self.assertIsNone(self.explainer.explainer,
                         "Explainer should be None before computation")
        
        # Compute SHAP values
        self.explainer.compute_shap_values(self.X_test)
        
        # Should now be stored
        self.assertIsNotNone(self.explainer.explainer,
                           "Explainer should be stored after computation")


if __name__ == '__main__':
    unittest.main()
