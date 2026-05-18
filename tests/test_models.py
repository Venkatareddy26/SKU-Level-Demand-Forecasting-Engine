"""Unit tests for forecasting models."""
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models import LightGBMForecaster


class TestLightGBMForecaster(unittest.TestCase):
    """Test cases for LightGBMForecaster class.
    
    Tests cover:
    - Requirement 2.1: Training produces non-null model and feature_cols
    - Requirement 2.2: Predict returns correct shape matching input rows
    - Requirement 2.3: Save persists model and metadata to disk
    - Requirement 2.4: Load restores model correctly
    - Requirement 2.5: Predict before training raises ValueError
    - Requirement 2.6: get_feature_importance returns sorted DataFrame
    """
    
    def setUp(self):
        """Set up test data and model instance."""
        # Create sample training data with multiple features
        np.random.seed(42)
        n_samples = 200
        
        self.X_train = pd.DataFrame({
            'lag_7': np.random.randint(50, 200, n_samples),
            'lag_14': np.random.randint(50, 200, n_samples),
            'rolling_mean_7': np.random.uniform(80, 150, n_samples),
            'rolling_std_7': np.random.uniform(10, 30, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'is_festival': np.random.randint(0, 2, n_samples),
            'price': np.random.uniform(80, 120, n_samples)
        })
        
        # Create target variable with some relationship to features
        self.y_train = (
            self.X_train['lag_7'] * 0.5 + 
            self.X_train['rolling_mean_7'] * 0.3 + 
            self.X_train['is_festival'] * 20 + 
            np.random.normal(0, 10, n_samples)
        )
        
        # Create validation data
        n_val = 50
        self.X_val = pd.DataFrame({
            'lag_7': np.random.randint(50, 200, n_val),
            'lag_14': np.random.randint(50, 200, n_val),
            'rolling_mean_7': np.random.uniform(80, 150, n_val),
            'rolling_std_7': np.random.uniform(10, 30, n_val),
            'day_of_week': np.random.randint(0, 7, n_val),
            'month': np.random.randint(1, 13, n_val),
            'is_festival': np.random.randint(0, 2, n_val),
            'price': np.random.uniform(80, 120, n_val)
        })
        
        self.y_val = (
            self.X_val['lag_7'] * 0.5 + 
            self.X_val['rolling_mean_7'] * 0.3 + 
            self.X_val['is_festival'] * 20 + 
            np.random.normal(0, 10, n_val)
        )
        
        # Create test data for prediction
        n_test = 30
        self.X_test = pd.DataFrame({
            'lag_7': np.random.randint(50, 200, n_test),
            'lag_14': np.random.randint(50, 200, n_test),
            'rolling_mean_7': np.random.uniform(80, 150, n_test),
            'rolling_std_7': np.random.uniform(10, 30, n_test),
            'day_of_week': np.random.randint(0, 7, n_test),
            'month': np.random.randint(1, 13, n_test),
            'is_festival': np.random.randint(0, 2, n_test),
            'price': np.random.uniform(80, 120, n_test)
        })
        
        # Initialize model
        self.model = LightGBMForecaster()
    
    def test_train_produces_non_null_model_and_features(self):
        """Test that train() produces non-null model and feature_cols.
        
        Validates: Requirement 2.1
        """
        # Train the model
        self.model.train(self.X_train, self.y_train, num_boost_round=50)
        
        # Check model is not None
        self.assertIsNotNone(self.model.model, "Model should not be None after training")
        
        # Check feature_cols is not None
        self.assertIsNotNone(self.model.feature_cols, "feature_cols should not be None after training")
        
        # Check feature_cols matches training data columns
        self.assertEqual(
            self.model.feature_cols, 
            self.X_train.columns.tolist(),
            "feature_cols should match training data columns"
        )
        
        # Check model has expected attributes
        self.assertTrue(hasattr(self.model.model, 'best_iteration'), 
                       "Model should have best_iteration attribute")
    
    def test_train_with_validation_data(self):
        """Test that train() works with validation data.
        
        Validates: Requirement 2.1
        """
        # Train with validation data
        self.model.train(
            self.X_train, self.y_train, 
            self.X_val, self.y_val, 
            num_boost_round=50
        )
        
        # Check model is trained
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.feature_cols)
    
    def test_predict_returns_correct_shape(self):
        """Test that predict() returns predictions with shape matching input rows.
        
        Validates: Requirement 2.2
        """
        # Train the model first
        self.model.train(self.X_train, self.y_train, num_boost_round=50)
        
        # Make predictions
        predictions = self.model.predict(self.X_test)
        
        # Check predictions shape matches input rows
        self.assertEqual(
            len(predictions), 
            len(self.X_test),
            f"Predictions length {len(predictions)} should match input rows {len(self.X_test)}"
        )
        
        # Check predictions is a numpy array
        self.assertIsInstance(predictions, np.ndarray, "Predictions should be a numpy array")
        
        # Check predictions are numeric
        self.assertTrue(np.issubdtype(predictions.dtype, np.number), 
                       "Predictions should be numeric")
        
        # Check no NaN values in predictions
        self.assertFalse(np.any(np.isnan(predictions)), 
                        "Predictions should not contain NaN values")
    
    def test_predict_with_different_input_sizes(self):
        """Test that predict() handles different input sizes correctly.
        
        Validates: Requirement 2.2
        """
        # Train the model
        self.model.train(self.X_train, self.y_train, num_boost_round=50)
        
        # Test with different input sizes
        for n_rows in [1, 10, 50, 100]:
            X_test_subset = self.X_test.head(n_rows) if n_rows <= len(self.X_test) else self.X_test
            if n_rows > len(self.X_test):
                # Create larger test set
                X_test_subset = pd.concat([self.X_test] * (n_rows // len(self.X_test) + 1)).head(n_rows)
            
            predictions = self.model.predict(X_test_subset)
            self.assertEqual(len(predictions), n_rows, 
                           f"Predictions should have {n_rows} rows")
    
    def test_save_persists_model_and_metadata(self):
        """Test that save() persists model and metadata to disk.
        
        Validates: Requirement 2.3
        """
        # Train the model
        self.model.train(self.X_train, self.y_train, num_boost_round=50)
        
        # Create temporary file for saving
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save the model
            self.model.save(tmp_path)
            
            # Check file exists
            self.assertTrue(os.path.exists(tmp_path), 
                          f"Model file should exist at {tmp_path}")
            
            # Check file is not empty
            file_size = os.path.getsize(tmp_path)
            self.assertGreater(file_size, 0, 
                             "Model file should not be empty")
            
            # Load the saved data to verify contents
            import pickle
            with open(tmp_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            # Check saved data contains model and feature_cols
            self.assertIn('model', saved_data, 
                         "Saved data should contain 'model' key")
            self.assertIn('feature_cols', saved_data, 
                         "Saved data should contain 'feature_cols' key")
            
            # Check feature_cols matches
            self.assertEqual(saved_data['feature_cols'], self.model.feature_cols,
                           "Saved feature_cols should match model's feature_cols")
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_load_restores_model_correctly(self):
        """Test that load() restores model and feature_cols correctly.
        
        Validates: Requirement 2.4
        """
        # Train the model
        self.model.train(self.X_train, self.y_train, num_boost_round=50)
        
        # Get predictions before saving
        predictions_before = self.model.predict(self.X_test)
        feature_cols_before = self.model.feature_cols.copy()
        
        # Create temporary file for saving
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save the model
            self.model.save(tmp_path)
            
            # Create a new model instance and load
            new_model = LightGBMForecaster()
            new_model.load(tmp_path)
            
            # Check model is loaded
            self.assertIsNotNone(new_model.model, 
                               "Loaded model should not be None")
            self.assertIsNotNone(new_model.feature_cols, 
                               "Loaded feature_cols should not be None")
            
            # Check feature_cols matches
            self.assertEqual(new_model.feature_cols, feature_cols_before,
                           "Loaded feature_cols should match original")
            
            # Check predictions match
            predictions_after = new_model.predict(self.X_test)
            np.testing.assert_array_almost_equal(
                predictions_before, predictions_after, decimal=5,
                err_msg="Predictions from loaded model should match original"
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_save_load_round_trip(self):
        """Test complete save/load round trip preserves model behavior.
        
        Validates: Requirements 2.3, 2.4
        """
        # Train the model
        self.model.train(self.X_train, self.y_train, num_boost_round=50)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save and load
            self.model.save(tmp_path)
            loaded_model = LightGBMForecaster()
            loaded_model.load(tmp_path)
            
            # Test multiple predictions to ensure consistency
            for _ in range(3):
                X_random = pd.DataFrame({
                    col: np.random.uniform(self.X_train[col].min(), 
                                          self.X_train[col].max(), 20)
                    for col in self.X_train.columns
                })
                
                pred_original = self.model.predict(X_random)
                pred_loaded = loaded_model.predict(X_random)
                
                np.testing.assert_array_almost_equal(
                    pred_original, pred_loaded, decimal=5,
                    err_msg="Loaded model predictions should match original"
                )
        
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_predict_before_training_raises_error(self):
        """Test that predict() before training raises ValueError.
        
        Validates: Requirement 2.5
        """
        # Create a new untrained model
        untrained_model = LightGBMForecaster()
        
        # Attempt to predict should raise ValueError
        with self.assertRaises(ValueError) as context:
            untrained_model.predict(self.X_test)
        
        # Check error message
        self.assertIn("Model not trained", str(context.exception),
                     "Error message should mention model not trained")
    
    def test_get_feature_importance_returns_sorted_dataframe(self):
        """Test that get_feature_importance() returns sorted DataFrame.
        
        Validates: Requirement 2.6
        """
        # Train the model
        self.model.train(self.X_train, self.y_train, num_boost_round=50)
        
        # Get feature importance
        importance_df = self.model.get_feature_importance(top_n=20)
        
        # Check it's a DataFrame
        self.assertIsInstance(importance_df, pd.DataFrame,
                            "Feature importance should be a DataFrame")
        
        # Check required columns exist
        self.assertIn('feature', importance_df.columns,
                     "DataFrame should have 'feature' column")
        self.assertIn('importance', importance_df.columns,
                     "DataFrame should have 'importance' column")
        
        # Check sorted by importance descending
        importance_values = importance_df['importance'].values
        self.assertTrue(
            all(importance_values[i] >= importance_values[i+1] 
                for i in range(len(importance_values)-1)),
            "Importance values should be sorted in descending order"
        )
        
        # Check number of features returned
        expected_n = min(20, len(self.X_train.columns))
        self.assertEqual(len(importance_df), expected_n,
                        f"Should return top {expected_n} features")
        
        # Check all features are from training data
        for feature in importance_df['feature']:
            self.assertIn(feature, self.X_train.columns,
                         f"Feature {feature} should be from training data")
    
    def test_get_feature_importance_with_different_top_n(self):
        """Test get_feature_importance() with different top_n values.
        
        Validates: Requirement 2.6
        """
        # Train the model
        self.model.train(self.X_train, self.y_train, num_boost_round=50)
        
        # Test different top_n values
        for top_n in [5, 10, 20]:
            importance_df = self.model.get_feature_importance(top_n=top_n)
            expected_n = min(top_n, len(self.X_train.columns))
            self.assertEqual(len(importance_df), expected_n,
                           f"Should return top {expected_n} features for top_n={top_n}")
    
    def test_get_feature_importance_before_training_raises_error(self):
        """Test that get_feature_importance() before training raises ValueError.
        
        Validates: Requirement 2.6
        """
        # Create a new untrained model
        untrained_model = LightGBMForecaster()
        
        # Attempt to get feature importance should raise ValueError
        with self.assertRaises(ValueError) as context:
            untrained_model.get_feature_importance()
        
        # Check error message
        self.assertIn("Model not trained", str(context.exception),
                     "Error message should mention model not trained")
    
    def test_custom_params(self):
        """Test that custom parameters are used during training."""
        # Create model with custom parameters
        custom_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 15,
            'learning_rate': 0.1,
            'verbose': -1
        }
        custom_model = LightGBMForecaster(params=custom_params)
        
        # Check params are set
        self.assertEqual(custom_model.params['num_leaves'], 15)
        self.assertEqual(custom_model.params['learning_rate'], 0.1)
        
        # Train and verify it works
        custom_model.train(self.X_train, self.y_train, num_boost_round=50)
        self.assertIsNotNone(custom_model.model)
        
        # Make predictions
        predictions = custom_model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_train_returns_self(self):
        """Test that train() returns self for method chaining."""
        result = self.model.train(self.X_train, self.y_train, num_boost_round=50)
        self.assertIs(result, self.model, "train() should return self")
    
    def test_load_returns_self(self):
        """Test that load() returns self for method chaining."""
        # Train and save
        self.model.train(self.X_train, self.y_train, num_boost_round=50)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            self.model.save(tmp_path)
            
            # Load and check return value
            new_model = LightGBMForecaster()
            result = new_model.load(tmp_path)
            self.assertIs(result, new_model, "load() should return self")
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == '__main__':
    unittest.main()
