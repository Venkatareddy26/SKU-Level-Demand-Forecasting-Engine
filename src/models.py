"""Forecasting models: LightGBM and NeuralProphet."""
import pandas as pd
import numpy as np
import lightgbm as lgb
from neuralprophet import NeuralProphet
import pickle
from datetime import datetime, timedelta

class LightGBMForecaster:
    """Global LightGBM model for all SKUs."""
    
    def __init__(self, params=None):
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        self.model = None
        self.feature_cols = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None, num_boost_round=1000):
        """Train LightGBM model."""
        print("Training LightGBM...")
        
        # Store feature columns
        self.feature_cols = X_train.columns.tolist()
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )
        
        print(f"✓ Training complete. Best iteration: {self.model.best_iteration}")
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X[self.feature_cols], num_iteration=self.model.best_iteration)
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance
    
    def save(self, path):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'feature_cols': self.feature_cols}, f)
        print(f"✓ Model saved to {path}")
    
    def load(self, path):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_cols = data['feature_cols']
        print(f"✓ Model loaded from {path}")
        return self


class NeuralProphetForecaster:
    """NeuralProphet model with festival regressors."""
    
    def __init__(self, growth='linear', seasonality_mode='multiplicative'):
        self.models = {}  # One model per category
        self.growth = growth
        self.seasonality_mode = seasonality_mode
    
    def prepare_data(self, df, date_col='date', target_col='sales', id_col='id'):
        """Prepare data in NeuralProphet format (ds, y)."""
        df_prophet = df[[date_col, target_col, id_col]].copy()
        df_prophet.columns = ['ds', 'y', 'ID']
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        return df_prophet
    
    def train(self, df, category_col='category', epochs=50):
        """Train one NeuralProphet model per category."""
        print("Training NeuralProphet models...")
        
        categories = df[category_col].unique()
        
        for category in categories:
            print(f"\nTraining model for category: {category}")
            
            # Filter data for this category
            cat_data = df[df[category_col] == category].copy()
            cat_data = self.prepare_data(cat_data)
            
            # Initialize model
            model = NeuralProphet(
                growth=self.growth,
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                epochs=epochs
            )
            
            # Add festival regressor if available
            if 'is_festival' in df.columns:
                cat_data['is_festival'] = df[df[category_col] == category]['is_festival'].values
                model.add_future_regressor('is_festival')
            
            # Train model
            metrics = model.fit(cat_data, freq='D')
            
            self.models[category] = model
            print(f"✓ Model trained for {category}")
        
        return self
    
    def predict(self, df, category, periods=56):
        """Make forecast for a category."""
        if category not in self.models:
            raise ValueError(f"No model trained for category: {category}")
        
        model = self.models[category]
        future = model.make_future_dataframe(df, periods=periods)
        forecast = model.predict(future)
        
        return forecast
    
    def save(self, path_prefix):
        """Save models to disk."""
        for category, model in self.models.items():
            path = f"{path_prefix}_{category}.pkl"
            with open(path, 'wb') as f:
                pickle.dump(model, f)
        print(f"✓ Models saved with prefix {path_prefix}")
    
    def load(self, path_prefix, categories):
        """Load models from disk."""
        for category in categories:
            path = f"{path_prefix}_{category}.pkl"
            with open(path, 'rb') as f:
                self.models[category] = pickle.load(f)
        print(f"✓ Models loaded from {path_prefix}")
        return self
