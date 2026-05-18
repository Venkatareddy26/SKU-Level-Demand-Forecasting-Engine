"""Training pipeline for demand forecasting models."""
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features import FeatureEngineer
from models import LightGBMForecaster
from metrics import evaluate_forecast
from explainer import DemandExplainer

def load_m5_data(data_path="data/raw"):
    """Load and prepare M5 dataset."""
    print("Loading M5 dataset...")
    
    # Load sales data
    sales_df = pd.read_csv(f"{data_path}/sales_train_validation.csv")
    calendar_df = pd.read_csv(f"{data_path}/calendar.csv")
    prices_df = pd.read_csv(f"{data_path}/sell_prices.csv")
    
    print(f"Sales shape: {sales_df.shape}")
    print(f"Calendar shape: {calendar_df.shape}")
    
    # Melt sales data to long format
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_long = sales_df.melt(
        id_vars=id_cols,
        var_name='d',
        value_name='sales'
    )
    
    # Merge with calendar
    sales_long = sales_long.merge(calendar_df[['d', 'date', 'wm_yr_wk']], on='d', how='left')
    
    # Merge with prices
    sales_long = sales_long.merge(
        prices_df,
        on=['store_id', 'item_id', 'wm_yr_wk'],
        how='left'
    )
    
    # Convert date
    sales_long['date'] = pd.to_datetime(sales_long['date'])
    
    print(f"[OK] Data loaded. Shape: {sales_long.shape}")
    return sales_long


def load_sample_data(data_path="data/sample_sales.csv"):
    """Load sample sales data for training."""
    print("Loading sample sales data...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"[OK] Sample data loaded. Shape: {df.shape}")
    return df


def train_lightgbm_model(df, test_days=60):
    """Train LightGBM model with temporal train/test split.
    
    Uses the last `test_days` days as validation (no random splitting)
    to prevent data leakage in time-series forecasting.
    
    Args:
        df: DataFrame with columns [id, date, sales, ...]
        test_days: Number of days to hold out for validation
    
    Returns:
        Tuple of (model, explainer, val_metrics)
    """
    print("\n" + "="*50)
    print("TRAINING LIGHTGBM MODEL")
    print("="*50)
    
    # Feature engineering
    fe = FeatureEngineer()
    df_features = fe.build_features(df, target_col='sales', date_col='date')
    
    # Remove rows with NaN (from lag features)
    df_features = df_features.dropna()
    
    # Define feature columns (exclude identifiers, target, and non-numeric)
    exclude_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                    'date', 'sales', 'd', 'wm_yr_wk', 'festival_name', 'category']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
    
    # TEMPORAL SPLIT — use last N days as validation (no data leakage)
    df_features = df_features.sort_values('date')
    cutoff_date = df_features['date'].max() - pd.Timedelta(days=test_days)
    
    train_df = df_features[df_features['date'] <= cutoff_date]
    val_df = df_features[df_features['date'] > cutoff_date]
    
    X_train = train_df[feature_cols]
    y_train = train_df['sales']
    X_val = val_df[feature_cols]
    y_val = val_df['sales']
    
    # Store training target for WRMSSE computation
    y_train_array = y_train.values
    
    print(f"\nTemporal split:")
    print(f"  Train: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(X_train)} rows)")
    print(f"  Val:   {val_df['date'].min().date()} to {val_df['date'].max().date()} ({len(X_val)} rows)")
    
    # Train model
    model = LightGBMForecaster()
    model.train(X_train, y_train, X_val, y_val, num_boost_round=500)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    train_metrics = evaluate_forecast(y_train, y_pred_train, y_train=y_train_array, metric='all')
    val_metrics = evaluate_forecast(y_val, y_pred_val, y_train=y_train_array, metric='all')
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print("\nTrain Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Feature importance
    print("\nTop 10 Features:")
    importance = model.get_feature_importance(top_n=10)
    print(importance.to_string(index=False))
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/lightgbm_model.pkl")
    
    # SHAP explainer
    print("\nComputing SHAP values...")
    explainer = DemandExplainer(model, feature_cols)
    explainer.compute_shap_values(X_val.head(500))
    
    # Example: Get top drivers for first validation sample
    print("\nExample: Top 3 Demand Drivers for Sample Prediction:")
    sample_row = X_val.iloc[0]
    drivers = explainer.get_top_drivers(sample_row, top_n=3)
    for i, driver in enumerate(drivers, 1):
        print(f"  {i}. {driver['explanation']}")
    
    return model, explainer, val_metrics

if __name__ == "__main__":
    # Check if M5 data exists
    if os.path.exists("data/raw/sales_train_validation.csv"):
        print("Found M5 dataset. Training on M5 data...")
        df = load_m5_data()
        
        # Sample data for faster training (remove for full training)
        print("\nSampling data for faster training...")
        sample_ids = df['id'].unique()[:100]  # First 100 SKUs
        df = df[df['id'].isin(sample_ids)]
        print(f"Sampled data shape: {df.shape}")
    elif os.path.exists("data/sample_sales.csv"):
        print("M5 dataset not found. Using sample data...")
        df = load_sample_data()
    else:
        print("[ERROR] No data found!")
        print("Run: python scripts/generate_sample_data.py")
        print("Or:  python scripts/download_data.py")
        sys.exit(1)
    
    # Train model
    model, explainer, metrics = train_lightgbm_model(df)
    
    print("\n" + "="*50)
    print("[OK] TRAINING COMPLETE")
    print("="*50)
    print(f"\nValidation RMSE: {metrics['RMSE']:.2f}")
    print(f"Validation MAPE: {metrics['MAPE']:.2f}%")
    if 'WRMSSE' in metrics:
        print(f"Validation WRMSSE: {metrics['WRMSSE']:.4f}")
    print("\nModel saved to: models/lightgbm_model.pkl")
