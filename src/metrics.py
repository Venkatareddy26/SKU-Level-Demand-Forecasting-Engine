"""Evaluation metrics for demand forecasting."""
import numpy as np
import pandas as pd

def calculate_wrmsse(y_true, y_pred, sales_train, weights=None):
    """
    Calculate Weighted Root Mean Squared Scaled Error (WRMSSE).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        sales_train: Training data for scaling
        weights: Optional weights per series
    
    Returns:
        WRMSSE score
    """
    # Calculate RMSSE per series
    rmsse_scores = []
    
    if isinstance(y_true, pd.DataFrame):
        for col in y_true.columns:
            true_vals = y_true[col].values
            pred_vals = y_pred[col].values if isinstance(y_pred, pd.DataFrame) else y_pred
            train_vals = sales_train[col].values if isinstance(sales_train, pd.DataFrame) else sales_train
            
            # Calculate MSE
            mse = np.mean((true_vals - pred_vals) ** 2)
            
            # Calculate scaling factor (mean squared difference in training)
            scale = np.mean(np.diff(train_vals) ** 2)
            scale = max(scale, 1e-10)  # Avoid division by zero
            
            # RMSSE
            rmsse = np.sqrt(mse / scale)
            rmsse_scores.append(rmsse)
    else:
        # Single series
        mse = np.mean((y_true - y_pred) ** 2)
        scale = np.mean(np.diff(sales_train) ** 2)
        scale = max(scale, 1e-10)
        rmsse = np.sqrt(mse / scale)
        rmsse_scores.append(rmsse)
    
    # Apply weights if provided
    if weights is None:
        weights = np.ones(len(rmsse_scores))
    
    weights = np.array(weights) / np.sum(weights)
    wrmsse = np.sum(np.array(rmsse_scores) * weights)
    
    return wrmsse

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def evaluate_forecast(y_true, y_pred, sales_train=None, metric='all'):
    """
    Evaluate forecast with multiple metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        sales_train: Training data (required for WRMSSE)
        metric: 'all', 'wrmsse', 'mape', 'mae', or 'rmse'
    
    Returns:
        Dictionary of metrics
    """
    results = {}
    
    if metric in ['all', 'mae']:
        results['MAE'] = calculate_mae(y_true, y_pred)
    
    if metric in ['all', 'rmse']:
        results['RMSE'] = calculate_rmse(y_true, y_pred)
    
    if metric in ['all', 'mape']:
        results['MAPE'] = calculate_mape(y_true, y_pred)
    
    if metric in ['all', 'wrmsse'] and sales_train is not None:
        results['WRMSSE'] = calculate_wrmsse(y_true, y_pred, sales_train)
    
    return results
