"""Evaluation metrics for demand forecasting."""
import numpy as np
import pandas as pd

def calculate_rmsse(y_true, y_pred, y_train):
    """
    Calculate Root Mean Squared Scaled Error for a single series.
    
    The scaling denominator is the mean squared successive difference
    in the training data (naive one-step-ahead baseline).
    
    Args:
        y_true: Actual values (test period)
        y_pred: Predicted values (test period)
        y_train: Training data for the same series (used for scaling)
    
    Returns:
        RMSSE score for the series
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    
    # Numerator: MSE of predictions
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Denominator: mean squared successive difference in training
    n = len(y_train)
    if n < 2:
        return np.nan
    scale = np.sum((y_train[1:] - y_train[:-1]) ** 2) / (n - 1)
    scale = max(scale, 1e-10)  # Avoid division by zero
    
    return np.sqrt(mse / scale)


def calculate_wrmsse(y_true_dict, y_pred_dict, y_train_dict, weights=None):
    """
    Calculate Weighted Root Mean Squared Scaled Error across multiple series.
    
    This is a simplified version of the M5 WRMSSE. The full M5 metric uses
    12-level hierarchical aggregation and revenue-based weights.
    
    Args:
        y_true_dict: Dict of {series_id: actual_values}
        y_pred_dict: Dict of {series_id: predicted_values}
        y_train_dict: Dict of {series_id: training_values}
        weights: Optional dict of {series_id: weight}. 
                 If None, uses equal weights.
    
    Returns:
        Weighted RMSSE score
    """
    series_ids = list(y_true_dict.keys())
    rmsse_scores = {}
    
    for sid in series_ids:
        rmsse_scores[sid] = calculate_rmsse(
            y_true_dict[sid], y_pred_dict[sid], y_train_dict[sid]
        )
    
    # Filter out NaN scores
    valid_ids = [sid for sid in series_ids if not np.isnan(rmsse_scores[sid])]
    if not valid_ids:
        return np.nan
    
    if weights is None:
        weights = {sid: 1.0 / len(valid_ids) for sid in valid_ids}
    
    # Normalize weights
    total_weight = sum(weights[sid] for sid in valid_ids)
    wrmsse = sum(rmsse_scores[sid] * weights[sid] / total_weight for sid in valid_ids)
    
    return wrmsse


def calculate_wrmsse_simple(y_true, y_pred, y_train):
    """
    Simplified WRMSSE for a single flattened series.
    
    Convenience wrapper when you just have arrays (not per-series dicts).
    
    Args:
        y_true: Actual values (array)
        y_pred: Predicted values (array)
        y_train: Training values for scaling (array)
    
    Returns:
        RMSSE score
    """
    return calculate_rmsse(y_true, y_pred, y_train)


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error.
    
    Skips zero-actual values to avoid division by zero.
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def evaluate_forecast(y_true, y_pred, y_train=None, metric='all'):
    """
    Evaluate forecast with multiple metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_train: Training data (required for WRMSSE/RMSSE)
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
    
    if metric in ['all', 'wrmsse'] and y_train is not None:
        results['WRMSSE'] = calculate_wrmsse_simple(y_true, y_pred, y_train)
    
    return results
