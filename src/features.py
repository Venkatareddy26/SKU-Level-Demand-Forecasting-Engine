"""Feature engineering for demand forecasting."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FeatureEngineer:
    """Build lag features, rolling stats, and festival flags."""
    
    def __init__(self, festival_calendar_path="data/festival_calendar.csv", verbose=True):
        self.verbose = verbose
        self.festival_df = pd.read_csv(festival_calendar_path)
        self.festival_df['date'] = pd.to_datetime(self.festival_df['date'])
        self.festival_lookup = (
            self.festival_df
            .groupby('date', as_index=False)['festival']
            .agg(lambda values: ', '.join(dict.fromkeys(values.astype(str))))
        )
        self.festival_dates = self.festival_lookup['date'].values.astype('datetime64[D]')
    
    def create_lag_features(self, df, target_col='sales', lags=[7, 14, 28, 364]):
        """Create lag features for time series."""
        df = df.copy()
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby('id')[target_col].shift(lag)
        return df
    
    def create_rolling_features(self, df, target_col='sales', windows=[7, 14, 28]):
        """Create rolling mean and std features.
        
        Uses .shift(1) to prevent look-ahead bias — the rolling window
        is computed on data BEFORE the current row, never including it.
        """
        df = df.copy()
        for window in windows:
            df[f'rolling_mean_{window}'] = df.groupby('id')[target_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby('id')[target_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )
        return df
    
    def add_festival_features(self, df, date_col='date'):
        """Add festival flags and days-to-festival features.
        
        Vectorized implementation — O(n × k) where k = number of festival dates,
        using numpy broadcasting instead of nested Python loops.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Mark exact festival dates via merge
        festival_lookup = self.festival_lookup.copy()
        festival_lookup.columns = [date_col, 'festival_name']
        df = df.merge(festival_lookup, on=date_col, how='left')
        df['festival_name'] = df['festival_name'].fillna('')
        df['is_festival'] = (df['festival_name'] != '').astype(int)
        
        # Vectorized days-to-festival calculation
        row_dates = df[date_col].values.astype('datetime64[D]')
        
        # Compute days-to-next-festival for each row (within 30-day window)
        days_to = np.full(len(df), 999, dtype=np.int32)
        for fd in self.festival_dates:
            diff = (fd - row_dates).astype('timedelta64[D]').astype(np.int32)
            mask = (diff >= 0) & (diff <= 30) & (diff < days_to)
            days_to[mask] = diff[mask]
        
        df['days_to_festival'] = days_to
        
        # Festival week flag (7 days before festival = pre-stocking period)
        df['is_festival_week'] = (df['days_to_festival'] <= 7).astype(int)
        
        return df
    
    def add_calendar_features(self, df, date_col='date'):
        """Add calendar-based features."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_month'] = df[date_col].dt.day
        df['week_of_year'] = df[date_col].dt.isocalendar().week.astype(int)
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 7).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 24).astype(int)
        df['year'] = df[date_col].dt.year
        
        return df
    
    def add_price_features(self, df, price_col='price'):
        """Add price-related features if price column exists."""
        df = df.copy()
        if price_col not in df.columns:
            return df
        
        # Price lag (previous day's price)
        df['price_lag_1'] = df.groupby('id')[price_col].shift(1)
        
        # Price change (day-over-day)
        df['price_change'] = df[price_col] - df['price_lag_1']
        df['price_change_pct'] = df['price_change'] / df['price_lag_1'].replace(0, np.nan)
        
        # Rolling average price (7-day)
        df['price_rolling_mean_7'] = df.groupby('id')[price_col].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        )
        
        # Price relative to rolling mean (proxy for discount detection)
        df['price_vs_avg'] = df[price_col] / df['price_rolling_mean_7'].replace(0, np.nan)
        
        return df
    
    def build_features(self, df, target_col='sales', date_col='date'):
        """Build complete feature set."""
        if self.verbose:
            print("Building features...")
        
        # Sort by id and date
        df = df.sort_values(['id', date_col]).reset_index(drop=True)
        
        # Add calendar features
        df = self.add_calendar_features(df, date_col)
        
        # Add festival features
        df = self.add_festival_features(df, date_col)
        
        # Add lag features
        df = self.create_lag_features(df, target_col)
        
        # Add rolling features
        df = self.create_rolling_features(df, target_col)
        
        # Add price features (if price column exists)
        df = self.add_price_features(df)
        
        if self.verbose:
            print(f"[OK] Features created. Shape: {df.shape}")
        return df
