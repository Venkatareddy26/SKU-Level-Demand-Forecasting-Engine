"""Feature engineering for demand forecasting."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FeatureEngineer:
    """Build lag features, rolling stats, and festival flags."""
    
    def __init__(self, festival_calendar_path="data/festival_calendar.csv"):
        self.festival_df = pd.read_csv(festival_calendar_path)
        self.festival_df['date'] = pd.to_datetime(self.festival_df['date'])
    
    def create_lag_features(self, df, target_col='sales', lags=[7, 14, 28, 364]):
        """Create lag features for time series."""
        df = df.copy()
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby('id')[target_col].shift(lag)
        return df
    
    def create_rolling_features(self, df, target_col='sales', windows=[7, 14, 28]):
        """Create rolling mean and std features."""
        df = df.copy()
        for window in windows:
            df[f'rolling_mean_{window}'] = df.groupby('id')[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby('id')[target_col].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        return df
    
    def add_festival_features(self, df, date_col='date'):
        """Add festival flags and days-to-festival features."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Initialize festival columns
        df['is_festival'] = 0
        df['days_to_festival'] = 999
        df['festival_name'] = ''
        
        # Mark festival dates and nearby dates
        for _, festival_row in self.festival_df.iterrows():
            festival_date = festival_row['date']
            festival_name = festival_row['festival']
            
            # Mark exact festival date
            mask = df[date_col] == festival_date
            df.loc[mask, 'is_festival'] = 1
            df.loc[mask, 'festival_name'] = festival_name
            
            # Calculate days to next festival (within 30 days)
            for idx, row in df.iterrows():
                days_diff = (festival_date - row[date_col]).days
                if 0 <= days_diff <= 30:
                    if days_diff < df.loc[idx, 'days_to_festival']:
                        df.loc[idx, 'days_to_festival'] = days_diff
        
        # Add festival week flag (7 days before festival)
        df['is_festival_week'] = (df['days_to_festival'] <= 7).astype(int)
        
        return df
    
    def add_calendar_features(self, df, date_col='date'):
        """Add calendar-based features."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_month'] = df[date_col].dt.day
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 7).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 24).astype(int)
        
        return df
    
    def build_features(self, df, target_col='sales', date_col='date'):
        """Build complete feature set."""
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
        
        print(f"✓ Features created. Shape: {df.shape}")
        return df
