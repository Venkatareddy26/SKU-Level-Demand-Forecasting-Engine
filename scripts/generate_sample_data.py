"""Generate sample sales data for testing."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(
    n_skus=20,
    start_date='2023-01-01',
    end_date='2024-12-31',
    festival_calendar_path='data/festival_calendar.csv',
    output_path='data/sample_sales.csv'
):
    """
    Generate synthetic sales data with seasonal patterns and festival spikes.
    
    Uses the actual festival_calendar.csv to ensure festival spikes align
    with the feature engineering pipeline.
    
    Args:
        n_skus: Number of SKUs to generate
        start_date: Start date for data
        end_date: End date for data
        festival_calendar_path: Path to festival calendar CSV
        output_path: Output CSV path
    """
    print(f"Generating sample data for {n_skus} SKUs...")
    
    # Date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Load festival dates from the ACTUAL calendar (not hardcoded)
    try:
        festival_df = pd.read_csv(festival_calendar_path)
        festival_df['date'] = pd.to_datetime(festival_df['date'])
        festival_dates = set(festival_df['date'].dt.date)
        
        # Pre-festival dates (7 days before each festival)
        pre_festival_dates = set()
        for fd in festival_df['date']:
            for d in range(1, 8):
                pre_festival_dates.add((fd - timedelta(days=d)).date())
        
        print(f"  Loaded {len(festival_dates)} festival dates from calendar")
    except FileNotFoundError:
        print("  Warning: festival_calendar.csv not found, using empty calendar")
        festival_dates = set()
        pre_festival_dates = set()
    
    data = []
    categories = ['Food', 'Beverage', 'Personal Care', 'Household']
    
    np.random.seed(42)  # Reproducibility
    
    for sku_id in range(1, n_skus + 1):
        sku_name = f"SKU_{sku_id:03d}"
        
        # Base demand (different for each SKU)
        base_demand = np.random.randint(50, 200)
        
        # Category (consistent per SKU)
        category = categories[(sku_id - 1) % len(categories)]
        
        # Base price (consistent per SKU with minor variation)
        base_price = np.random.uniform(50, 500)
        
        for date in dates:
            # Base sales with small trend
            day_num = (date - dates[0]).days
            trend = 1 + 0.0001 * day_num  # Slight upward trend
            sales = base_demand * trend
            
            # Weekly seasonality (weekend boost)
            if date.dayofweek >= 5:  # Saturday, Sunday
                sales *= 1.2
            
            # Monthly seasonality (month-end spike)
            if date.day >= 25:
                sales *= 1.15
            
            # Yearly seasonality (Q4 boost)
            if date.month in [10, 11, 12]:
                sales *= 1.3
            
            # Festival spike (from actual calendar)
            if date.date() in festival_dates:
                sales *= 2.5  # 150% increase during festivals
            
            # Pre-festival stocking period
            if date.date() in pre_festival_dates:
                sales *= 1.5
            
            # Add random noise
            sales *= np.random.uniform(0.85, 1.15)
            
            # Ensure non-negative
            sales = max(0, int(sales))
            
            # Price with slight daily variation
            price = base_price * np.random.uniform(0.95, 1.05)
            
            data.append({
                'id': sku_name,
                'date': date.strftime('%Y-%m-%d'),
                'sales': sales,
                'category': category,
                'price': round(price, 2)
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✓ Sample data generated: {len(df)} rows")
    print(f"✓ Saved to: {output_path}")
    print(f"\nData summary:")
    print(f"  SKUs: {n_skus}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total rows: {len(df)}")
    print(f"  Avg daily sales per SKU: {df['sales'].mean():.1f}")
    print(f"\nCategories:")
    print(df.groupby('category')['id'].nunique().to_string())
    
    return df

if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data(
        n_skus=20,
        start_date='2023-01-01',
        end_date='2024-12-31',
        output_path='data/sample_sales.csv'
    )
    
    print("\n✓ You can now use this data in the Streamlit dashboard!")
    print("  Upload: data/sample_sales.csv")
