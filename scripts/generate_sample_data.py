"""Generate sample sales data for testing."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(
    n_skus=10,
    start_date='2023-01-01',
    end_date='2024-12-31',
    output_path='data/sample_sales.csv'
):
    """
    Generate synthetic sales data with seasonal patterns and festival spikes.
    
    Args:
        n_skus: Number of SKUs to generate
        start_date: Start date for data
        end_date: End date for data
        output_path: Output CSV path
    """
    print(f"Generating sample data for {n_skus} SKUs...")
    
    # Date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Festival dates (approximate)
    festivals = {
        '2023-10-24': 'Diwali',
        '2023-10-15': 'Dussehra',
        '2024-01-15': 'Pongal',
        '2024-11-01': 'Diwali',
        '2024-10-12': 'Dussehra'
    }
    
    data = []
    
    for sku_id in range(1, n_skus + 1):
        sku_name = f"SKU_{sku_id:03d}"
        
        # Base demand (different for each SKU)
        base_demand = np.random.randint(50, 200)
        
        # Category (affects seasonality)
        category = np.random.choice(['Food', 'Beverage', 'Personal Care', 'Household'])
        
        for date in dates:
            # Base sales
            sales = base_demand
            
            # Weekly seasonality (weekend boost)
            if date.dayofweek >= 5:  # Saturday, Sunday
                sales *= 1.2
            
            # Monthly seasonality (month-end spike)
            if date.day >= 25:
                sales *= 1.15
            
            # Yearly seasonality (Q4 boost)
            if date.month in [10, 11, 12]:
                sales *= 1.3
            
            # Festival spike
            date_str = date.strftime('%Y-%m-%d')
            if date_str in festivals:
                sales *= 2.5  # 150% increase during festivals
            
            # Days before festival (pre-stocking)
            for festival_date in festivals.keys():
                festival_dt = datetime.strptime(festival_date, '%Y-%m-%d')
                days_diff = (festival_dt - date).days
                if 1 <= days_diff <= 7:
                    sales *= 1.5
            
            # Add random noise
            sales *= np.random.uniform(0.85, 1.15)
            
            # Ensure non-negative
            sales = max(0, int(sales))
            
            # Price (with some variation)
            price = np.random.uniform(50, 500)
            
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
    print(df['category'].value_counts())
    
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
