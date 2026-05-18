"""Streamlit dashboard for SKU-level demand forecasting."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append('src')

from features import FeatureEngineer
from models import LightGBMForecaster
from explainer import DemandExplainer

# Page config
st.set_page_config(
    page_title="SKU Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 SKU-Level Demand Forecasting Engine")
st.markdown("4-8 week item-level demand forecast with explainable drivers per SKU")

# Sidebar
st.sidebar.header("Configuration")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None

# Load model
@st.cache_resource
def load_model():
    """Load trained model."""
    try:
        model = LightGBMForecaster()
        model.load("models/lightgbm_model.pkl")
        return model
    except Exception as e:
        return None


def validate_csv(df):
    """Validate uploaded CSV has required columns and proper data types.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_cols = ['id', 'date', 'sales']
    
    # Check required columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}. Expected: id, date, sales"
    
    # Validate date parsing
    try:
        pd.to_datetime(df['date'])
    except Exception:
        return False, "Cannot parse 'date' column. Use YYYY-MM-DD format."
    
    # Validate sales is numeric
    if not pd.api.types.is_numeric_dtype(df['sales']):
        try:
            df['sales'] = pd.to_numeric(df['sales'])
        except Exception:
            return False, "'sales' column must be numeric."
    
    # Check for minimum data
    if len(df) < 30:
        return False, f"Need at least 30 rows of data, got {len(df)}."
    
    # Check per-SKU minimum
    sku_counts = df['id'].value_counts()
    small_skus = sku_counts[sku_counts < 30]
    if len(small_skus) > 0:
        return True, f"Warning: {len(small_skus)} SKUs have <30 data points. Forecasts may be unreliable."
    
    return True, ""


def recursive_forecast(model, fe, sku_data, feature_cols, forecast_days):
    """Generate multi-step recursive forecast.
    
    Each step predicts one day ahead, then feeds that prediction back
    as the lag feature for the next step. Calendar and festival features
    are computed from the future date.
    
    Args:
        model: Trained LightGBMForecaster
        fe: FeatureEngineer instance
        sku_data: Historical data for one SKU
        feature_cols: List of feature column names
        forecast_days: Number of days to forecast
    
    Returns:
        DataFrame with date and forecast columns
    """
    # Build features on historical data
    sku_features = fe.build_features(sku_data.copy(), target_col='sales', date_col='date')
    sku_features = sku_features.dropna()
    
    if len(sku_features) == 0:
        return pd.DataFrame(), sku_features
    
    # Get historical predictions for display
    hist_predictions = model.predict(sku_features[feature_cols])
    hist_df = sku_features[['date', 'sales']].copy()
    hist_df['forecast'] = hist_predictions
    
    # Prepare for recursive forecasting
    last_date = sku_data['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
    
    # Build a rolling history that we'll extend with predictions
    history = sku_data[['id', 'date', 'sales']].copy()
    if 'price' in sku_data.columns:
        history['price'] = sku_data['price']
    if 'category' in sku_data.columns:
        history['category'] = sku_data['category']
    
    future_preds = []
    sku_id = sku_data['id'].iloc[0]
    last_price = sku_data['price'].iloc[-1] if 'price' in sku_data.columns else None
    category = sku_data['category'].iloc[-1] if 'category' in sku_data.columns else None
    
    for future_date in future_dates:
        # Create a new row with the future date and placeholder sales
        new_row = {'id': sku_id, 'date': future_date, 'sales': 0}
        if last_price is not None:
            new_row['price'] = last_price
        if category is not None:
            new_row['category'] = category
        
        # Append to history
        new_row_df = pd.DataFrame([new_row])
        history = pd.concat([history, new_row_df], ignore_index=True)
        
        # Re-build features on the extended history
        temp_features = fe.build_features(history.copy(), target_col='sales', date_col='date')
        
        # Get features for the last row (the future date)
        last_row_features = temp_features.iloc[-1:]
        
        # Check if all required features are present
        missing_feats = [f for f in feature_cols if f not in last_row_features.columns]
        if missing_feats:
            break
        
        # Predict
        try:
            pred = model.predict(last_row_features[feature_cols])[0]
            pred = max(0, pred)  # Ensure non-negative
        except Exception:
            pred = future_preds[-1] if future_preds else hist_df['sales'].tail(7).mean()
        
        future_preds.append(pred)
        
        # Update the sales value in history so next iteration's lags use it
        history.iloc[-1, history.columns.get_loc('sales')] = pred
    
    future_df = pd.DataFrame({
        'date': future_dates[:len(future_preds)],
        'forecast': future_preds
    })
    
    return future_df, hist_df


def compute_confidence_bands(future_df, hist_df, confidence=0.80):
    """Compute prediction intervals based on historical residual spread.
    
    Uses the standard deviation of historical residuals to build
    intervals that widen over the forecast horizon.
    
    Args:
        future_df: DataFrame with forecast column
        hist_df: DataFrame with sales and forecast columns
        confidence: Confidence level (default 80%)
    
    Returns:
        future_df with upper and lower bound columns added
    """
    if len(hist_df) > 0 and 'forecast' in hist_df.columns:
        residuals = hist_df['sales'] - hist_df['forecast']
        residual_std = residuals.std()
    else:
        residual_std = future_df['forecast'].mean() * 0.2  # Fallback
    
    # Z-score for confidence level
    from scipy import stats
    try:
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
    except ImportError:
        z = 1.28  # Approx for 80% confidence
    
    # Widen intervals over the forecast horizon
    steps = np.arange(1, len(future_df) + 1)
    widening = np.sqrt(steps)  # Uncertainty grows with sqrt of horizon
    
    future_df = future_df.copy()
    future_df['upper'] = future_df['forecast'] + z * residual_std * widening
    future_df['lower'] = (future_df['forecast'] - z * residual_std * widening).clip(lower=0)
    
    return future_df


# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Sales CSV",
    type=['csv'],
    help="Upload historical sales data with columns: id, date, sales"
)

# Load sample data button
if os.path.exists("data/sample_sales.csv"):
    if st.sidebar.button("Load Sample Data", help="Load built-in sample data (20 SKUs, 2 years)"):
        st.session_state.data = pd.read_csv("data/sample_sales.csv")
        st.session_state.data['date'] = pd.to_datetime(st.session_state.data['date'])
        st.sidebar.success(f"[OK] Sample data loaded: {len(st.session_state.data)} rows")

st.sidebar.markdown("---")

forecast_weeks = st.sidebar.slider(
    "Forecast Horizon (weeks)",
    min_value=4,
    max_value=8,
    value=8,
    step=1
)

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Validate
    is_valid, message = validate_csv(df)
    
    if not is_valid:
        st.sidebar.error(f"[ERROR] {message}")
        st.stop()
    else:
        st.session_state.data = df
        st.sidebar.success(f"[OK] Data loaded: {len(df)} rows")
        if message:
            st.sidebar.warning(message)
    
    # Display data info
    with st.expander("Data Preview"):
        st.dataframe(df.head(10))
        st.write(f"Shape: {df.shape}")
        df['date'] = pd.to_datetime(df['date'])
        st.write(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        st.write(f"SKUs: {df['id'].nunique()}")

# Main content
if st.session_state.data is not None:
    df = st.session_state.data
    
    # SKU selector
    if 'id' in df.columns:
        sku_list = sorted(df['id'].unique())
        selected_sku = st.selectbox("Select SKU", sku_list)
        
        # Filter data for selected SKU
        sku_data = df[df['id'] == selected_sku].copy()
        sku_data['date'] = pd.to_datetime(sku_data['date'])
        sku_data = sku_data.sort_values('date')
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🔍 Drivers", "📊 Analytics"])
        
        with tab1:
            st.subheader(f"Demand Forecast: {selected_sku}")
            
            # Generate forecast button
            if st.button("Generate Forecast", type="primary"):
                with st.spinner("Generating forecast..."):
                    try:
                        # Load model
                        model = load_model()
                        
                        if model is None:
                            st.error("Model not found. Please train the model first: `python src/train.py`")
                        else:
                            # Feature engineering
                            fe = FeatureEngineer()
                            feature_cols = model.feature_cols
                            
                            # Recursive forecasting
                            forecast_days = forecast_weeks * 7
                            future_df, hist_df = recursive_forecast(
                                model, fe, sku_data, feature_cols, forecast_days
                            )
                            
                            if len(future_df) == 0:
                                st.warning("Not enough data for forecasting after feature engineering.")
                            else:
                                # Compute proper confidence bands
                                future_df = compute_confidence_bands(future_df, hist_df)
                                
                                # Plot
                                fig = go.Figure()
                                
                                # Historical actual (last 90 days)
                                fig.add_trace(go.Scatter(
                                    x=hist_df['date'].tail(90),
                                    y=hist_df['sales'].tail(90),
                                    mode='lines',
                                    name='Actual',
                                    line=dict(color='#3b82f6', width=2)
                                ))
                                
                                # Historical forecast (last 90 days)
                                fig.add_trace(go.Scatter(
                                    x=hist_df['date'].tail(90),
                                    y=hist_df['forecast'].tail(90),
                                    mode='lines',
                                    name='Fitted',
                                    line=dict(color='#22c55e', width=2, dash='dot')
                                ))
                                
                                # Future forecast
                                fig.add_trace(go.Scatter(
                                    x=future_df['date'],
                                    y=future_df['forecast'],
                                    mode='lines',
                                    name='Forecast',
                                    line=dict(color='#ef4444', width=2.5)
                                ))
                                
                                # Confidence band (upper)
                                fig.add_trace(go.Scatter(
                                    x=future_df['date'],
                                    y=future_df['upper'],
                                    mode='lines',
                                    name='80% CI Upper',
                                    line=dict(width=0),
                                    showlegend=False
                                ))
                                
                                # Confidence band (lower + fill)
                                fig.add_trace(go.Scatter(
                                    x=future_df['date'],
                                    y=future_df['lower'],
                                    mode='lines',
                                    name='80% CI',
                                    fill='tonexty',
                                    fillcolor='rgba(239,68,68,0.15)',
                                    line=dict(width=0),
                                ))
                                
                                fig.update_layout(
                                    title=f"Demand Forecast — {forecast_weeks} Weeks",
                                    xaxis_title="Date",
                                    yaxis_title="Sales Units",
                                    hovermode='x unified',
                                    height=500,
                                    legend=dict(orientation='h', yanchor='bottom', y=1.02)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Forecast summary
                                col1, col2, col3, col4 = st.columns(4)
                                
                                avg_daily = future_df['forecast'].mean()
                                with col1:
                                    st.metric("Avg Daily Demand", f"{avg_daily:.0f} units")
                                
                                with col2:
                                    total_forecast = future_df['forecast'].sum()
                                    st.metric(f"Total {forecast_weeks}-Week", f"{total_forecast:.0f} units")
                                
                                with col3:
                                    # Reorder point: avg daily × lead time (14 days) + safety stock
                                    safety_stock = future_df['forecast'].std() * 1.28  # 80% service level
                                    reorder_point = avg_daily * 14 + safety_stock
                                    st.metric("Reorder Point", f"{reorder_point:.0f} units")
                                
                                with col4:
                                    # Compare to historical average
                                    hist_avg = sku_data['sales'].tail(forecast_days).mean()
                                    if hist_avg > 0:
                                        change_pct = ((avg_daily - hist_avg) / hist_avg) * 100
                                        st.metric("vs Historical", f"{change_pct:+.1f}%")
                                    else:
                                        st.metric("vs Historical", "N/A")
                                
                                # Download forecast
                                csv = future_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Forecast CSV",
                                    csv,
                                    f"forecast_{selected_sku}.csv",
                                    "text/csv"
                                )
                    
                    except Exception as e:
                        st.error(f"Error generating forecast: {e}")
                        st.exception(e)
        
        with tab2:
            st.subheader("🔍 Demand Drivers")
            
            if st.button("Analyze Drivers"):
                with st.spinner("Computing demand drivers..."):
                    try:
                        model = load_model()
                        
                        if model is None:
                            st.error("Model not found. Train with: `python src/train.py`")
                        else:
                            # Feature engineering
                            fe = FeatureEngineer()
                            sku_features = fe.build_features(sku_data, target_col='sales', date_col='date')
                            sku_features_clean = sku_features.dropna()
                            
                            feature_cols = model.feature_cols
                            
                            # Create explainer
                            explainer = DemandExplainer(model, feature_cols)
                            explainer.compute_shap_values(sku_features_clean[feature_cols].tail(100))
                            
                            # Get drivers for latest prediction
                            latest_row = sku_features_clean[feature_cols].iloc[-1]
                            drivers = explainer.get_top_drivers(latest_row, top_n=5)
                            
                            st.markdown("### Top 5 Demand Drivers (Latest Period)")
                            
                            for i, driver in enumerate(drivers, 1):
                                impact_color = "🟢" if driver['direction'] == 'increase' else "🔴"
                                st.markdown(f"{i}. {impact_color} **{driver['explanation']}**")
                            
                            # Feature importance
                            st.markdown("### Overall Feature Importance")
                            importance_df = explainer.get_feature_importance_summary()
                            
                            fig = px.bar(
                                importance_df.head(15),
                                x='importance',
                                y='feature',
                                orientation='h',
                                title="Top 15 Features by SHAP Importance",
                                color='importance',
                                color_continuous_scale='RdYlGn'
                            )
                            fig.update_layout(height=500, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error analyzing drivers: {e}")
                        st.exception(e)
        
        with tab3:
            st.subheader("📊 Historical Analytics")
            
            # Sales trend
            fig = px.line(
                sku_data,
                x='date',
                y='sales',
                title=f"Sales Trend — {selected_sku}"
            )
            fig.update_traces(line_color='#3b82f6')
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Daily Sales", f"{sku_data['sales'].mean():.1f}")
            
            with col2:
                st.metric("Max Daily Sales", f"{sku_data['sales'].max():.0f}")
            
            with col3:
                st.metric("Std Deviation", f"{sku_data['sales'].std():.1f}")
            
            with col4:
                cv = (sku_data['sales'].std() / sku_data['sales'].mean()) * 100
                st.metric("Coefficient of Variation", f"{cv:.1f}%")

else:
    # Welcome screen
    st.info("👈 Upload a sales CSV file to get started")
    
    st.markdown("""
    ### Features
    - 📈 4-8 week SKU-level demand forecasting
    - 🎯 LightGBM global model with recursive multi-step prediction
    - 🎊 Indian festival calendar integration (97 festival dates, 12 festivals)
    - 🔍 SHAP-based explainable demand drivers
    - 📊 Interactive visualizations with confidence intervals
    
    ### Required CSV Format
    Your CSV should contain at least these columns:
    - `id`: SKU identifier
    - `date`: Date (YYYY-MM-DD format)
    - `sales`: Sales quantity
    
    **Optional columns**: `category`, `price`
    """)
    
    # Show sample data format
    sample_data = pd.DataFrame({
        'id': ['SKU_001', 'SKU_001', 'SKU_001'],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'sales': [120, 135, 128],
        'category': ['Food', 'Food', 'Food'],
        'price': [99.50, 99.50, 94.75]
    })
    
    st.markdown("### Sample Data Format")
    st.dataframe(sample_data)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**SKU Demand Forecasting Engine**

AI-powered demand forecasting for Indian retail 
with festival intelligence and explainable drivers.
""")
