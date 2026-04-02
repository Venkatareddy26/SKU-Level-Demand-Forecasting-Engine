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
        st.error(f"Error loading model: {e}")
        return None

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Sales CSV",
    type=['csv'],
    help="Upload historical sales data with columns: id, date, sales"
)

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
    st.session_state.data = df
    
    st.sidebar.success(f"✓ Data loaded: {len(df)} rows")
    
    # Display data info
    with st.expander("📋 Data Preview"):
        st.dataframe(df.head(10))
        st.write(f"Shape: {df.shape}")
        st.write(f"Date range: {df['date'].min()} to {df['date'].max()}")

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
                            st.error("Model not found. Please train the model first.")
                        else:
                            # Feature engineering
                            fe = FeatureEngineer()
                            sku_features = fe.build_features(sku_data, target_col='sales', date_col='date')
                            
                            # Get feature columns
                            feature_cols = [col for col in sku_features.columns if col not in 
                                          ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 
                                           'state_id', 'date', 'sales', 'd', 'wm_yr_wk', 'festival_name']]
                            
                            # Make predictions on historical data
                            sku_features_clean = sku_features.dropna()
                            if len(sku_features_clean) > 0:
                                predictions = model.predict(sku_features_clean[feature_cols])
                                
                                # Create forecast dataframe
                                forecast_df = sku_features_clean[['date', 'sales']].copy()
                                forecast_df['forecast'] = predictions
                                
                                # Generate future dates
                                last_date = forecast_df['date'].max()
                                future_dates = pd.date_range(
                                    start=last_date + timedelta(days=1),
                                    periods=forecast_weeks * 7,
                                    freq='D'
                                )
                                
                                # Simple future forecast (using last known features)
                                # In production, this would use proper recursive forecasting
                                last_features = sku_features_clean[feature_cols].iloc[-1:].copy()
                                future_preds = []
                                
                                for i in range(len(future_dates)):
                                    pred = model.predict(last_features)[0]
                                    future_preds.append(max(0, pred))  # Ensure non-negative
                                
                                future_df = pd.DataFrame({
                                    'date': future_dates,
                                    'forecast': future_preds
                                })
                                
                                # Plot
                                fig = go.Figure()
                                
                                # Historical actual
                                fig.add_trace(go.Scatter(
                                    x=forecast_df['date'].tail(90),
                                    y=forecast_df['sales'].tail(90),
                                    mode='lines',
                                    name='Actual',
                                    line=dict(color='blue', width=2)
                                ))
                                
                                # Historical forecast
                                fig.add_trace(go.Scatter(
                                    x=forecast_df['date'].tail(90),
                                    y=forecast_df['forecast'].tail(90),
                                    mode='lines',
                                    name='Fitted',
                                    line=dict(color='green', width=2, dash='dot')
                                ))
                                
                                # Future forecast
                                fig.add_trace(go.Scatter(
                                    x=future_df['date'],
                                    y=future_df['forecast'],
                                    mode='lines',
                                    name='Forecast',
                                    line=dict(color='red', width=2)
                                ))
                                
                                # Add confidence band (simple ±20%)
                                fig.add_trace(go.Scatter(
                                    x=future_df['date'],
                                    y=future_df['forecast'] * 1.2,
                                    mode='lines',
                                    name='Upper Bound',
                                    line=dict(width=0),
                                    showlegend=False
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=future_df['date'],
                                    y=future_df['forecast'] * 0.8,
                                    mode='lines',
                                    name='Lower Bound',
                                    fill='tonexty',
                                    fillcolor='rgba(255,0,0,0.2)',
                                    line=dict(width=0),
                                    showlegend=False
                                ))
                                
                                fig.update_layout(
                                    title=f"Demand Forecast - {forecast_weeks} Weeks",
                                    xaxis_title="Date",
                                    yaxis_title="Sales Units",
                                    hovermode='x unified',
                                    height=500
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Forecast summary
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    avg_forecast = future_df['forecast'].mean()
                                    st.metric("Avg Weekly Demand", f"{avg_forecast * 7:.0f} units")
                                
                                with col2:
                                    total_forecast = future_df['forecast'].sum()
                                    st.metric(f"Total {forecast_weeks}-Week Demand", f"{total_forecast:.0f} units")
                                
                                with col3:
                                    # Reorder point (simple: 2 weeks of avg demand)
                                    reorder_point = avg_forecast * 14
                                    st.metric("Suggested Reorder Point", f"{reorder_point:.0f} units")
                                
                                # Download forecast
                                csv = future_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Forecast CSV",
                                    csv,
                                    f"forecast_{selected_sku}.csv",
                                    "text/csv"
                                )
                            else:
                                st.warning("Not enough data for forecasting after feature engineering.")
                    
                    except Exception as e:
                        st.error(f"Error generating forecast: {e}")
                        st.exception(e)
        
        with tab2:
            st.subheader("🔍 Demand Drivers")
            
            if st.button("Analyze Drivers"):
                with st.spinner("Computing demand drivers..."):
                    try:
                        model = load_model()
                        
                        if model:
                            # Feature engineering
                            fe = FeatureEngineer()
                            sku_features = fe.build_features(sku_data, target_col='sales', date_col='date')
                            sku_features_clean = sku_features.dropna()
                            
                            feature_cols = [col for col in sku_features.columns if col not in 
                                          ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 
                                           'state_id', 'date', 'sales', 'd', 'wm_yr_wk', 'festival_name']]
                            
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
                                title="Top 15 Features by SHAP Importance"
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error analyzing drivers: {e}")
        
        with tab3:
            st.subheader("📊 Historical Analytics")
            
            # Sales trend
            fig = px.line(
                sku_data,
                x='date',
                y='sales',
                title=f"Sales Trend - {selected_sku}"
            )
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
    - 🎯 LightGBM & NeuralProphet models
    - 🎊 Indian festival calendar integration
    - 🔍 SHAP-based explainable drivers
    - 📊 Interactive visualizations
    
    ### Required CSV Format
    Your CSV should contain at least these columns:
    - `id`: SKU identifier
    - `date`: Date (YYYY-MM-DD format)
    - `sales`: Sales quantity
    
    ### Model Performance
    - Target WRMSSE: < 0.60
    - Trained on M5 Forecasting benchmark
    - Festival-aware predictions
    """)
    
    # Show sample data format
    sample_data = pd.DataFrame({
        'id': ['SKU_001', 'SKU_001', 'SKU_001'],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'sales': [120, 135, 128]
    })
    
    st.markdown("### Sample Data Format")
    st.dataframe(sample_data)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**SKU Demand Forecasting Engine**

Industry-grade MVP for FMCG distributors, 
kirana aggregators, and D2C brands.

Target: ₹75 crore ARR at 0.1% penetration
""")
