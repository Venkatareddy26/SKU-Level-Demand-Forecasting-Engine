"""SHAP-based explainability for demand drivers."""
import shap
import pandas as pd
import numpy as np

class DemandExplainer:
    """Extract top demand drivers using SHAP values."""
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def compute_shap_values(self, X, sample_size=1000):
        """Compute SHAP values for predictions."""
        print("Computing SHAP values...")
        
        # Sample data if too large
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model.model)
        self.shap_values = self.explainer.shap_values(X_sample[self.feature_names])
        
        print(f"[OK] SHAP values computed for {len(X_sample)} samples")
        return self.shap_values
    
    def get_top_drivers(self, X_row, top_n=3, baseline_prediction=None):
        """
        Get top N demand drivers for a specific prediction.
        
        Args:
            X_row: Single row of features (pandas Series or DataFrame)
            top_n: Number of top drivers to return
            baseline_prediction: Baseline prediction value
        
        Returns:
            List of dicts with driver info
        """
        if isinstance(X_row, pd.Series):
            X_row = X_row.to_frame().T
        
        # Get SHAP values for this row
        shap_vals = self.explainer.shap_values(X_row[self.feature_names])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        if shap_vals.ndim > 1:
            shap_vals = shap_vals[0]
        
        # Get feature values
        feature_vals = X_row[self.feature_names].values[0]
        
        # Create driver dataframe
        drivers_df = pd.DataFrame({
            'feature': self.feature_names,
            'value': feature_vals,
            'shap_value': shap_vals,
            'abs_shap': np.abs(shap_vals)
        }).sort_values('abs_shap', ascending=False)
        
        # Get top N drivers
        top_drivers = []
        for idx, row in drivers_df.head(top_n).iterrows():
            driver = {
                'feature': self._format_feature_name(row['feature']),
                'value': row['value'],
                'impact': row['shap_value'],
                'direction': 'increase' if row['shap_value'] > 0 else 'decrease',
                'explanation': self._generate_explanation(
                    row['feature'], 
                    row['value'], 
                    row['shap_value']
                )
            }
            top_drivers.append(driver)
        
        return top_drivers
    
    def _format_feature_name(self, feature):
        """Format feature name for display."""
        name_map = {
            'is_festival': 'Festival',
            'is_festival_week': 'Festival Week',
            'days_to_festival': 'Days to Festival',
            'is_weekend': 'Weekend',
            'day_of_week': 'Day of Week',
            'month': 'Month',
            'quarter': 'Quarter',
            'year': 'Year',
            'lag_7': '1-Week Lag',
            'lag_14': '2-Week Lag',
            'lag_28': '4-Week Lag',
            'lag_364': '52-Week Lag',
            'rolling_mean_7': '7-Day Average',
            'rolling_mean_14': '14-Day Average',
            'rolling_mean_28': '28-Day Average',
            'rolling_std_7': '7-Day Volatility',
            'rolling_std_14': '14-Day Volatility',
            'rolling_std_28': '28-Day Volatility',
            'price_change': 'Price Change',
            'price_change_pct': 'Price Change %',
            'price_vs_avg': 'Price vs Average',
            'price_lag_1': 'Yesterday Price',
            'price_rolling_mean_7': '7-Day Avg Price',
            'is_month_start': 'Month Start',
            'is_month_end': 'Month End',
        }
        return name_map.get(feature, feature.replace('_', ' ').title())
    
    def _generate_explanation(self, feature, value, impact):
        """Generate human-readable explanation."""
        feature_display = self._format_feature_name(feature)
        impact_abs = abs(impact)
        direction = "increasing" if impact > 0 else "decreasing"
        
        # Festival-specific explanations
        if 'festival' in feature.lower():
            if feature == 'is_festival' and value == 1:
                return f"Festival day {direction} demand by {impact_abs:.0f} units"
            elif feature == 'is_festival_week' and value == 1:
                return f"Festival week (pre-stocking) {direction} demand by {impact_abs:.0f} units"
            elif feature == 'days_to_festival':
                return f"{int(value)} days to festival {direction} demand by {impact_abs:.0f} units"
        
        # Weekend effect
        if feature == 'is_weekend' and value == 1:
            return f"Weekend {direction} demand by {impact_abs:.0f} units"
        
        # Price features
        if 'price' in feature.lower():
            if feature == 'price_change' and abs(value) > 0:
                price_dir = "increase" if value > 0 else "decrease"
                return f"Price {price_dir} (₹{abs(value):.0f}) {direction} demand by {impact_abs:.0f} units"
            elif feature == 'price_vs_avg':
                pct = (value - 1) * 100
                price_pos = "above" if pct > 0 else "below"
                return f"Price {abs(pct):.0f}% {price_pos} average {direction} demand by {impact_abs:.0f} units"
            return f"{feature_display} {direction} demand by {impact_abs:.0f} units"
        
        # Lag features
        if 'lag' in feature:
            return f"Previous sales pattern (lag={feature.split('_')[-1]}d) {direction} demand by {impact_abs:.0f} units"
        
        # Rolling averages
        if 'rolling_mean' in feature:
            window = feature.split('_')[-1]
            return f"Recent {window}-day trend {direction} demand by {impact_abs:.0f} units"
        
        if 'rolling_std' in feature:
            window = feature.split('_')[-1]
            return f"Recent {window}-day volatility {direction} demand by {impact_abs:.0f} units"
        
        # Calendar features
        if feature == 'month':
            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_name = month_names[int(value)] if 1 <= int(value) <= 12 else str(int(value))
            return f"Month ({month_name}) {direction} demand by {impact_abs:.0f} units"
        
        if feature == 'day_of_week':
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            day_name = day_names[int(value)] if 0 <= int(value) <= 6 else str(int(value))
            return f"Day of week ({day_name}) {direction} demand by {impact_abs:.0f} units"
        
        # Default explanation
        return f"{feature_display} (value: {value:.2f}) {direction} demand by {impact_abs:.0f} units"
    
    def get_feature_importance_summary(self):
        """Get overall feature importance across all predictions."""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        return importance_df
