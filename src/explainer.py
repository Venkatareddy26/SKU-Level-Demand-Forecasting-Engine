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
        
        print(f"✓ SHAP values computed for {len(X_sample)} samples")
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
        shap_vals = self.explainer.shap_values(X_row[self.feature_names])[0]
        
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
            'lag_7': '1-Week Lag',
            'lag_28': '4-Week Lag',
            'rolling_mean_7': '7-Day Average',
            'rolling_mean_28': '28-Day Average'
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
                return f"Festival week {direction} demand by {impact_abs:.0f} units"
            elif feature == 'days_to_festival':
                return f"{int(value)} days to festival {direction} demand by {impact_abs:.0f} units"
        
        # Weekend effect
        if feature == 'is_weekend' and value == 1:
            return f"Weekend {direction} demand by {impact_abs:.0f} units"
        
        # Lag features
        if 'lag' in feature:
            return f"Previous sales pattern {direction} demand by {impact_abs:.0f} units"
        
        # Rolling averages
        if 'rolling' in feature:
            return f"Recent trend {direction} demand by {impact_abs:.0f} units"
        
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
