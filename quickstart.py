"""Quick start script for SKU Demand Forecasting Engine."""
import os
import sys
import subprocess

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[ERROR] Python 3.8+ required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required packages."""
    print_header("Installing Dependencies")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n[OK] All dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("\n[ERROR] Failed to install dependencies")
        return False

def generate_sample_data():
    """Generate sample sales data."""
    print_header("Generating Sample Data")
    
    try:
        subprocess.check_call([sys.executable, "scripts/generate_sample_data.py"])
        return True
    except subprocess.CalledProcessError:
        print("\n[ERROR] Failed to generate sample data")
        return False

def check_m5_data():
    """Check if M5 dataset exists."""
    print_header("Checking M5 Dataset")
    
    if os.path.exists("data/raw/sales_train_validation.csv"):
        print("[OK] M5 dataset found")
        return True
    else:
        print("[WARN] M5 dataset not found")
        print("\nOptions:")
        print("1. Download using Kaggle API: python scripts/download_data.py")
        print("2. Manual download: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data")
        print("3. Use sample data for testing (already generated)")
        return False

def train_model():
    """Train the forecasting model."""
    print_header("Training Model")
    
    response = input("Train model now? (y/n): ")
    
    if response.lower() == 'y':
        try:
            subprocess.check_call([sys.executable, "src/train.py"])
            print("\n[OK] Model trained successfully")
            return True
        except subprocess.CalledProcessError:
            print("\n[ERROR] Model training failed")
            return False
    else:
        print("[WARN] Skipping model training")
        print("  You can train later with: python src/train.py")
        return False

def launch_dashboard():
    """Launch Streamlit dashboard."""
    print_header("Launching Dashboard")
    
    print("Starting Streamlit dashboard...")
    print("Dashboard will open at: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        subprocess.check_call([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\n[OK] Dashboard stopped")
    except subprocess.CalledProcessError:
        print("\n[ERROR] Failed to launch dashboard")

def main():
    """Main quickstart flow."""
    print_header("SKU Demand Forecasting Engine - Quick Start")
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Generate sample data
    generate_sample_data()
    
    # Check M5 data
    has_m5_data = check_m5_data()
    
    # Train model (works with either M5 or sample data)
    print_header("Model Training")
    print("The model can train on:")
    if has_m5_data:
        print("  [OK] M5 dataset (42,840 SKUs -- 2-3 hours)")
    print("  [OK] Sample data (20 SKUs -- ~5 minutes)")
    print()
    train_model()
    
    # Launch dashboard
    print_header("Setup Complete!")
    print("[OK] Dependencies installed")
    print("[OK] Sample data generated")
    if has_m5_data:
        print("[OK] M5 dataset available")
    if os.path.exists("models/lightgbm_model.pkl"):
        print("[OK] Model trained")
    else:
        print("[WARN] Model not yet trained (run: python src/train.py)")
    
    print("\nNext steps:")
    print("1. Launch dashboard: streamlit run app.py")
    print("2. Upload data/sample_sales.csv in the dashboard")
    print("3. Select a SKU and generate forecast")
    
    response = input("\nLaunch dashboard now? (y/n): ")
    if response.lower() == 'y':
        launch_dashboard()

if __name__ == "__main__":
    main()
