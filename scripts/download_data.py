"""Download M5 Forecasting dataset from Kaggle."""
import os
import sys

def download_m5_dataset():
    """Download M5 dataset using Kaggle API."""
    print("Downloading M5 Forecasting dataset...")
    print("\nPrerequisites:")
    print("1. Install Kaggle CLI: pip install kaggle")
    print("2. Setup Kaggle API credentials: https://www.kaggle.com/docs/api")
    print("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
    
    try:
        import kaggle
        
        # Create data directory
        os.makedirs("data/raw", exist_ok=True)
        
        # Download M5 dataset
        kaggle.api.competition_download_files(
            "m5-forecasting-accuracy",
            path="data/raw",
            quiet=False
        )
        
        print("\n[OK] Download complete!")
        print("Extracting files...")
        
        import zipfile
        with zipfile.ZipFile("data/raw/m5-forecasting-accuracy.zip", 'r') as zip_ref:
            zip_ref.extractall("data/raw")
        
        print("[OK] Extraction complete!")
        print("\nDataset files:")
        for file in os.listdir("data/raw"):
            if file.endswith('.csv'):
                print(f"  - {file}")
                
    except ImportError:
        print("\n[ERROR] Kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        print("\nManual download:")
        print("1. Visit: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data")
        print("2. Download all files to data/raw/")
        sys.exit(1)

if __name__ == "__main__":
    download_m5_dataset()
