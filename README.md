# 📊 SKU-Level Demand Forecasting Engine

> **AI-powered demand forecasting for Indian retail with festival intelligence**

Transform historical sales data into accurate 4-8 week SKU-level forecasts with explainable AI drivers. Built specifically for FMCG distributors, kirana aggregators, and D2C brands operating in India.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B.svg)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-green.svg)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.44.0-orange.svg)](https://shap.readthedocs.io/)

---

## 🎯 Why This Matters

**The Problem**: Traditional forecasting models miss Indian festival spikes (Diwali, Holi, Navratri, Pongal), leading to stockouts and excess inventory costs.

**The Solution**: This engine has **97 Indian festival dates across 12 festivals** baked in (2019-2026), with LightGBM + SHAP explainability for transparent, actionable forecasts.

**Target Metrics**: WRMSSE < 0.60 on M5 benchmark, MAPE < 25%.

## ✨ Key Features

### 🎯 **Forecasting Engine**
- LightGBM global model — one model across all SKUs using lag/rolling features
- Recursive multi-step forecasting (each prediction feeds back as input for next step)
- Temporal train/test splitting (no data leakage)
- Trained on M5 benchmark (42,840 SKUs, 5.4 years of Walmart data)

### 🎊 **Indian Festival Intelligence**
- **12 festivals × 8 years = 97 festival dates** (2019-2026)
- Diwali, Holi, Dussehra, Pongal, Ugadi, Navratri, Ganesh Chaturthi, Onam, Eid al-Fitr, Eid al-Adha, Raksha Bandhan, Christmas
- Pre-festival stocking patterns (7-day lead time)
- **Competitive Moat**: Western models miss these critical demand signals

### 🔍 **Explainable AI**
- SHAP-based demand drivers per SKU
- Human-readable explanations: *"Festival week (pre-stocking) increasing demand by 340 units"*
- Price-aware drivers: *"Price 15% below average increasing demand by 120 units"*
- Top 5 drivers with impact quantification

### 📈 **Interactive Dashboard**
- Upload CSV → Select SKU → Generate forecast (3 clicks)
- Recursive 4-8 week forecast with residual-based widening confidence bands
- Historical analytics with trend visualization
- Reorder point calculation with safety stock (80% service level)
- Download forecast CSV for ERP integration

### 🔧 **Feature Engineering**
- Lag features: 7, 14, 28, 364 days
- Rolling statistics: mean & std with `.shift(1)` (no look-ahead bias)
- Calendar: day of week, month, quarter, year, weekend, month start/end
- Festival: is_festival, is_festival_week, days_to_festival
- Price: lag, change, change %, rolling average, price vs average

## 🚀 Quick Start (5 Minutes)

### **Option 1: Automated Setup** (Recommended)
```bash
python quickstart.py
```
This will:
1. Check Python version (3.8+ required)
2. Install all dependencies
3. Generate sample data (20 SKUs, 2 years)
4. Optionally train model
5. Launch dashboard

### **Option 2: Manual Setup**
```bash
# 1. Install dependencies (2-3 minutes)
pip install -r requirements.txt

# 2. Generate sample data (30 seconds)
python scripts/generate_sample_data.py

# 3. Train model (5 minutes for sample data)
python src/train.py

# 4. Launch dashboard (opens at http://localhost:8501)
streamlit run app.py
```

### **Test the Dashboard**
1. Upload `data/sample_sales.csv`
2. Select "SKU_001" from dropdown
3. Click "Generate Forecast"
4. View 8-week prediction with confidence bands
5. Switch to "Drivers" tab → Click "Analyze Drivers"
6. See top 5 demand drivers with explanations

## 📁 Project Structure & Code Statistics

```
📦 SKU-Level Demand Forecasting Engine (1,888 lines of Python)
│
├── 📊 app.py (502 lines)                    # Streamlit dashboard with recursive forecasting
├── 🚀 quickstart.py (133 lines)             # Automated setup & dependency installer
│
├── 📂 src/ (819 lines)                      # Core forecasting engine
│   ├── features.py (132 lines)              # 60+ feature engineering (vectorized)
│   ├── models.py (192 lines)                # LightGBM + NeuralProphet dual models
│   ├── explainer.py (184 lines)             # SHAP explainability with retail logic
│   ├── metrics.py (133 lines)               # WRMSSE, MAPE, MAE, RMSE
│   └── train.py (178 lines)                 # End-to-end training pipeline
│
├── 📂 data/                                 # Festival calendar & sample data
│   ├── festival_calendar.csv (97 dates)     # 12 Indian festivals (2019-2026)
│   └── sample_sales.csv (generated)         # 20 SKUs × 2 years
│
├── 📂 scripts/ (183 lines)                  # Data utilities
│   ├── download_data.py (48 lines)          # M5 Kaggle downloader
│   └── generate_sample_data.py (135 lines)  # Synthetic data with festival spikes
│
└── 📂 tests/ (231 lines)                    # Unit tests
    ├── test_features.py (138 lines)         # Feature engineering tests
    └── test_metrics.py (93 lines)           # Metrics calculation tests
```

### Code Quality Metrics
- **Total Lines**: 1,888 Python LOC
- **Test Coverage**: 231 test lines (12% test-to-code ratio)
- **Largest Module**: app.py (502 lines) - Full-featured dashboard
- **Core Engine**: 819 lines (features + models + explainer + metrics + train)
- **Documentation**: Comprehensive docstrings in all modules
- **Architecture**: Global model (single LightGBM for all SKUs)
- **Performance**: Vectorized NumPy operations, no nested Python loops

## 📊 Performance Benchmarks

### Target Metrics
| Metric | Target | How We'll Measure |
|--------|--------|-------------------|
| **WRMSSE** | < 0.60 | Temporal holdout on M5 test period |
| **MAPE** | < 25% | On validation set (last 60 days) |

> **Note**: Run `python src/train.py` to train and see actual metrics. Naive seasonal baseline WRMSSE is ~0.90.

### Model Architecture
| Model | Role | Speed |
|-------|------|-------|
| **LightGBM (Global)** | Primary forecaster — one model across all SKUs | < 1 sec inference |
| **NeuralProphet (Per Category)** | Alternative — Prophet + AR-Net + regressors | 2-3 sec inference |

## 🛠 Tech Stack & Implementation Details

### Core ML Stack
| Component | Version | Role | Implementation Highlights |
|-----------|---------|------|---------------------------|
| **LightGBM** | 4.1.0 | Primary forecaster | Global model (single model for all SKUs), early stopping (50 rounds), feature importance via gain, pickle persistence |
| **NeuralProphet** | 0.7.0 | Alternative model | Per-category models, Prophet decomposition + AR-Net, festival regressors, lazy import for fast dashboard loading |
| **SHAP** | 0.44.0 | Explainability | TreeExplainer for LightGBM, per-prediction drivers, retail-aware explanations (price, festivals, lags) |

### Dashboard & Visualization
| Component | Version | Features |
|-----------|---------|----------|
| **Streamlit** | 1.29.0 | Interactive dashboard (502 lines), CSV validation, recursive forecasting, confidence intervals |
| **Plotly** | 5.18.0 | Interactive charts (actual vs fitted, forecast with CI, SHAP bars) |

### Data Processing
- **Pandas 2.0.3**: DataFrame operations, temporal sorting
- **NumPy 1.24.3**: Vectorized festival computation (O(n × k) using broadcasting)
- **Scikit-learn 1.3.0**: Train/test split, preprocessing

### Architecture Highlights
1. **Global Model**: Single LightGBM for all SKUs (faster than per-SKU models)
2. **Recursive Forecasting**: Multi-step ahead with lag feature updates (each prediction feeds back)
3. **Vectorized Features**: NumPy broadcasting for O(n × k) festival computation (no nested loops)
4. **Look-Ahead Bias Prevention**: `.shift(1)` on rolling windows (window computed on data BEFORE current row)
5. **Confidence Intervals**: Residual-based with horizon widening (uncertainty grows as √t)
6. **Price Intelligence**: Automatic discount detection (price vs 7-day rolling average)
7. **Temporal Splitting**: Train/test split respects time order (no data leakage)

## 📖 How to Use

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Generate sample data**: `python scripts/generate_sample_data.py`
3. **Train model**: `python src/train.py`
4. **Launch dashboard**: `streamlit run app.py`
5. **Upload CSV**: Use `data/sample_sales.csv` or your own data
6. **Select SKU**: Choose from dropdown
7. **Generate forecast**: Click button to see 8-week prediction
8. **Analyze drivers**: View top 5 demand drivers with SHAP

### CSV Format Required
```csv
id,date,sales,category,price
SKU_001,2024-01-01,120,Food,99.50
SKU_001,2024-01-02,135,Food,99.50
```

**Required columns**: `id`, `date`, `sales`  
**Optional columns**: `category`, `price`

## 🧪 Testing

```bash
# Run all tests
python run_tests.py

# Run specific test
python -m unittest tests.test_features
python -m unittest tests.test_metrics
```

## 💼 Commercial Information

### Pricing Model

| Tier | SKU Count | Monthly Price | Features |
|------|-----------|---------------|----------|
| **Starter** | 1-50 | ₹30,000 | Dashboard, CSV upload, basic support |
| **Growth** | 51-200 | ₹60,000 | + API access, email alerts, priority support |
| **Enterprise** | 201-500 | ₹1,00,000 | + ERP integration, custom models, dedicated support |

**API Integration**: ₹5-20L/year (one-time setup + annual maintenance)

### Target Market

| Segment | Count in India | Potential |
|---------|----------------|-----------|
| **FMCG Distributors** | 15,000+ | Primary target |
| **Kirana Aggregators** | 50+ (Jumbotail, StoreKing) | High value |
| **D2C Brands** | 5,000+ | Growing segment |
| **Cold-Chain Logistics** | 200+ | Perishable demand |

**Conservative TAM**: ₹75 crore ARR at 0.1% overall penetration

### Competitive Moat

1. **Indian Festival Calendar**: 12 festivals baked in (competitors use Western models)
2. **Explainable AI**: SHAP drivers build trust with non-technical buyers
3. **Price Intelligence**: Automatic discount/promotion impact detection
4. **Fast Time-to-Value**: 5-minute setup, immediate forecasts

## 🗺 Product Roadmap

### Q2 2026 — API & Mobile
- [ ] FastAPI REST API with authentication
- [ ] Weather data integration (cold drink seasonality)
- [ ] Automated email alerts for reorder points

### Q3 2026 — Integrations
- [ ] ERP integrations (Tally, Zoho, SAP)
- [ ] Promotion modeling (discount impact on demand)
- [ ] Multi-store forecasting

### Q4 2026 — Scale
- [ ] Price elasticity modeling
- [ ] Inventory optimization module (safety stock, EOQ)
- [ ] White-label solution for aggregators

## 📄 License

MIT License - Free for commercial use. See [LICENSE](LICENSE) for details.

## 📞 Contact

- **Email**: thor47222@gmail.com
- **GitHub**: [@Venkatareddy26](https://github.com/Venkatareddy26)
- **Project**: [SKU-Level-Demand-Forecasting-Engine](https://github.com/Venkatareddy26/SKU-Level-Demand-Forecasting-Engine)

## 🙏 Acknowledgments

- **M5 Competition** — Walmart benchmark dataset
- **Meta** — NeuralProphet library
- **Microsoft** — LightGBM library
- **Streamlit** — Dashboard framework

---

**Built with ❤️ for Indian retail | Powered by LightGBM, NeuralProphet, and SHAP**
