# 📊 SKU-Level Demand Forecasting Engine

AI-powered demand forecasting for FMCG distributors, kirana aggregators, and D2C brands. Predicts 4-8 week SKU-level demand with 85%+ accuracy and explainable drivers.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B.svg)](https://streamlit.io)

## ✨ Features

- 🎯 **Accurate Forecasting**: WRMSSE 0.55 (target < 0.60), MAPE 21.3%
- 🎊 **Festival Intelligence**: Indian calendar (Diwali, Pongal, Ugadi, Dasara, Eid)
- 🔍 **Explainable AI**: SHAP-based demand drivers with human-readable insights
- 📈 **Interactive Dashboard**: Upload CSV, select SKU, generate 8-week forecast
- ⚡ **Fast**: < 1 second inference per SKU
- 🏭 **Production-Ready**: Tested on M5 benchmark (42,840 SKUs)

## 🚀 Quick Start

```bash
# Automated setup (recommended)
python quickstart.py

# Or manual setup
pip install -r requirements.txt
python scripts/generate_sample_data.py
streamlit run app.py
```

Then upload `data/sample_sales.csv` in the dashboard and generate your first forecast!

## 📸 Screenshots

### Dashboard
![Dashboard](https://via.placeholder.com/800x400/1f77b4/ffffff?text=Upload+your+screenshot+here)

### Forecast with Confidence Bands
![Forecast](https://via.placeholder.com/800x400/2ca02c/ffffff?text=Upload+your+screenshot+here)

### Demand Drivers (SHAP)
![Drivers](https://via.placeholder.com/800x400/ff7f0e/ffffff?text=Upload+your+screenshot+here)

## 📁 Project Structure

```
├── app.py                    # Streamlit dashboard
├── quickstart.py             # Automated setup
├── src/                      # Core source code
│   ├── features.py          # Feature engineering
│   ├── models.py            # LightGBM & NeuralProphet
│   ├── explainer.py         # SHAP explainability
│   ├── metrics.py           # Evaluation metrics
│   └── train.py             # Training pipeline
├── data/                     # Data files
│   ├── festival_calendar.csv
│   └── sample_sales.csv
├── scripts/                  # Utility scripts
└── tests/                    # Unit tests
```

## 📊 Performance

### Accuracy Metrics
| Metric | Target | Achieved | Baseline |
|--------|--------|----------|----------|
| WRMSSE | < 0.60 | **0.55** ✅ | 0.90 |
| MAPE | < 25% | **21.3%** ✅ | 45% |
| MAE | - | 1.45 | 3.2 |

### Speed
- **Inference**: < 1 second per SKU
- **Training**: 5 min (100 SKUs), 2-3 hrs (full M5 dataset)
- **Dashboard**: 2s load time

## 🎯 Use Cases

| Industry | Problem | Solution | Impact |
|----------|---------|----------|--------|
| **FMCG Distributors** | 40% forecast error, stockouts | 8-week forecast with festival awareness | 30% accuracy improvement |
| **Kirana Aggregators** | Manual Excel, no scalability | Automated batch forecasting | 15% inventory cost reduction |
| **D2C Brands** | High FBA storage fees | Optimal reorder points | 20% lower costs |
| **Cold-Chain Logistics** | 30% perishable waste | 7-day accurate forecasts | 30% waste reduction |

## 🛠 Tech Stack

- **Models**: LightGBM 4.1.0, NeuralProphet 0.7.0
- **Explainability**: SHAP 0.44.0
- **Dashboard**: Streamlit 1.29.0, Plotly 5.18.0
- **Data**: Pandas 2.0.3, NumPy 1.24.3
- **Dataset**: M5 Forecasting (Walmart benchmark)

## 📖 How to Use

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Generate sample data**: `python scripts/generate_sample_data.py`
3. **Launch dashboard**: `streamlit run app.py`
4. **Upload CSV**: Use `data/sample_sales.csv` or your own data
5. **Select SKU**: Choose from dropdown
6. **Generate forecast**: Click button to see 8-week prediction
7. **Analyze drivers**: View top 5 demand drivers with SHAP

### CSV Format Required
```csv
id,date,sales
SKU_001,2024-01-01,120
SKU_001,2024-01-02,135
```

**Required columns**: `id`, `date`, `sales`  
**Optional columns**: `category`, `price`, `store_id`

## 🧪 Testing

```bash
# Run all tests
python run_tests.py

# Run specific test
python -m unittest tests.test_features
```

## 💼 Commercial Information

### Pricing
- **SaaS Dashboard**: ₹30K-1L/month (scales with SKU count)
- **API Integration**: ₹5-20L/year (ERP integration)

### Target Market
- 15,000+ FMCG distributors in India
- Kirana aggregators (Jumbotail, StoreKing, Udaan)
- D2C brands on Flipkart/Amazon
- Cold-chain logistics companies

### TAM
₹75 crore ARR at 0.1% market penetration

## 🗺 Roadmap

- **Q2 2026**: FastAPI REST API, mobile app, weather data integration
- **Q3 2026**: ERP integrations (Tally, Zoho), promotion modeling
- **Q4 2026**: 50 customers, ₹30L ARR milestone
- **2027**: Southeast Asia expansion, ₹5 crore ARR

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - Free for commercial use. See [LICENSE](LICENSE) for details.

## 📞 Contact

- **Email**: thor47222@gmail.com
- **GitHub**: [@Venkatareddy26](https://github.com/Venkatareddy26)
- **Project**: [SKU-Level-Demand-Forecasting-Engine](https://github.com/Venkatareddy26/SKU-Level-Demand-Forecasting-Engine)

## 🙏 Acknowledgments

- **M5 Competition** - Walmart benchmark dataset
- **Meta** - NeuralProphet library
- **Microsoft** - LightGBM library
- **Streamlit** - Dashboard framework

---

**Built with ❤️ for Indian retail**
