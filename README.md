# 🚀 Qortfolio V2 - Quantitative Finance Platform

Qortfolio V2 is a professional, modular platform for quantitative finance, combining advanced options analytics, volatility modeling, and portfolio management. Built for reliability, extensibility, and real-time analysis, it is designed for both research and professional trading environments.

## 🎯 Key Features

- **Options Analytics**: Real-time BTC/ETH options data from Deribit, Black-Scholes pricing, full Greeks (Delta, Gamma, Theta, Vega, Rho)
- **Volatility Analysis**: ML-based volatility forecasting for 10+ cryptocurrencies (yfinance integration)
- **Gamma Exposure**: Real-time gamma exposure monitoring at both position and portfolio levels
- **Volatility Surfaces**: 3D implied volatility surface construction and visualization
- **IV vs RV Analysis**: Implied vs realized volatility, term structure, and premium/discount analytics
- **Statistical Dashboard**: Comprehensive statistical analysis, correlation monitoring, and risk metrics
- **Robust Data Validation**: Automated data cleaning, validation, and error handling throughout
- **Extensible Architecture**: Modular codebase for analytics, strategies, and dashboard components

## 🛠️ Critical Bug Fixes & Improvements

- ✅ **Time-to-maturity calculation**: Fixed legacy mathematical bug (now uses `total_seconds() / (365.25 * 24 * 3600)`)
- ✅ **ML model shape consistency**: All volatility models (MLP, RNN, LSTM, GRU) validated and fixed
- ✅ **Dashboard stability**: Improved error handling and session management

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git
- Virtual environment (recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/mhmdfasihi/qortfolio-v2.git
cd qortfolio-v2

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Configuration
- **Environment Variables**: Copy `.env.example` to `.env` and set your API keys and environment settings.
- **YAML Configs**: Edit files in `config/` (e.g., `crypto_mapping.yaml`, `api_config.yaml`) for coin mappings and API endpoints.

### Running the Dashboard
```bash
streamlit run src/dashboard/main_dashboard.py
```

## 🗂️ Project Structure

```
qortfolio-v2/
├── src/
│   ├── core/         # Core utilities, config, logging, time utils (critical bug fixes)
│   ├── data/         # Data collectors (Deribit, yfinance), processors, storage
│   ├── models/       # Options pricing (Black-Scholes), Greeks, ML volatility models
│   ├── analytics/    # Volatility surfaces, PnL simulation, risk/statistical analysis
│   ├── dashboard/    # Streamlit app, UI components, pages
├── config/           # YAML configuration files
├── tests/            # Unit, integration, and system tests
├── requirements.txt  # Python dependencies
└── README.md
```

## ⚙️ Technology Stack
- **Python 3.9+**
- **Streamlit** (dashboard UI)
- **yfinance, Deribit API** (data sources)
- **Pandas, NumPy, SciPy** (data processing)
- **scikit-learn, keras** (ML models)
- **QuantLib, riskfolio-lib** (financial analytics)
- **Plotly** (visualizations)
- **PyYAML, python-dotenv** (configuration)

## 🧪 Testing & Quality
- All core modules are covered by unit and integration tests (`tests/`)
- Run tests with:
  ```bash
  pytest --cov=src
  ```
- Target: **>90% test coverage** for all critical utilities and analytics

## 🤝 Contributing
- Follow the modular structure and add new features in the appropriate directory
- Ensure all new code is tested and documented
- Update configuration and documentation as needed

## 📄 License
- AGPLv3 or commercial license. See source headers for details.

---
For detailed development roadmap, feature requirements, and handoff protocols, see the `docs/` and `config/` directories.
