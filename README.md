# 🚀 Qortfolio V2 - Quantitative Finance Platform

A professional quantitative finance platform combining options analytics with volatility modeling and portfolio management.

## 🎯 Features

### Phase 1 (Current Development)
- **Options Analytics**: Real-time BTC/ETH options data, Black-Scholes pricing, Greeks calculations
- **Volatility Analysis**: ML-based volatility forecasting for 10+ cryptocurrencies
- **Gamma Exposure**: Real-time gamma exposure monitoring and risk management
- **Volatility Surfaces**: 3D implied volatility surface visualization
- **IV vs RV Analysis**: Implied vs realized volatility comparative analysis
- **Statistical Dashboard**: Comprehensive statistical analysis and correlation monitoring

### Critical Bug Fixes
- ✅ Fixed time-to-maturity mathematical calculation
- ✅ Corrected ML model shape inconsistencies
- ✅ Resolved dashboard stability issues

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git
- Virtual environment (recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/qortfolio-v2.git
cd qortfolio-v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Copy environment file
cp .env.example .env
# Edit .env with your API keys