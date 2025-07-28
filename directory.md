qortfolio-v2/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── setup.py
├── pyproject.toml
├── 
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                    # Configuration management
│   │   ├── logging.py                   # Logging framework
│   │   ├── exceptions.py                # Custom exceptions
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── time_utils.py           # ⚠️ CRITICAL: Fix time calculation bug
│   │       ├── math_utils.py           # Financial mathematics utilities
│   │       ├── validation.py           # Data validation utilities
│   │       └── api_utils.py            # API helper functions
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collectors/
│   │   │   ├── __init__.py
│   │   │   ├── base_collector.py       # Abstract base collector
│   │   │   ├── crypto_collector.py     # yfinance integration
│   │   │   ├── deribit_collector.py    # Deribit API integration
│   │   │   └── data_manager.py         # Data coordination
│   │   ├── processors/
│   │   │   ├── __init__.py
│   │   │   ├── options_processor.py    # Options data processing
│   │   │   ├── volatility_processor.py # Volatility calculations
│   │   │   └── data_cleaner.py         # Data validation/cleaning
│   │   └── storage/
│   │       ├── __init__.py
│   │       ├── cache_manager.py        # Data caching
│   │       └── data_store.py           # Local data storage
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── options/
│   │   │   ├── __init__.py
│   │   │   ├── black_scholes.py        # Black-Scholes model
│   │   │   ├── greeks.py               # Greeks calculations
│   │   │   ├── gamma_exposure.py       # ⭐ NEW: Gamma exposure
│   │   │   └── option_chain.py         # Options chain modeling
│   │   ├── volatility/
│   │   │   ├── __init__.py
│   │   │   ├── ml_forecaster.py        # ML volatility forecasting
│   │   │   ├── statistical_vol.py      # Statistical volatility
│   │   │   ├── rnn_forecaster.py       # RNN/LSTM/GRU models
│   │   │   └── vol_surface.py          # ⭐ NEW: Volatility surfaces
│   │   └── portfolio/
│   │       ├── __init__.py
│   │       ├── allocation_engine.py    # Asset allocation
│   │       ├── risk_manager.py         # Risk management
│   │       └── pnl_calculator.py       # P&L calculations
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── options/
│   │   │   ├── __init__.py
│   │   │   ├── iv_rv_analyzer.py       # ⭐ NEW: IV vs RV analysis
│   │   │   ├── ratio_analyzer.py       # ⭐ NEW: Call/Put ratios
│   │   │   ├── flow_analyzer.py        # Options flow analysis
│   │   │   └── strategy_analyzer.py    # Options strategies
│   │   ├── risk/
│   │   │   ├── __init__.py
│   │   │   ├── var_calculator.py       # Value at Risk
│   │   │   ├── cvar_calculator.py      # ⭐ NEW: Conditional VaR
│   │   │   └── stress_tester.py        # Stress testing
│   │   ├── statistical/
│   │   │   ├── __init__.py
│   │   │   ├── correlation_analyzer.py # ⭐ NEW: Statistical analysis
│   │   │   ├── distribution_analyzer.py
│   │   │   └── seasonal_analyzer.py
│   │   └── strategies/
│   │       ├── __init__.py
│   │       ├── strategy_tester.py      # Strategy backtesting
│   │       └── multi_leg_builder.py    # Multi-leg strategies
│   │
│   └── dashboard/
│       ├── __init__.py
│       ├── main.py                     # Streamlit main app
│       ├── pages/
│       │   ├── __init__.py
│       │   ├── options_analytics.py    # Options analysis page
│       │   ├── volatility_analysis.py  # Volatility analysis page
│       │   ├── vol_surfaces.py         # ⭐ NEW: 3D volatility surfaces
│       │   ├── statistical_dashboard.py # ⭐ NEW: Statistical analysis
│       │   ├── portfolio_management.py # Portfolio management page
│       │   ├── risk_dashboard.py       # Risk monitoring page
│       │   └── strategy_testing.py     # Strategy testing page
│       ├── components/
│       │   ├── __init__.py
│       │   ├── charts.py               # Chart components
│       │   ├── tables.py               # Data table components
│       │   ├── indicators.py           # KPI indicators
│       │   └── forms.py                # Input forms
│       └── utils/
│           ├── __init__.py
│           ├── layout.py               # Layout utilities
│           ├── session.py              # Session management
│           └── helpers.py              # UI helper functions
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # Pytest configuration
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── test_time_utils.py      # ⚠️ CRITICAL: Time calculation tests
│   │   │   ├── test_math_utils.py
│   │   │   └── test_validation.py
│   │   ├── data/
│   │   │   ├── test_collectors.py
│   │   │   └── test_processors.py
│   │   ├── models/
│   │   │   ├── test_black_scholes.py
│   │   │   ├── test_greeks.py
│   │   │   └── test_volatility.py
│   │   └── analytics/
│   │       ├── test_iv_rv.py
│   │       └── test_risk.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_data_flow.py
│   │   ├── test_api_integration.py
│   │   └── test_dashboard.py
│   └── fixtures/
│       ├── __init__.py
│       ├── sample_data.py
│       └── mock_responses.py
│
├── config/
│   ├── crypto_mapping.yaml            # Crypto name to ticker mapping
│   ├── api_config.yaml                # API configuration settings
│   ├── dashboard_config.yaml          # Dashboard settings
│   └── model_config.yaml              # ML model configurations
│
├── docs/
│   ├── README.md
│   ├── api_reference.md
│   ├── user_guide.md
│   ├── technical_architecture.md
│   ├── development_notes.md
│   └── deployment_guide.md
│
├── scripts/
│   ├── setup_environment.py           # Environment setup
│   ├── data_collection_test.py        # Data collection testing
│   ├── model_training.py              # ML model training
│   └── deployment_prepare.py          # Deployment preparation
│
└── notebooks/
    ├── exploration/
    │   ├── options_analysis.ipynb
    │   ├── volatility_research.ipynb
    │   └── statistical_analysis.ipynb
    ├── validation/
    │   ├── model_validation.ipynb
    │   └── backtest_validation.ipynb
    └── research/
        ├── new_features.ipynb
        └── optimization.ipynb