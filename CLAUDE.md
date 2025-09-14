# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qortfolio V2 is a professional quantitative finance platform focused on cryptocurrency options analytics and portfolio management. The application uses Python 3.11, Reflex.dev for the UI framework, MongoDB for data persistence, and optional Redis for caching.

## Development Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements-dev.txt

# Start the Reflex development server
reflex run
# Serves at http://localhost:3000

# Run tests with coverage
pytest -v --cov=src --cov-report=html

# Code formatting and linting
black .
mypy .
flake8 .
```

### Docker Development
```bash
# Start database services
docker compose up -d mongodb redis

# Build and run the application
docker compose up --build qortfolio_app

# Ports: UI (3000), MongoDB (27017), API (8000)
```

### Frontend (React/Reflex)
```bash
# Navigate to .web directory for frontend operations
cd .web

# Development server
npm run dev

# Production build
npm run export

# Production server
npm run prod
```

## Architecture Overview

The project follows a dual-structure pattern:

### Main Application Structure (`qortfolio_v2/`)
- **Entry Point**: `qortfolio_v2.py` - Main Reflex application
- **Pages**: `pages/` - UI page components (options_analytics, volatility, portfolio, risk)
- **Components**: `components/` - Reusable UI components and navigation
- **State Management**:
  - `state.py` - Base application state and options state
  - `volatility_state.py` - Volatility analysis state
  - `portfolio_state.py` - Portfolio management state
  - `risk_state.py` - Risk analysis state

### Core Library (`src/`)
- **Analytics**: `analytics/` - Financial analytics and calculations
- **Models**: `models/` - Financial models (Black-Scholes, Greeks, etc.)
- **Data**: `data/` - Data collection, processing, and management
- **Core**: `core/` - Database operations, utilities, and shared functionality
- **Dashboard**: `dashboard/` - Dashboard-specific logic
- **Qortfolio**: `qortfolio/` - Alternative application entry point

### Key Technical Details

**Database Operations**: Located in `src/core/database/` with connection handling and CRUD operations

**Time Calculations**: Uses 365.25 day-year basis for time-to-maturity calculations (see `src/core/utils/time_utils.py`)

**Options Analytics**: Crypto coin-settled options with custom Greeks calculations in `src/models/` and `src/analytics/`

**Configuration**:
- Reflex config in `rxconfig.py`
- Crypto sectors mapping in `config/crypto_sectors.json`
- Environment variables via `.env` or shell

## Testing

Tests are organized under `tests/` with:
- `tests/unit/` - Unit tests
- `tests/integration/` - Integration tests
- `tests/system/` - System tests
- `tests/fixtures/` - Test fixtures

Some tests require Docker MongoDB to be running. Use `pytest -v --cov=src --cov-report=html` for comprehensive test runs with coverage reports.

## Documentation

**Critical**: The `docs/` folder contains comprehensive project documentation that must be reviewed for any development work:

- **Project Structure**: `docs/qortfolio_v2_project_files.txt` - Complete directory structure and file organization
- **Development Roadmap**: `docs/qortfolio_v2_roadmap.txt` - 8-week development plan with phases and priorities
- **Feature Specifications**: `docs/qortfolio_v2_features.txt` - Detailed feature matrix (68 features across 8 categories)
- **Technical Compass**: `docs/compass_artifact.txt` - Additional technical guidance

**Always check these documentation files first** before making any changes to understand:
- Development phases and priorities
- Feature dependencies and bottlenecks
- Critical legacy issues that need addressing
- Proper implementation standards and testing requirements

## Important Notes

- When working with financial calculations, ensure crypto-specific adjustments for coin-settled options
- Database indexes and operations are critical for performance - see `src/core/database/`
- The application supports both local development and Docker containerization
- Frontend is built with React/Reflex and uses Radix UI components
- WebSocket support available for real-time data updates
- Follow the 8-week development roadmap and prioritize critical bugs from Phase 1