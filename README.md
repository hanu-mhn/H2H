# Algo-Trading System

A comprehensive Python-based algorithmic trading system with machine learning automation, designed for NIFTY 50 stock analysis and automated trading.

## 🚀 Features

- **Yahoo Finance Integration**: Real-time and historical data for NIFTY 50 stocks
- **RSI + Moving Average Strategy**: Technical analysis-based trading signals
- **Machine Learning**: Multiple ML algorithms for next-day price prediction
- **Google Sheets Automation**: Automated trade logging and portfolio tracking
- **Telegram Alerts**: Real-time notifications for signals and trades
- **Comprehensive Backtesting**: 6-month backtesting with performance analytics
- **Risk Management**: Position sizing, stop-loss, and portfolio risk controls
- **Modular Architecture**: Clean, maintainable code structure
- **Google Sheets Integration**: Automated trade logging with P&L tracking
- **Telegram Alerts**: Real-time signal notifications and error alerts
- **Portfolio Analytics**: Comprehensive performance metrics and visualization

## Project Structure

```
algo_trading_system/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_fetcher.py          # Yahoo Finance API integration
│   │   └── data_processor.py        # Data cleaning and preprocessing
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base_strategy.py         # Abstract base strategy class
│   │   ├── rsi_ma_strategy.py       # RSI + Moving Average strategy
│   │   └── technical_indicators.py  # Technical analysis indicators
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── features.py              # Feature engineering
│   │   ├── models.py                # ML models (Decision Tree, Logistic Regression)
│   │   └── predictor.py             # Prediction engine
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── google_sheets.py         # Google Sheets automation
│   │   └── telegram_bot.py          # Telegram alerts
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── backtest_engine.py       # Backtesting framework
│   │   └── performance_metrics.py   # Portfolio analytics
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py                # Configuration management
│   │   ├── logger.py                # Logging utilities
│   │   └── helpers.py               # Helper functions
│   └── main.py                      # Main application entry point
├── config/
│   ├── config.yaml                  # Main configuration
│   └── nifty50_stocks.yaml          # NIFTY 50 stock symbols
├── tests/                           # Unit tests
├── notebooks/                       # Jupyter notebooks for analysis
├── logs/                           # Application logs
├── data/                           # Data storage
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore file
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd algo-trading-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your credentials:
# - Google Sheets API credentials
# - Telegram Bot Token
# - Other API keys
```

### 3. Google Sheets Setup
1. Create a Google Cloud Project
2. Enable Google Sheets API
3. Create service account credentials
4. Download JSON credentials file
5. Share your Google Sheet with the service account email

### 4. Telegram Bot Setup
1. Create a bot using @BotFather on Telegram
2. Get the bot token
3. Add token to .env file

## Usage

### Run the Trading System
```bash
python src/main.py
```

### Run Backtesting
```bash
python -m src.backtesting.backtest_engine --symbol RELIANCE --start-date 2024-02-01 --end-date 2024-08-01
```

### Run ML Model Training
```bash
python -m src.ml.models --train --symbol RELIANCE
```

## Trading Strategy

### RSI + Moving Average Crossover
- **Buy Signal**: RSI < 30 AND 20-DMA crosses above 50-DMA
- **Sell Signal**: RSI > 70 OR 20-DMA crosses below 50-DMA
- **Risk Management**: Stop-loss at 5%, Take-profit at 10%

### Machine Learning Enhancement
- Features: RSI, MACD, Volume, Moving Averages, Bollinger Bands
- Models: Decision Tree, Logistic Regression
- Prediction: Next-day price movement (Up/Down)

## Google Sheets Integration

The system automatically logs:
- **Trade Log**: Entry/exit points, P&L, timestamps
- **Summary P&L**: Daily, weekly, monthly performance
- **Win Ratio**: Success rate and strategy metrics

## Telegram Alerts

Receive real-time notifications for:
- Buy/Sell signals
- Trade executions
- System errors
- Daily performance summary

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_strategy.py

# Run with coverage
pytest --cov=src tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not use it for actual trading without proper testing and risk management. The authors are not responsible for any financial losses.
