import sys
sys.path.append('.')

# Test imports one by one
modules_to_test = [
    "src.utils.logger",
    "src.utils.helpers", 
    "src.utils.config",
    "src.data.data_fetcher",
    "src.strategies.rsi_ma_strategy",
    "src.ml.models",
    "src.ml.feature_engineer",
    "src.ml.predictor",
    "src.integrations.google_sheets",
    "src.integrations.telegram_bot",
    "src.backtesting.backtest_engine",
    "src.main"
]

for module in modules_to_test:
    try:
        print(f"Testing {module}...")
        __import__(module)
        print(f"✓ {module} imported successfully")
    except Exception as e:
        print(f"✗ {module} failed: {e}")
        break
