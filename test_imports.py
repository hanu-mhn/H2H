#!/usr/bin/env python3
"""
Test script to verify all imports and basic functionality.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test all module imports."""
    print("🧪 Testing module imports...")
    
    try:
        print("  ✓ Testing config module...")
        from src.utils.config import Config
        
        print("  ✓ Testing logger module...")
        from src.utils.logger import setup_logger
        
        print("  ✓ Testing data fetcher...")
        from src.data.data_fetcher import DataFetcher
        
        print("  ✓ Testing strategy module...")
        from src.strategies.rsi_ma_strategy import RSIMAStrategy
        
        print("  ✓ Testing ML predictor...")
        from src.ml.predictor import MLPredictor
        
        print("  ✓ Testing Google Sheets integration...")
        from src.integrations.google_sheets import GoogleSheetsLogger
        
        print("  ✓ Testing Telegram bot...")
        from src.integrations.telegram_bot import TelegramBot
        
        print("  ✓ Testing backtest engine...")
        from src.backtesting.backtest_engine import BacktestEngine
        
        print("  ✓ Testing main system...")
        from src.main import AlgoTradingSystem
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False


def test_basic_functionality():
    """Test basic system initialization."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        from src.utils.config import Config
        
        # Test config loading
        config = Config()
        print(f"  ✓ Config loaded successfully")
        
        # Test getting some config values
        stocks = config.get('stocks.symbols', [])
        print(f"  ✓ Found {len(stocks)} configured stocks")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("🚀 Algo Trading System - Import Test")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! System is ready to run.")
        print("\nTo start the trading system, run:")
        print("  python run.py")
        print("\nOr to run from src directory:")
        print("  python src/main.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
