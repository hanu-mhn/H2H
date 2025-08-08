#!/usr/bin/env python3
"""
Quick system test to verify all components are working
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root and src directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

async def test_imports():
    """Test all critical imports."""
    print("🔍 Testing imports...")
    
    try:
        # Test data module
        from src.data.data_fetcher import DataFetcher
        print("✅ Data module imported successfully")
        
        # Test strategies module
        from src.strategies.rsi_ma_strategy import RSIMAStrategy
        print("✅ Strategies module imported successfully")
        
        # Test ML module
        from src.ml.predictor import MLPredictor
        print("✅ ML module imported successfully")
        
        # Test backtesting module
        from src.backtesting.backtest_engine import BacktestEngine
        print("✅ Backtesting module imported successfully")
        
        # Test integrations
        from src.integrations.telegram_bot import TelegramBot
        print("✅ Telegram integration imported successfully")
        
        # Test utils
        from src.utils.config import Config
        print("✅ Utils module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

async def test_telegram_bot():
    """Test Telegram bot connection."""
    print("\n📱 Testing Telegram bot...")
    
    try:
        from src.integrations.telegram_bot import TelegramBot
        
        # Initialize bot
        bot = TelegramBot()
        
        # Test sending a message
        test_signal = {
            'symbol': 'TEST.NS',
            'action': 'BUY',
            'price': 100.0,
            'confidence': 0.85,
            'reason': '🧪 System Test - Algo Trading Bot is operational! 🚀'
        }
        await bot.send_signal_alert(test_signal)
        
        print("✅ Telegram bot test message sent successfully")
        return True
        
    except Exception as e:
        print(f"❌ Telegram bot error: {e}")
        return False

async def test_data_fetcher():
    """Test data fetching."""
    print("\n📊 Testing data fetcher...")
    
    try:
        from src.data.data_fetcher import DataFetcher
        
        fetcher = DataFetcher()
        
        # Test fetching a small sample of data
        data = await fetcher.get_stock_data("RELIANCE.NS", period="5d")
        
        if data is not None and len(data) > 0:
            print(f"✅ Data fetcher working - got {len(data)} days of data")
            return True
        else:
            print("❌ Data fetcher returned empty data")
            return False
            
    except Exception as e:
        print(f"❌ Data fetcher error: {e}")
        return False

async def test_google_sheets():
    """Test Google Sheets integration."""
    print("\n📊 Testing Google Sheets integration...")
    
    try:
        from src.integrations.google_sheets import GoogleSheetsLogger
        from datetime import datetime
        
        # Initialize Google Sheets logger
        sheets_logger = GoogleSheetsLogger()
        
        # Test signal
        test_signal = {
            'symbol': 'TEST.NS',
            'timestamp': datetime.now(),
            'action': 'BUY',
            'price': 1500.50,
            'quantity': 10,
            'strategy': 'RSI_MA_Crossover',
            'confidence': 0.85,
            'combined_confidence': 0.82,
            'reason': '🧪 Google Sheets integration test'
        }
        
        await sheets_logger.log_signals([test_signal])
        
        print("✅ Google Sheets integration test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Google Sheets integration error: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 ALGO TRADING SYSTEM - QUICK TEST")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Data Fetcher", test_data_fetcher),
        ("Google Sheets", test_google_sheets),
        ("Telegram Bot", test_telegram_bot),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL SYSTEMS GO! Your algo trading system is ready!")
        print("\n🚀 To start trading, run: .\.venv\Scripts\python.exe src\main.py")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Test cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
