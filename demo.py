#!/usr/bin/env python3
"""
Simple demo script to test the trading system without complex scheduling.
"""

import sys
import os
import asyncio

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_basic_functionality():
    """Test basic functionality without running the full system."""
    print("🧪 Testing Basic Functionality")
    print("=" * 40)
    
    try:
        # Test configuration loading
        print("📝 Testing configuration...")
        from src.utils.config import Config
        config = Config()
        print(f"✓ Config loaded successfully")
        
        # Get sample stocks
        stocks = config.get('stocks.symbols', [])[:3]  # Get first 3 stocks
        print(f"✓ Found {len(stocks)} test stocks: {stocks}")
        
        print("\n💰 Testing data fetching...")
        from src.data.data_fetcher import DataFetcher
        data_fetcher = DataFetcher()
        print("✓ Data fetcher initialized")
        
        print("\n📊 Testing strategy...")
        from src.strategies.rsi_ma_strategy import RSIMAStrategy
        strategy = RSIMAStrategy()
        print("✓ Strategy initialized")
        
        print("\n🤖 Testing ML components...")
        from src.ml.predictor import MLPredictor
        ml_predictor = MLPredictor()
        print("✓ ML predictor initialized")
        
        print("\n✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_data_fetching():
    """Test actual data fetching."""
    print("\n🔍 Testing Data Fetching")
    print("=" * 40)
    
    try:
        from src.data.data_fetcher import DataFetcher
        from src.utils.config import Config
        
        config = Config()
        data_fetcher = DataFetcher()
        
        # Test with one stock
        test_symbol = "RELIANCE.NS"
        print(f"📈 Fetching data for {test_symbol}...")
        
        data = await data_fetcher.fetch_stock_data(test_symbol, period="5d")
        
        if data is not None and not data.empty:
            print(f"✓ Successfully fetched {len(data)} days of data")
            print(f"✓ Latest price: ₹{data['Close'].iloc[-1]:.2f}")
            print(f"✓ Data columns: {list(data.columns)}")
            return True
        else:
            print("❌ No data received")
            return False
            
    except Exception as e:
        print(f"❌ Data fetching error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function."""
    print("🚀 Algo Trading System - Simple Demo")
    print("=" * 50)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic tests failed!")
        return 1
    
    # Test async data fetching
    print("\n" + "=" * 50)
    try:
        result = asyncio.run(test_data_fetching())
        if not result:
            print("\n❌ Data fetching test failed!")
            return 1
    except Exception as e:
        print(f"\n❌ Async test error: {str(e)}")
        return 1
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! System is working correctly.")
    print("\nNext steps:")
    print("1. Set up your .env file with Telegram and Google Sheets credentials")
    print("2. Run: python run.py")
    print("3. Or run the main system: python src/main.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
