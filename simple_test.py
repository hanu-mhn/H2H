import sys
import os
sys.path.insert(0, '.')

print("Starting simple test...")

try:
    print("1. Testing config...")
    from src.utils.config import Config
    config = Config()
    print("   ✓ Config loaded")
    
    print("2. Testing data fetcher...")
    from src.data.data_fetcher import DataFetcher
    data_fetcher = DataFetcher()
    print("   ✓ Data fetcher loaded")
    
    print("3. Testing strategy...")
    from src.strategies.rsi_ma_strategy import RSIMAStrategy
    strategy = RSIMAStrategy()
    print("   ✓ Strategy loaded")
    
    print("✅ Basic components working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
