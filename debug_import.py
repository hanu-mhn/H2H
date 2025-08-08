import sys
import os
sys.path.insert(0, '.')

# Test importing the actual config module
print("Testing actual Config import...")

try:
    print("1. Importing config module...")
    import src.utils.config
    print("   ✓ Module imported")
    
    print("2. Getting Config class...")
    Config = src.utils.config.Config
    print("   ✓ Class reference obtained")
    
    print("3. Creating Config instance...")
    config = Config()
    print("   ✓ Config instance created!")
    
    print("4. Testing config methods...")
    stocks = config.get('stocks.symbols', [])
    print(f"   ✓ Got {len(stocks)} stocks")
    
    print("✅ Full Config class test passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
