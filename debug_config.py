import sys
import os
sys.path.insert(0, '.')

print("Testing config step by step...")

print("1. Testing imports...")
import yaml
from dotenv import load_dotenv
print("   ✓ Imports OK")

print("2. Testing YAML loading...")
config_data = yaml.safe_load(open('config/trading_config.yaml'))
print("   ✓ Main config loaded")

print("3. Testing stocks loading...")
stocks_data = yaml.safe_load(open('config/nifty50_stocks.yaml'))
print(f"   ✓ Stocks loaded: {len(stocks_data.get('nifty50_stocks', []))}")

print("4. Testing dotenv...")
load_dotenv()
print("   ✓ Environment variables loaded")

print("✅ All config components work individually!")

# Now test the actual Config class
print("\n5. Testing Config class...")
try:
    from src.utils.config import Config
    print("   Config class imported, now initializing...")
    config = Config()
    print("   ✓ Config class initialized successfully!")
except Exception as e:
    print(f"   ❌ Config class error: {e}")
    import traceback
    traceback.print_exc()
