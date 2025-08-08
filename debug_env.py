import sys
import os
sys.path.insert(0, '.')

print("Testing env vars loading...")

import os
from dotenv import load_dotenv

print("1. Loading .env file...")
load_dotenv()
print("   ✓ .env loaded")

print("2. Testing env var access...")
env_vars = {
    'google_sheets_credentials': os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH'),
    'google_sheet_id': os.getenv('GOOGLE_SHEET_ID'),
    'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
    'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
    'trading_mode': os.getenv('TRADING_MODE', 'paper'),
    'initial_capital': float(os.getenv('INITIAL_CAPITAL', '100000')),
    'log_level': os.getenv('LOG_LEVEL', 'INFO'),
    'log_file': os.getenv('LOG_FILE', 'logs/trading.log')
}
print(f"   ✓ Environment variables loaded: {len(env_vars)}")

print("✅ Environment loading works!")

# Now test just the problematic method
print("\n3. Testing Config._load_env_vars method...")
import yaml

class TestConfig:
    def __init__(self):
        self.config_path = "config/trading_config.yaml"
        self.stocks_path = "config/nifty50_stocks.yaml"
        self.logger = None
        
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        with open(self.stocks_path, 'r') as file:
            self.stocks_config = yaml.safe_load(file)
        
        # Test the problematic method
        self._load_env_vars()
        print("   ✓ _load_env_vars completed!")
    
    def _load_env_vars(self):
        """Test version of load env vars."""
        from dotenv import load_dotenv
        load_dotenv()
        
        self.env_vars = {
            'google_sheets_credentials': os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH'),
            'google_sheet_id': os.getenv('GOOGLE_SHEET_ID'),
            'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
            'trading_mode': os.getenv('TRADING_MODE', 'paper'),
            'initial_capital': float(os.getenv('INITIAL_CAPITAL', '100000')),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'log_file': os.getenv('LOG_FILE', 'logs/trading.log')
        }

try:
    test_config = TestConfig()
    print("✅ Test config class works!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
