import sys
import os
sys.path.insert(0, '.')

print("Testing minimal config creation...")

import yaml
import logging

class MinimalConfig:
    def __init__(self):
        print("Creating minimal config...")
        with open('config/trading_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        print("Config loaded successfully!")

try:
    config = MinimalConfig()
    print("✅ Minimal config works!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
