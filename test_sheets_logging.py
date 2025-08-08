#!/usr/bin/env python3
"""
Test Google Sheets logging functionality
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_google_sheets_logging():
    """Test Google Sheets signal logging."""
    print("📊 TESTING GOOGLE SHEETS LOGGING")
    print("=" * 40)
    
    try:
        from src.integrations.google_sheets import GoogleSheetsLogger
        
        # Initialize the logger
        print("🔧 Initializing Google Sheets logger...")
        sheets_logger = GoogleSheetsLogger()
        
        # Test signal data
        test_signals = [
            {
                'symbol': 'RELIANCE.NS',
                'timestamp': datetime.now(),
                'action': 'BUY',
                'price': 2500.50,
                'quantity': 10,
                'strategy': 'RSI_MA_Crossover',
                'confidence': 0.85,
                'combined_confidence': 0.82,
                'ml_confidence': 0.78,
                'ml_action': 'BUY',
                'reason': 'RSI oversold (28.5) + MA crossover + ML confirmation',
                'rsi': 28.5,
                'ma_20': 2480.00,
                'ma_50': 2520.00,
                'volume_ratio': 1.25,
                'portfolio_value': 100000.00,
                'position_size': 0.25
            },
            {
                'symbol': 'TCS.NS',
                'timestamp': datetime.now(),
                'action': 'SELL',
                'price': 3200.75,
                'quantity': 5,
                'strategy': 'RSI_MA_Crossover',
                'confidence': 0.92,
                'combined_confidence': 0.89,
                'ml_confidence': 0.86,
                'ml_action': 'SELL',
                'reason': 'RSI overbought (73.2) + MA bearish crossover',
                'rsi': 73.2,
                'ma_20': 3180.00,
                'ma_50': 3220.00,
                'volume_ratio': 1.45,
                'portfolio_value': 98500.00,
                'position_size': 0.16
            }
        ]
        
        print("📝 Logging test signals to Google Sheets...")
        await sheets_logger.log_signals(test_signals)
        
        print("✅ Google Sheets logging test completed!")
        print(f"🔗 Check your spreadsheet: https://docs.google.com/spreadsheets/d/1hi-88rJqFcw0R_vPzJ0oXVMepO5WzM1QWE1I--FPNDI")
        
        return True
        
    except Exception as e:
        print(f"❌ Google Sheets logging test failed: {e}")
        return False

async def main():
    """Run the test."""
    success = await test_google_sheets_logging()
    
    if success:
        print("\n🎉 GOOGLE SHEETS LOGGING IS WORKING!")
        print("Your trading system will now log all signals to your spreadsheet.")
    else:
        print("\n❌ Google Sheets logging needs attention.")
    
    return 0 if success else 1

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
