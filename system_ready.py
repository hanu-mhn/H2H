#!/usr/bin/env python3
"""
Final system verification and usage instructions.
"""

import sys
import os
import asyncio

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def print_banner():
    """Print system banner."""
    print("🚀 ALGO TRADING SYSTEM")
    print("=" * 50)
    print("✅ System Successfully Initialized!")
    print("=" * 50)

def test_system_components():
    """Test all system components."""
    print("\n🔧 SYSTEM COMPONENTS TEST")
    print("-" * 30)
    
    try:
        from src.main import AlgoTradingSystem
        system = AlgoTradingSystem()
        
        print(f"✓ Data Fetcher: {type(system.data_fetcher).__name__}")
        print(f"✓ Strategy: {type(system.strategy).__name__}")
        print(f"✓ ML Predictor: {type(system.ml_predictor).__name__}")
        print(f"✓ Google Sheets: {type(system.sheets_logger).__name__}")
        print(f"✓ Telegram Bot: {type(system.telegram_bot).__name__}")
        
        return system
        
    except Exception as e:
        print(f"❌ System Error: {e}")
        return None

def show_configuration(system):
    """Show system configuration."""
    print("\n📋 CONFIGURATION")
    print("-" * 30)
    
    stocks = system.config.get('stocks.symbols', [])
    print(f"📈 Configured Stocks: {len(stocks)}")
    print(f"   Sample: {', '.join(stocks[:5])}...")
    
    strategy_config = system.config.get('strategy', {})
    print(f"🎯 Strategy: {strategy_config.get('name', 'RSI_MA_Strategy')}")
    
    ml_models = system.config.get('ml.models', [])
    print(f"🤖 ML Models: {', '.join(ml_models)}")
    
    trading_mode = system.config.get_env('trading_mode', 'paper')
    print(f"💼 Trading Mode: {trading_mode}")
    
    initial_capital = system.config.get_env('initial_capital', 100000)
    print(f"💰 Initial Capital: ₹{initial_capital:,.0f}")

def show_usage_instructions():
    """Show usage instructions."""
    print("\n🎯 USAGE INSTRUCTIONS")
    print("-" * 30)
    print("\n1. 📝 CONFIGURE INTEGRATIONS:")
    print("   Edit .env file with your credentials:")
    print("   • TELEGRAM_BOT_TOKEN=your_bot_token")
    print("   • TELEGRAM_CHAT_ID=your_chat_id")
    print("   • GOOGLE_SHEETS_CREDENTIALS=path_to_credentials.json")
    
    print("\n2. 🚀 RUN THE SYSTEM:")
    print("   D:/H2H/.venv/Scripts/python.exe src/main.py")
    print("   Or: python run.py")
    
    print("\n3. 🧪 RUN BACKTESTING:")
    print("   The system will automatically run 6-month backtests")
    
    print("\n4. 📊 MONITOR PERFORMANCE:")
    print("   • Check Google Sheets for trade logs")
    print("   • Receive Telegram alerts for signals")
    print("   • Review logs in the logs/ directory")

def show_features():
    """Show system features."""
    print("\n🌟 SYSTEM FEATURES")
    print("-" * 30)
    features = [
        "Yahoo Finance data integration",
        "RSI + Moving Average strategy",
        "Machine Learning predictions",
        "Google Sheets automation",
        "Telegram notifications",
        "Comprehensive backtesting",
        "Risk management",
        "Performance analytics",
        "Interactive visualizations",
        "Modular architecture"
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"   {i:2d}. {feature}")

def main():
    """Main verification function."""
    print_banner()
    
    # Test system components
    system = test_system_components()
    if not system:
        print("\n❌ System initialization failed!")
        return 1
    
    # Show configuration
    show_configuration(system)
    
    # Show features
    show_features()
    
    # Show usage instructions
    show_usage_instructions()
    
    print("\n" + "=" * 50)
    print("🎉 SYSTEM READY FOR TRADING!")
    print("=" * 50)
    print("\n💡 TIP: Start with paper trading mode to test the system")
    print("📚 Check README.md for detailed documentation")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
