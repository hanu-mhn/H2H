#!/usr/bin/env python3
"""
Safe launcher for the trading system that handles initialization gracefully.
"""

import sys
import os
import asyncio
import signal

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\n🛑 Shutdown signal received. Stopping trading system...")
    sys.exit(0)

async def main():
    """Main launcher function."""
    print("🚀 Algo Trading System Launcher")
    print("=" * 40)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("📋 Initializing system components...")
        
        # Import and initialize the system
        from src.main import AlgoTradingSystem
        system = AlgoTradingSystem()
        
        print("✅ System initialized successfully!")
        print("\n🎯 Trading System Status:")
        print("  • Mode: Paper Trading (Safe)")
        print("  • Data Source: Yahoo Finance")
        print("  • Strategy: RSI + Moving Average")
        print("  • ML Enhancement: Enabled")
        print("  • Risk Management: Active")
        
        print("\n⚠️  IMPORTANT NOTES:")
        print("  • This is PAPER TRADING mode - no real money involved")
        print("  • System will scan for signals every 5 minutes")
        print("  • Press Ctrl+C to stop the system safely")
        print("  • Check logs/ directory for detailed information")
        
        print("\n🟢 Starting trading system...")
        print("=" * 40)
        
        # Run the trading system
        await system.run()
        
    except KeyboardInterrupt:
        print("\n👋 Trading system stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error starting trading system: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("  1. Check if all dependencies are installed")
        print("  2. Verify configuration files are present")
        print("  3. Ensure .env file is configured")
        print("  4. Check logs/ directory for detailed errors")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
