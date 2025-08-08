#!/usr/bin/env python3
"""
Run script for the Algo Trading System.

This script should be run from the project root directory.
"""

import sys
import os
import asyncio

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.main import AlgoTradingSystem


async def main():
    """Main entry point."""
    print("🚀 Starting Algo Trading System...")
    print("=" * 50)
    
    try:
        system = AlgoTradingSystem()
        await system.run()
    except KeyboardInterrupt:
        print("\n👋 Trading system stopped by user")
    except Exception as e:
        print(f"❌ Error starting trading system: {str(e)}")
        print("Please check your configuration and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
