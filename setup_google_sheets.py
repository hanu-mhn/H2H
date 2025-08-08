#!/usr/bin/env python3
"""
Google Sheets Setup and Test Script

This script will:
1. Test the credentials
2. Create a new spreadsheet for trading logs
3. Set up the proper headers
4. Update the .env file with the spreadsheet ID
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def setup_google_sheets():
    """Set up Google Sheets integration."""
    print("📊 GOOGLE SHEETS SETUP")
    print("=" * 40)
    
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        
        # Load credentials
        print("🔑 Loading Google Sheets credentials...")
        
        credentials_path = "config/google_sheets_credentials.json"
        if not os.path.exists(credentials_path):
            print(f"❌ Credentials file not found: {credentials_path}")
            return False
        
        # Set up the scope
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # Authenticate
        creds = Credentials.from_service_account_file(credentials_path, scopes=scope)
        client = gspread.authorize(creds)
        
        print("✅ Google Sheets authentication successful!")
        
        # Create a new spreadsheet
        print("📝 Creating new trading log spreadsheet...")
        
        spreadsheet_name = "Algo Trading Log - NIFTY 50"
        spreadsheet = client.create(spreadsheet_name)
        
        print(f"✅ Created spreadsheet: {spreadsheet_name}")
        print(f"📋 Spreadsheet ID: {spreadsheet.id}")
        print(f"🔗 URL: https://docs.google.com/spreadsheets/d/{spreadsheet.id}")
        
        # Get the default worksheet
        worksheet = spreadsheet.sheet1
        worksheet.update_title("Trading_Log")
        
        # Set up headers
        print("📊 Setting up trading log headers...")
        
        headers = [
            "Timestamp",
            "Date", 
            "Time",
            "Symbol",
            "Action",
            "Quantity",
            "Price",
            "Total_Value",
            "Strategy",
            "Confidence",
            "RSI",
            "MA_20",
            "MA_50",
            "Volume_Ratio",
            "ML_Prediction",
            "ML_Confidence",
            "Reason",
            "Portfolio_Value",
            "Position_Size",
            "P&L",
            "Cumulative_P&L"
        ]
        
        # Update the first row with headers
        worksheet.update('A1', [headers])
        
        # Format headers (bold)
        worksheet.format('A1:U1', {
            'textFormat': {'bold': True},
            'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
        })
        
        print("✅ Headers set up successfully!")
        
        # Update .env file
        print("🔧 Updating .env file...")
        
        with open('.env', 'r') as f:
            content = f.read()
        
        # Replace the placeholder spreadsheet ID
        content = content.replace(
            'GOOGLE_SHEETS_SPREADSHEET_ID=your_spreadsheet_id_here',
            f'GOOGLE_SHEETS_SPREADSHEET_ID={spreadsheet.id}'
        )
        
        with open('.env', 'w') as f:
            f.write(content)
        
        print("✅ .env file updated with spreadsheet ID!")
        
        # Test writing some sample data
        print("🧪 Testing data write...")
        
        sample_data = [
            "2025-08-08 12:00:00",
            "2025-08-08",
            "12:00:00", 
            "RELIANCE.NS",
            "BUY",
            "10",
            "2500.50",
            "25005.00",
            "RSI_MA_Crossover",
            "0.85",
            "28.5",
            "2480.00",
            "2520.00",
            "1.25",
            "BUY",
            "0.78",
            "RSI oversold + MA crossover",
            "100000.00",
            "0.25",
            "0.00",
            "0.00"
        ]
        
        worksheet.append_row(sample_data)
        print("✅ Sample data written successfully!")
        
        print("\n" + "=" * 40)
        print("🎉 GOOGLE SHEETS SETUP COMPLETE!")
        print("=" * 40)
        print(f"📋 Spreadsheet ID: {spreadsheet.id}")
        print(f"🔗 URL: https://docs.google.com/spreadsheets/d/{spreadsheet.id}")
        print(f"📧 Service Account: n50alert@gen-lang-client-0710060197.iam.gserviceaccount.com")
        print("\n✅ Your trading system is now ready to log trades to Google Sheets!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: .\.venv\Scripts\pip.exe install gspread google-auth")
        return False
    except Exception as e:
        print(f"❌ Error setting up Google Sheets: {e}")
        return False

async def test_integration():
    """Test the complete Google Sheets integration."""
    print("\n🧪 TESTING INTEGRATION")
    print("=" * 40)
    
    try:
        from src.integrations.google_sheets import GoogleSheetsLogger
        
        # Initialize the logger
        sheets_logger = GoogleSheetsLogger()
        
        # Test logging a signal
        test_signal = {
            'symbol': 'TESTSTOCK.NS',
            'timestamp': '2025-08-08 12:30:00',
            'action': 'BUY',
            'price': 1500.75,
            'strategy_confidence': 0.92,
            'ml_confidence': 0.88,
            'combined_confidence': 0.90
        }
        
        print("📝 Testing signal logging...")
        await sheets_logger.log_signals([test_signal])
        
        print("✅ Google Sheets integration test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

async def main():
    """Run the setup process."""
    # First, set up Google Sheets
    setup_success = await setup_google_sheets()
    
    if setup_success:
        # Then test the integration
        test_success = await test_integration()
        
        if test_success:
            print("\n🚀 READY TO TRADE!")
            print("Your algo trading system can now:")
            print("• ✅ Log all trades to Google Sheets")
            print("• ✅ Send Telegram notifications")
            print("• ✅ Calculate technical indicators")
            print("• ✅ Generate ML predictions")
            print("\nStart trading with: .\.venv\Scripts\python.exe src\main.py")
            return 0
        else:
            print("\n⚠️  Setup complete but integration test failed.")
            return 1
    else:
        print("\n❌ Google Sheets setup failed.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Setup cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
