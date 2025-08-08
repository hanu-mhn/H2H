#!/usr/bin/env python3
"""
Manual Google Sheets Setup Instructions

Since we need to enable the Google Drive API, let's set this up manually.
"""

print("📊 GOOGLE SHEETS MANUAL SETUP")
print("=" * 50)

print("\n🔧 STEP 1: Enable Google Drive API")
print("-" * 30)
print("1. Go to: https://console.cloud.google.com/apis/library/drive.googleapis.com")
print("2. Make sure your project 'gen-lang-client-0710060197' is selected")
print("3. Click 'ENABLE' for Google Drive API")
print("4. Wait 2-3 minutes for the API to be fully enabled")

print("\n📝 STEP 2: Create Trading Log Spreadsheet")
print("-" * 40)
print("1. Go to: https://sheets.google.com")
print("2. Create a new spreadsheet")
print("3. Name it: 'Algo Trading Log - NIFTY 50'")
print("4. Copy the spreadsheet ID from the URL")
print("   (the long string between '/d/' and '/edit')")

print("\n🔗 STEP 3: Share Spreadsheet")
print("-" * 30)
print("1. Click 'Share' in the top-right of Google Sheets")
print("2. Add this email: n50alert@gen-lang-client-0710060197.iam.gserviceaccount.com")
print("3. Give it 'Editor' permissions")
print("4. Click 'Send'")

print("\n📋 STEP 4: Set Up Headers")
print("-" * 25)
print("Add these headers to row 1 of your spreadsheet:")

headers = [
    "Timestamp", "Date", "Time", "Symbol", "Action", "Quantity", "Price", 
    "Total_Value", "Strategy", "Confidence", "RSI", "MA_20", "MA_50", 
    "Volume_Ratio", "ML_Prediction", "ML_Confidence", "Reason", 
    "Portfolio_Value", "Position_Size", "P&L", "Cumulative_P&L"
]

for i, header in enumerate(headers, 1):
    print(f"{chr(64+i)}1: {header}")

print("\n⚙️  STEP 5: Update Configuration")
print("-" * 35)
print("After creating the spreadsheet, run this command with your spreadsheet ID:")
print("python update_spreadsheet_id.py YOUR_SPREADSHEET_ID")

print("\n✅ CREDENTIALS ALREADY CONFIGURED")
print("-" * 35)
print("✅ Service account: n50alert@gen-lang-client-0710060197.iam.gserviceaccount.com")
print("✅ Credentials file: config/google_sheets_credentials.json")
print("✅ Project ID: gen-lang-client-0710060197")

print("\n🚀 After completing these steps, your trading system will be ready!")
