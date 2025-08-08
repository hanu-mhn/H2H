#!/usr/bin/env python3
"""
Test Google Sheets credentials
"""

import sys
import os

def test_credentials():
    """Test if the Google Sheets credentials are valid."""
    print("🔑 TESTING GOOGLE SHEETS CREDENTIALS")
    print("=" * 40)
    
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        
        credentials_path = "config/google_sheets_credentials.json"
        
        if not os.path.exists(credentials_path):
            print(f"❌ Credentials file not found: {credentials_path}")
            return False
        
        print("📁 Credentials file found ✅")
        
        # Test loading credentials
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        creds = Credentials.from_service_account_file(credentials_path, scopes=scope)
        print("🔐 Credentials loaded successfully ✅")
        
        # Test authentication
        client = gspread.authorize(creds)
        print("🔗 Google Sheets authentication successful ✅")
        
        # Get service account email
        with open(credentials_path, 'r') as f:
            import json
            cred_data = json.load(f)
            service_email = cred_data.get('client_email', 'Unknown')
        
        print(f"\n📧 Service Account: {service_email}")
        print(f"🏢 Project ID: {cred_data.get('project_id', 'Unknown')}")
        
        print("\n✅ CREDENTIALS ARE VALID!")
        print("\n📋 NEXT STEPS:")
        print("1. Enable Google Drive API at:")
        print("   https://console.cloud.google.com/apis/library/drive.googleapis.com")
        print("2. Create a Google Sheet and share it with:")
        print(f"   {service_email}")
        print("3. Run: python update_spreadsheet_id.py YOUR_SPREADSHEET_ID")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing credentials: {e}")
        return False

if __name__ == "__main__":
    test_credentials()
