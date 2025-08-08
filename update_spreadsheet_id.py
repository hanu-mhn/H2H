#!/usr/bin/env python3
"""
Update spreadsheet ID in .env file
"""

import sys
import os

def update_spreadsheet_id(spreadsheet_id):
    """Update the spreadsheet ID in .env file."""
    
    if not spreadsheet_id:
        print("❌ Please provide a spreadsheet ID")
        return False
    
    print(f"🔧 Updating .env with spreadsheet ID: {spreadsheet_id}")
    
    try:
        # Read current .env file
        with open('.env', 'r') as f:
            content = f.read()
        
        # Replace the placeholder
        if 'GOOGLE_SHEETS_SPREADSHEET_ID=your_spreadsheet_id_here' in content:
            content = content.replace(
                'GOOGLE_SHEETS_SPREADSHEET_ID=your_spreadsheet_id_here',
                f'GOOGLE_SHEETS_SPREADSHEET_ID={spreadsheet_id}'
            )
        elif 'GOOGLE_SHEETS_SPREADSHEET_ID=' in content:
            # Update existing ID
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('GOOGLE_SHEETS_SPREADSHEET_ID='):
                    lines[i] = f'GOOGLE_SHEETS_SPREADSHEET_ID={spreadsheet_id}'
                    break
            content = '\n'.join(lines)
        else:
            # Add new line
            content += f'\nGOOGLE_SHEETS_SPREADSHEET_ID={spreadsheet_id}\n'
        
        # Write back to file
        with open('.env', 'w') as f:
            f.write(content)
        
        print("✅ .env file updated successfully!")
        print(f"✅ Google Sheets URL: https://docs.google.com/spreadsheets/d/{spreadsheet_id}")
        print("\n🎉 Google Sheets integration is now ready!")
        print("🚀 Start trading with: .\.venv\Scripts\python.exe src\main.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating .env file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_spreadsheet_id.py YOUR_SPREADSHEET_ID")
        print("\nExample:")
        print("python update_spreadsheet_id.py 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms")
        sys.exit(1)
    
    spreadsheet_id = sys.argv[1].strip()
    success = update_spreadsheet_id(spreadsheet_id)
    sys.exit(0 if success else 1)
