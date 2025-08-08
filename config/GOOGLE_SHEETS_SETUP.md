# Google Sheets Setup Instructions

## Step 1: Create a Google Cloud Project and Enable APIs

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Sheets API:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Sheets API"
   - Click "Enable"

## Step 2: Create Service Account Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "Service Account"
3. Fill in the service account details:
   - Name: `algo-trading-bot`
   - Description: `Service account for algorithmic trading system`
4. Click "Create and Continue"
5. Skip the optional steps and click "Done"

## Step 3: Generate and Download JSON Key

1. Click on the service account you just created
2. Go to the "Keys" tab
3. Click "Add Key" > "Create new key"
4. Select "JSON" format
5. Click "Create" - this will download the JSON file
6. Rename the downloaded file to `google_sheets_credentials.json`
7. Move it to the `config/` folder in your project

## Step 4: Create Google Sheets and Share Access

1. Create a new Google Sheet for your trading log
2. Copy the spreadsheet ID from the URL (the long string between `/d/` and `/edit`)
3. Update the `GOOGLE_SHEETS_SPREADSHEET_ID` in your `.env` file
4. Share the spreadsheet with your service account email:
   - Click "Share" in Google Sheets
   - Add the service account email (found in the JSON file as "client_email")
   - Give it "Editor" permissions

## Step 5: Update Environment Variables

Update your `.env` file with:
```
GOOGLE_SHEETS_SPREADSHEET_ID=your_actual_spreadsheet_id
GOOGLE_SHEETS_WORKSHEET_NAME=Trading_Log
```

## Example Spreadsheet Structure

Your Google Sheet should have columns like:
- Date
- Symbol
- Action (BUY/SELL)
- Quantity
- Price
- Total Value
- Strategy
- Reason
- Portfolio Value
- P&L

The system will automatically create these headers if they don't exist.

## Security Note

- Never commit the `google_sheets_credentials.json` file to version control
- Keep your service account credentials secure
- The JSON file contains private keys that provide access to your Google account
