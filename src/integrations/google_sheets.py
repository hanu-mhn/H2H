"""
Google Sheets integration for trade logging and analytics.
"""

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime, timedelta
import json

from ..utils.logger import setup_logger
from ..utils.config import Config
from ..utils.helpers import format_currency, format_percentage


class GoogleSheetsLogger:
    """Google Sheets integration for automated trade logging."""
    
    def __init__(self):
        """Initialize Google Sheets logger."""
        self.logger = setup_logger(__name__)
        self.config = Config()
        self.sheets_config = self.config.get_sheets_config()
        
        # Google Sheets client
        self.client = None
        self.sheet = None
        self.worksheets = {}
        
        # Initialize connection
        self._initialize_connection()
        
    def _initialize_connection(self) -> None:
        """Initialize Google Sheets connection."""
        try:
            credentials_path = self.sheets_config.get('credentials_path')
            sheet_id = self.sheets_config.get('sheet_id')
            
            if not credentials_path or not sheet_id:
                self.logger.error("Missing Google Sheets credentials or sheet ID")
                return
            
            # Define the scope
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # Load credentials
            credentials = Credentials.from_service_account_file(
                credentials_path, scopes=scope
            )
            
            # Create client
            self.client = gspread.authorize(credentials)
            
            # Open the spreadsheet
            self.sheet = self.client.open_by_key(sheet_id)
            
            # Initialize worksheets
            self._setup_worksheets()
            
            self.logger.info("Google Sheets connection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Google Sheets connection: {str(e)}")
            self.client = None
            self.sheet = None
    
    def _setup_worksheets(self) -> None:
        """Setup and initialize worksheets."""
        try:
            worksheet_names = self.sheets_config.get('worksheets', {})
            
            for ws_key, ws_name in worksheet_names.items():
                try:
                    # Try to get existing worksheet
                    worksheet = self.sheet.worksheet(ws_name)
                except gspread.WorksheetNotFound:
                    # Create new worksheet if it doesn't exist
                    worksheet = self.sheet.add_worksheet(title=ws_name, rows=1000, cols=20)
                    self._initialize_worksheet_headers(ws_key, worksheet)
                
                self.worksheets[ws_key] = worksheet
            
            self.logger.info(f"Setup {len(self.worksheets)} worksheets")
            
        except Exception as e:
            self.logger.error(f"Error setting up worksheets: {str(e)}")
    
    def _initialize_worksheet_headers(self, ws_key: str, worksheet) -> None:
        """Initialize headers for different worksheet types."""
        try:
            if ws_key == 'trade_log':
                headers = [
                    'Timestamp', 'Symbol', 'Action', 'Quantity', 'Price', 
                    'Total Value', 'Strategy', 'Confidence', 'Stop Loss', 
                    'Take Profit', 'Status', 'Exit Price', 'Exit Time', 
                    'P&L', 'P&L %', 'Reason'
                ]
            elif ws_key == 'summary_pl':
                headers = [
                    'Date', 'Starting Value', 'Ending Value', 'Daily P&L', 
                    'Daily P&L %', 'Total Trades', 'Winning Trades', 
                    'Win Rate', 'Best Trade', 'Worst Trade', 'Notes'
                ]
            elif ws_key == 'win_ratio':
                headers = [
                    'Symbol', 'Total Trades', 'Winning Trades', 'Losing Trades',
                    'Win Rate %', 'Avg Win', 'Avg Loss', 'Profit Factor',
                    'Best Trade', 'Worst Trade', 'Last Updated'
                ]
            elif ws_key == 'daily_stats':
                headers = [
                    'Date', 'Portfolio Value', 'Daily Return %', 'Cumulative Return %',
                    'Sharpe Ratio', 'Max Drawdown %', 'Volatility %', 'Trades Count',
                    'Win Rate %', 'Strategy Performance', 'Market Condition'
                ]
            else:
                headers = ['Timestamp', 'Data']  # Default headers
            
            worksheet.append_row(headers)
            
            # Format headers
            header_range = f"A1:{chr(65 + len(headers) - 1)}1"
            worksheet.format(header_range, {
                'textFormat': {'bold': True},
                'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
            })
            
        except Exception as e:
            self.logger.error(f"Error initializing headers for {ws_key}: {str(e)}")
    
    async def log_signals(self, signals: List[Dict[str, Any]]) -> None:
        """
        Log trading signals to Google Sheets.
        
        Args:
            signals: List of trading signals
        """
        if not self.client or not self.sheet:
            self.logger.warning("Google Sheets not available for logging signals")
            return
        
        try:
            # Try to get or create the Trading_Log worksheet
            try:
                worksheet = self.sheet.worksheet("Trading_Log")
            except gspread.WorksheetNotFound:
                # Create the worksheet if it doesn't exist
                worksheet = self.sheet.add_worksheet(title="Trading_Log", rows=1000, cols=26)
                # Set up headers
                headers = [
                    "Timestamp", "Date", "Time", "Symbol", "Action", "Quantity", "Price",
                    "Total_Value", "Strategy", "Confidence", "RSI", "MA_20", "MA_50",
                    "Volume_Ratio", "ML_Prediction", "ML_Confidence", "Reason",
                    "Portfolio_Value", "Position_Size", "P&L", "Cumulative_P&L"
                ]
                worksheet.update('A1', [headers])
                # Format headers
                worksheet.format('A1:U1', {
                    'textFormat': {'bold': True},
                    'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
                })
            
            for signal in signals:
                now = datetime.now()
                row_data = [
                    signal.get('timestamp', now).strftime('%Y-%m-%d %H:%M:%S'),
                    now.strftime('%Y-%m-%d'),
                    now.strftime('%H:%M:%S'),
                    signal.get('symbol', ''),
                    signal.get('action', ''),
                    signal.get('quantity', 0),
                    signal.get('price', 0),
                    signal.get('quantity', 0) * signal.get('price', 0),
                    signal.get('strategy', 'RSI_MA_Crossover'),
                    round(signal.get('combined_confidence', signal.get('confidence', 0)), 4),
                    signal.get('rsi', ''),
                    signal.get('ma_20', ''),
                    signal.get('ma_50', ''),
                    signal.get('volume_ratio', ''),
                    signal.get('ml_action', ''),
                    round(signal.get('ml_confidence', 0), 4),
                    signal.get('reason', ''),
                    signal.get('portfolio_value', ''),
                    signal.get('position_size', ''),
                    '',  # P&L (to be calculated later)
                    ''   # Cumulative P&L (to be calculated later)
                ]
                
                worksheet.append_row(row_data)
            
            self.logger.info(f"Logged {len(signals)} signals to Google Sheets")
            
        except Exception as e:
            self.logger.error(f"Error logging signals to Google Sheets: {str(e)}")
    
    async def log_trade_execution(self, trade: Dict[str, Any]) -> None:
        """
        Log trade execution details.
        
        Args:
            trade: Trade execution details
        """
        if not self.client or 'trade_log' not in self.worksheets:
            return
        
        try:
            worksheet = self.worksheets['trade_log']
            
            # Find the corresponding signal row and update it
            # For now, append as new row (can be enhanced to update existing rows)
            row_data = [
                trade.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                trade.get('symbol', ''),
                trade.get('action', ''),
                trade.get('quantity', 0),
                trade.get('entry_price', 0),
                trade.get('quantity', 0) * trade.get('entry_price', 0),
                trade.get('strategy', ''),
                trade.get('confidence', 0),
                trade.get('stop_loss', ''),
                trade.get('take_profit', ''),
                'EXECUTED',
                trade.get('exit_price', ''),
                trade.get('exit_time', ''),
                trade.get('pnl', ''),
                trade.get('pnl_percent', ''),
                trade.get('reason', '')
            ]
            
            worksheet.append_row(row_data)
            
        except Exception as e:
            self.logger.error(f"Error logging trade execution: {str(e)}")
    
    async def log_daily_summary(self, summary: Dict[str, Any]) -> None:
        """
        Log daily performance summary.
        
        Args:
            summary: Daily performance summary
        """
        if not self.client or 'summary_pl' not in self.worksheets:
            return
        
        try:
            worksheet = self.worksheets['summary_pl']
            
            row_data = [
                summary.get('date', datetime.now().strftime('%Y-%m-%d')),
                summary.get('starting_value', 0),
                summary.get('ending_value', 0),
                summary.get('daily_pnl', 0),
                round(summary.get('daily_pnl_percent', 0) * 100, 2),
                summary.get('total_trades', 0),
                summary.get('winning_trades', 0),
                round(summary.get('win_rate', 0) * 100, 2),
                summary.get('best_trade', 0),
                summary.get('worst_trade', 0),
                summary.get('notes', '')
            ]
            
            worksheet.append_row(row_data)
            
        except Exception as e:
            self.logger.error(f"Error logging daily summary: {str(e)}")
    
    async def update_win_ratio_stats(self, symbol: str, stats: Dict[str, Any]) -> None:
        """
        Update win ratio statistics for a symbol.
        
        Args:
            symbol: Stock symbol
            stats: Win ratio statistics
        """
        if not self.client or 'win_ratio' not in self.worksheets:
            return
        
        try:
            worksheet = self.worksheets['win_ratio']
            
            # Find existing row for symbol or create new one
            all_records = worksheet.get_all_records()
            symbol_row = None
            
            for i, record in enumerate(all_records, start=2):  # Start from row 2 (after header)
                if record.get('Symbol') == symbol:
                    symbol_row = i
                    break
            
            row_data = [
                symbol,
                stats.get('total_trades', 0),
                stats.get('winning_trades', 0),
                stats.get('losing_trades', 0),
                round(stats.get('win_rate', 0) * 100, 2),
                round(stats.get('avg_win', 0), 2),
                round(stats.get('avg_loss', 0), 2),
                round(stats.get('profit_factor', 0), 2),
                round(stats.get('best_trade', 0), 2),
                round(stats.get('worst_trade', 0), 2),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
            
            if symbol_row:
                # Update existing row
                for i, value in enumerate(row_data, start=1):
                    worksheet.update_cell(symbol_row, i, value)
            else:
                # Append new row
                worksheet.append_row(row_data)
            
        except Exception as e:
            self.logger.error(f"Error updating win ratio stats: {str(e)}")
    
    async def log_backtest_results(self, results: Dict[str, Any]) -> None:
        """
        Log backtest results to Google Sheets.
        
        Args:
            results: Backtest results
        """
        if not self.client:
            return
        
        try:
            # Create or get backtest worksheet
            worksheet_name = "Backtest Results"
            
            try:
                worksheet = self.sheet.worksheet(worksheet_name)
            except gspread.WorksheetNotFound:
                worksheet = self.sheet.add_worksheet(title=worksheet_name, rows=1000, cols=15)
                
                # Initialize headers
                headers = [
                    'Symbol', 'Start Date', 'End Date', 'Strategy', 'Initial Capital',
                    'Final Value', 'Total Return %', 'Annualized Return %', 'Sharpe Ratio',
                    'Max Drawdown %', 'Win Rate %', 'Total Trades', 'Avg Trade',
                    'Best Trade', 'Worst Trade'
                ]
                worksheet.append_row(headers)
            
            # Add backtest results
            row_data = [
                results.get('symbol', ''),
                results.get('start_date', ''),
                results.get('end_date', ''),
                results.get('strategy', ''),
                results.get('initial_capital', 0),
                results.get('final_value', 0),
                round(results.get('total_return', 0) * 100, 2),
                round(results.get('annualized_return', 0) * 100, 2),
                round(results.get('sharpe_ratio', 0), 2),
                round(results.get('max_drawdown', 0) * 100, 2),
                round(results.get('win_rate', 0) * 100, 2),
                results.get('total_trades', 0),
                round(results.get('avg_trade', 0), 2),
                round(results.get('best_trade', 0), 2),
                round(results.get('worst_trade', 0), 2)
            ]
            
            worksheet.append_row(row_data)
            
        except Exception as e:
            self.logger.error(f"Error logging backtest results: {str(e)}")
    
    async def get_daily_summary(self, date: str = None) -> Dict[str, Any]:
        """
        Get daily summary from Google Sheets.
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
        
        Returns:
            Daily summary dictionary
        """
        if not self.client or 'summary_pl' not in self.worksheets:
            return {}
        
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
            
            worksheet = self.worksheets['summary_pl']
            all_records = worksheet.get_all_records()
            
            for record in reversed(all_records):  # Start from most recent
                if record.get('Date') == date:
                    return {
                        'date': record.get('Date'),
                        'starting_value': float(record.get('Starting Value', 0)),
                        'ending_value': float(record.get('Ending Value', 0)),
                        'daily_pnl': float(record.get('Daily P&L', 0)),
                        'daily_pnl_percent': float(record.get('Daily P&L %', 0)) / 100,
                        'total_trades': int(record.get('Total Trades', 0)),
                        'winning_trades': int(record.get('Winning Trades', 0)),
                        'win_rate': float(record.get('Win Rate', 0)) / 100
                    }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting daily summary: {str(e)}")
            return {}
    
    async def get_portfolio_performance(self, days: int = 30) -> Dict[str, Any]:
        """
        Get portfolio performance for the last N days.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Performance metrics dictionary
        """
        if not self.client or 'summary_pl' not in self.worksheets:
            return {}
        
        try:
            worksheet = self.worksheets['summary_pl']
            all_records = worksheet.get_all_records()
            
            # Filter records for the last N days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            filtered_records = []
            for record in all_records:
                try:
                    record_date = datetime.strptime(record.get('Date', ''), '%Y-%m-%d')
                    if start_date <= record_date <= end_date:
                        filtered_records.append(record)
                except ValueError:
                    continue
            
            if not filtered_records:
                return {}
            
            # Calculate performance metrics
            daily_returns = [float(r.get('Daily P&L %', 0)) / 100 for r in filtered_records]
            total_trades = sum(int(r.get('Total Trades', 0)) for r in filtered_records)
            winning_trades = sum(int(r.get('Winning Trades', 0)) for r in filtered_records)
            
            # Portfolio values
            starting_value = float(filtered_records[0].get('Starting Value', 0))
            ending_value = float(filtered_records[-1].get('Ending Value', 0))
            
            # Performance calculations
            total_return = (ending_value - starting_value) / starting_value if starting_value > 0 else 0
            avg_daily_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0
            volatility = pd.Series(daily_returns).std() if len(daily_returns) > 1 else 0
            
            # Sharpe ratio (assuming 5% risk-free rate)
            risk_free_daily = 0.05 / 252  # Daily risk-free rate
            sharpe_ratio = (avg_daily_return - risk_free_daily) / volatility if volatility > 0 else 0
            
            # Win rate
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            return {
                'period_days': days,
                'starting_value': starting_value,
                'ending_value': ending_value,
                'total_return': total_return,
                'avg_daily_return': avg_daily_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'records_count': len(filtered_records)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio performance: {str(e)}")
            return {}
    
    async def export_data(self, worksheet_name: str, format: str = 'csv') -> Optional[str]:
        """
        Export worksheet data to file.
        
        Args:
            worksheet_name: Name of worksheet to export
            format: Export format ('csv', 'json')
        
        Returns:
            File path of exported data or None if failed
        """
        try:
            if worksheet_name not in self.worksheets:
                self.logger.error(f"Worksheet {worksheet_name} not found")
                return None
            
            worksheet = self.worksheets[worksheet_name]
            all_records = worksheet.get_all_records()
            
            if not all_records:
                self.logger.warning(f"No data found in worksheet {worksheet_name}")
                return None
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{worksheet_name}_{timestamp}.{format}"
            
            if format == 'csv':
                df = pd.DataFrame(all_records)
                df.to_csv(filename, index=False)
            elif format == 'json':
                with open(filename, 'w') as f:
                    json.dump(all_records, f, indent=2, default=str)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return None
            
            self.logger.info(f"Exported {len(all_records)} records to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {str(e)}")
            return None
    
    def is_connected(self) -> bool:
        """Check if Google Sheets connection is active."""
        return self.client is not None and self.sheet is not None
