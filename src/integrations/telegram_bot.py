"""
Telegram bot integration for alerts and notifications.
"""

import asyncio
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, time
from telegram import Bot
from telegram.error import TelegramError

from ..utils.logger import setup_logger
from ..utils.config import Config
from ..utils.helpers import format_currency, format_percentage, is_market_open


class TelegramBot:
    """Telegram bot for trading alerts and notifications."""
    
    def __init__(self):
        """Initialize Telegram bot."""
        self.logger = setup_logger(__name__)
        self.config = Config()
        self.telegram_config = self.config.get_telegram_config()
        
        # Bot configuration
        self.bot_token = self.telegram_config.get('bot_token')
        self.chat_id = self.telegram_config.get('chat_id')
        self.bot = None
        
        # Message settings
        self.send_signals = self.telegram_config.get('send_signals', True)
        self.send_errors = self.telegram_config.get('send_errors', True)
        self.send_daily_summary = self.telegram_config.get('send_daily_summary', True)
        
        # Quiet hours (no notifications during these hours)
        quiet_hours = self.telegram_config.get('quiet_hours', {})
        self.quiet_start = quiet_hours.get('start', 22)  # 10 PM
        self.quiet_end = quiet_hours.get('end', 8)       # 8 AM
        
        # Initialize bot
        self._initialize_bot()
        
    def _initialize_bot(self) -> None:
        """Initialize Telegram bot connection."""
        try:
            if not self.bot_token or not self.chat_id:
                self.logger.warning("Telegram bot token or chat ID not configured")
                return
            
            self.bot = Bot(token=self.bot_token)
            self.logger.info("Telegram bot initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Telegram bot: {str(e)}")
            self.bot = None
    
    def _is_quiet_time(self) -> bool:
        """Check if current time is within quiet hours."""
        current_hour = datetime.now().hour
        
        if self.quiet_start <= self.quiet_end:
            # Same day quiet hours (e.g., 22:00 to 23:59)
            return self.quiet_start <= current_hour <= self.quiet_end
        else:
            # Overnight quiet hours (e.g., 22:00 to 08:00)
            return current_hour >= self.quiet_start or current_hour <= self.quiet_end
    
    async def _send_message(self, message: str, parse_mode: str = 'HTML', 
                           silent: bool = False) -> bool:
        """
        Send message to Telegram.
        
        Args:
            message: Message text
            parse_mode: Message formatting mode
            silent: Whether to send silently
        
        Returns:
            True if successful, False otherwise
        """
        if not self.bot or not self.chat_id:
            return False
        
        try:
            # Check quiet hours for non-critical messages
            if silent and self._is_quiet_time():
                self.logger.debug("Skipping message due to quiet hours")
                return False
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
                disable_notification=silent
            )
            
            return True
            
        except TelegramError as e:
            self.logger.error(f"Telegram error: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {str(e)}")
            return False
    
    async def send_signal_alert(self, signal: Dict[str, Any]) -> None:
        """
        Send trading signal alert.
        
        Args:
            signal: Trading signal dictionary
        """
        if not self.send_signals or signal.get('action') == 'HOLD':
            return
        
        try:
            # Create signal message
            action = signal.get('action', 'UNKNOWN')
            symbol = signal.get('symbol', 'UNKNOWN')
            price = signal.get('price', 0)
            confidence = signal.get('confidence', 0)
            timestamp = signal.get('timestamp', datetime.now())
            reason = signal.get('reason', '')
            
            # Determine emoji based on action
            emoji = "🟢" if action == 'BUY' else "🔴" if action == 'SELL' else "⚪"
            
            # Format message
            message = f"""
{emoji} <b>TRADING SIGNAL</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Action:</b> {action}
<b>Price:</b> ₹{price:.2f}
<b>Confidence:</b> {confidence:.1%}
<b>Time:</b> {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

<b>Strategy Analysis:</b>
{reason}

<b>Risk Management:</b>
Stop Loss: ₹{signal.get('stop_loss', 'N/A')}
Take Profit: ₹{signal.get('take_profit', 'N/A')}

<i>⚠️ This is an automated signal. Please conduct your own analysis before trading.</i>
            """
            
            await self._send_message(message.strip())
            self.logger.info(f"Sent signal alert for {symbol}: {action}")
            
        except Exception as e:
            self.logger.error(f"Error sending signal alert: {str(e)}")
    
    async def send_trade_execution_alert(self, trade: Dict[str, Any]) -> None:
        """
        Send trade execution alert.
        
        Args:
            trade: Trade execution details
        """
        try:
            action = trade.get('action', 'UNKNOWN')
            symbol = trade.get('symbol', 'UNKNOWN')
            quantity = trade.get('quantity', 0)
            price = trade.get('price', 0)
            total_value = quantity * price
            
            emoji = "✅" if action == 'BUY' else "💰" if action == 'SELL' else "📊"
            
            message = f"""
{emoji} <b>TRADE EXECUTED</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Action:</b> {action}
<b>Quantity:</b> {quantity:,} shares
<b>Price:</b> ₹{price:.2f}
<b>Total Value:</b> {format_currency(total_value)}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<b>Portfolio Impact:</b>
Position Size: {trade.get('position_percent', 0):.1%} of portfolio
            """
            
            await self._send_message(message.strip())
            self.logger.info(f"Sent trade execution alert for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error sending trade execution alert: {str(e)}")
    
    async def send_daily_summary(self, summary: Dict[str, Any]) -> None:
        """
        Send daily performance summary.
        
        Args:
            summary: Daily performance summary
        """
        if not self.send_daily_summary:
            return
        
        try:
            date = summary.get('date', datetime.now().strftime('%Y-%m-%d'))
            starting_value = summary.get('starting_value', 0)
            ending_value = summary.get('ending_value', 0)
            daily_pnl = summary.get('daily_pnl', 0)
            daily_pnl_percent = summary.get('daily_pnl_percent', 0)
            total_trades = summary.get('total_trades', 0)
            winning_trades = summary.get('winning_trades', 0)
            win_rate = summary.get('win_rate', 0)
            
            # Determine emoji based on performance
            performance_emoji = "📈" if daily_pnl > 0 else "📉" if daily_pnl < 0 else "➡️"
            
            message = f"""
{performance_emoji} <b>DAILY SUMMARY - {date}</b> {performance_emoji}

<b>Portfolio Performance:</b>
Starting Value: {format_currency(starting_value)}
Ending Value: {format_currency(ending_value)}
Daily P&L: {format_currency(daily_pnl)} ({format_percentage(daily_pnl_percent)})

<b>Trading Activity:</b>
Total Trades: {total_trades}
Winning Trades: {winning_trades}
Win Rate: {format_percentage(win_rate)}

<b>Market Status:</b>
{self._get_market_status_text()}

<i>💡 Keep following your strategy and managing risk!</i>
            """
            
            await self._send_message(message.strip(), silent=True)
            self.logger.info("Sent daily summary")
            
        except Exception as e:
            self.logger.error(f"Error sending daily summary: {str(e)}")
    
    async def send_backtest_summary(self, results: Dict[str, Any]) -> None:
        """
        Send backtest results summary.
        
        Args:
            results: Backtest results
        """
        try:
            symbol = results.get('symbol', 'UNKNOWN')
            start_date = results.get('start_date', '')
            end_date = results.get('end_date', '')
            total_return = results.get('total_return', 0)
            sharpe_ratio = results.get('sharpe_ratio', 0)
            max_drawdown = results.get('max_drawdown', 0)
            win_rate = results.get('win_rate', 0)
            total_trades = results.get('total_trades', 0)
            
            message = f"""
📊 <b>BACKTEST RESULTS</b> 📊

<b>Symbol:</b> {symbol}
<b>Period:</b> {start_date} to {end_date}

<b>Performance Metrics:</b>
Total Return: {format_percentage(total_return)}
Sharpe Ratio: {sharpe_ratio:.2f}
Max Drawdown: {format_percentage(max_drawdown)}
Win Rate: {format_percentage(win_rate)}
Total Trades: {total_trades}

<b>Strategy:</b> RSI + Moving Average Crossover

<i>📈 Historical performance doesn't guarantee future results.</i>
            """
            
            await self._send_message(message.strip())
            self.logger.info(f"Sent backtest summary for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error sending backtest summary: {str(e)}")
    
    async def send_error_alert(self, error_message: str, severity: str = 'ERROR') -> None:
        """
        Send error alert.
        
        Args:
            error_message: Error message
            severity: Error severity level
        """
        if not self.send_errors:
            return
        
        try:
            emoji = "🚨" if severity == 'CRITICAL' else "⚠️" if severity == 'WARNING' else "❌"
            
            message = f"""
{emoji} <b>{severity}</b> {emoji}

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<b>Error Details:</b>
{error_message}

<b>System Status:</b>
Market Open: {'Yes' if is_market_open() else 'No'}

<i>🔧 Please check the system logs for more details.</i>
            """
            
            await self._send_message(message.strip())
            self.logger.info(f"Sent error alert: {severity}")
            
        except Exception as e:
            self.logger.error(f"Error sending error alert: {str(e)}")
    
    async def send_startup_notification(self) -> None:
        """Send system startup notification."""
        try:
            message = f"""
🚀 <b>ALGO TRADING SYSTEM STARTED</b> 🚀

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<b>System Status:</b>
Market Open: {'Yes' if is_market_open() else 'No'}
Configuration: Loaded ✅
Data Connection: Active ✅
Strategies: Loaded ✅

<b>Active Strategies:</b>
• RSI + Moving Average Crossover
• ML Predictions (Decision Tree, Logistic Regression)

<i>📊 System is ready for trading!</i>
            """
            
            await self._send_message(message.strip())
            self.logger.info("Sent startup notification")
            
        except Exception as e:
            self.logger.error(f"Error sending startup notification: {str(e)}")
    
    async def send_shutdown_notification(self) -> None:
        """Send system shutdown notification."""
        try:
            message = f"""
🛑 <b>ALGO TRADING SYSTEM SHUTDOWN</b> 🛑

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<b>Final Status:</b>
All positions monitored ✅
Data saved ✅
Logs updated ✅

<i>💤 System going offline. See you next trading session!</i>
            """
            
            await self._send_message(message.strip())
            self.logger.info("Sent shutdown notification")
            
        except Exception as e:
            self.logger.error(f"Error sending shutdown notification: {str(e)}")
    
    async def send_weekly_performance(self, performance: Dict[str, Any]) -> None:
        """
        Send weekly performance report.
        
        Args:
            performance: Weekly performance data
        """
        try:
            week_return = performance.get('weekly_return', 0)
            total_trades = performance.get('total_trades', 0)
            win_rate = performance.get('win_rate', 0)
            best_trade = performance.get('best_trade', 0)
            worst_trade = performance.get('worst_trade', 0)
            
            message = f"""
📅 <b>WEEKLY PERFORMANCE REPORT</b> 📅

<b>Performance Overview:</b>
Weekly Return: {format_percentage(week_return)}
Total Trades: {total_trades}
Win Rate: {format_percentage(win_rate)}

<b>Trade Analysis:</b>
Best Trade: {format_currency(best_trade)}
Worst Trade: {format_currency(worst_trade)}

<b>Portfolio Health:</b>
Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}
Max Drawdown: {format_percentage(performance.get('max_drawdown', 0))}

<i>📈 Keep up the systematic approach!</i>
            """
            
            await self._send_message(message.strip(), silent=True)
            self.logger.info("Sent weekly performance report")
            
        except Exception as e:
            self.logger.error(f"Error sending weekly performance: {str(e)}")
    
    async def send_custom_alert(self, title: str, message: str, 
                               priority: str = 'normal') -> None:
        """
        Send custom alert message.
        
        Args:
            title: Alert title
            message: Alert message
            priority: Priority level ('low', 'normal', 'high', 'critical')
        """
        try:
            emoji_map = {
                'low': 'ℹ️',
                'normal': '📢',
                'high': '⚠️',
                'critical': '🚨'
            }
            
            emoji = emoji_map.get(priority, '📢')
            
            formatted_message = f"""
{emoji} <b>{title.upper()}</b> {emoji}

{message}

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
<b>Priority:</b> {priority.upper()}
            """
            
            silent = priority in ['low', 'normal']
            await self._send_message(formatted_message.strip(), silent=silent)
            self.logger.info(f"Sent custom alert: {title}")
            
        except Exception as e:
            self.logger.error(f"Error sending custom alert: {str(e)}")
    
    def _get_market_status_text(self) -> str:
        """Get market status text."""
        if is_market_open():
            return "🟢 Market is OPEN"
        else:
            return "🔴 Market is CLOSED"
    
    async def test_connection(self) -> bool:
        """
        Test Telegram bot connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if not self.bot:
                return False
            
            # Send test message
            test_message = f"""
🧪 <b>TELEGRAM BOT TEST</b> 🧪

Connection test successful!
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<i>Your algo trading alerts are ready! 🚀</i>
            """
            
            result = await self._send_message(test_message.strip())
            
            if result:
                self.logger.info("Telegram bot connection test successful")
            else:
                self.logger.error("Telegram bot connection test failed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error testing Telegram connection: {str(e)}")
            return False
    
    def is_configured(self) -> bool:
        """Check if Telegram bot is properly configured."""
        return self.bot is not None and self.chat_id is not None
