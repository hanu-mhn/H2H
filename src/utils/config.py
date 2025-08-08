"""
Configuration management for the algo trading system.
"""

import os
import yaml
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging


class Config:
    """Configuration manager for the trading system."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration."""
        self.config_path = config_path or "config/trading_config.yaml"
        self.stocks_path = "config/nifty50_stocks.yaml"
        
        # Initialize without logger first to avoid circular dependencies
        self.logger = None
        
        # Load configurations
        self._load_config()
        self._load_stocks()
        self._load_env_vars()
        
        # Set up logger after configuration is loaded
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> None:
        """Load main configuration file."""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            self.config = {}
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            self.config = {}
    
    def _load_stocks(self) -> None:
        """Load stock symbols configuration."""
        try:
            with open(self.stocks_path, 'r') as file:
                self.stocks_config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Stocks config file not found: {self.stocks_path}")
            self.stocks_config = {'nifty50_stocks': []}
        except yaml.YAMLError as e:
            print(f"Error parsing stocks config file: {e}")
            self.stocks_config = {'nifty50_stocks': []}
    
    def _load_env_vars(self) -> None:
        """Load environment variables."""
        from dotenv import load_dotenv
        load_dotenv()
        
        self.env_vars = {
            'google_sheets_credentials': os.getenv('GOOGLE_SHEETS_CREDENTIALS'),
            'google_sheet_id': os.getenv('GOOGLE_SHEETS_SPREADSHEET_ID'),
            'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
            'trading_mode': os.getenv('TRADING_MODE', 'paper'),
            'initial_capital': float(os.getenv('INITIAL_CAPITAL', '100000')),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'log_file': os.getenv('LOG_FILE', 'logs/trading.log')
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_env(self, key: str, default: Any = None) -> Any:
        """Get environment variable."""
        return self.env_vars.get(key, default)
    
    def get_nifty50_stocks(self) -> List[str]:
        """Get list of NIFTY 50 stock symbols."""
        return self.stocks_config.get('nifty50_stocks', [])
    
    def get_stock_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get metadata for a specific stock."""
        metadata = self.stocks_config.get('stock_metadata', {})
        return metadata.get(symbol, {})
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration."""
        return {
            'mode': self.get_env('trading_mode'),
            'initial_capital': self.get_env('initial_capital'),
            'max_position_size': self.get('trading.max_position_size', 0.1),
            'stop_loss_percent': self.get('trading.stop_loss_percent', 0.05),
            'take_profit_percent': self.get('trading.take_profit_percent', 0.10),
            'commission': self.get('trading.commission', 0.001)
        }
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get strategy configuration."""
        return self.get('strategy', {})
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get ML configuration."""
        return self.get('ml', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get('data', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        base_config = self.get('logging', {})
        base_config.update({
            'level': self.get_env('log_level'),
            'file': self.get_env('log_file')
        })
        return base_config
    
    def get_sheets_config(self) -> Dict[str, Any]:
        """Get Google Sheets configuration."""
        base_config = self.get('google_sheets', {})
        base_config.update({
            'credentials_path': self.get_env('google_sheets_credentials'),
            'sheet_id': self.get_env('google_sheet_id')
        })
        return base_config
    
    def get_telegram_config(self) -> Dict[str, Any]:
        """Get Telegram configuration."""
        base_config = self.get('telegram', {})
        base_config.update({
            'bot_token': self.get_env('telegram_bot_token'),
            'chat_id': self.get_env('telegram_chat_id')
        })
        return base_config
    
    def update_config(self, key: str, value: Any) -> None:
        """Update configuration value."""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            # Ensure directory exists
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            
            self.logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def validate_config(self) -> bool:
        """Validate configuration completeness."""
        required_env_vars = [
            'google_sheets_credentials',
            'google_sheet_id',
            'telegram_bot_token',
            'telegram_chat_id'
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not self.get_env(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.logger.error(f"Missing required environment variables: {missing_vars}")
            return False
        
        return True
