<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Copilot Instructions for Algo-Trading System

## Project Context
This is a Python-based algorithmic trading system that implements:
- RSI and Moving Average crossover trading strategies
- Machine learning models for price prediction
- Automated Google Sheets integration for trade logging
- Telegram bot integration for real-time alerts
- Comprehensive backtesting framework

## Code Style Guidelines
- Follow PEP 8 Python style guidelines
- Use type hints for all function parameters and return values
- Implement comprehensive error handling and logging
- Write docstrings for all classes and functions
- Use async/await for I/O operations where applicable

## Architecture Patterns
- Use dependency injection for external services (Google Sheets, Telegram)
- Implement strategy pattern for trading algorithms
- Use factory pattern for creating different ML models
- Follow single responsibility principle for each module

## Trading Strategy Guidelines
- All trading strategies should inherit from base_strategy.py
- Implement proper risk management (stop-loss, position sizing)
- Include backtesting capabilities for all strategies
- Log all trading decisions with reasoning

## Machine Learning Guidelines
- Feature engineering should be in separate modules
- Implement cross-validation for model training
- Include model performance metrics and validation
- Use ensemble methods when appropriate

## Integration Guidelines
- All external API calls should have retry logic
- Implement rate limiting for API requests
- Use environment variables for sensitive data
- Include comprehensive error handling for third-party services

## Testing Guidelines
- Write unit tests for all core functions
- Include integration tests for external services
- Mock external API calls in tests
- Maintain >80% code coverage

## Logging Guidelines
- Use structured logging with appropriate log levels
- Include trade execution logs with timestamps
- Log all API errors and retry attempts
- Implement log rotation for production use
