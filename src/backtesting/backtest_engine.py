"""
Backtesting engine for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import asyncio
from datetime import datetime, timedelta
import logging

from ..utils.logger import setup_logger, trading_logger
from ..utils.config import Config
from ..utils.helpers import calculate_returns, calculate_sharpe_ratio, calculate_max_drawdown
from ..data.data_fetcher import DataFetcher
from ..strategies.base_strategy import BaseStrategy
from .performance_metrics import PerformanceAnalyzer


class BacktestEngine:
    """Comprehensive backtesting engine for trading strategies."""
    
    def __init__(self, strategy: BaseStrategy, initial_capital: float = 100000):
        """
        Initialize backtest engine.
        
        Args:
            strategy: Trading strategy to backtest
            initial_capital: Starting capital for backtesting
        """
        self.logger = setup_logger(__name__)
        self.config = Config()
        self.trading_config = self.config.get_trading_config()
        
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.data_fetcher = DataFetcher()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Backtest state
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: {'quantity': int, 'avg_price': float}}
        self.trades = []
        self.portfolio_history = []
        self.signals_history = []
        
        # Trading costs and constraints
        self.commission_rate = self.trading_config.get('commission', 0.001)  # 0.1%
        self.slippage_rate = 0.001  # 0.1% slippage
        self.max_position_size = self.trading_config.get('max_position_size', 0.1)
        
        self.logger.info(f"Backtest engine initialized with ₹{initial_capital:,.2f} capital")
    
    async def run_backtest(self, symbol: str, start_date: str, end_date: str,
                          benchmark: str = "^NSEI") -> Dict[str, Any]:
        """
        Run comprehensive backtest for a symbol.
        
        Args:
            symbol: Stock symbol to backtest
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            benchmark: Benchmark symbol for comparison
        
        Returns:
            Backtest results dictionary
        """
        try:
            self.logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
            
            # Reset backtest state
            self._reset_backtest()
            
            # Fetch historical data
            data = await self.data_fetcher.fetch_stock_data(
                symbol, start_date=start_date, end_date=end_date
            )
            
            if data.empty:
                raise ValueError(f"No data available for {symbol} in the specified period")
            
            # Fetch benchmark data
            benchmark_data = None
            try:
                benchmark_data = await self.data_fetcher.fetch_stock_data(
                    benchmark, start_date=start_date, end_date=end_date
                )
            except Exception as e:
                self.logger.warning(f"Could not fetch benchmark data: {str(e)}")
            
            # Add technical indicators
            from ..strategies.technical_indicators import TechnicalIndicators
            tech_indicators = TechnicalIndicators()
            data = tech_indicators.calculate_all_indicators(data)
            
            # Run backtest simulation
            await self._simulate_trading(symbol, data)
            
            # Calculate performance metrics
            results = self._calculate_results(symbol, start_date, end_date, benchmark_data)
            
            # Generate detailed analysis
            detailed_analysis = self.performance_analyzer.analyze_backtest_results(
                self.trades, self.portfolio_history, results
            )
            
            results.update(detailed_analysis)
            
            self.logger.info(f"Backtest completed for {symbol}. Total return: {results['total_return']:.2%}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}")
            raise
    
    async def _simulate_trading(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Simulate trading based on strategy signals.
        
        Args:
            symbol: Stock symbol
            data: Historical price data with indicators
        """
        for i in range(50, len(data)):  # Start after warm-up period for indicators
            current_date = data.index[i]
            current_data = data.iloc[:i+1]  # Data up to current point
            
            # Generate signal
            signal = self.strategy.generate_signal(current_data)
            
            # Record signal
            self.signals_history.append({
                'date': current_date,
                'symbol': symbol,
                'signal': signal.copy()
            })
            
            # Execute trades based on signal
            if signal['action'] in ['BUY', 'SELL']:
                await self._execute_trade(symbol, signal, data.iloc[i], current_date)
            
            # Update portfolio value
            self._update_portfolio_value(symbol, data.iloc[i]['Close'], current_date)
    
    async def _execute_trade(self, symbol: str, signal: Dict[str, Any], 
                           price_data: pd.Series, date: datetime) -> None:
        """
        Execute a trade based on signal.
        
        Args:
            symbol: Stock symbol
            signal: Trading signal
            price_data: Current price data
            date: Current date
        """
        try:
            action = signal['action']
            price = price_data['Close']
            
            # Apply slippage
            if action == 'BUY':
                execution_price = price * (1 + self.slippage_rate)
            else:  # SELL
                execution_price = price * (1 - self.slippage_rate)
            
            # Calculate position size
            position_size = self.strategy.calculate_position_size(signal, self.portfolio_value)
            
            if position_size <= 0:
                return
            
            # Limit position size
            max_value = self.portfolio_value * self.max_position_size
            if position_size * execution_price > max_value:
                position_size = int(max_value / execution_price)
            
            if action == 'BUY':
                await self._execute_buy(symbol, position_size, execution_price, signal, date)
            elif action == 'SELL':
                await self._execute_sell(symbol, position_size, execution_price, signal, date)
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
    
    async def _execute_buy(self, symbol: str, quantity: int, price: float,
                          signal: Dict[str, Any], date: datetime) -> None:
        """Execute buy order."""
        total_cost = quantity * price
        commission = total_cost * self.commission_rate
        total_with_commission = total_cost + commission
        
        # Check if we have enough cash
        if total_with_commission > self.cash:
            # Adjust quantity based on available cash
            available_for_trade = self.cash * 0.98  # Keep 2% buffer
            quantity = int(available_for_trade / (price * (1 + self.commission_rate)))
            
            if quantity <= 0:
                return
            
            total_cost = quantity * price
            commission = total_cost * self.commission_rate
            total_with_commission = total_cost + commission
        
        # Update cash
        self.cash -= total_with_commission
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
        
        current_qty = self.positions[symbol]['quantity']
        current_avg = self.positions[symbol]['avg_price']
        
        new_qty = current_qty + quantity
        new_avg = ((current_qty * current_avg) + (quantity * price)) / new_qty
        
        self.positions[symbol] = {'quantity': new_qty, 'avg_price': new_avg}
        
        # Record trade
        trade = {
            'date': date,
            'symbol': symbol,
            'action': 'BUY',
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'total_cost': total_with_commission,
            'signal_confidence': signal.get('confidence', 0),
            'portfolio_value_before': self.portfolio_value,
            'cash_after': self.cash,
            'position_after': self.positions[symbol]['quantity']
        }
        
        self.trades.append(trade)
        
        # Log trade
        trading_logger.log_trade(
            symbol, 'BUY', price, quantity, date.strftime('%Y-%m-%d %H:%M:%S'),
            self.strategy.name, signal.get('confidence', 0)
        )
    
    async def _execute_sell(self, symbol: str, quantity: int, price: float,
                           signal: Dict[str, Any], date: datetime) -> None:
        """Execute sell order."""
        # Check if we have the position to sell
        if symbol not in self.positions or self.positions[symbol]['quantity'] <= 0:
            return
        
        # Limit quantity to available shares
        available_qty = self.positions[symbol]['quantity']
        quantity = min(quantity, available_qty)
        
        if quantity <= 0:
            return
        
        total_proceeds = quantity * price
        commission = total_proceeds * self.commission_rate
        net_proceeds = total_proceeds - commission
        
        # Calculate P&L
        avg_buy_price = self.positions[symbol]['avg_price']
        pnl = (price - avg_buy_price) * quantity - commission
        pnl_percent = pnl / (avg_buy_price * quantity) if avg_buy_price > 0 else 0
        
        # Update cash
        self.cash += net_proceeds
        
        # Update position
        new_qty = self.positions[symbol]['quantity'] - quantity
        if new_qty <= 0:
            del self.positions[symbol]
        else:
            self.positions[symbol]['quantity'] = new_qty
        
        # Record trade
        trade = {
            'date': date,
            'symbol': symbol,
            'action': 'SELL',
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'net_proceeds': net_proceeds,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'signal_confidence': signal.get('confidence', 0),
            'portfolio_value_before': self.portfolio_value,
            'cash_after': self.cash,
            'avg_buy_price': avg_buy_price
        }
        
        self.trades.append(trade)
        
        # Log trade
        trading_logger.log_trade(
            symbol, 'SELL', price, quantity, date.strftime('%Y-%m-%d %H:%M:%S'),
            self.strategy.name, signal.get('confidence', 0), pnl=pnl
        )
    
    def _update_portfolio_value(self, symbol: str, current_price: float, date: datetime) -> None:
        """Update portfolio value and record history."""
        # Calculate position values
        position_value = 0
        if symbol in self.positions:
            position_value = self.positions[symbol]['quantity'] * current_price
        
        # Total portfolio value
        self.portfolio_value = self.cash + position_value
        
        # Record portfolio history
        self.portfolio_history.append({
            'date': date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position_value': position_value,
            'current_price': current_price
        })
    
    def _calculate_results(self, symbol: str, start_date: str, end_date: str,
                          benchmark_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate backtest results and metrics."""
        if not self.portfolio_history:
            return {}
        
        # Create portfolio series
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        # Basic metrics
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        
        # Risk metrics
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = calculate_sharpe_ratio(portfolio_returns)
        max_drawdown, dd_start, dd_end = calculate_max_drawdown(portfolio_df['portfolio_value'])
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        
        total_trades = len([t for t in self.trades if 'pnl' in t])
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if avg_loss != 0 else float('inf')
        
        # Benchmark comparison
        benchmark_return = 0
        beta = 0
        alpha = 0
        
        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_start = benchmark_data['Close'].iloc[0]
            benchmark_end = benchmark_data['Close'].iloc[-1]
            benchmark_return = (benchmark_end - benchmark_start) / benchmark_start
            
            # Calculate beta and alpha
            benchmark_returns = benchmark_data['Close'].pct_change().dropna()
            if len(benchmark_returns) > 1 and len(portfolio_returns) > 1:
                # Align returns
                common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
                if len(common_dates) > 10:
                    portfolio_aligned = portfolio_returns.loc[common_dates]
                    benchmark_aligned = benchmark_returns.loc[common_dates]
                    
                    covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
                    benchmark_variance = np.var(benchmark_aligned)
                    
                    if benchmark_variance > 0:
                        beta = covariance / benchmark_variance
                        alpha = portfolio_aligned.mean() - beta * benchmark_aligned.mean()
        
        results = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'strategy': self.strategy.name,
            'initial_capital': self.initial_capital,
            'final_value': self.portfolio_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_start': dd_start,
            'max_drawdown_end': dd_end,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'best_trade': max([t.get('pnl', 0) for t in self.trades]) if self.trades else 0,
            'worst_trade': min([t.get('pnl', 0) for t in self.trades]) if self.trades else 0,
            'benchmark_return': benchmark_return,
            'alpha': alpha,
            'beta': beta,
            'excess_return': total_return - benchmark_return,
            'portfolio_history': self.portfolio_history,
            'trades': self.trades,
            'signals_history': self.signals_history
        }
        
        return results
    
    def _reset_backtest(self) -> None:
        """Reset backtest state for new run."""
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.portfolio_history.clear()
        self.signals_history.clear()
    
    async def run_parameter_optimization(self, symbol: str, start_date: str, end_date: str,
                                       parameter_ranges: Dict[str, List]) -> Dict[str, Any]:
        """
        Run parameter optimization for strategy.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            parameter_ranges: Dictionary of parameter ranges to test
        
        Returns:
            Optimization results
        """
        try:
            self.logger.info(f"Starting parameter optimization for {symbol}")
            
            # Generate parameter combinations
            from itertools import product
            
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            param_combinations = list(product(*param_values))
            
            results = []
            
            for i, params in enumerate(param_combinations):
                try:
                    # Set strategy parameters
                    param_dict = dict(zip(param_names, params))
                    self.strategy.set_parameters(param_dict)
                    
                    # Run backtest
                    backtest_result = await self.run_backtest(symbol, start_date, end_date)
                    
                    # Store result with parameters
                    result = {
                        'parameters': param_dict.copy(),
                        'total_return': backtest_result['total_return'],
                        'sharpe_ratio': backtest_result['sharpe_ratio'],
                        'max_drawdown': backtest_result['max_drawdown'],
                        'win_rate': backtest_result['win_rate'],
                        'total_trades': backtest_result['total_trades']
                    }
                    
                    results.append(result)
                    
                    self.logger.debug(f"Optimization {i+1}/{len(param_combinations)}: "
                                    f"Return={result['total_return']:.2%}, "
                                    f"Sharpe={result['sharpe_ratio']:.2f}")
                    
                except Exception as e:
                    self.logger.warning(f"Error in optimization iteration {i}: {str(e)}")
                    continue
            
            if not results:
                raise ValueError("No successful optimization runs")
            
            # Find best parameters based on Sharpe ratio
            best_result = max(results, key=lambda x: x['sharpe_ratio'])
            
            optimization_summary = {
                'symbol': symbol,
                'total_combinations': len(param_combinations),
                'successful_runs': len(results),
                'best_parameters': best_result['parameters'],
                'best_performance': {
                    'total_return': best_result['total_return'],
                    'sharpe_ratio': best_result['sharpe_ratio'],
                    'max_drawdown': best_result['max_drawdown'],
                    'win_rate': best_result['win_rate']
                },
                'all_results': results
            }
            
            self.logger.info(f"Parameter optimization completed. Best Sharpe ratio: {best_result['sharpe_ratio']:.2f}")
            return optimization_summary
            
        except Exception as e:
            self.logger.error(f"Error in parameter optimization: {str(e)}")
            raise
    
    async def run_walk_forward_analysis(self, symbol: str, start_date: str, end_date: str,
                                      train_period_months: int = 6, 
                                      test_period_months: int = 1) -> Dict[str, Any]:
        """
        Run walk-forward analysis to test strategy robustness.
        
        Args:
            symbol: Stock symbol
            start_date: Analysis start date
            end_date: Analysis end date
            train_period_months: Training period in months
            test_period_months: Testing period in months
        
        Returns:
            Walk-forward analysis results
        """
        try:
            self.logger.info(f"Starting walk-forward analysis for {symbol}")
            
            # Parse dates
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            results = []
            current_start = start
            
            while current_start < end:
                # Define training period
                train_end = current_start + timedelta(days=train_period_months * 30)
                
                # Define test period
                test_start = train_end
                test_end = test_start + timedelta(days=test_period_months * 30)
                
                if test_end > end:
                    test_end = end
                
                if test_start >= end:
                    break
                
                try:
                    # Run backtest on test period
                    backtest_result = await self.run_backtest(
                        symbol,
                        test_start.strftime('%Y-%m-%d'),
                        test_end.strftime('%Y-%m-%d')
                    )
                    
                    period_result = {
                        'train_start': current_start.strftime('%Y-%m-%d'),
                        'train_end': train_end.strftime('%Y-%m-%d'),
                        'test_start': test_start.strftime('%Y-%m-%d'),
                        'test_end': test_end.strftime('%Y-%m-%d'),
                        'total_return': backtest_result['total_return'],
                        'sharpe_ratio': backtest_result['sharpe_ratio'],
                        'max_drawdown': backtest_result['max_drawdown'],
                        'win_rate': backtest_result['win_rate'],
                        'total_trades': backtest_result['total_trades']
                    }
                    
                    results.append(period_result)
                    
                    self.logger.debug(f"Walk-forward period {test_start.strftime('%Y-%m')} to "
                                    f"{test_end.strftime('%Y-%m')}: Return={period_result['total_return']:.2%}")
                    
                except Exception as e:
                    self.logger.warning(f"Error in walk-forward period: {str(e)}")
                
                # Move to next period
                current_start = test_start
            
            if not results:
                raise ValueError("No successful walk-forward periods")
            
            # Calculate summary statistics
            returns = [r['total_return'] for r in results]
            sharpe_ratios = [r['sharpe_ratio'] for r in results]
            
            summary = {
                'symbol': symbol,
                'analysis_period': f"{start_date} to {end_date}",
                'total_periods': len(results),
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'avg_sharpe': np.mean(sharpe_ratios),
                'std_sharpe': np.std(sharpe_ratios),
                'positive_periods': len([r for r in returns if r > 0]),
                'consistency_ratio': len([r for r in returns if r > 0]) / len(returns),
                'period_results': results
            }
            
            self.logger.info(f"Walk-forward analysis completed. Consistency ratio: {summary['consistency_ratio']:.2%}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward analysis: {str(e)}")
            raise
