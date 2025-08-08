"""
Performance metrics and analytics for backtesting and live trading.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

from ..utils.logger import setup_logger
from ..utils.helpers import calculate_sharpe_ratio, calculate_max_drawdown, format_currency, format_percentage


class PerformanceAnalyzer:
    """Comprehensive performance analysis for trading strategies."""
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.logger = setup_logger(__name__)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def analyze_backtest_results(self, trades: List[Dict[str, Any]], 
                                portfolio_history: List[Dict[str, Any]],
                                basic_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of backtest results.
        
        Args:
            trades: List of trades executed
            portfolio_history: Portfolio value history
            basic_metrics: Basic performance metrics
        
        Returns:
            Detailed analysis results
        """
        try:
            analysis = {}
            
            if not trades or not portfolio_history:
                return analysis
            
            # Convert to DataFrames for analysis
            trades_df = pd.DataFrame(trades)
            portfolio_df = pd.DataFrame(portfolio_history)
            portfolio_df.set_index('date', inplace=True)
            
            # Trade analysis
            analysis['trade_analysis'] = self._analyze_trades(trades_df)
            
            # Portfolio performance analysis
            analysis['portfolio_analysis'] = self._analyze_portfolio_performance(portfolio_df)
            
            # Risk analysis
            analysis['risk_analysis'] = self._analyze_risk_metrics(portfolio_df, trades_df)
            
            # Drawdown analysis
            analysis['drawdown_analysis'] = self._analyze_drawdowns(portfolio_df)
            
            # Monthly/yearly performance
            analysis['period_analysis'] = self._analyze_period_performance(portfolio_df)
            
            # Strategy effectiveness
            analysis['strategy_analysis'] = self._analyze_strategy_effectiveness(trades_df)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in performance analysis: {str(e)}")
            return {}
    
    def _analyze_trades(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual trades."""
        if trades_df.empty:
            return {}
        
        # Filter to only trades with P&L (sell trades)
        pnl_trades = trades_df[trades_df['action'] == 'SELL'].copy()
        
        if pnl_trades.empty:
            return {'message': 'No completed trades found'}
        
        # Basic trade statistics
        total_trades = len(pnl_trades)
        winning_trades = len(pnl_trades[pnl_trades['pnl'] > 0])
        losing_trades = len(pnl_trades[pnl_trades['pnl'] < 0])
        
        # P&L statistics
        total_pnl = pnl_trades['pnl'].sum()
        avg_pnl = pnl_trades['pnl'].mean()
        avg_win = pnl_trades[pnl_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = pnl_trades[pnl_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Trade duration analysis (if we have holding periods)
        # This would require matching buy/sell pairs - simplified for now
        
        # Confidence analysis
        if 'signal_confidence' in pnl_trades.columns:
            avg_confidence_winners = pnl_trades[pnl_trades['pnl'] > 0]['signal_confidence'].mean()
            avg_confidence_losers = pnl_trades[pnl_trades['pnl'] < 0]['signal_confidence'].mean()
        else:
            avg_confidence_winners = avg_confidence_losers = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else float('inf'),
            'best_trade': pnl_trades['pnl'].max(),
            'worst_trade': pnl_trades['pnl'].min(),
            'avg_confidence_winners': avg_confidence_winners,
            'avg_confidence_losers': avg_confidence_losers,
            'pnl_std': pnl_trades['pnl'].std(),
            'total_commission': pnl_trades['commission'].sum()
        }
    
    def _analyze_portfolio_performance(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze portfolio performance over time."""
        if portfolio_df.empty:
            return {}
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        portfolio_df['cumulative_returns'] = (1 + portfolio_df['returns']).cumprod() - 1
        
        # Basic statistics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] - portfolio_df['portfolio_value'].iloc[0]) / portfolio_df['portfolio_value'].iloc[0]
        
        # Annualized metrics
        days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Volatility
        daily_vol = portfolio_df['returns'].std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        # Risk-adjusted returns
        sharpe_ratio = calculate_sharpe_ratio(portfolio_df['returns'].dropna())
        
        # Other metrics
        calmar_ratio = annualized_return / abs(calculate_max_drawdown(portfolio_df['portfolio_value'])[0]) if calculate_max_drawdown(portfolio_df['portfolio_value'])[0] != 0 else 0
        
        # Consistency metrics
        positive_days = len(portfolio_df[portfolio_df['returns'] > 0])
        total_days = len(portfolio_df['returns'].dropna())
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'daily_volatility': daily_vol,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'positive_days': positive_days,
            'total_trading_days': total_days,
            'positive_day_ratio': positive_days / total_days if total_days > 0 else 0,
            'best_day': portfolio_df['returns'].max(),
            'worst_day': portfolio_df['returns'].min(),
            'avg_daily_return': portfolio_df['returns'].mean(),
            'skewness': portfolio_df['returns'].skew(),
            'kurtosis': portfolio_df['returns'].kurtosis()
        }
    
    def _analyze_risk_metrics(self, portfolio_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze risk metrics."""
        if portfolio_df.empty:
            return {}
        
        returns = portfolio_df['returns'].dropna()
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Maximum drawdown
        max_dd, dd_start, dd_end = calculate_max_drawdown(portfolio_df['portfolio_value'])
        
        # Drawdown duration
        dd_duration = (dd_end - dd_start).days if dd_start and dd_end else 0
        
        # Ulcer Index (measure of downside risk)
        portfolio_df['running_max'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['running_max']) / portfolio_df['running_max']
        ulcer_index = np.sqrt(np.mean(portfolio_df['drawdown'] ** 2))
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0
        
        # Sortino ratio
        sortino_ratio = returns.mean() / downside_deviation if downside_deviation != 0 else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_drawdown': max_dd,
            'max_drawdown_duration_days': dd_duration,
            'ulcer_index': ulcer_index,
            'downside_deviation': downside_deviation,
            'sortino_ratio': sortino_ratio,
            'tail_ratio': abs(returns.quantile(0.95) / returns.quantile(0.05)) if returns.quantile(0.05) != 0 else 0
        }
    
    def _analyze_drawdowns(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze drawdown patterns."""
        if portfolio_df.empty:
            return {}
        
        # Calculate drawdowns
        portfolio_df['running_max'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['running_max']) / portfolio_df['running_max']
        
        # Find drawdown periods
        in_drawdown = portfolio_df['drawdown'] < 0
        drawdown_periods = []
        
        start_dd = None
        for i, (date, is_dd) in enumerate(in_drawdown.items()):
            if is_dd and start_dd is None:
                start_dd = date
            elif not is_dd and start_dd is not None:
                drawdown_periods.append({
                    'start': start_dd,
                    'end': date,
                    'duration_days': (date - start_dd).days,
                    'max_drawdown': portfolio_df.loc[start_dd:date, 'drawdown'].min()
                })
                start_dd = None
        
        # Handle ongoing drawdown
        if start_dd is not None:
            drawdown_periods.append({
                'start': start_dd,
                'end': portfolio_df.index[-1],
                'duration_days': (portfolio_df.index[-1] - start_dd).days,
                'max_drawdown': portfolio_df.loc[start_dd:, 'drawdown'].min()
            })
        
        if not drawdown_periods:
            return {'message': 'No significant drawdown periods found'}
        
        # Analyze drawdown statistics
        dd_durations = [dd['duration_days'] for dd in drawdown_periods]
        dd_magnitudes = [abs(dd['max_drawdown']) for dd in drawdown_periods]
        
        return {
            'total_drawdown_periods': len(drawdown_periods),
            'avg_drawdown_duration': np.mean(dd_durations),
            'max_drawdown_duration': max(dd_durations),
            'avg_drawdown_magnitude': np.mean(dd_magnitudes),
            'max_drawdown_magnitude': max(dd_magnitudes),
            'time_in_drawdown_percent': sum(dd_durations) / len(portfolio_df) * 100,
            'drawdown_periods': drawdown_periods[-5:]  # Last 5 drawdown periods
        }
    
    def _analyze_period_performance(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by different time periods."""
        if portfolio_df.empty:
            return {}
        
        # Monthly performance
        monthly_returns = portfolio_df['portfolio_value'].resample('M').last().pct_change().dropna()
        
        # Quarterly performance  
        quarterly_returns = portfolio_df['portfolio_value'].resample('Q').last().pct_change().dropna()
        
        # Yearly performance
        yearly_returns = portfolio_df['portfolio_value'].resample('Y').last().pct_change().dropna()
        
        analysis = {}
        
        # Monthly analysis
        if len(monthly_returns) > 0:
            analysis['monthly'] = {
                'avg_return': monthly_returns.mean(),
                'std_return': monthly_returns.std(),
                'best_month': monthly_returns.max(),
                'worst_month': monthly_returns.min(),
                'positive_months': len(monthly_returns[monthly_returns > 0]),
                'total_months': len(monthly_returns),
                'positive_month_ratio': len(monthly_returns[monthly_returns > 0]) / len(monthly_returns)
            }
        
        # Quarterly analysis
        if len(quarterly_returns) > 0:
            analysis['quarterly'] = {
                'avg_return': quarterly_returns.mean(),
                'std_return': quarterly_returns.std(),
                'best_quarter': quarterly_returns.max(),
                'worst_quarter': quarterly_returns.min(),
                'positive_quarters': len(quarterly_returns[quarterly_returns > 0]),
                'total_quarters': len(quarterly_returns)
            }
        
        # Yearly analysis
        if len(yearly_returns) > 0:
            analysis['yearly'] = {
                'avg_return': yearly_returns.mean(),
                'std_return': yearly_returns.std(),
                'best_year': yearly_returns.max(),
                'worst_year': yearly_returns.min(),
                'positive_years': len(yearly_returns[yearly_returns > 0]),
                'total_years': len(yearly_returns)
            }
        
        return analysis
    
    def _analyze_strategy_effectiveness(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze strategy-specific effectiveness metrics."""
        if trades_df.empty:
            return {}
        
        pnl_trades = trades_df[trades_df['action'] == 'SELL'].copy()
        
        if pnl_trades.empty:
            return {}
        
        analysis = {}
        
        # Confidence-based analysis
        if 'signal_confidence' in pnl_trades.columns:
            # Bin trades by confidence levels
            confidence_bins = pd.cut(pnl_trades['signal_confidence'], 
                                   bins=[0, 0.5, 0.7, 0.9, 1.0], 
                                   labels=['Low', 'Medium', 'High', 'Very High'])
            
            confidence_analysis = pnl_trades.groupby(confidence_bins).agg({
                'pnl': ['count', 'mean', 'sum', lambda x: (x > 0).sum()],
                'pnl_percent': 'mean'
            }).round(4)
            
            analysis['confidence_analysis'] = confidence_analysis.to_dict()
        
        # Time-based analysis
        if 'date' in pnl_trades.columns:
            pnl_trades['hour'] = pd.to_datetime(pnl_trades['date']).dt.hour
            pnl_trades['day_of_week'] = pd.to_datetime(pnl_trades['date']).dt.dayofweek
            
            # Performance by hour of day
            hourly_performance = pnl_trades.groupby('hour')['pnl'].agg(['count', 'mean', 'sum']).round(2)
            analysis['hourly_performance'] = hourly_performance.to_dict()
            
            # Performance by day of week
            daily_performance = pnl_trades.groupby('day_of_week')['pnl'].agg(['count', 'mean', 'sum']).round(2)
            analysis['daily_performance'] = daily_performance.to_dict()
        
        # Symbol-based analysis (if multiple symbols)
        if 'symbol' in pnl_trades.columns and pnl_trades['symbol'].nunique() > 1:
            symbol_performance = pnl_trades.groupby('symbol').agg({
                'pnl': ['count', 'mean', 'sum', lambda x: (x > 0).sum()],
                'pnl_percent': 'mean'
            }).round(4)
            
            analysis['symbol_performance'] = symbol_performance.to_dict()
        
        return analysis
    
    def create_performance_report(self, analysis_results: Dict[str, Any],
                                basic_metrics: Dict[str, Any]) -> str:
        """
        Create a comprehensive text report of performance analysis.
        
        Args:
            analysis_results: Detailed analysis results
            basic_metrics: Basic performance metrics
        
        Returns:
            Formatted performance report
        """
        try:
            report = []
            report.append("=" * 80)
            report.append("TRADING STRATEGY PERFORMANCE REPORT")
            report.append("=" * 80)
            report.append("")
            
            # Basic Metrics
            report.append("📊 BASIC PERFORMANCE METRICS")
            report.append("-" * 40)
            report.append(f"Initial Capital: {format_currency(basic_metrics.get('initial_capital', 0))}")
            report.append(f"Final Value: {format_currency(basic_metrics.get('final_value', 0))}")
            report.append(f"Total Return: {format_percentage(basic_metrics.get('total_return', 0))}")
            report.append(f"Annualized Return: {format_percentage(basic_metrics.get('annualized_return', 0))}")
            report.append(f"Sharpe Ratio: {basic_metrics.get('sharpe_ratio', 0):.2f}")
            report.append(f"Maximum Drawdown: {format_percentage(basic_metrics.get('max_drawdown', 0))}")
            report.append("")
            
            # Trade Analysis
            if 'trade_analysis' in analysis_results:
                trade_analysis = analysis_results['trade_analysis']
                report.append("📈 TRADE ANALYSIS")
                report.append("-" * 40)
                report.append(f"Total Trades: {trade_analysis.get('total_trades', 0)}")
                report.append(f"Winning Trades: {trade_analysis.get('winning_trades', 0)}")
                report.append(f"Win Rate: {format_percentage(trade_analysis.get('win_rate', 0))}")
                report.append(f"Average Win: {format_currency(trade_analysis.get('avg_win', 0))}")
                report.append(f"Average Loss: {format_currency(trade_analysis.get('avg_loss', 0))}")
                report.append(f"Profit Factor: {trade_analysis.get('profit_factor', 0):.2f}")
                report.append(f"Best Trade: {format_currency(trade_analysis.get('best_trade', 0))}")
                report.append(f"Worst Trade: {format_currency(trade_analysis.get('worst_trade', 0))}")
                report.append("")
            
            # Risk Analysis
            if 'risk_analysis' in analysis_results:
                risk_analysis = analysis_results['risk_analysis']
                report.append("⚠️ RISK ANALYSIS")
                report.append("-" * 40)
                report.append(f"Value at Risk (95%): {format_percentage(risk_analysis.get('var_95', 0))}")
                report.append(f"Value at Risk (99%): {format_percentage(risk_analysis.get('var_99', 0))}")
                report.append(f"Sortino Ratio: {risk_analysis.get('sortino_ratio', 0):.2f}")
                report.append(f"Maximum Drawdown Duration: {risk_analysis.get('max_drawdown_duration_days', 0)} days")
                report.append("")
            
            # Portfolio Analysis
            if 'portfolio_analysis' in analysis_results:
                portfolio_analysis = analysis_results['portfolio_analysis']
                report.append("💼 PORTFOLIO ANALYSIS")
                report.append("-" * 40)
                report.append(f"Daily Volatility: {format_percentage(portfolio_analysis.get('daily_volatility', 0))}")
                report.append(f"Annualized Volatility: {format_percentage(portfolio_analysis.get('annualized_volatility', 0))}")
                report.append(f"Positive Trading Days: {portfolio_analysis.get('positive_days', 0)} / {portfolio_analysis.get('total_trading_days', 0)}")
                report.append(f"Best Day: {format_percentage(portfolio_analysis.get('best_day', 0))}")
                report.append(f"Worst Day: {format_percentage(portfolio_analysis.get('worst_day', 0))}")
                report.append("")
            
            # Period Performance
            if 'period_analysis' in analysis_results:
                period_analysis = analysis_results['period_analysis']
                if 'monthly' in period_analysis:
                    monthly = period_analysis['monthly']
                    report.append("📅 MONTHLY PERFORMANCE")
                    report.append("-" * 40)
                    report.append(f"Average Monthly Return: {format_percentage(monthly.get('avg_return', 0))}")
                    report.append(f"Best Month: {format_percentage(monthly.get('best_month', 0))}")
                    report.append(f"Worst Month: {format_percentage(monthly.get('worst_month', 0))}")
                    report.append(f"Positive Months: {monthly.get('positive_months', 0)} / {monthly.get('total_months', 0)}")
                    report.append("")
            
            report.append("=" * 80)
            report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error creating performance report: {str(e)}")
            return "Error generating performance report"
    
    def export_analysis_to_excel(self, analysis_results: Dict[str, Any],
                                basic_metrics: Dict[str, Any],
                                filename: str = None) -> str:
        """
        Export analysis results to Excel file.
        
        Args:
            analysis_results: Analysis results dictionary
            basic_metrics: Basic metrics dictionary
            filename: Output filename (optional)
        
        Returns:
            Path to created Excel file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"backtest_analysis_{timestamp}.xlsx"
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Basic metrics
                basic_df = pd.DataFrame([basic_metrics])
                basic_df.to_excel(writer, sheet_name='Basic Metrics', index=False)
                
                # Trade analysis
                if 'trade_analysis' in analysis_results:
                    trade_df = pd.DataFrame([analysis_results['trade_analysis']])
                    trade_df.to_excel(writer, sheet_name='Trade Analysis', index=False)
                
                # Portfolio analysis
                if 'portfolio_analysis' in analysis_results:
                    portfolio_df = pd.DataFrame([analysis_results['portfolio_analysis']])
                    portfolio_df.to_excel(writer, sheet_name='Portfolio Analysis', index=False)
                
                # Risk analysis
                if 'risk_analysis' in analysis_results:
                    risk_df = pd.DataFrame([analysis_results['risk_analysis']])
                    risk_df.to_excel(writer, sheet_name='Risk Analysis', index=False)
                
                # Period analysis
                if 'period_analysis' in analysis_results:
                    period_data = analysis_results['period_analysis']
                    if 'monthly' in period_data:
                        monthly_df = pd.DataFrame([period_data['monthly']])
                        monthly_df.to_excel(writer, sheet_name='Monthly Performance', index=False)
            
            self.logger.info(f"Analysis exported to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error exporting analysis to Excel: {str(e)}")
            return ""
