"""
Visualization utilities for backtesting and performance analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ..utils.logger import setup_logger


class PerformanceVisualizer:
    """Create visualizations for trading performance analysis."""
    
    def __init__(self):
        """Initialize performance visualizer."""
        self.logger = setup_logger(__name__)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Color scheme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8'
        }
    
    def create_portfolio_performance_chart(self, portfolio_history: List[Dict[str, Any]],
                                         benchmark_data: Optional[List[Dict[str, Any]]] = None,
                                         save_path: Optional[str] = None) -> go.Figure:
        """
        Create portfolio performance chart with benchmark comparison.
        
        Args:
            portfolio_history: Portfolio value history
            benchmark_data: Benchmark data for comparison (optional)
            save_path: Path to save the chart (optional)
        
        Returns:
            Plotly figure object
        """
        try:
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Portfolio Value', 'Daily Returns', 'Cumulative Returns'),
                row_heights=[0.5, 0.25, 0.25],
                vertical_spacing=0.08
            )
            
            # Convert to DataFrame
            portfolio_df = pd.DataFrame(portfolio_history)
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df.set_index('date', inplace=True)
            
            # Calculate returns
            portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
            portfolio_df['cumulative_returns'] = (1 + portfolio_df['returns']).cumprod() - 1
            
            # Portfolio value
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color=self.colors['primary'], width=2)
                ),
                row=1, col=1
            )
            
            # Benchmark comparison if provided
            if benchmark_data:
                benchmark_df = pd.DataFrame(benchmark_data)
                benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
                benchmark_df.set_index('date', inplace=True)
                
                # Normalize benchmark to same starting value
                initial_value = portfolio_df['portfolio_value'].iloc[0]
                benchmark_normalized = (benchmark_df['close'] / benchmark_df['close'].iloc[0]) * initial_value
                
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_df.index,
                        y=benchmark_normalized,
                        mode='lines',
                        name='Benchmark',
                        line=dict(color=self.colors['secondary'], width=2, dash='dash')
                    ),
                    row=1, col=1
                )
            
            # Daily returns
            colors = ['red' if x < 0 else 'green' for x in portfolio_df['returns']]
            fig.add_trace(
                go.Bar(
                    x=portfolio_df.index,
                    y=portfolio_df['returns'] * 100,
                    name='Daily Returns (%)',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Cumulative returns
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['cumulative_returns'] * 100,
                    mode='lines',
                    name='Cumulative Returns (%)',
                    line=dict(color=self.colors['success'], width=2),
                    fill='tonexty'
                ),
                row=3, col=1
            )
            
            # Update layout
            fig.update_layout(
                title='Portfolio Performance Analysis',
                height=800,
                showlegend=True,
                template='plotly_white'
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
            fig.update_yaxes(title_text="Cumulative (%)", row=3, col=1)
            
            if save_path:
                fig.write_html(save_path)
                self.logger.info(f"Portfolio performance chart saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating portfolio performance chart: {str(e)}")
            return go.Figure()
    
    def create_drawdown_chart(self, portfolio_history: List[Dict[str, Any]],
                            save_path: Optional[str] = None) -> go.Figure:
        """
        Create drawdown chart showing underwater periods.
        
        Args:
            portfolio_history: Portfolio value history
            save_path: Path to save the chart (optional)
        
        Returns:
            Plotly figure object
        """
        try:
            # Convert to DataFrame
            portfolio_df = pd.DataFrame(portfolio_history)
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df.set_index('date', inplace=True)
            
            # Calculate drawdown
            portfolio_df['running_max'] = portfolio_df['portfolio_value'].expanding().max()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['running_max']) / portfolio_df['running_max']
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Portfolio Value vs Running Maximum', 'Drawdown Periods'),
                row_heights=[0.6, 0.4],
                vertical_spacing=0.1
            )
            
            # Portfolio value and running max
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color=self.colors['primary'], width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['running_max'],
                    mode='lines',
                    name='Running Maximum',
                    line=dict(color=self.colors['success'], width=1, dash='dash')
                ),
                row=1, col=1
            )
            
            # Drawdown
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['drawdown'] * 100,
                    mode='lines',
                    name='Drawdown (%)',
                    line=dict(color=self.colors['danger'], width=2),
                    fill='tonexty'
                ),
                row=2, col=1
            )
            
            # Add zero line for drawdown
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
            
            # Update layout
            fig.update_layout(
                title='Portfolio Drawdown Analysis',
                height=600,
                showlegend=True,
                template='plotly_white'
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            
            if save_path:
                fig.write_html(save_path)
                self.logger.info(f"Drawdown chart saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating drawdown chart: {str(e)}")
            return go.Figure()
    
    def create_trade_analysis_chart(self, trades: List[Dict[str, Any]],
                                  save_path: Optional[str] = None) -> go.Figure:
        """
        Create trade analysis visualization.
        
        Args:
            trades: List of trades
            save_path: Path to save the chart (optional)
        
        Returns:
            Plotly figure object
        """
        try:
            # Convert to DataFrame and filter sell trades (completed trades)
            trades_df = pd.DataFrame(trades)
            pnl_trades = trades_df[trades_df['action'] == 'SELL'].copy()
            
            if pnl_trades.empty:
                return go.Figure().add_annotation(text="No completed trades found")
            
            pnl_trades['date'] = pd.to_datetime(pnl_trades['date'])
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('P&L Distribution', 'P&L Over Time', 'Win/Loss by Confidence', 'Trade Returns'),
                specs=[[{"type": "histogram"}, {"type": "scatter"}],
                       [{"type": "box"}, {"type": "bar"}]]
            )
            
            # P&L Distribution
            fig.add_trace(
                go.Histogram(
                    x=pnl_trades['pnl'],
                    nbinsx=30,
                    name='P&L Distribution',
                    marker_color=self.colors['primary'],
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # P&L Over Time
            colors = ['red' if x < 0 else 'green' for x in pnl_trades['pnl']]
            fig.add_trace(
                go.Scatter(
                    x=pnl_trades['date'],
                    y=pnl_trades['pnl'],
                    mode='markers',
                    name='P&L Over Time',
                    marker=dict(color=colors, size=8),
                    text=pnl_trades['symbol'],
                    hovertemplate='<b>%{text}</b><br>P&L: $%{y:.2f}<br>Date: %{x}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Win/Loss by Confidence (if available)
            if 'signal_confidence' in pnl_trades.columns:
                confidence_bins = pd.cut(pnl_trades['signal_confidence'], 
                                       bins=[0, 0.5, 0.7, 0.9, 1.0], 
                                       labels=['Low', 'Medium', 'High', 'Very High'])
                pnl_trades['confidence_bin'] = confidence_bins
                
                for bin_name in ['Low', 'Medium', 'High', 'Very High']:
                    bin_data = pnl_trades[pnl_trades['confidence_bin'] == bin_name]['pnl']
                    if not bin_data.empty:
                        fig.add_trace(
                            go.Box(
                                y=bin_data,
                                name=f'{bin_name} Confidence',
                                boxpoints='outliers'
                            ),
                            row=2, col=1
                        )
            
            # Trade Returns
            if 'pnl_percent' in pnl_trades.columns:
                colors = ['red' if x < 0 else 'green' for x in pnl_trades['pnl_percent']]
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(pnl_trades))),
                        y=pnl_trades['pnl_percent'] * 100,
                        name='Returns (%)',
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title='Trade Analysis Dashboard',
                height=700,
                showlegend=True,
                template='plotly_white'
            )
            
            if save_path:
                fig.write_html(save_path)
                self.logger.info(f"Trade analysis chart saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating trade analysis chart: {str(e)}")
            return go.Figure()
    
    def create_returns_distribution_chart(self, portfolio_history: List[Dict[str, Any]],
                                        save_path: Optional[str] = None) -> go.Figure:
        """
        Create returns distribution analysis chart.
        
        Args:
            portfolio_history: Portfolio value history
            save_path: Path to save the chart (optional)
        
        Returns:
            Plotly figure object
        """
        try:
            # Convert to DataFrame
            portfolio_df = pd.DataFrame(portfolio_history)
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df.set_index('date', inplace=True)
            
            # Calculate returns
            portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change().dropna()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Daily Returns Distribution', 'Returns Q-Q Plot', 
                               'Monthly Returns Heatmap', 'Rolling Volatility'),
                specs=[[{"type": "histogram"}, {"type": "scatter"}],
                       [{"type": "heatmap"}, {"type": "scatter"}]]
            )
            
            # Daily returns histogram
            fig.add_trace(
                go.Histogram(
                    x=portfolio_df['returns'] * 100,
                    nbinsx=50,
                    name='Returns Distribution',
                    marker_color=self.colors['primary'],
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Q-Q plot (simplified)
            returns_sorted = np.sort(portfolio_df['returns'])
            normal_quantiles = np.random.normal(0, portfolio_df['returns'].std(), len(returns_sorted))
            normal_quantiles.sort()
            
            fig.add_trace(
                go.Scatter(
                    x=normal_quantiles * 100,
                    y=returns_sorted * 100,
                    mode='markers',
                    name='Q-Q Plot',
                    marker=dict(color=self.colors['secondary'], size=4)
                ),
                row=1, col=2
            )
            
            # Add diagonal line for Q-Q plot
            min_val = min(normal_quantiles.min(), returns_sorted.min()) * 100
            max_val = max(normal_quantiles.max(), returns_sorted.max()) * 100
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(dash='dash', color='red')
                ),
                row=1, col=2
            )
            
            # Monthly returns heatmap
            portfolio_df['year'] = portfolio_df.index.year
            portfolio_df['month'] = portfolio_df.index.month
            monthly_returns = portfolio_df.groupby(['year', 'month'])['returns'].sum().unstack(fill_value=0)
            
            if not monthly_returns.empty:
                fig.add_trace(
                    go.Heatmap(
                        z=monthly_returns.values * 100,
                        x=monthly_returns.columns,
                        y=monthly_returns.index,
                        colorscale='RdYlGn',
                        name='Monthly Returns (%)'
                    ),
                    row=2, col=1
                )
            
            # Rolling volatility
            portfolio_df['rolling_vol'] = portfolio_df['returns'].rolling(window=30).std() * np.sqrt(252) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['rolling_vol'],
                    mode='lines',
                    name='30-Day Rolling Volatility (%)',
                    line=dict(color=self.colors['warning'], width=2)
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title='Returns Distribution Analysis',
                height=700,
                showlegend=True,
                template='plotly_white'
            )
            
            if save_path:
                fig.write_html(save_path)
                self.logger.info(f"Returns distribution chart saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating returns distribution chart: {str(e)}")
            return go.Figure()
    
    def create_risk_metrics_dashboard(self, analysis_results: Dict[str, Any],
                                    save_path: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive risk metrics dashboard.
        
        Args:
            analysis_results: Analysis results from PerformanceAnalyzer
            save_path: Path to save the chart (optional)
        
        Returns:
            Plotly figure object
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Risk Metrics Gauge', 'VaR Analysis', 
                               'Sharpe vs Volatility', 'Risk-Return Scatter'),
                specs=[[{"type": "indicator"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Extract metrics
            portfolio_metrics = analysis_results.get('portfolio_analysis', {})
            risk_metrics = analysis_results.get('risk_analysis', {})
            
            # Sharpe Ratio Gauge
            sharpe_ratio = portfolio_metrics.get('sharpe_ratio', 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=sharpe_ratio,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Sharpe Ratio"},
                    gauge={
                        'axis': {'range': [-2, 3]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-2, 0], 'color': "lightgray"},
                            {'range': [0, 1], 'color': "gray"},
                            {'range': [1, 2], 'color': "lightgreen"},
                            {'range': [2, 3], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 2
                        }
                    }
                ),
                row=1, col=1
            )
            
            # VaR Analysis
            var_metrics = ['var_95', 'var_99', 'cvar_95', 'cvar_99']
            var_values = [abs(risk_metrics.get(metric, 0)) * 100 for metric in var_metrics]
            var_labels = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%']
            
            fig.add_trace(
                go.Bar(
                    x=var_labels,
                    y=var_values,
                    name='Value at Risk',
                    marker_color=self.colors['danger']
                ),
                row=1, col=2
            )
            
            # Risk metrics could be expanded with more data points
            # For now, showing basic metrics
            
            # Update layout
            fig.update_layout(
                title='Risk Metrics Dashboard',
                height=600,
                showlegend=True,
                template='plotly_white'
            )
            
            if save_path:
                fig.write_html(save_path)
                self.logger.info(f"Risk metrics dashboard saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating risk metrics dashboard: {str(e)}")
            return go.Figure()
    
    def save_all_charts(self, portfolio_history: List[Dict[str, Any]], 
                       trades: List[Dict[str, Any]],
                       analysis_results: Dict[str, Any],
                       output_dir: str = "charts") -> Dict[str, str]:
        """
        Generate and save all performance charts.
        
        Args:
            portfolio_history: Portfolio value history
            trades: List of trades
            analysis_results: Analysis results
            output_dir: Output directory for charts
        
        Returns:
            Dictionary of chart names and file paths
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            chart_files = {}
            
            # Portfolio performance chart
            portfolio_chart = self.create_portfolio_performance_chart(portfolio_history)
            portfolio_path = os.path.join(output_dir, "portfolio_performance.html")
            portfolio_chart.write_html(portfolio_path)
            chart_files['portfolio_performance'] = portfolio_path
            
            # Drawdown chart
            drawdown_chart = self.create_drawdown_chart(portfolio_history)
            drawdown_path = os.path.join(output_dir, "drawdown_analysis.html")
            drawdown_chart.write_html(drawdown_path)
            chart_files['drawdown_analysis'] = drawdown_path
            
            # Trade analysis chart
            trade_chart = self.create_trade_analysis_chart(trades)
            trade_path = os.path.join(output_dir, "trade_analysis.html")
            trade_chart.write_html(trade_path)
            chart_files['trade_analysis'] = trade_path
            
            # Returns distribution chart
            returns_chart = self.create_returns_distribution_chart(portfolio_history)
            returns_path = os.path.join(output_dir, "returns_distribution.html")
            returns_chart.write_html(returns_path)
            chart_files['returns_distribution'] = returns_path
            
            # Risk metrics dashboard
            risk_chart = self.create_risk_metrics_dashboard(analysis_results)
            risk_path = os.path.join(output_dir, "risk_metrics.html")
            risk_chart.write_html(risk_path)
            chart_files['risk_metrics'] = risk_path
            
            self.logger.info(f"All charts saved to {output_dir}")
            return chart_files
            
        except Exception as e:
            self.logger.error(f"Error saving charts: {str(e)}")
            return {}
