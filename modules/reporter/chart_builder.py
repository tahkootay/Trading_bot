"""
Interactive chart generation using Plotly.js for HTML reports.
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from .config import ReporterConfig


class ChartBuilder:
    """Builds interactive charts for HTML reports using Plotly.js."""
    
    def __init__(self, config: ReporterConfig):
        self.config = config
    
    async def build_all_charts(self, results_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Build all charts for the report.
        
        Args:
            results_data: Backtest results data
            
        Returns:
            Dictionary with chart_name -> HTML content
        """
        
        charts = {}
        
        # Build individual charts
        charts['equity_curve'] = await self.build_equity_curve(results_data)
        charts['trade_distribution'] = await self.build_trade_distribution(results_data)
        charts['monthly_returns'] = await self.build_monthly_returns(results_data)
        charts['risk_metrics'] = await self.build_risk_metrics(results_data)
        
        return charts
    
    async def build_equity_curve(self, results_data: Dict[str, Any]) -> str:
        """Build equity curve chart with drawdown."""
        
        equity_data = results_data.get('equity_curve', [])
        
        if not equity_data:
            return '<p>No equity data available</p>'
        
        # Prepare data
        timestamps = []
        equity_values = []
        drawdown_values = []
        
        for point in equity_data:
            try:
                timestamp = point.get('timestamp', '')
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamps.append(dt.strftime('%Y-%m-%d %H:%M:%S'))
                equity_values.append(point.get('equity', 0))
                drawdown_values.append(-point.get('drawdown', 0))  # Negative for visual effect
            except:
                continue
        
        if not timestamps:
            return '<p>Invalid equity data</p>'
        
        # Create Plotly chart data
        chart_data = {
            'data': [
                {
                    'x': timestamps,
                    'y': equity_values,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Equity',
                    'line': {'color': '#2E7D32', 'width': 2},
                    'yaxis': 'y'
                },
                {
                    'x': timestamps,
                    'y': drawdown_values,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Drawdown %',
                    'line': {'color': '#D32F2F', 'width': 1},
                    'fill': 'tonexty',
                    'fillcolor': 'rgba(211, 47, 47, 0.1)',
                    'yaxis': 'y2'
                }
            ],
            'layout': {
                'title': {
                    'text': 'Portfolio Equity Curve with Drawdown',
                    'font': {'size': 16}
                },
                'xaxis': {
                    'title': 'Date',
                    'type': 'date'
                },
                'yaxis': {
                    'title': 'Equity ($)',
                    'side': 'left',
                    'tickformat': ',.0f'
                },
                'yaxis2': {
                    'title': 'Drawdown (%)',
                    'side': 'right',
                    'overlaying': 'y',
                    'tickformat': '.1f'
                },
                'legend': {
                    'x': 0.02,
                    'y': 0.98
                },
                'hovermode': 'x unified',
                'showlegend': True,
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white'
            }
        }
        
        chart_id = 'equity-curve-chart'
        
        return f"""
        <div id="{chart_id}" style="height: 400px;"></div>
        <script>
            Plotly.newPlot('{chart_id}', {json.dumps(chart_data['data'])}, {json.dumps(chart_data['layout'])}, {{responsive: true}});
        </script>
        """
    
    async def build_trade_distribution(self, results_data: Dict[str, Any]) -> str:
        """Build trade P&L distribution histogram."""
        
        trades = results_data.get('trades', [])
        
        if not trades:
            return '<p>No trade data available</p>'
        
        # Extract P&L values
        pnls = [trade.get('pnl', 0) for trade in trades]
        
        if not pnls:
            return '<p>No P&L data available</p>'
        
        # Create histogram data
        chart_data = {
            'data': [
                {
                    'x': pnls,
                    'type': 'histogram',
                    'name': 'Trade P&L Distribution',
                    'marker': {
                        'color': pnls,
                        'colorscale': [
                            [0, '#D32F2F'],      # Red for losses
                            [0.5, '#FFC107'],    # Yellow for break-even
                            [1, '#2E7D32']       # Green for profits
                        ],
                        'cmin': min(pnls),
                        'cmax': max(pnls)
                    },
                    'nbinsx': min(20, len(trades) // 2) if len(trades) > 4 else 10
                }
            ],
            'layout': {
                'title': {
                    'text': 'Trade P&L Distribution',
                    'font': {'size': 16}
                },
                'xaxis': {
                    'title': 'Profit/Loss ($)',
                    'tickformat': ',.0f'
                },
                'yaxis': {
                    'title': 'Number of Trades'
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'bargap': 0.1
            }
        }
        
        chart_id = 'trade-distribution-chart'
        
        return f"""
        <div id="{chart_id}" style="height: 300px;"></div>
        <script>
            Plotly.newPlot('{chart_id}', {json.dumps(chart_data['data'])}, {json.dumps(chart_data['layout'])}, {{responsive: true}});
        </script>
        """
    
    async def build_monthly_returns(self, results_data: Dict[str, Any]) -> str:
        """Build monthly returns heatmap."""
        
        daily_returns = results_data.get('daily_returns', [])
        
        if not daily_returns:
            return '<p>No daily returns data available</p>'
        
        # Group returns by month
        monthly_returns = {}
        for day_return in daily_returns:
            try:
                date_str = day_return.get('date', '')
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                month_key = dt.strftime('%Y-%m')
                
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = []
                
                monthly_returns[month_key].append(day_return.get('return_pct', 0))
            except:
                continue
        
        if not monthly_returns:
            return '<p>Insufficient data for monthly analysis</p>'
        
        # Calculate monthly totals
        months = sorted(monthly_returns.keys())
        month_labels = []
        month_returns = []
        
        for month in months:
            try:
                dt = datetime.strptime(month, '%Y-%m')
                month_labels.append(dt.strftime('%b %Y'))
                # Sum of daily returns for the month
                month_total = sum(monthly_returns[month])
                month_returns.append(month_total)
            except:
                continue
        
        if not month_returns:
            return '<p>No valid monthly data</p>'
        
        # Create bar chart
        colors = ['#2E7D32' if x >= 0 else '#D32F2F' for x in month_returns]
        
        chart_data = {
            'data': [
                {
                    'x': month_labels,
                    'y': month_returns,
                    'type': 'bar',
                    'name': 'Monthly Returns',
                    'marker': {'color': colors},
                    'text': [f'{x:+.2f}%' for x in month_returns],
                    'textposition': 'outside'
                }
            ],
            'layout': {
                'title': {
                    'text': 'Monthly Returns',
                    'font': {'size': 16}
                },
                'xaxis': {
                    'title': 'Month'
                },
                'yaxis': {
                    'title': 'Return (%)',
                    'tickformat': '.2f'
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'bargap': 0.3
            }
        }
        
        chart_id = 'monthly-returns-chart'
        
        return f"""
        <div id="{chart_id}" style="height: 300px;"></div>
        <script>
            Plotly.newPlot('{chart_id}', {json.dumps(chart_data['data'])}, {json.dumps(chart_data['layout'])}, {{responsive: true}});
        </script>
        """
    
    async def build_risk_metrics(self, results_data: Dict[str, Any]) -> str:
        """Build risk metrics comparison chart."""
        
        metrics = results_data.get('performance_metrics', {})
        
        # Risk metrics to display
        risk_metrics = {
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Sortino Ratio': metrics.get('sortino_ratio', 0),
            'Profit Factor': metrics.get('profit_factor', 0),
            'Win Rate (%)': metrics.get('win_rate_pct', 0),
            'Return/Drawdown': metrics.get('return_to_drawdown_ratio', 0)
        }
        
        # Benchmark values for comparison
        benchmarks = {
            'Sharpe Ratio': 1.0,
            'Sortino Ratio': 1.5,
            'Profit Factor': 1.5,
            'Win Rate (%)': 50.0,
            'Return/Drawdown': 3.0
        }
        
        metric_names = list(risk_metrics.keys())
        metric_values = list(risk_metrics.values())
        benchmark_values = [benchmarks.get(name, 0) for name in metric_names]
        
        # Create grouped bar chart
        chart_data = {
            'data': [
                {
                    'x': metric_names,
                    'y': metric_values,
                    'type': 'bar',
                    'name': 'Strategy',
                    'marker': {'color': '#1976D2'},
                    'text': [f'{x:.2f}' for x in metric_values],
                    'textposition': 'outside'
                },
                {
                    'x': metric_names,
                    'y': benchmark_values,
                    'type': 'bar',
                    'name': 'Benchmark',
                    'marker': {'color': '#FFC107', 'opacity': 0.7},
                    'text': [f'{x:.2f}' for x in benchmark_values],
                    'textposition': 'outside'
                }
            ],
            'layout': {
                'title': {
                    'text': 'Risk Metrics vs Benchmarks',
                    'font': {'size': 16}
                },
                'xaxis': {
                    'title': 'Metrics'
                },
                'yaxis': {
                    'title': 'Value'
                },
                'barmode': 'group',
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'bargap': 0.3,
                'legend': {
                    'x': 0.02,
                    'y': 0.98
                }
            }
        }
        
        chart_id = 'risk-metrics-chart'
        
        return f"""
        <div id="{chart_id}" style="height: 300px;"></div>
        <script>
            Plotly.newPlot('{chart_id}', {json.dumps(chart_data['data'])}, {json.dumps(chart_data['layout'])}, {{responsive: true}});
        </script>
        """