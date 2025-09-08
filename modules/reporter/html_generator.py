"""
HTML report generation with professional styling and layout.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from .template_manager import TemplateManager
from .config import ReporterConfig


class HTMLReportGenerator:
    """Generates complete HTML reports with styling and interactive elements."""
    
    def __init__(self, config: ReporterConfig):
        self.config = config
        self.template_manager = TemplateManager()
    
    async def generate_report(
        self,
        results_data: Dict[str, Any],
        charts: Dict[str, str],
        template: str = "comprehensive"
    ) -> str:
        """
        Generate complete HTML report.
        
        Args:
            results_data: Backtest results data
            charts: Dictionary of chart HTML/JS code
            template: Template name to use
            
        Returns:
            Complete HTML content
        """
        
        # Parse and enrich data
        enriched_data = self._enrich_data(results_data)
        
        # Generate content sections
        sections = await self._generate_sections(enriched_data, charts, template)
        
        # Load template and build report
        template_html = self.template_manager.load_template(template)
        
        # Replace placeholders
        html_content = template_html.format(
            title=self._generate_title(enriched_data),
            styles=self._generate_styles(),
            scripts=self._generate_scripts(),
            header=sections['header'],
            summary=sections['summary'],
            candlestick_chart=sections['candlestick_chart'],
            detailed_tables=sections['detailed_tables'],
            footer=sections['footer']
        )
        
        return html_content
    
    def _enrich_data(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich results data with additional calculated metrics."""
        
        enriched = results_data.copy()
        
        # Add calculated fields
        backtest_info = enriched.get('backtest_info', {})
        metrics = enriched.get('performance_metrics', {})
        trades = enriched.get('trades', [])
        
        # Calculate additional metrics
        duration_days = 0
        if 'start_date' in backtest_info and 'end_date' in backtest_info:
            try:
                start_dt = datetime.fromisoformat(backtest_info['start_date'].replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(backtest_info['end_date'].replace('Z', '+00:00'))
                duration_days = (end_dt - start_dt).days
                backtest_info['duration_days'] = duration_days
            except:
                pass
        
        # Trade frequency
        if duration_days > 0 and len(trades) > 0:
            metrics['trades_per_day'] = len(trades) / duration_days
            metrics['avg_days_between_trades'] = duration_days / len(trades)
        else:
            metrics['trades_per_day'] = 0
            metrics['avg_days_between_trades'] = 0
        
        # Risk-adjusted returns
        if metrics.get('max_drawdown_pct', 0) > 0:
            metrics['return_to_drawdown_ratio'] = metrics.get('total_return_pct', 0) / metrics['max_drawdown_pct']
        else:
            metrics['return_to_drawdown_ratio'] = 0
        
        # Add performance grade
        metrics['performance_grade'] = self._calculate_performance_grade(metrics)
        
        enriched['backtest_info'] = backtest_info
        enriched['performance_metrics'] = metrics
        
        return enriched
    
    def _calculate_performance_grade(self, metrics: Dict[str, Any]) -> str:
        """Calculate overall performance grade A-F."""
        
        score = 0
        
        # Return score (0-30 points)
        total_return = metrics.get('total_return_pct', 0)
        if total_return >= 50:
            score += 30
        elif total_return >= 20:
            score += 25
        elif total_return >= 10:
            score += 20
        elif total_return >= 5:
            score += 15
        elif total_return >= 0:
            score += 10
        
        # Win rate score (0-25 points)
        win_rate = metrics.get('win_rate_pct', 0)
        if win_rate >= 70:
            score += 25
        elif win_rate >= 60:
            score += 20
        elif win_rate >= 50:
            score += 15
        elif win_rate >= 40:
            score += 10
        elif win_rate >= 30:
            score += 5
        
        # Sharpe ratio score (0-25 points)
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe >= 2.0:
            score += 25
        elif sharpe >= 1.5:
            score += 20
        elif sharpe >= 1.0:
            score += 15
        elif sharpe >= 0.5:
            score += 10
        elif sharpe >= 0:
            score += 5
        
        # Drawdown score (0-20 points)
        drawdown = metrics.get('max_drawdown_pct', 100)
        if drawdown <= 5:
            score += 20
        elif drawdown <= 10:
            score += 15
        elif drawdown <= 20:
            score += 10
        elif drawdown <= 30:
            score += 5
        
        # Convert to grade
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 55:
            return "C"
        elif score >= 50:
            return "C-"
        elif score >= 45:
            return "D"
        else:
            return "F"
    
    async def _generate_sections(
        self,
        results_data: Dict[str, Any],
        charts: Dict[str, str],
        template: str
    ) -> Dict[str, str]:
        """Generate all report sections."""
        
        sections = {}
        
        # Header section
        sections['header'] = self._generate_header_section(results_data)
        
        # Summary section
        sections['summary'] = self._generate_summary_section(results_data)
        
        # Candlestick chart section  
        sections['candlestick_chart'] = await self._generate_candlestick_section(results_data)
        
        # Detailed tables section
        sections['detailed_tables'] = self._generate_tables_section(results_data)
        
        # Footer section
        sections['footer'] = self._generate_footer_section(results_data)
        
        return sections
    
    def _generate_header_section(self, results_data: Dict[str, Any]) -> str:
        """Generate report header with title and key info."""
        
        backtest_info = results_data.get('backtest_info', {})
        metrics = results_data.get('performance_metrics', {})
        
        strategy_name = backtest_info.get('name', 'Trading Strategy')
        start_date = backtest_info.get('start_date', 'N/A')
        end_date = backtest_info.get('end_date', 'N/A')
        total_return = metrics.get('total_return_pct', 0)
        grade = metrics.get('performance_grade', 'N/A')
        
        # Format dates
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            start_formatted = start_dt.strftime('%B %d, %Y')
            end_formatted = end_dt.strftime('%B %d, %Y')
        except:
            start_formatted = start_date
            end_formatted = end_date
        
        return f"""
        <div class="report-header">
            <div class="header-main">
                <h1 class="strategy-title">{strategy_name}</h1>
                <div class="header-subtitle">Backtest Analysis Report</div>
            </div>
            <div class="header-info">
                <div class="info-grid">
                    <div class="info-item">
                        <span class="info-label">Period</span>
                        <span class="info-value">{start_formatted} - {end_formatted}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Total Return</span>
                        <span class="info-value return-{'positive' if total_return >= 0 else 'negative'}">{total_return:+.2f}%</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Performance Grade</span>
                        <span class="info-value grade-{grade.replace('+', 'plus').replace('-', 'minus')}">{grade}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Generated</span>
                        <span class="info-value">{datetime.now().strftime('%B %d, %Y at %I:%M %p')}</span>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_summary_section(self, results_data: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        
        metrics = results_data.get('performance_metrics', {})
        backtest_info = results_data.get('backtest_info', {})
        
        # Key metrics
        total_return = metrics.get('total_return_pct', 0)
        max_drawdown = metrics.get('max_drawdown_pct', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        win_rate = metrics.get('win_rate_pct', 0)
        total_trades = metrics.get('total_trades', 0)
        profit_factor = metrics.get('profit_factor', 0)
        
        # Capital info
        initial_capital = backtest_info.get('initial_capital', 0)
        final_capital = backtest_info.get('final_capital', 0)
        
        return f"""
        <div class="summary-section">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card capital">
                    <h3>Capital</h3>
                    <div class="metric-row">
                        <span class="metric-label">Initial</span>
                        <span class="metric-value">${initial_capital:,.2f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Final</span>
                        <span class="metric-value">${final_capital:,.2f}</span>
                    </div>
                    <div class="metric-highlight">
                        <span class="metric-label">Net Profit</span>
                        <span class="metric-value profit-{'positive' if final_capital >= initial_capital else 'negative'}">${final_capital - initial_capital:+,.2f}</span>
                    </div>
                </div>
                
                <div class="summary-card performance">
                    <h3>Performance</h3>
                    <div class="metric-row">
                        <span class="metric-label">Total Return</span>
                        <span class="metric-value return-{'positive' if total_return >= 0 else 'negative'}">{total_return:+.2f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Sharpe Ratio</span>
                        <span class="metric-value">{sharpe_ratio:.3f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Max Drawdown</span>
                        <span class="metric-value drawdown">{max_drawdown:.2f}%</span>
                    </div>
                </div>
                
                <div class="summary-card trading">
                    <h3>Trading Activity</h3>
                    <div class="metric-row">
                        <span class="metric-label">Total Trades</span>
                        <span class="metric-value">{total_trades}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Win Rate</span>
                        <span class="metric-value">{win_rate:.1f}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Profit Factor</span>
                        <span class="metric-value">{profit_factor:.2f}</span>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_charts_section(self, charts: Dict[str, str]) -> str:
        """Generate charts section with interactive visualizations."""
        
        return f"""
        <div class="charts-section">
            <h2>Performance Analysis</h2>
            
            <div class="chart-container">
                <div class="chart-header">
                    <h3>Equity Curve</h3>
                    <div class="chart-description">Portfolio value over time with drawdown periods highlighted</div>
                </div>
                <div class="chart-content">
                    {charts.get('equity_curve', '<p>Equity curve chart not available</p>')}
                </div>
            </div>
            
            <div class="charts-grid">
                <div class="chart-container">
                    <div class="chart-header">
                        <h3>Trade Distribution</h3>
                    </div>
                    <div class="chart-content">
                        {charts.get('trade_distribution', '<p>Trade distribution chart not available</p>')}
                    </div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-header">
                        <h3>Monthly Returns</h3>
                    </div>
                    <div class="chart-content">
                        {charts.get('monthly_returns', '<p>Monthly returns chart not available</p>')}
                    </div>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-header">
                    <h3>Risk Metrics</h3>
                </div>
                <div class="chart-content">
                    {charts.get('risk_metrics', '<p>Risk metrics chart not available</p>')}
                </div>
            </div>
        </div>
        """
    
    def _generate_trade_analysis_section(self, results_data: Dict[str, Any]) -> str:
        """Generate detailed trade analysis."""
        
        trades = results_data.get('trades', [])
        metrics = results_data.get('performance_metrics', {})
        
        if not trades:
            return """
            <div class="trade-analysis-section">
                <h2>Trade Analysis</h2>
                <div class="no-data">
                    <p>No trades were executed during the backtest period.</p>
                    <p>This could indicate that the strategy conditions were not met or the strategy is too conservative.</p>
                </div>
            </div>
            """
        
        # Calculate trade statistics
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        avg_winner = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loser = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        return f"""
        <div class="trade-analysis-section">
            <h2>Trade Analysis</h2>
            
            <div class="trade-stats-grid">
                <div class="stat-card">
                    <h4>Trade Summary</h4>
                    <div class="stat-row">
                        <span class="stat-label">Total Trades</span>
                        <span class="stat-value">{len(trades)}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Winning Trades</span>
                        <span class="stat-value text-success">{len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Losing Trades</span>
                        <span class="stat-value text-danger">{len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)</span>
                    </div>
                </div>
                
                <div class="stat-card">
                    <h4>Average Trade Performance</h4>
                    <div class="stat-row">
                        <span class="stat-label">Average Winner</span>
                        <span class="stat-value text-success">${avg_winner:.2f}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Average Loser</span>
                        <span class="stat-value text-danger">${avg_loser:.2f}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Win/Loss Ratio</span>
                        <span class="stat-value">{abs(avg_winner/avg_loser) if avg_loser != 0 else 0:.2f}</span>
                    </div>
                </div>
                
                <div class="stat-card">
                    <h4>Trade Duration</h4>
                    <div class="stat-row">
                        <span class="stat-label">Average Duration</span>
                        <span class="stat-value">{metrics.get("avg_trade_duration_minutes", 0):.0f} minutes</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Trades per Day</span>
                        <span class="stat-value">{metrics.get("trades_per_day", 0):.2f}</span>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_risk_analysis_section(self, results_data: Dict[str, Any]) -> str:
        """Generate risk analysis section."""
        
        metrics = results_data.get('performance_metrics', {})
        
        return f"""
        <div class="risk-analysis-section">
            <h2>Risk Analysis</h2>
            
            <div class="risk-grid">
                <div class="risk-card">
                    <h4>Drawdown Analysis</h4>
                    <div class="risk-metric">
                        <span class="risk-label">Maximum Drawdown</span>
                        <span class="risk-value drawdown-value">{metrics.get('max_drawdown_pct', 0):.2f}%</span>
                    </div>
                    <div class="risk-assessment">
                        {self._get_drawdown_assessment(metrics.get('max_drawdown_pct', 0))}
                    </div>
                </div>
                
                <div class="risk-card">
                    <h4>Risk-Adjusted Returns</h4>
                    <div class="risk-metric">
                        <span class="risk-label">Sharpe Ratio</span>
                        <span class="risk-value">{metrics.get('sharpe_ratio', 0):.3f}</span>
                    </div>
                    <div class="risk-metric">
                        <span class="risk-label">Sortino Ratio</span>
                        <span class="risk-value">{metrics.get('sortino_ratio', 0):.3f}</span>
                    </div>
                </div>
                
                <div class="risk-card">
                    <h4>Consistency</h4>
                    <div class="risk-metric">
                        <span class="risk-label">Max Consecutive Losses</span>
                        <span class="risk-value">{metrics.get('max_consecutive_losses', 0)}</span>
                    </div>
                    <div class="risk-metric">
                        <span class="risk-label">Profit Factor</span>
                        <span class="risk-value">{metrics.get('profit_factor', 0):.2f}</span>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _get_drawdown_assessment(self, drawdown: float) -> str:
        """Get risk assessment text based on drawdown."""
        if drawdown <= 5:
            return '<span class="risk-low">Low Risk</span> - Excellent drawdown control'
        elif drawdown <= 10:
            return '<span class="risk-moderate">Moderate Risk</span> - Acceptable drawdown'
        elif drawdown <= 20:
            return '<span class="risk-high">High Risk</span> - Significant drawdown periods'
        else:
            return '<span class="risk-extreme">Extreme Risk</span> - Very high drawdown'
    
    def _generate_tables_section(self, results_data: Dict[str, Any]) -> str:
        """Generate detailed data tables."""
        
        trades = results_data.get('trades', [])
        
        if not trades:
            return """
            <div class="tables-section">
                <h2>Trade Details</h2>
                <div class="no-data">
                    <p>No trade data available.</p>
                </div>
            </div>
            """
        
        # Sort trades in reverse chronological order (most recent first)
        sorted_trades = sorted(trades, key=lambda x: x.get('exit_time', ''), reverse=True)
        
        # Generate trade table rows - show ALL trades, not just first 50
        trade_rows = ""
        trade_data_js = []  # For storing trade data for modal
        
        for i, trade in enumerate(sorted_trades):
            entry_time = trade.get('entry_time', '')
            exit_time = trade.get('exit_time', '')
            direction = trade.get('direction', '')
            entry_price = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            pnl = trade.get('pnl', 0)
            duration = trade.get('duration_minutes', 0)
            entry_reason = trade.get('entry_reason', 'N/A')
            exit_reason = trade.get('exit_reason', 'N/A')
            quantity = trade.get('quantity', 0)
            commission = trade.get('commission', 0)
            
            # Format times
            try:
                entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                entry_formatted = entry_dt.strftime('%m/%d %H:%M')
            except:
                entry_formatted = entry_time
            
            try:
                exit_dt = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                exit_formatted = exit_dt.strftime('%m/%d %H:%M')
            except:
                exit_formatted = exit_time
            
            pnl_class = 'text-success' if pnl >= 0 else 'text-danger'
            
            # Store trade data for modal
            trade_data = {
                'entry_time': entry_time,
                'exit_time': exit_time,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'quantity': quantity,
                'pnl': pnl,
                'commission': commission,
                'duration_minutes': duration,
                'entry_reason': entry_reason,
                'exit_reason': exit_reason
            }
            trade_data_js.append(trade_data)
            
            trade_rows += f"""
            <tr onclick="showTradeDetails(tradeData[{i}])" style="cursor: pointer;">
                <td>{i+1}</td>
                <td>{entry_formatted}</td>
                <td>{exit_formatted}</td>
                <td><span class="badge badge-{direction.lower()}">{direction}</span></td>
                <td>${entry_price:.4f}</td>
                <td>${exit_price:.4f}</td>
                <td class="{pnl_class}">${pnl:+.2f}</td>
                <td>{duration:.0f}m</td>
            </tr>
            """
        
        import json
        
        return f"""
        <div class="tables-section">
            <h2>Trade Details</h2>
            <p style="color: #6c757d; margin-bottom: 20px;">
                Showing all {len(sorted_trades)} trades (most recent first). Click on a trade to see detailed information.
            </p>
            <div class="table-container">
                <table class="trade-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Entry Time</th>
                            <th>Exit Time</th>
                            <th>Direction</th>
                            <th>Entry Price</th>
                            <th>Exit Price</th>
                            <th>P&L</th>
                            <th>Duration</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trade_rows}
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Modal for trade details -->
        <div id="tradeModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <span class="close" onclick="closeModal()">&times;</span>
                    <h2>Trade Details</h2>
                </div>
                <div class="modal-body" id="tradeModalBody">
                    <!-- Trade details will be inserted here by JavaScript -->
                </div>
            </div>
        </div>
        
        <script>
        // Store trade data for modal access
        var tradeData = {json.dumps(trade_data_js)};
        </script>
        """
    
    def _generate_footer_section(self, results_data: Dict[str, Any]) -> str:
        """Generate report footer."""
        
        return f"""
        <div class="report-footer">
            <div class="footer-content">
                <div class="footer-left">
                    <p>Generated by Trading Bot Analysis System</p>
                    <p>Report created on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>
                <div class="footer-right">
                    <p class="disclaimer">
                        <strong>Disclaimer:</strong> Past performance does not guarantee future results. 
                        This backtest represents historical simulation and may not reflect actual trading conditions.
                    </p>
                </div>
            </div>
        </div>
        """
    
    def _generate_title(self, results_data: Dict[str, Any]) -> str:
        """Generate HTML title."""
        strategy_name = results_data.get('backtest_info', {}).get('name', 'Trading Strategy')
        return f"{strategy_name} - Backtest Report"
    
    def _generate_styles(self) -> str:
        """Generate CSS styles for the report."""
        
        theme = self.config.theme
        return self.template_manager.load_styles(theme)
    
    def _generate_scripts(self) -> str:
        """Generate JavaScript for interactive elements."""
        
        return self.template_manager.load_scripts()
    
    async def _generate_candlestick_section(self, results_data: Dict[str, Any]) -> str:
        """Generate candlestick chart with trade entry/exit points."""
        import pandas as pd
        import json
        from pathlib import Path
        
        try:
            # Get backtest period
            backtest_info = results_data.get('backtest_info', {})
            start_date = backtest_info.get('start_date', '')
            end_date = backtest_info.get('end_date', '')
            
            # Find appropriate CSV file with OHLC data
            data_dir = Path("data/raw")
            csv_files = list(data_dir.glob("SOLUSDT_5m_*.csv"))
            
            # Find file that covers our backtest period
            best_file = None
            for csv_file in csv_files:
                # Check if filename suggests it covers our period
                if "20250807" in str(csv_file) and "20250906" in str(csv_file):
                    best_file = csv_file
                    break
            
            if not best_file and csv_files:
                # Fallback to any available file
                best_file = sorted(csv_files)[-1]
            
            if not best_file:
                return '<div class="no-data">No OHLC data available for candlestick chart</div>'
            
            # Read OHLC data
            df = pd.read_csv(best_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter data to backtest period if dates available
            if start_date and end_date:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
            
            # Keep all original data - don't sample/reduce candles
            # User specifically wants to see all candles with horizontal scrolling
            
            # Get trades data
            trades = results_data.get('trades', [])
            
            # Convert timestamps to ISO format for better compatibility  
            timestamps_str = [ts.isoformat() for ts in df['timestamp']]
            
            # Prepare chart data
            chart_html = f"""
            <div class="charts-section">
                <h2>📊 Price Chart with Trade Signals</h2>
                <div class="chart-container">
                    <div class="chart-header">
                        <h3>Candlestick Chart</h3>
                        <div class="chart-description">
                            Price movement with entry and exit points ({len(df)} candles, {len(trades)} trades)<br>
                            <small style="color: #888;">💡 Use the slider below to navigate through time periods. Drag to zoom, double-click to reset.</small>
                        </div>
                    </div>
                    <div class="chart-content">
                        <div id="candlestick-chart" style="width:100%;height:600px;"></div>
                        <div id="chart-status" style="margin-top:10px; color:#666; text-align:center;">Loading chart...</div>
                    </div>
                </div>
            </div>
            
            <script>
            document.addEventListener('DOMContentLoaded', function() {{
                try {{
                    // Status update
                    document.getElementById('chart-status').textContent = 'Preparing data...';
                    
                    // Candlestick chart data preparation
                    var candleData = {{
                        x: {json.dumps(timestamps_str)},
                        open: {json.dumps(df['open'].tolist())},
                        high: {json.dumps(df['high'].tolist())},
                        low: {json.dumps(df['low'].tolist())},
                        close: {json.dumps(df['close'].tolist())},
                        type: 'candlestick',
                        name: 'SOLUSDT',
                        increasing: {{line: {{color: '#26a69a'}}}},
                        decreasing: {{line: {{color: '#ef5350'}}}}
                    }};
                    
                    // Trade entry/exit points - show all trades
                    var allTrades = {json.dumps(trades)};
                    
                    var entryPoints = {{
                        x: allTrades.map(function(trade) {{ return trade.entry_time; }}),
                        y: allTrades.map(function(trade) {{ return trade.entry_price; }}),
                        mode: 'markers',
                        type: 'scatter',
                        name: 'Entry Points',
                        marker: {{
                            color: allTrades.map(function(trade) {{ 
                                return trade.direction === 'LONG' ? '#2196F3' : '#FF5722';
                            }}),
                            size: 10,
                            symbol: allTrades.map(function(trade) {{
                                return trade.direction === 'LONG' ? 'triangle-up' : 'triangle-down';
                            }})
                        }},
                        text: allTrades.map(function(trade) {{
                            return 'Entry ' + trade.direction + ' $' + trade.entry_price.toFixed(4);
                        }}),
                        hovertemplate: '%{{text}}<extra></extra>'
                    }};
                    
                    var exitPoints = {{
                        x: allTrades.map(function(trade) {{ return trade.exit_time; }}),
                        y: allTrades.map(function(trade) {{ return trade.exit_price; }}),
                        mode: 'markers', 
                        type: 'scatter',
                        name: 'Exit Points',
                        marker: {{
                            color: allTrades.map(function(trade) {{
                                return trade.pnl > 0 ? '#4CAF50' : '#F44336';
                            }}),
                            size: 8,
                            symbol: 'x'
                        }},
                        text: allTrades.map(function(trade) {{
                            return 'Exit $' + trade.exit_price.toFixed(4) + ' P&L: $' + trade.pnl.toFixed(2);
                        }}),
                        hovertemplate: '%{{text}}<extra></extra>'
                    }};
                    
                    var layout = {{
                        title: 'Trading Strategy Performance - All {len(df)} Candles',
                        xaxis: {{
                            title: 'Time',
                            type: 'date',
                            rangeslider: {{
                                visible: true,
                                thickness: 0.1
                            }},
                            range: [
                                '{timestamps_str[0]}',
                                '{timestamps_str[min(200, len(timestamps_str)-1)]}'
                            ]
                        }},
                        yaxis: {{
                            title: 'Price (USDT)'
                        }},
                        showlegend: true,
                        hovermode: 'x unified',
                        margin: {{l: 60, r: 30, t: 80, b: 100}}
                    }};
                    
                    var config = {{
                        responsive: true,
                        displayModeBar: true,
                        displaylogo: false,
                        scrollZoom: true,
                        modeBarButtonsToRemove: ['lasso2d', 'select2d']
                    }};
                    
                    document.getElementById('chart-status').textContent = 'Rendering chart...';
                    
                    Plotly.newPlot('candlestick-chart', [candleData, entryPoints, exitPoints], layout, config)
                        .then(function() {{
                            document.getElementById('chart-status').textContent = 'Chart loaded successfully!';
                            setTimeout(function() {{
                                document.getElementById('chart-status').style.display = 'none';
                            }}, 2000);
                        }})
                        .catch(function(error) {{
                            console.error('Chart error:', error);
                            document.getElementById('chart-status').textContent = 'Error loading chart: ' + error.message;
                        }});
                        
                }} catch (error) {{
                    console.error('Chart initialization error:', error);
                    document.getElementById('chart-status').textContent = 'Error initializing chart: ' + error.message;
                }}
            }});
            </script>
            """
            
            return chart_html
            
        except Exception as e:
            return f'<div class="no-data">Error generating candlestick chart: {str(e)}</div>'