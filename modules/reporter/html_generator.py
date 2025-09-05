"""
HTML report generation with professional styling and layout.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from template_manager import TemplateManager
from config import ReporterConfig


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
            performance_charts=sections['performance_charts'],
            trade_analysis=sections['trade_analysis'],
            risk_analysis=sections['risk_analysis'],
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
        
        # Performance charts section
        sections['performance_charts'] = self._generate_charts_section(charts)
        
        # Trade analysis section
        sections['trade_analysis'] = self._generate_trade_analysis_section(results_data)
        
        # Risk analysis section
        sections['risk_analysis'] = self._generate_risk_analysis_section(results_data)
        
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
        
        # Generate trade table rows
        trade_rows = ""
        for i, trade in enumerate(trades[:50]):  # Limit to first 50 trades
            entry_time = trade.get('entry_time', '')
            exit_time = trade.get('exit_time', '')
            direction = trade.get('direction', '')
            entry_price = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            pnl = trade.get('pnl', 0)
            duration = trade.get('duration_minutes', 0)
            
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
            
            trade_rows += f"""
            <tr>
                <td>{i+1}</td>
                <td>{entry_formatted}</td>
                <td>{exit_formatted}</td>
                <td><span class="badge badge-{direction.lower()}">{direction}</span></td>
                <td>${entry_price:.2f}</td>
                <td>${exit_price:.2f}</td>
                <td class="{pnl_class}">${pnl:+.2f}</td>
                <td>{duration:.0f}m</td>
            </tr>
            """
        
        return f"""
        <div class="tables-section">
            <h2>Trade Details</h2>
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
                {f'<p class="table-note">Showing first 50 trades of {len(trades)} total</p>' if len(trades) > 50 else ''}
            </div>
        </div>
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