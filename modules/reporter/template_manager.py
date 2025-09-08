"""
HTML template management and styling for reports.
"""

from pathlib import Path
from typing import Dict, Any


class TemplateManager:
    """Manages HTML templates and styling for reports."""
    
    def load_template(self, template_name: str) -> str:
        """Load HTML template by name."""
        
        templates = {
            'comprehensive': self._comprehensive_template(),
            'summary': self._summary_template(),
            'professional': self._professional_template(),
            'minimal': self._minimal_template()
        }
        
        return templates.get(template_name, templates['comprehensive'])
    
    def load_styles(self, theme: str = "professional") -> str:
        """Load CSS styles for specified theme."""
        
        base_styles = self._base_styles()
        theme_styles = self._theme_styles(theme)
        
        return base_styles + theme_styles
    
    def load_scripts(self) -> str:
        """Load JavaScript code for interactive elements."""
        
        return """
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script>
        // Responsive chart resizing
        window.addEventListener('resize', function() {
            var charts = document.querySelectorAll('[id*="chart"]');
            charts.forEach(function(chart) {
                if (chart && Plotly) {
                    Plotly.Plots.resize(chart);
                }
            });
        });
        
        // Print functionality
        function printReport() {
            window.print();
        }
        
        // Toggle sections
        function toggleSection(sectionId) {
            var section = document.getElementById(sectionId);
            if (section) {
                section.style.display = section.style.display === 'none' ? 'block' : 'none';
            }
        }
        
        // Modal functionality
        function showTradeDetails(tradeData) {
            var modal = document.getElementById('tradeModal');
            var modalBody = document.getElementById('tradeModalBody');
            
            var details = `
                <div class="modal-detail">
                    <span class="modal-detail-label">Время входа:</span>
                    <span class="modal-detail-value">${tradeData.entry_time}</span>
                </div>
                <div class="modal-detail">
                    <span class="modal-detail-label">Время выхода:</span>
                    <span class="modal-detail-value">${tradeData.exit_time}</span>
                </div>
                <div class="modal-detail">
                    <span class="modal-detail-label">Направление:</span>
                    <span class="modal-detail-value">${tradeData.direction}</span>
                </div>
                <div class="modal-detail">
                    <span class="modal-detail-label">Цена входа:</span>
                    <span class="modal-detail-value">$${parseFloat(tradeData.entry_price).toFixed(4)}</span>
                </div>
                <div class="modal-detail">
                    <span class="modal-detail-label">Цена выхода:</span>
                    <span class="modal-detail-value">$${parseFloat(tradeData.exit_price).toFixed(4)}</span>
                </div>
                <div class="modal-detail">
                    <span class="modal-detail-label">Количество:</span>
                    <span class="modal-detail-value">${parseFloat(tradeData.quantity).toFixed(4)}</span>
                </div>
                <div class="modal-detail">
                    <span class="modal-detail-label">P&L:</span>
                    <span class="modal-detail-value" style="color: ${tradeData.pnl >= 0 ? '#27ae60' : '#e74c3c'}">$${parseFloat(tradeData.pnl).toFixed(2)}</span>
                </div>
                <div class="modal-detail">
                    <span class="modal-detail-label">Комиссия:</span>
                    <span class="modal-detail-value">$${parseFloat(tradeData.commission).toFixed(2)}</span>
                </div>
                <div class="modal-detail">
                    <span class="modal-detail-label">Длительность:</span>
                    <span class="modal-detail-value">${tradeData.duration_minutes} мин</span>
                </div>
                <div class="modal-detail">
                    <span class="modal-detail-label">Причина входа:</span>
                    <span class="modal-detail-value">${tradeData.entry_reason}</span>
                </div>
                <div class="modal-detail">
                    <span class="modal-detail-label">Причина выхода:</span>
                    <span class="modal-detail-value">${tradeData.exit_reason}</span>
                </div>
            `;
            
            modalBody.innerHTML = details;
            modal.style.display = 'block';
        }
        
        function closeModal() {
            var modal = document.getElementById('tradeModal');
            modal.style.display = 'none';
        }
        
        // Close modal when clicking outside of it
        window.onclick = function(event) {
            var modal = document.getElementById('tradeModal');
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }
        </script>
        """
    
    def _comprehensive_template(self) -> str:
        """Comprehensive report template with all sections."""
        
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>{styles}</style>
        </head>
        <body>
            <div class="report-container">
                {header}
                {summary}
                {candlestick_chart}
                {detailed_tables}
                {footer}
            </div>
            {scripts}
        </body>
        </html>
        """
    
    def _summary_template(self) -> str:
        """Summary template with key metrics only."""
        
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>{styles}</style>
        </head>
        <body>
            <div class="report-container">
                {header}
                {summary}
                {performance_charts}
                {footer}
            </div>
            {scripts}
        </body>
        </html>
        """
    
    def _professional_template(self) -> str:
        """Professional template with enhanced styling."""
        
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>{styles}</style>
        </head>
        <body>
            <div class="report-container professional">
                <div class="report-toolbar">
                    <button onclick="printReport()" class="btn btn-print">🖨️ Print Report</button>
                </div>
                {header}
                {summary}
                {candlestick_chart}
                {detailed_tables}
                {footer}
            </div>
            {scripts}
        </body>
        </html>
        """
    
    def _minimal_template(self) -> str:
        """Minimal template for quick overview."""
        
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>{styles}</style>
        </head>
        <body>
            <div class="report-container minimal">
                {header}
                {summary}
                {footer}
            </div>
            {scripts}
        </body>
        </html>
        """
    
    def _base_styles(self) -> str:
        """Base CSS styles for all themes."""
        
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .report-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .report-toolbar {
            text-align: right;
            margin-bottom: 20px;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .btn-print {
            background-color: #1976D2;
            color: white;
        }
        
        .btn-print:hover {
            background-color: #1565C0;
        }
        
        h1, h2, h3, h4 {
            margin-bottom: 16px;
            color: #2c3e50;
        }
        
        h1 {
            font-size: 2.5em;
            text-align: center;
            margin-bottom: 8px;
        }
        
        h2 {
            font-size: 1.8em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
            margin-top: 40px;
        }
        
        h3 {
            font-size: 1.4em;
            color: #34495e;
        }
        
        h4 {
            font-size: 1.2em;
            color: #5a6c7d;
        }
        
        .report-header {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
        }
        
        .strategy-title {
            font-size: 2.5em;
            margin-bottom: 8px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .header-subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            margin-bottom: 20px;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .info-item {
            text-align: center;
        }
        
        .info-label {
            display: block;
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 4px;
        }
        
        .info-value {
            display: block;
            font-size: 1.2em;
            font-weight: bold;
        }
        
        .return-positive {
            color: #27ae60;
        }
        
        .return-negative {
            color: #e74c3c;
        }
        
        .grade-Aplus, .grade-A, .grade-Aminus {
            color: #27ae60;
        }
        
        .grade-Bplus, .grade-B, .grade-Bminus {
            color: #f39c12;
        }
        
        .grade-Cplus, .grade-C, .grade-Cminus {
            color: #e67e22;
        }
        
        .grade-D, .grade-F {
            color: #e74c3c;
        }
        
        .summary-section {
            margin: 40px 0;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }
        
        .summary-card {
            background: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .summary-card h3 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.3em;
        }
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .metric-label {
            color: #7f8c8d;
            font-size: 0.95em;
        }
        
        .metric-value {
            font-weight: 600;
            font-size: 1.05em;
        }
        
        .metric-highlight {
            background: #e8f4fd;
            padding: 12px;
            border-radius: 6px;
            margin-top: 15px;
        }
        
        .metric-highlight .metric-value {
            font-size: 1.2em;
            font-weight: bold;
        }
        
        .profit-positive {
            color: #27ae60;
        }
        
        .profit-negative {
            color: #e74c3c;
        }
        
        .drawdown {
            color: #e74c3c;
        }
        
        .charts-section {
            margin: 40px 0;
        }
        
        .chart-container {
            margin: 30px 0;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .chart-header {
            background: #f8f9fa;
            padding: 20px 25px;
            border-bottom: 1px solid #dee2e6;
        }
        
        .chart-header h3 {
            margin: 0;
            color: #495057;
        }
        
        .chart-description {
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .chart-content {
            padding: 20px;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        @media (max-width: 768px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }
            
            .summary-grid {
                grid-template-columns: 1fr;
            }
            
            .info-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .trade-analysis-section,
        .risk-analysis-section {
            margin: 40px 0;
        }
        
        .trade-stats-grid,
        .risk-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin-top: 25px;
        }
        
        .stat-card,
        .risk-card {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .stat-card h4,
        .risk-card h4 {
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        
        .stat-row,
        .risk-metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        
        .stat-label,
        .risk-label {
            color: #6c757d;
            font-size: 0.9em;
        }
        
        .stat-value,
        .risk-value {
            font-weight: 600;
        }
        
        .text-success {
            color: #28a745;
        }
        
        .text-danger {
            color: #dc3545;
        }
        
        .risk-assessment {
            margin-top: 10px;
            font-size: 0.9em;
        }
        
        .risk-low {
            color: #28a745;
            font-weight: 600;
        }
        
        .risk-moderate {
            color: #ffc107;
            font-weight: 600;
        }
        
        .risk-high {
            color: #fd7e14;
            font-weight: 600;
        }
        
        .risk-extreme {
            color: #dc3545;
            font-weight: 600;
        }
        
        .drawdown-value {
            color: #dc3545;
            font-weight: bold;
        }
        
        .tables-section {
            margin: 40px 0;
        }
        
        .table-container {
            overflow-x: auto;
            margin-top: 20px;
        }
        
        .trade-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .trade-table th {
            background: #f8f9fa;
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
            color: #495057;
            border-bottom: 2px solid #dee2e6;
        }
        
        .trade-table td {
            padding: 10px 15px;
            border-bottom: 1px solid #f1f3f4;
        }
        
        .trade-table tr:hover {
            background: #f8f9fa;
            cursor: pointer;
        }
        
        .badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .badge-long {
            background: #d4edda;
            color: #155724;
        }
        
        .badge-short {
            background: #f8d7da;
            color: #721c24;
        }
        
        .table-note {
            margin-top: 10px;
            color: #6c757d;
            font-style: italic;
            text-align: center;
        }
        
        .no-data {
            text-align: center;
            padding: 40px;
            color: #6c757d;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .report-footer {
            margin-top: 60px;
            padding: 30px 0;
            border-top: 2px solid #dee2e6;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .footer-content {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        
        .footer-left p {
            margin-bottom: 5px;
            color: #6c757d;
        }
        
        .footer-right {
            max-width: 50%;
        }
        
        .disclaimer {
            font-size: 0.85em;
            color: #6c757d;
            font-style: italic;
        }
        
        @media (max-width: 768px) {
            .footer-content {
                flex-direction: column;
                gap: 20px;
            }
            
            .footer-right {
                max-width: 100%;
            }
        }
        
        /* Modal styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.4);
        }
        
        .modal-content {
            background-color: #fefefe;
            margin: 15% auto;
            padding: 20px;
            border: none;
            border-radius: 8px;
            width: 80%;
            max-width: 600px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover,
        .close:focus {
            color: black;
            text-decoration: none;
        }
        
        .modal-header {
            padding-bottom: 15px;
            border-bottom: 1px solid #dee2e6;
            margin-bottom: 20px;
        }
        
        .modal-body {
            line-height: 1.6;
        }
        
        .modal-detail {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 5px 0;
        }
        
        .modal-detail-label {
            font-weight: 600;
            color: #495057;
        }
        
        .modal-detail-value {
            color: #212529;
        }
        
        @media print {
            body {
                background: white;
            }
            
            .report-container {
                box-shadow: none;
                max-width: none;
            }
            
            .report-toolbar {
                display: none;
            }
            
            .chart-container {
                page-break-inside: avoid;
            }
            
            .summary-card,
            .stat-card,
            .risk-card {
                page-break-inside: avoid;
            }
            
            .modal {
                display: none !important;
            }
        }
        """
    
    def _theme_styles(self, theme: str) -> str:
        """Theme-specific CSS styles."""
        
        themes = {
            'light': """
                .report-header {
                    background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
                }
                
                .summary-card {
                    border-left-color: #74b9ff;
                }
            """,
            'dark': """
                body {
                    background-color: #2c3e50;
                    color: #ecf0f1;
                }
                
                .report-container {
                    background-color: #34495e;
                    color: #ecf0f1;
                }
                
                .report-header {
                    background: linear-gradient(135deg, #2c3e50 0%, #4a6741 100%);
                }
                
                .summary-card,
                .stat-card,
                .risk-card,
                .chart-container {
                    background: #3c4858;
                    color: #ecf0f1;
                }
            """,
            'blue': """
                .report-header {
                    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                }
                
                h2 {
                    border-bottom-color: #2980b9;
                }
                
                .summary-card {
                    border-left-color: #3498db;
                }
            """,
            'professional': """
                .report-header {
                    background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                }
                
                .summary-card {
                    border-left-color: #2c3e50;
                    background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
                }
                
                .stat-card,
                .risk-card {
                    border-left: 3px solid #3498db;
                }
            """
        }
        
        return themes.get(theme, themes['professional'])