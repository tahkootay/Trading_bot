"""
Automated report generation module for statistical analysis.

This module provides functionality to generate comprehensive HTML and PDF reports
from statistical analysis results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt

# HTML template generation
from jinja2 import Template

# Try to import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class ReportGenerator:
    """Automated report generation for statistical analysis results."""
    
    def __init__(self, output_dir: str = "output/reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_data = None
        self.plots_data = {}
        
    def load_analysis_data(self, analysis_results: Dict[str, Any]) -> None:
        """
        Load analysis results data.
        
        Args:
            analysis_results: Results from StatisticalAnalyzer
        """
        self.analysis_data = analysis_results
        print("✅ Analysis data loaded for report generation")
    
    def encode_plot_to_base64(self, plot_path: str) -> str:
        """
        Encode plot image to base64 for HTML embedding.
        
        Args:
            plot_path: Path to plot image
            
        Returns:
            Base64 encoded image string
        """
        if not Path(plot_path).exists():
            return ""
        
        with open(plot_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        return f"data:image/png;base64,{encoded_string}"
    
    def generate_html_report(self, 
                           heatmap_path: Optional[str] = None,
                           distribution_plots: Optional[List[str]] = None,
                           output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            heatmap_path: Path to correlation heatmap
            distribution_plots: List of distribution plot paths
            output_path: Custom output path for report
            
        Returns:
            Path to generated HTML report
        """
        if self.analysis_data is None:
            raise ValueError("No analysis data loaded. Call load_analysis_data() first.")
        
        # Prepare template data
        template_data = {
            'title': 'Statistical Analysis Report',
            'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'analysis_data': self.analysis_data
        }
        
        # Add encoded plots
        if heatmap_path and Path(heatmap_path).exists():
            template_data['heatmap'] = self.encode_plot_to_base64(heatmap_path)
        
        if distribution_plots:
            template_data['distribution_plots'] = [
                self.encode_plot_to_base64(plot_path) 
                for plot_path in distribution_plots 
                if Path(plot_path).exists()
            ]
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 40px; 
            background-color: #f8f9fa;
            color: #333;
        }
        .header { 
            text-align: center; 
            margin-bottom: 40px; 
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }
        .section { 
            margin: 30px 0; 
            padding: 25px; 
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .section h2 { 
            color: #2c3e50; 
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .stat-label {
            color: #7f8c8d;
            font-size: 14px;
        }
        .plot-container { 
            text-align: center; 
            margin: 30px 0;
        }
        .plot-container img { 
            max-width: 100%; 
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .correlation-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .correlation-table th, .correlation-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .correlation-table th {
            background-color: #3498db;
            color: white;
        }
        .correlation-table tr:hover {
            background-color: #f5f5f5;
        }
        .positive-corr { color: #27ae60; font-weight: bold; }
        .negative-corr { color: #e74c3c; font-weight: bold; }
        .metadata { 
            font-size: 12px; 
            color: #7f8c8d; 
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: #ecf0f1;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated on {{ generation_date }}</p>
    </div>
    
    <!-- Data Overview -->
    <div class="section">
        <h2>📊 Data Overview</h2>
        {% if analysis_data.data_overview %}
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ analysis_data.data_overview.total_rows }}</div>
                <div class="stat-label">Total Rows</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ analysis_data.data_overview.symbol }}</div>
                <div class="stat-label">Symbol</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ analysis_data.data_overview.timeframe }}</div>
                <div class="stat-label">Timeframe</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ analysis_data.data_overview.date_range.start[:10] }}</div>
                <div class="stat-label">Start Date</div>
            </div>
        </div>
        {% endif %}
    </div>
    
    <!-- Extremes Analysis -->
    <div class="section">
        <h2>🎯 Extremes Analysis</h2>
        {% if analysis_data.extremes_analysis %}
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ analysis_data.extremes_analysis.total_extremes }}</div>
                <div class="stat-label">Total Extremes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ analysis_data.extremes_analysis.maxima_count }}</div>
                <div class="stat-label">Maxima</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ analysis_data.extremes_analysis.minima_count }}</div>
                <div class="stat-label">Minima</div>
            </div>
            {% if analysis_data.extremes_analysis.strength_stats %}
            <div class="stat-card">
                <div class="stat-value">${{ "%.2f"|format(analysis_data.extremes_analysis.strength_stats.mean) }}</div>
                <div class="stat-label">Average Strength</div>
            </div>
            {% endif %}
        </div>
        
        {% if analysis_data.extremes_analysis.rsi_at_extremes %}
        <h3>RSI at Extremes</h3>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ "%.1f"|format(analysis_data.extremes_analysis.rsi_at_extremes.mean) }}</div>
                <div class="stat-label">Mean RSI</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ analysis_data.extremes_analysis.rsi_at_extremes.oversold_count }}</div>
                <div class="stat-label">Oversold (<30)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ analysis_data.extremes_analysis.rsi_at_extremes.overbought_count }}</div>
                <div class="stat-label">Overbought (>70)</div>
            </div>
        </div>
        {% endif %}
        {% endif %}
    </div>
    
    <!-- Correlation Heatmap -->
    {% if heatmap %}
    <div class="section">
        <h2>🔥 Correlation Heatmap</h2>
        <div class="plot-container">
            <img src="{{ heatmap }}" alt="Correlation Heatmap">
        </div>
    </div>
    {% endif %}
    
    <!-- Strong Correlations -->
    {% if analysis_data.correlation_insights %}
    <div class="section">
        <h2>🔗 Strong Correlations</h2>
        <table class="correlation-table">
            <thead>
                <tr>
                    <th>Variable 1</th>
                    <th>Variable 2</th>
                    <th>Correlation</th>
                </tr>
            </thead>
            <tbody>
                {% for corr in analysis_data.correlation_insights.strong_correlations[:10] %}
                <tr>
                    <td>{{ corr.variable1 }}</td>
                    <td>{{ corr.variable2 }}</td>
                    <td class="{% if corr.correlation > 0 %}positive-corr{% else %}negative-corr{% endif %}">
                        {{ "%.3f"|format(corr.correlation) }}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}
    
    <!-- Distribution Plots -->
    {% if distribution_plots %}
    <div class="section">
        <h2>📈 Distribution Analysis</h2>
        {% for plot in distribution_plots %}
        <div class="plot-container">
            <img src="{{ plot }}" alt="Distribution Plot">
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    <!-- Indicator Statistics -->
    {% if analysis_data.indicator_statistics %}
    <div class="section">
        <h2>📊 Indicator Statistics</h2>
        <table class="correlation-table">
            <thead>
                <tr>
                    <th>Indicator</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
            </thead>
            <tbody>
                {% for indicator, stats in analysis_data.indicator_statistics.items() %}
                <tr>
                    <td>{{ indicator.upper() }}</td>
                    <td>{{ "%.2f"|format(stats.mean) }}</td>
                    <td>{{ "%.2f"|format(stats.median) }}</td>
                    <td>{{ "%.2f"|format(stats.std) }}</td>
                    <td>{{ "%.2f"|format(stats.min) }}</td>
                    <td>{{ "%.2f"|format(stats.max) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}
    
    <div class="metadata">
        <p>Generated by Trading Bot Statistical Analysis Module v2.0</p>
        <p>Analysis Date: {{ generation_date }}</p>
    </div>
</body>
</html>
        """
        
        # Render template
        template = Template(html_template)
        html_content = template.render(**template_data)
        
        # Save HTML report
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"statistical_report_{timestamp}.html"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 HTML report generated: {output_path}")
        
        return str(output_path)
    
    def generate_pdf_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate PDF report (requires reportlab).
        
        Args:
            output_path: Custom output path for PDF
            
        Returns:
            Path to generated PDF report
        """
        if not HAS_REPORTLAB:
            raise ImportError("reportlab not installed. Cannot generate PDF reports.")
        
        if self.analysis_data is None:
            raise ValueError("No analysis data loaded. Call load_analysis_data() first.")
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"statistical_report_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.HexColor('#2c3e50')
        )
        
        story.append(Paragraph("Statistical Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Generation info
        info_text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 30))
        
        # Data Overview
        if 'data_overview' in self.analysis_data:
            story.append(Paragraph("Data Overview", styles['Heading2']))
            
            overview_data = self.analysis_data['data_overview']
            overview_table_data = [
                ['Metric', 'Value'],
                ['Total Rows', str(overview_data.get('total_rows', 'N/A'))],
                ['Symbol', overview_data.get('symbol', 'N/A')],
                ['Timeframe', overview_data.get('timeframe', 'N/A')],
                ['Start Date', overview_data.get('date_range', {}).get('start', 'N/A')[:10]],
                ['End Date', overview_data.get('date_range', {}).get('end', 'N/A')[:10]]
            ]
            
            overview_table = Table(overview_table_data)
            overview_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(overview_table)
            story.append(Spacer(1, 20))
        
        # Extremes Analysis
        if 'extremes_analysis' in self.analysis_data:
            story.append(Paragraph("Extremes Analysis", styles['Heading2']))
            
            extremes_data = self.analysis_data['extremes_analysis']
            extremes_table_data = [
                ['Metric', 'Value'],
                ['Total Extremes', str(extremes_data.get('total_extremes', 'N/A'))],
                ['Maxima Count', str(extremes_data.get('maxima_count', 'N/A'))],
                ['Minima Count', str(extremes_data.get('minima_count', 'N/A'))]
            ]
            
            if 'strength_stats' in extremes_data:
                strength_stats = extremes_data['strength_stats']
                extremes_table_data.extend([
                    ['Average Strength', f"${strength_stats.get('mean', 0):.2f}"],
                    ['Max Strength', f"${strength_stats.get('max', 0):.2f}"]
                ])
            
            extremes_table = Table(extremes_table_data)
            extremes_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(extremes_table)
            story.append(Spacer(1, 20))
        
        # Strong Correlations
        if 'correlation_insights' in self.analysis_data:
            story.append(Paragraph("Strong Correlations (|r| > 0.5)", styles['Heading2']))
            
            correlations = self.analysis_data['correlation_insights']['strong_correlations'][:10]
            if correlations:
                corr_table_data = [['Variable 1', 'Variable 2', 'Correlation']]
                for corr in correlations:
                    corr_table_data.append([
                        corr['variable1'],
                        corr['variable2'],
                        f"{corr['correlation']:.3f}"
                    ])
                
                corr_table = Table(corr_table_data)
                corr_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(corr_table)
                story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        
        print(f"📄 PDF report generated: {output_path}")
        
        return str(output_path)