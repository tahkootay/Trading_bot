#!/usr/bin/env python3
"""
Module 3: HTML Report Generation System

Converts backtest results (JSON) into interactive HTML reports with charts and analysis.
Provides professional-quality reports suitable for sharing and documentation.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .html_generator import HTMLReportGenerator
from .chart_builder import ChartBuilder
from .template_manager import TemplateManager
from .config import ReporterConfig


class ReportGenerator:
    """Main report generation orchestrator."""
    
    def __init__(self, config: ReporterConfig):
        self.config = config
        self.html_generator = HTMLReportGenerator(config)
        self.chart_builder = ChartBuilder(config)
        self.template_manager = TemplateManager()
    
    async def generate_report(
        self,
        results_file: str,
        output_file: Optional[str] = None,
        template: str = "comprehensive"
    ) -> str:
        """
        Generate HTML report from backtest results.
        
        Args:
            results_file: Path to JSON results file from backtester
            output_file: Optional path for output HTML file
            template: Report template to use
            
        Returns:
            Path to generated HTML report
        """
        
        print(f"📊 Generating report from: {results_file}")
        print(f"📋 Template: {template}")
        
        # Load backtest results
        results_data = await self._load_results(results_file)
        
        # Generate auto filename if not provided
        if not output_file:
            output_file = self._generate_output_filename(results_data, results_file)
        
        print(f"💾 Output file: {output_file}")
        
        # Build interactive charts
        charts = await self.chart_builder.build_all_charts(results_data)
        
        # Generate HTML report
        html_content = await self.html_generator.generate_report(
            results_data=results_data,
            charts=charts,
            template=template
        )
        
        # Save HTML report
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Generate summary
        self._print_summary(results_data, str(output_path))
        
        return str(output_path)
    
    async def _load_results(self, results_file: str) -> Dict[str, Any]:
        """Load backtest results from JSON file."""
        
        file_path = Path(results_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Validate required structure
            required_keys = ['backtest_info', 'performance_metrics', 'trades', 'equity_curve']
            missing_keys = [key for key in required_keys if key not in data]
            
            if missing_keys:
                # Try legacy format
                if 'strategy_info' in data:
                    data['backtest_info'] = data['strategy_info']
                if 'trade_summary' in data:
                    data['trades'] = data['trade_summary'].get('trades', [])
                
                # Check again
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    raise ValueError(f"Missing required data sections: {missing_keys}")
            
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in results file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load results: {e}")
    
    def _generate_output_filename(self, results_data: Dict[str, Any], results_file: str) -> str:
        """Generate output filename based on results data."""
        
        # Extract strategy name and timestamp
        backtest_info = results_data.get('backtest_info', {})
        strategy_name = backtest_info.get('name', 'Strategy').replace(' ', '_')
        
        # Try to get timestamp from backtest or use current time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if 'backtest_timestamp' in backtest_info:
            try:
                dt = datetime.fromisoformat(backtest_info['backtest_timestamp'].replace('Z', '+00:00'))
                timestamp = dt.strftime("%Y%m%d_%H%M%S")
            except:
                pass
        
        # Generate filename
        filename = f"{strategy_name}_Report_{timestamp}.html"
        
        # Use same directory as results file
        results_path = Path(results_file)
        output_dir = results_path.parent / "reports" if results_path.parent.name != "reports" else results_path.parent
        
        return str(output_dir / filename)
    
    def _print_summary(self, results_data: Dict[str, Any], output_path: str) -> None:
        """Print report generation summary."""
        
        backtest_info = results_data.get('backtest_info', {})
        metrics = results_data.get('performance_metrics', {})
        trades = results_data.get('trades', [])
        equity_points = results_data.get('equity_curve', [])
        
        print("\n" + "="*60)
        print("📊 REPORT GENERATED SUCCESSFULLY")
        print("="*60)
        
        print(f"Strategy: {backtest_info.get('name', 'Unknown')}")
        print(f"Period: {backtest_info.get('start_date', 'N/A')} to {backtest_info.get('end_date', 'N/A')}")
        print(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"Total Trades: {len(trades)}")
        print(f"Data Points: {len(equity_points)}")
        
        print(f"\n📄 Report saved to: {output_path}")
        
        # File size
        try:
            file_size = Path(output_path).stat().st_size
            size_mb = file_size / 1024 / 1024
            print(f"📁 File size: {size_mb:.2f} MB")
        except:
            pass
        
        print("="*60)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="Generate HTML reports from backtest results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic report generation
  python main.py --results ./output/backtests/my_backtest.json
  
  # Generate with specific template and output location
  python main.py --results ./results.json --template professional --output ./reports/analysis.html
  
  # Generate multiple reports
  python main.py --results ./backtests/*.json --template summary --batch
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--results", "-r",
        required=True,
        help="Path to backtest results JSON file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        help="Output HTML file path (defaults to auto-generated)"
    )
    
    parser.add_argument(
        "--template", "-t",
        choices=["summary", "comprehensive", "professional", "minimal"],
        default="comprehensive",
        help="Report template (default: comprehensive)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file (YAML)"
    )
    
    parser.add_argument(
        "--charts-only",
        action="store_true",
        help="Generate only charts (for debugging)"
    )
    
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Generate static charts instead of interactive"
    )
    
    parser.add_argument(
        "--theme",
        choices=["light", "dark", "blue", "professional"],
        default="professional",
        help="Report color theme (default: professional)"
    )
    
    return parser


async def main():
    """Main entry point."""
    
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config:
            config = ReporterConfig.from_file(args.config)
        else:
            config = ReporterConfig(
                theme=args.theme,
                interactive_charts=not args.no_interactive,
                charts_only=args.charts_only
            )
        
        # Create generator and run
        generator = ReportGenerator(config)
        output_path = await generator.generate_report(
            results_file=args.results,
            output_file=args.output,
            template=args.template
        )
        
        print(f"\n✅ Report generation completed!")
        print(f"🌐 Open in browser: file://{Path(output_path).absolute()}")
        
    except KeyboardInterrupt:
        print("\n🛑 Report generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())