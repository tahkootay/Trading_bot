"""
Command-line interface for the Extremes Analyzer module.

This module provides command-line access to extremes detection and analysis
functionality with flexible parameters for different use cases.
"""

import argparse
import sys
from pathlib import Path
import json

from .analyzer import ExtremesAnalyzer
from .data_filter import DataFilter


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Extremes Analyzer - Statistical Analysis of Price Extremes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default settings
  python -m modules.extremes_analyzer --data data/SOLUSDT_5m_2024.csv

  # Custom threshold and symbol
  python -m modules.extremes_analyzer --data data/BTCUSDT_1h.csv --symbol BTCUSDT --timeframe 1h --threshold 50

  # Save results without showing plots
  python -m modules.extremes_analyzer --data data/SOLUSDT_5m.csv --no-plots --save-only

  # Custom output directory
  python -m modules.extremes_analyzer --data data/SOLUSDT_5m.csv --output output/my_analysis

  # Enrich indicators file with extremes and analytical data
  python -m modules.extremes_analyzer --enrich-indicators --data data/processed/SOLUSDT_indicators.csv --output data/processed/SOLUSDT_enriched.csv

  # Run statistical analysis with correlation heatmap and reports
  python -m modules.extremes_analyzer --statistical-analysis --data data/processed/SOLUSDT_enriched.csv --symbol SOLUSDT --timeframe 1h

  # Statistical analysis with timeframe filter and PDF report
  python -m modules.extremes_analyzer --statistical-analysis --data data/processed/SOLUSDT_enriched.csv --timeframe-filter 1h --report-format both

  # Filter Bollinger Bands lower touches (low <= bb_lower)
  python -m modules.extremes_analyzer --filter-bb-lower --data data/processed/SOLUSDT_enriched.csv

  # Filter RSI oversold conditions with custom threshold
  python -m modules.extremes_analyzer --filter-rsi-oversold --data data/processed/SOLUSDT_enriched.csv --rsi-threshold 25

  # Custom condition filtering
  python -m modules.extremes_analyzer --filter-custom --data data/processed/SOLUSDT_enriched.csv --condition "(low <= bb_lower) & (rsi < 30)"
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to CSV file with OHLCV data'
    )
    
    # Optional arguments
    parser.add_argument(
        '--symbol',
        type=str,
        default='SOLUSDT',
        help='Trading symbol for analysis (default: SOLUSDT)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        default='5m',
        help='Timeframe for analysis (default: 5m)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=3.0,
        help='Minimum price movement threshold in USDT (default: 3.0)'
    )
    
    parser.add_argument(
        '--window',
        type=int,
        default=5,
        help='Window size for extremes detection (default: 5)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output/extremes',
        help='Output directory for results (default: output/extremes)'
    )
    
    # Mode flags
    parser.add_argument(
        '--enrich-indicators',
        action='store_true',
        help='Enrich indicators file with extremes and analytical data'
    )
    
    parser.add_argument(
        '--statistical-analysis',
        action='store_true',
        help='Run comprehensive statistical analysis on enriched data'
    )
    
    parser.add_argument(
        '--filter-bb-lower',
        action='store_true',
        help='Filter rows where low <= bb_lower (Bollinger lower band touches)'
    )
    
    parser.add_argument(
        '--filter-bb-upper',
        action='store_true',
        help='Filter rows where high >= bb_upper (Bollinger upper band touches)'
    )
    
    parser.add_argument(
        '--filter-rsi-oversold',
        action='store_true',
        help='Filter rows where RSI <= threshold (oversold conditions)'
    )
    
    parser.add_argument(
        '--filter-rsi-overbought',
        action='store_true',
        help='Filter rows where RSI >= threshold (overbought conditions)'
    )
    
    parser.add_argument(
        '--filter-custom',
        action='store_true',
        help='Filter rows using custom pandas condition'
    )
    
    # Behavior flags
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation and display'
    )
    
    parser.add_argument(
        '--save-only',
        action='store_true',
        help='Save results but don\'t show interactive plots'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Don\'t save results to files'
    )
    
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Save results in JSON format only (skip CSV)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Extremes Analyzer v2.0.0'
    )
    
    # Statistical analysis specific options
    parser.add_argument(
        '--timeframe-filter',
        help='Filter data by specific timeframe (e.g., 1h, 5m)'
    )
    
    parser.add_argument(
        '--report-format',
        choices=['html', 'pdf', 'both'],
        default='html',
        help='Report format for statistical analysis (default: html)'
    )
    
    parser.add_argument(
        '--no-reports',
        action='store_true',
        help='Skip report generation in statistical analysis'
    )
    
    # Filtering specific options
    parser.add_argument(
        '--rsi-threshold',
        type=float,
        default=30.0,
        help='RSI threshold for oversold/overbought filtering (default: 30.0 for oversold, 70.0 for overbought)'
    )
    
    parser.add_argument(
        '--condition',
        type=str,
        help='Custom pandas condition string for filtering (e.g., "(low <= bb_lower) & (rsi < 30)")'
    )
    
    parser.add_argument(
        '--include-near-misses',
        action='store_true',
        help='Include near misses in Bollinger Bands filtering'
    )
    
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.001,
        help='Tolerance for near misses as percentage (default: 0.001 = 0.1%)'
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> bool:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        True if arguments are valid
    """
    # Check data file exists
    if not Path(args.data).exists():
        print(f"❌ Error: Data file not found: {args.data}")
        return False
    
    # Check threshold is positive
    if args.threshold <= 0:
        print(f"❌ Error: Threshold must be positive, got: {args.threshold}")
        return False
    
    # Check window size is valid
    if args.window < 1:
        print(f"❌ Error: Window size must be at least 1, got: {args.window}")
        return False
    
    return True


def main():
    """Main entry point for command-line interface."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_args(args):
        sys.exit(1)
    
    try:
        if not args.quiet:
            print("🚀 Starting Extremes Analysis...")
            print(f"📊 Data: {args.data}")
            print(f"💰 Symbol: {args.symbol}")
            print(f"⏰ Timeframe: {args.timeframe}")
            print(f"📏 Threshold: ${args.threshold} USDT")
            print(f"🪟 Window Size: {args.window}")
            print()
        
        # Initialize analyzer
        analyzer = ExtremesAnalyzer(
            min_threshold_usdt=args.threshold,
            window_size=args.window,
            output_dir=args.output
        )
        
        # Check if we're in enrich indicators mode
        if args.enrich_indicators:
            if not args.output or args.output == 'output/extremes':
                # Default enriched output path
                input_path = Path(args.data)
                output_path = input_path.parent / f"{input_path.stem}_enriched.csv"
            else:
                output_path = args.output
            
            print(f"🔄 Enriching indicators mode...")
            result_path = analyzer.enrich_indicators_file(
                indicators_path=args.data,
                output_path=str(output_path),
                symbol=args.symbol,
                timeframe=args.timeframe
            )
            
            if not args.quiet:
                print(f"\n✅ Indicators file enriched successfully!")
                print(f"📁 Output: {result_path}")
            
            return
        
        # Check if we're in statistical analysis mode
        if args.statistical_analysis:
            if not Path(args.data).exists():
                print(f"❌ Error: Enriched data file not found: {args.data}")
                sys.exit(1)
            
            print(f"🔄 Statistical analysis mode...")
            results = analyzer.run_statistical_analysis(
                enriched_data_path=args.data,
                symbol=args.symbol,
                timeframe=args.timeframe,
                timeframe_filter=args.timeframe_filter,
                generate_reports=not args.no_reports,
                report_format=args.report_format
            )
            
            if results['success']:
                if not args.quiet:
                    print(f"\n✅ Statistical analysis completed successfully!")
                    print(f"🎯 Extremes analyzed: {results['data_overview']['extremes_found']}")
                    if 'reports' in results['file_paths']:
                        print("📄 Generated reports:")
                        for report_path in results['file_paths']['reports']:
                            print(f"   - {report_path}")
            else:
                print(f"❌ Statistical analysis failed: {results['message']}")
                sys.exit(1)
            
            return
        
        # Check if we're in filtering mode
        if any([args.filter_bb_lower, args.filter_bb_upper, args.filter_rsi_oversold, 
                args.filter_rsi_overbought, args.filter_custom]):
            
            if not Path(args.data).exists():
                print(f"❌ Error: Data file not found: {args.data}")
                sys.exit(1)
            
            print(f"🔄 Data filtering mode...")
            
            # Initialize data filter
            data_filter = DataFilter(output_dir=str(Path(args.output) / "filtered_data"))
            data_filter.load_data(args.data)
            
            filter_results = []
            
            # Bollinger Bands lower touches
            if args.filter_bb_lower:
                try:
                    result_path = data_filter.filter_bollinger_lower_touch(
                        include_near_misses=args.include_near_misses,
                        tolerance=args.tolerance
                    )
                    if result_path:
                        filter_results.append(("Bollinger Lower Band Touches", result_path))
                except Exception as e:
                    print(f"❌ Error in BB Lower filtering: {e}")
            
            # Bollinger Bands upper touches
            if args.filter_bb_upper:
                try:
                    result_path = data_filter.filter_bollinger_upper_touch(
                        include_near_misses=args.include_near_misses,
                        tolerance=args.tolerance
                    )
                    if result_path:
                        filter_results.append(("Bollinger Upper Band Touches", result_path))
                except Exception as e:
                    print(f"❌ Error in BB Upper filtering: {e}")
            
            # RSI oversold
            if args.filter_rsi_oversold:
                try:
                    threshold = args.rsi_threshold if args.rsi_threshold <= 50 else 30.0
                    result_path = data_filter.filter_rsi_oversold(rsi_threshold=threshold)
                    if result_path:
                        filter_results.append((f"RSI Oversold (<= {threshold})", result_path))
                except Exception as e:
                    print(f"❌ Error in RSI oversold filtering: {e}")
            
            # RSI overbought
            if args.filter_rsi_overbought:
                try:
                    threshold = args.rsi_threshold if args.rsi_threshold >= 50 else 70.0
                    result_path = data_filter.filter_rsi_overbought(rsi_threshold=threshold)
                    if result_path:
                        filter_results.append((f"RSI Overbought (>= {threshold})", result_path))
                except Exception as e:
                    print(f"❌ Error in RSI overbought filtering: {e}")
            
            # Custom condition
            if args.filter_custom:
                if not args.condition:
                    print("❌ Error: --condition argument required for custom filtering")
                    sys.exit(1)
                
                try:
                    result_path = data_filter.filter_custom_condition(
                        condition_string=args.condition,
                        description=f"Custom: {args.condition}"
                    )
                    if result_path:
                        filter_results.append((f"Custom Condition", result_path))
                except Exception as e:
                    print(f"❌ Error in custom filtering: {e}")
            
            # Summary
            if filter_results and not args.quiet:
                print(f"\n✅ Data filtering completed!")
                print(f"📁 Generated {len(filter_results)} filtered datasets:")
                for description, path in filter_results:
                    print(f"   - {description}: {path}")
            elif not filter_results:
                print("⚠️ No data matched any of the filtering conditions")
            
            return
        
        # Standard analysis mode
        # Determine behavior flags
        show_plots = not args.no_plots and not args.save_only
        save_results = not args.no_save
        
        # Run analysis
        results = analyzer.run_full_analysis(
            data_path=args.data,
            symbol=args.symbol,
            timeframe=args.timeframe,
            save_results=save_results,
            show_plots=show_plots
        )
        
        # Handle results
        if results['success']:
            if not args.quiet:
                print("\n✅ Analysis completed successfully!")
            
            # Save in specific format if requested
            if save_results and args.json_only and not results['extremes_data'].empty:
                analyzer.save_results("json")
        
        else:
            print(f"❌ Analysis failed: {results['message']}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()