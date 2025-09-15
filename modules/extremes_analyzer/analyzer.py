"""
Main analyzer class that coordinates extremes detection and analysis.

This module provides the main interface for the extremes analysis workflow,
integrating data loading, extremes detection, and visualization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

from .extremes_detector import ExtremesDetector
from .visualizer import ExtremesVisualizer
from .statistical_analyzer import StatisticalAnalyzer
from .report_generator import ReportGenerator


class ExtremesAnalyzer:
    """Main class for extremes analysis workflow."""
    
    def __init__(
        self, 
        min_threshold_usdt: float = 3.0, 
        window_size: int = 5,
        output_dir: str = "output/extremes"
    ):
        """
        Initialize the extremes analyzer.
        
        Args:
            min_threshold_usdt: Minimum price movement in USDT to qualify as extreme
            window_size: Window size for local extrema detection
            output_dir: Directory for saving results
        """
        self.detector = ExtremesDetector(min_threshold_usdt, window_size)
        self.visualizer = ExtremesVisualizer()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store analysis results
        self.price_data: Optional[pd.DataFrame] = None
        self.extremes_data: Optional[pd.DataFrame] = None
        self.analysis_metadata: Dict[str, Any] = {}
    
    def load_data(self, data_path: str, symbol: str = "SOLUSDT", timeframe: str = "5m") -> pd.DataFrame:
        """
        Load OHLCV data from CSV file.
        
        Args:
            data_path: Path to CSV file with OHLCV data
            symbol: Trading symbol for metadata
            timeframe: Timeframe for metadata
            
        Returns:
            Loaded DataFrame with OHLCV data
        """
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Validate required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add metadata as attributes
        df.attrs['symbol'] = symbol
        df.attrs['timeframe'] = timeframe
        
        self.price_data = df
        
        # Update analysis metadata
        self.analysis_metadata.update({
            'symbol': symbol,
            'timeframe': timeframe,
            'data_start': df['timestamp'].iloc[0].isoformat(),
            'data_end': df['timestamp'].iloc[-1].isoformat(),
            'total_candles': len(df),
            'analysis_timestamp': datetime.now().isoformat()
        })
        
        print(f"✅ Loaded {len(df)} candles for {symbol} ({timeframe})")
        print(f"📅 Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        
        return df
    
    def analyze(self) -> pd.DataFrame:
        """
        Perform extremes detection and analysis.
        
        Returns:
            DataFrame with detected extremes and analysis
        """
        if self.price_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("🔍 Detecting extremes...")
        extremes_df = self.detector.detect_extremes(self.price_data)
        
        self.extremes_data = extremes_df
        
        # Update metadata
        if not extremes_df.empty:
            self.analysis_metadata.update({
                'extremes_found': len(extremes_df),
                'maxima_count': len(extremes_df[extremes_df['extreme_type'] == 'max']),
                'minima_count': len(extremes_df[extremes_df['extreme_type'] == 'min']),
                'avg_strength': float(extremes_df['strength'].mean()),
                'max_strength': float(extremes_df['strength'].max())
            })
        else:
            self.analysis_metadata.update({
                'extremes_found': 0,
                'maxima_count': 0,
                'minima_count': 0,
                'avg_strength': 0.0,
                'max_strength': 0.0
            })
        
        print(f"🎯 Found {len(extremes_df)} extremes")
        
        return extremes_df
    
    def visualize(self, show_plots: bool = True, save_plots: bool = True) -> None:
        """
        Create visualizations for the analysis.
        
        Args:
            show_plots: Whether to display plots
            save_plots: Whether to save plots to files
        """
        if self.price_data is None or self.extremes_data is None:
            raise ValueError("No analysis data available. Run analyze() first.")
        
        symbol = self.analysis_metadata.get('symbol', 'UNKNOWN')
        timeframe = self.analysis_metadata.get('timeframe', 'UNKNOWN')
        
        title = f"{symbol} Price Chart with Extremes ({timeframe})"
        
        # Create plots
        if show_plots or save_plots:
            try:
                print("📊 Creating visualizations...")
                
                # Price chart with extremes
                save_path = None
                if save_plots:
                    save_path = self.output_dir / f"{symbol}_{timeframe}_extremes_chart.png"
                
                self.visualizer.plot_price_with_extremes(
                    self.price_data, 
                    self.extremes_data,
                    title=title,
                    save_path=save_path if save_plots else None
                )
                
                # Summary statistics
                if not self.extremes_data.empty:
                    save_path = None
                    if save_plots:
                        save_path = self.output_dir / f"{symbol}_{timeframe}_extremes_summary.png"
                    
                    self.visualizer.plot_extremes_summary(
                        self.extremes_data,
                        save_path=save_path if save_plots else None
                    )
            except Exception as e:
                print(f"⚠️  Visualization failed: {e}")
                print("Analysis results are still available.")
    
    def save_results(self, format: str = "csv") -> str:
        """
        Save analysis results to file.
        
        Args:
            format: Output format ("csv" or "json")
            
        Returns:
            Path to saved file
        """
        if self.extremes_data is None:
            raise ValueError("No analysis data available. Run analyze() first.")
        
        symbol = self.analysis_metadata.get('symbol', 'UNKNOWN')
        timeframe = self.analysis_metadata.get('timeframe', 'UNKNOWN')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "csv":
            filename = f"{symbol}_{timeframe}_extremes_{timestamp}.csv"
            filepath = self.output_dir / filename
            self.extremes_data.to_csv(filepath, index=False)
        elif format.lower() == "json":
            filename = f"{symbol}_{timeframe}_extremes_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Combine extremes data and metadata
            output_data = {
                'metadata': self.analysis_metadata,
                'extremes': self.extremes_data.to_dict('records')
            }
            
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"💾 Results saved to: {filepath}")
        return str(filepath)
    
    def print_summary(self) -> None:
        """Print analysis summary to console."""
        if self.extremes_data is None:
            print("No analysis data available.")
            return
        
        print("\n" + "="*60)
        print("📈 EXTREMES ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Symbol: {self.analysis_metadata.get('symbol', 'N/A')}")
        print(f"Timeframe: {self.analysis_metadata.get('timeframe', 'N/A')}")
        print(f"Analysis Period: {self.analysis_metadata.get('data_start', 'N/A')[:19]} to {self.analysis_metadata.get('data_end', 'N/A')[:19]}")
        print(f"Total Candles: {self.analysis_metadata.get('total_candles', 0):,}")
        
        print(f"\n🎯 EXTREMES DETECTED:")
        print(f"Total Extremes: {self.analysis_metadata.get('extremes_found', 0)}")
        print(f"Local Maxima: {self.analysis_metadata.get('maxima_count', 0)}")
        print(f"Local Minima: {self.analysis_metadata.get('minima_count', 0)}")
        
        if self.analysis_metadata.get('extremes_found', 0) > 0:
            print(f"\n📊 MOVEMENT STATISTICS:")
            print(f"Average Strength: ${self.analysis_metadata.get('avg_strength', 0):.2f} USDT")
            print(f"Maximum Strength: ${self.analysis_metadata.get('max_strength', 0):.2f} USDT")
        
        print("="*60)
        
        # Show detailed table
        if not self.extremes_data.empty:
            self.visualizer.print_extremes_table(self.extremes_data)
    
    def run_full_analysis(
        self, 
        data_path: str, 
        symbol: str = "SOLUSDT", 
        timeframe: str = "5m",
        save_results: bool = True,
        show_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete analysis workflow.
        
        Args:
            data_path: Path to CSV data file
            symbol: Trading symbol
            timeframe: Timeframe
            save_results: Whether to save results to files
            show_plots: Whether to display plots
            
        Returns:
            Dictionary with analysis results and metadata
        """
        try:
            # Load data
            self.load_data(data_path, symbol, timeframe)
            
            # Analyze
            extremes_df = self.analyze()
            
            # Print summary
            self.print_summary()
            
            # Visualize
            self.visualize(show_plots=show_plots, save_plots=save_results)
            
            # Save results
            if save_results and not extremes_df.empty:
                self.save_results("csv")
                self.save_results("json")
            
            return {
                'metadata': self.analysis_metadata,
                'extremes_data': extremes_df,
                'success': True,
                'message': f"Analysis completed successfully. Found {len(extremes_df)} extremes."
            }
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'metadata': self.analysis_metadata,
                'extremes_data': pd.DataFrame(),
                'success': False,
                'message': error_msg
            }
    
    def enrich_indicators_file(
        self, 
        indicators_path: str, 
        output_path: str, 
        symbol: str = "SOLUSDT", 
        timeframe: str = "1h"
    ) -> str:
        """
        Enrich indicators file with extremes detection and additional analytical data.
        
        Args:
            indicators_path: Path to CSV file with indicators data
            output_path: Path for enriched output file
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Path to enriched file
        """
        print(f"📊 Enriching indicators file: {indicators_path}")
        
        # Load indicators data
        if not Path(indicators_path).exists():
            raise FileNotFoundError(f"Indicators file not found: {indicators_path}")
        
        df = pd.read_csv(indicators_path)
        
        # Validate required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required OHLCV columns: {missing_cols}")
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} rows with indicators")
        
        # Extract OHLCV for extremes detection
        ohlcv_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Detect extremes
        print("🔍 Detecting price extremes...")
        extremes_df = self.detector.detect_extremes(ohlcv_df)
        
        # Add extremes markers to main dataframe
        df = self._add_extremes_markers(df, extremes_df)
        
        # Add indicator extremes
        df = self._add_indicator_extremes(df)
        
        # Add additional analytical columns
        df = self._add_analytical_columns(df)
        
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save enriched data
        df.to_csv(output_path, index=False)
        
        print(f"✅ Enriched file saved to: {output_path}")
        print(f"📈 Added columns: extremes markers, indicator extremes, analytical data")
        print(f"🎯 Total extremes found: {len(extremes_df) if not extremes_df.empty else 0}")
        
        return str(output_path)
    
    def _add_extremes_markers(self, df: pd.DataFrame, extremes_df: pd.DataFrame) -> pd.DataFrame:
        """Add price extremes markers to the main dataframe."""
        # Initialize extremes columns
        df['is_extreme'] = False
        df['extreme_type'] = ''
        df['extreme_strength'] = 0.0
        df['extreme_price_change'] = 0.0
        
        if not extremes_df.empty:
            # Map extremes to main dataframe
            for _, extreme in extremes_df.iterrows():
                # Find matching timestamp
                mask = df['timestamp'] == extreme['timestamp']
                if mask.any():
                    idx = df[mask].index[0]
                    df.loc[idx, 'is_extreme'] = True
                    df.loc[idx, 'extreme_type'] = extreme['extreme_type']
                    df.loc[idx, 'extreme_strength'] = extreme['strength']
                    df.loc[idx, 'extreme_price_change'] = extreme.get('price_change', 0.0)
        
        return df
    
    def _add_indicator_extremes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator extremes detection."""
        # RSI extremes
        if 'rsi' in df.columns:
            df['rsi_oversold'] = df['rsi'] < 30
            df['rsi_overbought'] = df['rsi'] > 70
            df['rsi_extreme'] = df['rsi_oversold'] | df['rsi_overbought']
        
        # Bollinger Bands extremes
        if all(col in df.columns for col in ['close', 'bb_upper', 'bb_lower']):
            df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['close'] < 0.05
            df['bb_breakout_upper'] = df['close'] > df['bb_upper']
            df['bb_breakout_lower'] = df['close'] < df['bb_lower']
            df['bb_extreme'] = df['bb_breakout_upper'] | df['bb_breakout_lower']
        
        # MACD extremes
        if all(col in df.columns for col in ['macd_line', 'macd_signal']):
            df['macd_crossover'] = (
                (df['macd_line'] > df['macd_signal']) & 
                (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
            )
            df['macd_crossunder'] = (
                (df['macd_line'] < df['macd_signal']) & 
                (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
            )
            df['macd_signal_change'] = df['macd_crossover'] | df['macd_crossunder']
        
        return df
    
    def _add_analytical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional analytical columns."""
        # Price momentum
        df['price_change_1h'] = df['close'].pct_change(1) * 100
        df['price_change_24h'] = df['close'].pct_change(24) * 100 if len(df) > 24 else 0
        
        # Volume analysis
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_spike'] = df['volume'] > (df['volume_sma_10'] * 2)
        
        # Volatility
        df['high_low_spread'] = ((df['high'] - df['low']) / df['close']) * 100
        df['open_close_change'] = ((df['close'] - df['open']) / df['open']) * 100
        
        # Support/Resistance levels (simplified)
        df['resistance_level'] = df['high'].rolling(window=20).max()
        df['support_level'] = df['low'].rolling(window=20).min()
        df['near_resistance'] = abs(df['close'] - df['resistance_level']) / df['close'] < 0.01
        df['near_support'] = abs(df['close'] - df['support_level']) / df['close'] < 0.01
        
        return df
    
    def run_statistical_analysis(
        self,
        enriched_data_path: str,
        symbol: str = "SOLUSDT",
        timeframe: str = "1h",
        timeframe_filter: Optional[str] = None,
        generate_reports: bool = True,
        report_format: str = "html"
    ) -> Dict[str, Any]:
        """
        Run comprehensive statistical analysis on enriched data.
        
        Args:
            enriched_data_path: Path to enriched CSV data
            symbol: Trading symbol
            timeframe: Timeframe
            timeframe_filter: Optional filter for specific timeframe
            generate_reports: Whether to generate reports
            report_format: Report format ('html', 'pdf', 'both')
            
        Returns:
            Dictionary with analysis results and file paths
        """
        print(f"🚀 Starting statistical analysis for {symbol} ({timeframe})")
        
        try:
            # Initialize statistical analyzer
            stat_analyzer = StatisticalAnalyzer(
                output_dir=str(self.output_dir / "statistical_analysis")
            )
            
            # Load enriched data
            df = stat_analyzer.load_enriched_data(enriched_data_path, timeframe_filter)
            
            # Calculate correlations
            print("📊 Calculating correlations...")
            correlation_matrix = stat_analyzer.calculate_correlations()
            
            # Create correlation heatmap
            print("🔥 Creating correlation heatmap...")
            heatmap_path = stat_analyzer.create_correlation_heatmap(interactive=False)
            heatmap_interactive_path = stat_analyzer.create_correlation_heatmap(interactive=True)
            
            # Analyze extremes distribution
            print("🎯 Analyzing extremes distribution...")
            extremes_analysis = stat_analyzer.analyze_extremes_distribution()
            
            # Create distribution plots
            print("📈 Creating distribution plots...")
            distribution_plots = stat_analyzer.create_distribution_plots()
            
            # Generate statistical summary
            print("📊 Generating statistical summary...")
            statistical_summary = stat_analyzer.generate_statistical_summary()
            
            # Save analysis results
            results_path = stat_analyzer.save_analysis_results()
            
            results = {
                'success': True,
                'message': 'Statistical analysis completed successfully',
                'data_overview': {
                    'total_rows': len(df),
                    'extremes_found': extremes_analysis.get('total_extremes', 0),
                    'symbol': symbol,
                    'timeframe': timeframe
                },
                'file_paths': {
                    'heatmap_static': heatmap_path,
                    'heatmap_interactive': heatmap_interactive_path,
                    'distribution_plots': distribution_plots,
                    'analysis_results': results_path
                },
                'analysis_data': stat_analyzer.analysis_results
            }
            
            # Generate reports if requested
            if generate_reports:
                print("📄 Generating reports...")
                report_generator = ReportGenerator(
                    output_dir=str(self.output_dir / "reports")
                )
                
                report_generator.load_analysis_data(stat_analyzer.analysis_results)
                
                report_paths = []
                
                if report_format in ['html', 'both']:
                    html_report = report_generator.generate_html_report(
                        heatmap_path=heatmap_path,
                        distribution_plots=distribution_plots
                    )
                    report_paths.append(html_report)
                
                if report_format in ['pdf', 'both']:
                    try:
                        pdf_report = report_generator.generate_pdf_report()
                        report_paths.append(pdf_report)
                    except ImportError:
                        print("⚠️ reportlab not installed, skipping PDF report")
                
                results['file_paths']['reports'] = report_paths
            
            # Print summary
            print("\n" + "="*60)
            print("📈 STATISTICAL ANALYSIS SUMMARY")
            print("="*60)
            print(f"Symbol: {symbol}")
            print(f"Timeframe: {timeframe}")
            print(f"Total Rows Analyzed: {len(df):,}")
            print(f"Extremes Found: {extremes_analysis.get('total_extremes', 0)}")
            print(f"Strong Correlations Found: {len(statistical_summary.get('correlation_insights', {}).get('strong_correlations', []))}")
            print(f"Analysis Results: {results_path}")
            
            if generate_reports and 'reports' in results['file_paths']:
                print("Generated Reports:")
                for report_path in results['file_paths']['reports']:
                    print(f"  - {report_path}")
            
            print("="*60)
            
            return results
            
        except Exception as e:
            error_msg = f"Statistical analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'message': error_msg,
                'data_overview': {},
                'file_paths': {},
                'analysis_data': {}
            }