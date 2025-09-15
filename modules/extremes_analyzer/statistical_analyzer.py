"""
Statistical analysis module for extremes data.

This module provides advanced statistical analysis capabilities including
correlation heatmaps, distribution analysis, and automated report generation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Try to import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("Warning: reportlab not installed. PDF reports will not be available.")

warnings.filterwarnings('ignore', category=FutureWarning)


class StatisticalAnalyzer:
    """Advanced statistical analysis for extremes and indicators data."""
    
    def __init__(self, output_dir: str = "output/statistical_analysis"):
        """
        Initialize statistical analyzer.
        
        Args:
            output_dir: Directory for saving analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.analysis_results = {}
        
        # Configure plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_enriched_data(self, file_path: str, timeframe_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Load enriched data file with optional timeframe filtering.
        
        Args:
            file_path: Path to enriched CSV file
            timeframe_filter: Optional timeframe to filter (e.g., '1h', '5m')
            
        Returns:
            Loaded and filtered DataFrame
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Apply timeframe filter if specified
        if timeframe_filter and 'timeframe' in df.columns:
            df = df[df['timeframe'] == timeframe_filter]
            print(f"📊 Filtered data to {timeframe_filter}: {len(df)} rows")
        
        # Clean data - remove rows with all NaN values in indicators
        indicator_cols = [col for col in df.columns if col in 
                         ['rsi', 'sma', 'ema', 'macd_line', 'atr', 'bb_upper', 'bb_middle', 'bb_lower']]
        
        if indicator_cols:
            df = df.dropna(subset=indicator_cols, how='all')
        
        self.df = df
        print(f"✅ Loaded {len(df)} rows for statistical analysis")
        
        return df
    
    def calculate_correlations(self, include_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix for numerical columns.
        
        Args:
            include_columns: Specific columns to include in analysis
            
        Returns:
            Correlation matrix DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_enriched_data() first.")
        
        # Select numerical columns for correlation analysis
        if include_columns:
            numeric_cols = [col for col in include_columns if col in self.df.columns]
        else:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove ID-like columns and timestamps
            exclude_cols = ['timestamp', 'volume', 'volume_sma_10']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Calculate correlations
        correlation_matrix = self.df[numeric_cols].corr()
        
        # Store in results
        self.analysis_results['correlation_matrix'] = correlation_matrix
        self.analysis_results['numeric_columns'] = numeric_cols
        
        print(f"📊 Calculated correlations for {len(numeric_cols)} numerical columns")
        
        return correlation_matrix
    
    def create_correlation_heatmap(self, save_path: Optional[str] = None, 
                                 interactive: bool = False) -> str:
        """
        Create correlation heatmap visualization.
        
        Args:
            save_path: Optional custom save path
            interactive: Whether to create interactive plotly heatmap
            
        Returns:
            Path to saved heatmap
        """
        if 'correlation_matrix' not in self.analysis_results:
            self.calculate_correlations()
        
        corr_matrix = self.analysis_results['correlation_matrix']
        
        if interactive:
            # Create interactive heatmap with plotly
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap - Indicators and Extremes"
            )
            
            fig.update_layout(
                title_x=0.5,
                width=1000,
                height=800
            )
            
            if not save_path:
                save_path = self.output_dir / "correlation_heatmap_interactive.html"
            
            fig.write_html(save_path)
            
        else:
            # Create static heatmap with matplotlib/seaborn
            plt.figure(figsize=(14, 12))
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8}
            )
            
            plt.title('Correlation Heatmap - Indicators and Extremes', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            if not save_path:
                save_path = self.output_dir / "correlation_heatmap.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📈 Correlation heatmap saved to: {save_path}")
        return str(save_path)
    
    def analyze_extremes_distribution(self) -> Dict[str, Any]:
        """
        Analyze distribution of extremes and their characteristics.
        
        Returns:
            Dictionary with distribution analysis results
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_enriched_data() first.")
        
        # Filter extremes data
        extremes_df = self.df[self.df['is_extreme'] == True].copy()
        
        if extremes_df.empty:
            return {"error": "No extremes found in data"}
        
        results = {
            'total_extremes': len(extremes_df),
            'maxima_count': len(extremes_df[extremes_df['extreme_type'] == 'max']),
            'minima_count': len(extremes_df[extremes_df['extreme_type'] == 'min']),
            'strength_stats': extremes_df['extreme_strength'].describe().to_dict(),
            'price_change_stats': extremes_df['extreme_price_change'].describe().to_dict()
        }
        
        # RSI analysis at extremes
        if 'rsi' in extremes_df.columns:
            rsi_at_extremes = extremes_df['rsi'].dropna()
            if not rsi_at_extremes.empty:
                results['rsi_at_extremes'] = {
                    'mean': float(rsi_at_extremes.mean()),
                    'median': float(rsi_at_extremes.median()),
                    'std': float(rsi_at_extremes.std()),
                    'oversold_count': int((rsi_at_extremes < 30).sum()),
                    'overbought_count': int((rsi_at_extremes > 70).sum())
                }
        
        # Bollinger Bands analysis at extremes
        if all(col in extremes_df.columns for col in ['bb_upper', 'bb_lower', 'close']):
            bb_touches = extremes_df[
                (extremes_df['close'] >= extremes_df['bb_upper']) | 
                (extremes_df['close'] <= extremes_df['bb_lower'])
            ]
            results['bb_extreme_touches'] = len(bb_touches)
            results['bb_extreme_ratio'] = len(bb_touches) / len(extremes_df)
        
        # Volume analysis at extremes
        if 'volume_spike' in extremes_df.columns:
            volume_spikes_at_extremes = extremes_df['volume_spike'].sum()
            results['volume_spikes_at_extremes'] = int(volume_spikes_at_extremes)
            results['volume_spike_ratio'] = volume_spikes_at_extremes / len(extremes_df)
        
        self.analysis_results['extremes_distribution'] = results
        
        print(f"📊 Analyzed {len(extremes_df)} extremes distribution")
        
        return results
    
    def create_distribution_plots(self, save_dir: Optional[str] = None) -> List[str]:
        """
        Create various distribution plots for extremes analysis.
        
        Args:
            save_dir: Directory to save plots
            
        Returns:
            List of saved plot paths
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_enriched_data() first.")
        
        if not save_dir:
            save_dir = self.output_dir / "distributions"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        extremes_df = self.df[self.df['is_extreme'] == True].copy()
        saved_plots = []
        
        if extremes_df.empty:
            print("⚠️ No extremes found for distribution plots")
            return saved_plots
        
        # 1. Extreme strength distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        extremes_df['extreme_strength'].hist(bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Extreme Strength')
        plt.xlabel('Strength (USDT)')
        plt.ylabel('Frequency')
        
        # 2. RSI distribution at extremes
        plt.subplot(2, 2, 2)
        if 'rsi' in extremes_df.columns:
            rsi_values = extremes_df['rsi'].dropna()
            if not rsi_values.empty:
                rsi_values.hist(bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                plt.axvline(30, color='red', linestyle='--', label='Oversold')
                plt.axvline(70, color='red', linestyle='--', label='Overbought')
                plt.legend()
        plt.title('RSI Distribution at Extremes')
        plt.xlabel('RSI')
        plt.ylabel('Frequency')
        
        # 3. Price change distribution
        plt.subplot(2, 2, 3)
        extremes_df['extreme_price_change'].hist(bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.title('Price Change Distribution at Extremes')
        plt.xlabel('Price Change (USDT)')
        plt.ylabel('Frequency')
        
        # 4. Extreme types pie chart
        plt.subplot(2, 2, 4)
        extreme_counts = extremes_df['extreme_type'].value_counts()
        plt.pie(extreme_counts.values, labels=extreme_counts.index, autopct='%1.1f%%')
        plt.title('Distribution of Extreme Types')
        
        plt.tight_layout()
        
        plot_path = save_dir / "extremes_distributions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(str(plot_path))
        
        # Create boxplot for indicators at extremes vs non-extremes
        if len(self.df[self.df['is_extreme'] == False]) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            indicators = ['rsi', 'extreme_strength', 'high_low_spread', 'price_change_1h']
            for i, indicator in enumerate(indicators):
                if indicator in self.df.columns:
                    row, col = i // 2, i % 2
                    
                    # Prepare data for boxplot
                    extreme_data = extremes_df[indicator].dropna()
                    non_extreme_data = self.df[self.df['is_extreme'] == False][indicator].dropna()
                    
                    if not extreme_data.empty and not non_extreme_data.empty:
                        data_for_box = [non_extreme_data, extreme_data]
                        labels = ['Non-Extremes', 'Extremes']
                        
                        axes[row, col].boxplot(data_for_box, labels=labels)
                        axes[row, col].set_title(f'{indicator} Distribution')
                        axes[row, col].grid(True, alpha=0.3)
            
            plt.tight_layout()
            boxplot_path = save_dir / "indicators_boxplots.png"
            plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(str(boxplot_path))
        
        print(f"📊 Created {len(saved_plots)} distribution plots in {save_dir}")
        
        return saved_plots
    
    def generate_statistical_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistical summary.
        
        Returns:
            Dictionary with statistical summary
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_enriched_data() first.")
        
        summary = {
            'data_overview': {
                'total_rows': len(self.df),
                'date_range': {
                    'start': self.df['timestamp'].min().isoformat(),
                    'end': self.df['timestamp'].max().isoformat()
                },
                'symbol': self.df['symbol'].iloc[0] if 'symbol' in self.df.columns else 'Unknown',
                'timeframe': self.df['timeframe'].iloc[0] if 'timeframe' in self.df.columns else 'Unknown'
            }
        }
        
        # Add extremes analysis
        if 'extremes_distribution' in self.analysis_results:
            summary['extremes_analysis'] = self.analysis_results['extremes_distribution']
        else:
            summary['extremes_analysis'] = self.analyze_extremes_distribution()
        
        # Add correlation insights
        if 'correlation_matrix' in self.analysis_results:
            corr_matrix = self.analysis_results['correlation_matrix']
            
            # Find strongest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    if not pd.isna(corr_value) and abs(corr_value) > 0.5:
                        corr_pairs.append({
                            'variable1': col1,
                            'variable2': col2,
                            'correlation': float(corr_value)
                        })
            
            # Sort by absolute correlation value
            corr_pairs = sorted(corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
            
            summary['correlation_insights'] = {
                'strong_correlations': corr_pairs[:10],  # Top 10 strongest correlations
                'total_correlations_analyzed': len(corr_matrix.columns)
            }
        
        # Add indicator statistics
        indicator_stats = {}
        key_indicators = ['rsi', 'sma', 'ema', 'macd_line', 'atr', 'bb_upper', 'bb_middle', 'bb_lower']
        
        for indicator in key_indicators:
            if indicator in self.df.columns:
                series = self.df[indicator].dropna()
                if not series.empty:
                    indicator_stats[indicator] = {
                        'mean': float(series.mean()),
                        'median': float(series.median()),
                        'std': float(series.std()),
                        'min': float(series.min()),
                        'max': float(series.max()),
                        'count': int(series.count())
                    }
        
        summary['indicator_statistics'] = indicator_stats
        
        self.analysis_results['statistical_summary'] = summary
        
        print("📊 Generated comprehensive statistical summary")
        
        return summary
    
    def save_analysis_results(self, file_path: Optional[str] = None) -> str:
        """
        Save analysis results to JSON file.
        
        Args:
            file_path: Optional custom file path
            
        Returns:
            Path to saved file
        """
        if not self.analysis_results:
            raise ValueError("No analysis results to save. Run analysis methods first.")
        
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.output_dir / f"statistical_analysis_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        import json
        
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            return obj
        
        # Deep convert the analysis results
        def deep_convert(item):
            if isinstance(item, dict):
                return {key: deep_convert(value) for key, value in item.items()}
            elif isinstance(item, list):
                return [deep_convert(element) for element in item]
            else:
                return convert_numpy_types(item)
        
        converted_results = deep_convert(self.analysis_results)
        
        with open(file_path, 'w') as f:
            json.dump(converted_results, f, indent=2, default=str)
        
        print(f"💾 Analysis results saved to: {file_path}")
        
        return str(file_path)