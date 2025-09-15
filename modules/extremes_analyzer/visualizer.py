"""
Visualization tools for extremes analysis.

This module provides plotting functionality for price charts with
detected extremes and basic analysis visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import Optional, Tuple
import numpy as np


class ExtremesVisualizer:
    """Creates visualizations for extremes analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size for plots (width, height)
        """
        self.figsize = figsize
        plt.style.use('default')
    
    def plot_price_with_extremes(
        self, 
        price_data: pd.DataFrame, 
        extremes_data: pd.DataFrame,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot price chart with detected extremes.
        
        Args:
            price_data: OHLCV DataFrame with timestamp column
            extremes_data: DataFrame with detected extremes
            title: Optional plot title
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[3, 1])
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(price_data['timestamp']):
            price_data = price_data.copy()
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        
        # Plot price line
        ax1.plot(price_data['timestamp'], price_data['close'], 
                color='black', linewidth=1, label='Close Price')
        
        # Plot extremes
        if not extremes_data.empty:
            # Ensure extremes timestamps are datetime
            extremes_plot = extremes_data.copy()
            if not pd.api.types.is_datetime64_any_dtype(extremes_plot['timestamp']):
                extremes_plot['timestamp'] = pd.to_datetime(extremes_plot['timestamp'])
            
            # Plot maxima (red dots)
            maxima = extremes_plot[extremes_plot['extreme_type'] == 'max']
            if not maxima.empty:
                ax1.scatter(maxima['timestamp'], maxima['extreme_price'], 
                           color='red', s=60, marker='v', label='Local Maxima', zorder=5)
            
            # Plot minima (green dots) 
            minima = extremes_plot[extremes_plot['extreme_type'] == 'min']
            if not minima.empty:
                ax1.scatter(minima['timestamp'], minima['extreme_price'], 
                           color='green', s=60, marker='^', label='Local Minima', zorder=5)
        
        # Format price chart
        ax1.set_ylabel('Price (USDT)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        if title:
            ax1.set_title(title, fontsize=14, fontweight='bold')
        
        # Plot volume
        ax2.bar(price_data['timestamp'], price_data['volume'], 
                color='lightblue', alpha=0.7, width=0.8)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_extremes_summary(
        self, 
        extremes_data: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot summary statistics for detected extremes.
        
        Args:
            extremes_data: DataFrame with detected extremes
            save_path: Optional path to save the plot
        """
        if extremes_data.empty:
            print("No extremes data to plot.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. Distribution of price changes
        ax1 = axes[0, 0]
        extremes_data['strength'].hist(bins=20, ax=ax1, alpha=0.7, color='skyblue')
        ax1.set_title('Distribution of Price Movement Strength')
        ax1.set_xlabel('Price Change (USDT)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 2. Extremes by type
        ax2 = axes[0, 1]
        type_counts = extremes_data['extreme_type'].value_counts()
        ax2.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                colors=['lightcoral', 'lightgreen'])
        ax2.set_title('Distribution by Extreme Type')
        
        # 3. RSI distribution
        ax3 = axes[1, 0]
        valid_rsi = extremes_data['rsi14'].dropna()
        if not valid_rsi.empty:
            valid_rsi.hist(bins=15, ax=ax3, alpha=0.7, color='orange')
            ax3.set_title('RSI Distribution at Extremes')
            ax3.set_xlabel('RSI Value')
            ax3.set_ylabel('Frequency')
            ax3.axvline(30, color='red', linestyle='--', alpha=0.7, label='Oversold')
            ax3.axvline(70, color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No RSI data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('RSI Distribution at Extremes')
        ax3.grid(True, alpha=0.3)
        
        # 4. Trend context
        ax4 = axes[1, 1]
        trend_counts = extremes_data['trend_context'].value_counts()
        if not trend_counts.empty:
            trend_counts.plot(kind='bar', ax=ax4, color=['lightblue', 'lightcoral', 'lightgray'])
            ax4.set_title('Trend Context (vs EMA200)')
            ax4.set_ylabel('Count')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No trend data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Trend Context (vs EMA200)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary plot saved to: {save_path}")
        
        plt.show()
    
    def print_extremes_table(self, extremes_data: pd.DataFrame, max_rows: int = 10) -> None:
        """
        Print formatted table of extremes data.
        
        Args:
            extremes_data: DataFrame with detected extremes
            max_rows: Maximum number of rows to display
        """
        if extremes_data.empty:
            print("No extremes detected.")
            return
        
        # Select key columns for display
        display_cols = [
            'timestamp', 'extreme_type', 'extreme_price', 'strength',
            'rsi14', 'trend_context'
        ]
        
        available_cols = [col for col in display_cols if col in extremes_data.columns]
        display_data = extremes_data[available_cols].head(max_rows)
        
        print(f"\n🎯 Detected Extremes Summary (showing {len(display_data)} of {len(extremes_data)}):")
        print("=" * 80)
        
        for _, row in display_data.iterrows():
            print(f"📅 {row['timestamp']}")
            print(f"   Type: {row['extreme_type'].upper():<6} Price: ${row['extreme_price']:>8.2f}")
            if 'strength' in row and not pd.isna(row['strength']):
                print(f"   Movement: ${row['strength']:>6.2f}")
            if 'rsi14' in row and not pd.isna(row['rsi14']):
                print(f"   RSI: {row['rsi14']:>6.1f}")
            if 'trend_context' in row:
                print(f"   Trend: {row['trend_context']}")
            print()
        
        print(f"📊 Total extremes found: {len(extremes_data)}")
        if len(extremes_data) > 0:
            avg_strength = extremes_data['strength'].mean()
            print(f"📈 Average movement strength: ${avg_strength:.2f} USDT")