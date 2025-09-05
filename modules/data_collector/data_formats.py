"""
Data formatting utilities for different output formats.
"""

import json
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any
from pathlib import Path

import pandas as pd

from bybit_client import Candle


class OutputFormat(Enum):
    """Supported output formats."""
    CSV = "CSV"
    JSON = "JSON"
    PARQUET = "PARQUET"


class DataFormatter:
    """Handles data formatting and conversion to different output formats."""
    
    def candles_to_dataframe(self, candles: List[Candle]) -> pd.DataFrame:
        """Convert list of Candle objects to pandas DataFrame."""
        
        if not candles:
            return pd.DataFrame(columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe'
            ])
        
        data = []
        for candle in candles:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume,
                'symbol': candle.symbol,
                'timeframe': candle.timeframe
            })
        
        df = pd.DataFrame(data)
        
        # Ensure timestamp is datetime and sorted
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def candles_to_dict(self, candles: List[Candle]) -> Dict[str, Any]:
        """Convert list of Candle objects to dictionary structure."""
        
        if not candles:
            return {
                'symbol': '',
                'timeframe': '',
                'data': [],
                'metadata': {
                    'total_candles': 0,
                    'start_time': None,
                    'end_time': None,
                    'collection_time': datetime.now().isoformat()
                }
            }
        
        data = []
        for candle in candles:
            data.append({
                'timestamp': candle.timestamp.isoformat(),
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })
        
        # Sort by timestamp
        data.sort(key=lambda x: x['timestamp'])
        
        return {
            'symbol': candles[0].symbol,
            'timeframe': candles[0].timeframe,
            'data': data,
            'metadata': {
                'total_candles': len(data),
                'start_time': data[0]['timestamp'] if data else None,
                'end_time': data[-1]['timestamp'] if data else None,
                'collection_time': datetime.now().isoformat()
            }
        }
    
    def dataframe_to_csv(self, df: pd.DataFrame, file_path: Path) -> None:
        """Save DataFrame to CSV file."""
        df.to_csv(file_path, index=False)
    
    def dict_to_json(self, data: Dict[str, Any], file_path: Path) -> None:
        """Save dictionary to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def dataframe_to_parquet(self, df: pd.DataFrame, file_path: Path) -> None:
        """Save DataFrame to Parquet file."""
        df.to_parquet(file_path, index=False)
    
    def validate_data(self, candles: List[Candle]) -> Dict[str, Any]:
        """Validate candle data and return validation report."""
        
        if not candles:
            return {
                'is_valid': False,
                'errors': ['No data provided'],
                'warnings': [],
                'statistics': {}
            }
        
        errors = []
        warnings = []
        
        # Check for required fields
        for i, candle in enumerate(candles):
            if candle.open <= 0:
                errors.append(f"Invalid open price at index {i}: {candle.open}")
            if candle.high <= 0:
                errors.append(f"Invalid high price at index {i}: {candle.high}")
            if candle.low <= 0:
                errors.append(f"Invalid low price at index {i}: {candle.low}")
            if candle.close <= 0:
                errors.append(f"Invalid close price at index {i}: {candle.close}")
            if candle.volume < 0:
                errors.append(f"Invalid volume at index {i}: {candle.volume}")
            
            # OHLC validation
            if candle.high < max(candle.open, candle.close, candle.low):
                errors.append(f"High price is not highest at index {i}")
            if candle.low > min(candle.open, candle.close, candle.high):
                errors.append(f"Low price is not lowest at index {i}")
        
        # Check for duplicates
        timestamps = [candle.timestamp for candle in candles]
        if len(timestamps) != len(set(timestamps)):
            warnings.append("Duplicate timestamps found")
        
        # Check for gaps (basic check)
        sorted_candles = sorted(candles, key=lambda x: x.timestamp)
        if len(sorted_candles) > 1:
            gaps = 0
            for i in range(1, len(sorted_candles)):
                time_diff = sorted_candles[i].timestamp - sorted_candles[i-1].timestamp
                # This is a simplified gap detection - would need timeframe-specific logic
                if time_diff.total_seconds() > 3600:  # More than 1 hour gap
                    gaps += 1
            
            if gaps > 0:
                warnings.append(f"Found {gaps} potential data gaps")
        
        # Calculate statistics
        if candles:
            prices = [candle.close for candle in candles]
            volumes = [candle.volume for candle in candles]
            
            statistics = {
                'total_candles': len(candles),
                'date_range': {
                    'start': sorted_candles[0].timestamp.isoformat(),
                    'end': sorted_candles[-1].timestamp.isoformat()
                },
                'price_statistics': {
                    'min': min(prices),
                    'max': max(prices),
                    'first': sorted_candles[0].close,
                    'last': sorted_candles[-1].close
                },
                'volume_statistics': {
                    'min': min(volumes),
                    'max': max(volumes),
                    'total': sum(volumes)
                }
            }
        else:
            statistics = {}
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'statistics': statistics
        }
    
    def generate_summary_report(self, candles: List[Candle], file_path: Path) -> Dict[str, Any]:
        """Generate a summary report for the collected data."""
        
        validation = self.validate_data(candles)
        
        if not candles:
            report = {
                'summary': 'No data collected',
                'validation': validation,
                'recommendations': ['Check symbol name and date range', 'Verify exchange connectivity']
            }
        else:
            df = self.candles_to_dataframe(candles)
            
            # Basic analysis
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
            volatility = df['close'].pct_change().std() * 100
            avg_volume = df['volume'].mean()
            
            report = {
                'summary': f"Successfully collected {len(candles)} candles for {candles[0].symbol}",
                'data_overview': {
                    'symbol': candles[0].symbol,
                    'timeframe': candles[0].timeframe,
                    'total_candles': len(candles),
                    'date_range': {
                        'start': df['timestamp'].iloc[0].isoformat(),
                        'end': df['timestamp'].iloc[-1].isoformat()
                    }
                },
                'market_summary': {
                    'opening_price': float(df['open'].iloc[0]),
                    'closing_price': float(df['close'].iloc[-1]),
                    'price_change_percent': round(price_change, 2),
                    'highest_price': float(df['high'].max()),
                    'lowest_price': float(df['low'].min()),
                    'average_volume': round(avg_volume, 2),
                    'total_volume': round(df['volume'].sum(), 2),
                    'volatility_percent': round(volatility, 2)
                },
                'validation': validation,
                'file_info': {
                    'path': str(file_path),
                    'format': file_path.suffix[1:].upper(),
                    'size_mb': round(file_path.stat().st_size / 1024 / 1024, 2) if file_path.exists() else 0
                }
            }
            
            # Recommendations
            recommendations = []
            if validation['warnings']:
                recommendations.extend([f"Warning: {w}" for w in validation['warnings']])
            if volatility > 5:
                recommendations.append("High volatility detected - consider risk management")
            if avg_volume < 1000:
                recommendations.append("Low average volume - liquidity may be limited")
            
            report['recommendations'] = recommendations if recommendations else ["Data looks good!"]
        
        return report