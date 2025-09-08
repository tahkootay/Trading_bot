#!/usr/bin/env python3
"""
Module 1: Historical Data Collection System

Automated collection of historical futures data from Bybit exchange.
Supports multiple timeframes, date ranges, and output formats.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .bybit_client import BybitDataCollector
from .data_formats import DataFormatter, OutputFormat
from .config import DataCollectorConfig


class HistoricalDataCollector:
    """Main historical data collection system."""
    
    def __init__(self, config: DataCollectorConfig):
        self.config = config
        self.collector = BybitDataCollector(
            testnet=config.testnet,
            rate_limit=config.rate_limit
        )
        self.formatter = DataFormatter()
        
    async def collect_data(
        self,
        symbol: str,
        timeframes: List[str],
        start_date: datetime,
        end_date: datetime,
        output_format: OutputFormat = OutputFormat.CSV
    ) -> Dict[str, str]:
        """
        Collect historical data for specified parameters.
        
        Args:
            symbol: Trading pair (e.g., SOLUSDT, BTCUSDT)
            timeframes: List of timeframes (1m, 5m, 15m, 1h, etc.)
            start_date: Start date for data collection
            end_date: End date for data collection
            output_format: Output format (CSV, JSON, PARQUET)
            
        Returns:
            Dictionary with timeframe -> file_path mapping
        """
        results = {}
        
        print(f"📊 Starting data collection for {symbol}")
        print(f"📅 Period: {start_date.date()} to {end_date.date()}")
        print(f"⏰ Timeframes: {', '.join(timeframes)}")
        print(f"💾 Output format: {output_format.value}")
        
        for timeframe in timeframes:
            print(f"\n🔍 Collecting {timeframe} data...")
            
            try:
                # Collect data from Bybit
                candles = await self.collector.get_historical_data(
                    symbol=symbol,
                    interval=timeframe,
                    start_time=start_date,
                    end_time=end_date
                )
                
                if not candles:
                    print(f"⚠️ No data found for {timeframe}")
                    continue
                
                # Format and save data
                output_path = await self._save_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles=candles,
                    output_format=output_format,
                    start_date=start_date,
                    end_date=end_date
                )
                
                results[timeframe] = str(output_path)
                print(f"✅ Saved {len(candles)} candles to {output_path}")
                
            except Exception as e:
                print(f"❌ Error collecting {timeframe} data: {e}")
                continue
        
        # Generate metadata
        metadata_path = await self._generate_metadata(
            symbol=symbol,
            timeframes=list(results.keys()),
            start_date=start_date,
            end_date=end_date,
            file_paths=results
        )
        
        print(f"\n📋 Metadata saved to {metadata_path}")
        print(f"🎉 Data collection completed! {len(results)} timeframes collected.")
        
        return results
    
    async def _save_data(
        self,
        symbol: str,
        timeframe: str,
        candles: List,
        output_format: OutputFormat,
        start_date: datetime,
        end_date: datetime
    ) -> Path:
        """Save candles data in specified format."""
        
        # Create filename
        date_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        filename = f"{symbol}_{timeframe}_{date_str}"
        
        # Ensure output directory exists
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if output_format == OutputFormat.CSV:
            file_path = output_dir / f"{filename}.csv"
            df = self.formatter.candles_to_dataframe(candles)
            df.to_csv(file_path, index=False)
            
        elif output_format == OutputFormat.JSON:
            file_path = output_dir / f"{filename}.json"
            data = self.formatter.candles_to_dict(candles)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        elif output_format == OutputFormat.PARQUET:
            file_path = output_dir / f"{filename}.parquet"
            df = self.formatter.candles_to_dataframe(candles)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, file_path)
            
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return file_path
    
    async def _generate_metadata(
        self,
        symbol: str,
        timeframes: List[str],
        start_date: datetime,
        end_date: datetime,
        file_paths: Dict[str, str]
    ) -> Path:
        """Generate metadata file for collected data."""
        
        metadata = {
            "collection_info": {
                "symbol": symbol,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "collection_timestamp": datetime.now().isoformat(),
                "timeframes": timeframes
            },
            "files": file_paths,
            "statistics": {}
        }
        
        # Add basic statistics for each file
        for timeframe, file_path in file_paths.items():
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    metadata["statistics"][timeframe] = {
                        "total_candles": len(df),
                        "first_timestamp": df['timestamp'].iloc[0],
                        "last_timestamp": df['timestamp'].iloc[-1],
                        "price_range": {
                            "min": float(df['low'].min()),
                            "max": float(df['high'].max())
                        }
                    }
            except Exception as e:
                metadata["statistics"][timeframe] = {"error": str(e)}
        
        # Save metadata
        output_dir = Path(self.config.output_directory)
        metadata_path = output_dir / f"{symbol}_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Collect historical futures data from Bybit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 7 days of 5m data for SOLUSDT
  python main.py --symbol SOLUSDT --timeframe 5m --period week
  
  # Collect multiple timeframes for custom date range  
  python main.py --symbol BTCUSDT --timeframes 1m,5m,1h --start 2024-01-01 --end 2024-01-07
  
  # Collect all timeframes and save as Parquet
  python main.py --symbol ETHUSDT --all-timeframes --period month --format parquet
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--symbol", "-s",
        required=True,
        help="Trading pair symbol (e.g., SOLUSDT, BTCUSDT)"
    )
    
    # Timeframe arguments (mutually exclusive group)
    timeframe_group = parser.add_mutually_exclusive_group(required=True)
    timeframe_group.add_argument(
        "--timeframe", "-t",
        help="Single timeframe (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)"
    )
    timeframe_group.add_argument(
        "--timeframes", "-T",
        help="Multiple timeframes comma-separated (e.g., 1m,5m,1h)"
    )
    timeframe_group.add_argument(
        "--all-timeframes", "-A",
        action="store_true",
        help="Collect all available timeframes"
    )
    
    # Date range arguments (mutually exclusive group)
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--period", "-p",
        choices=["hour", "day", "week", "month", "3months", "6months", "year"],
        help="Standard time period from now backwards"
    )
    date_group.add_argument(
        "--start",
        help="Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM)"
    )
    
    parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD or YYYY-MM-DD HH:MM), defaults to now"
    )
    
    # Output options
    parser.add_argument(
        "--format", "-f",
        choices=["csv", "json", "parquet"],
        default="csv",
        help="Output format (default: csv)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="./data/raw",
        help="Output directory (default: ./data/raw)"
    )
    
    # Connection options
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use testnet instead of mainnet"
    )
    
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=10,
        help="API rate limit (requests per second, default: 10)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file (YAML)"
    )
    
    return parser


def align_time_to_timeframe(dt: datetime, timeframe: str, is_start: bool = True) -> datetime:
    """
    Align datetime to timeframe boundaries.
    
    Args:
        dt: Datetime to align
        timeframe: Timeframe string (e.g., "5m", "1h")
        is_start: If True, align to start of interval, else to end
        
    Returns:
        Aligned datetime
    """
    # Get timeframe in minutes
    timeframe_map = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, 
        "12h": 720, "1d": 1440
    }
    
    if timeframe not in timeframe_map:
        return dt
    
    minutes = timeframe_map[timeframe]
    
    if minutes < 60:  # Sub-hourly alignment
        # Align to minute boundary
        aligned_minute = (dt.minute // minutes) * minutes
        aligned = dt.replace(minute=aligned_minute, second=0, microsecond=0)
        
        if not is_start and aligned < dt:
            # For end time, move to next interval if we're not exactly on boundary
            aligned += timedelta(minutes=minutes)
            
    elif minutes < 1440:  # Hourly alignment
        hours = minutes // 60
        aligned_hour = (dt.hour // hours) * hours
        aligned = dt.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)
        
        if not is_start and aligned < dt:
            aligned += timedelta(hours=hours)
            
    else:  # Daily alignment
        aligned = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        
        if not is_start and aligned < dt:
            aligned += timedelta(days=1)
    
    return aligned


def parse_date_range(args) -> tuple[datetime, datetime]:
    """Parse date range from arguments with proper timeframe alignment."""
    
    # Get timeframe for alignment (from first timeframe if multiple)
    if hasattr(args, 'timeframe') and args.timeframe:
        timeframe = args.timeframe
    elif hasattr(args, 'timeframes') and args.timeframes:
        timeframe = args.timeframes.split(',')[0].strip()
    else:
        timeframe = "5m"  # Default fallback
    
    if args.period:
        # Standard periods - align to current time boundaries
        raw_end = datetime.now()
        
        if args.period == "hour":
            raw_start = raw_end - timedelta(hours=1)
        elif args.period == "day":
            raw_start = raw_end - timedelta(days=1)
        elif args.period == "week":
            raw_start = raw_end - timedelta(weeks=1)
        elif args.period == "month":
            raw_start = raw_end - timedelta(days=30)
        elif args.period == "3months":
            raw_start = raw_end - timedelta(days=90)
        elif args.period == "6months":
            raw_start = raw_end - timedelta(days=180)
        elif args.period == "year":
            raw_start = raw_end - timedelta(days=365)
        else:
            raise ValueError(f"Unknown period: {args.period}")
        
        # Align boundaries to timeframe
        start_date = align_time_to_timeframe(raw_start, timeframe, is_start=True)
        # For end date, align to the most recent closed candle
        end_date = align_time_to_timeframe(raw_end, timeframe, is_start=True)  # Use start alignment to get closed candle
            
    else:
        # Custom date range
        try:
            # Try parsing with time
            try:
                start_date = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
            except ValueError:
                # Try parsing date only
                start_date = datetime.strptime(args.start, "%Y-%m-%d")
                
            if args.end:
                try:
                    end_date = datetime.strptime(args.end, "%Y-%m-%d %H:%M")
                except ValueError:
                    end_date = datetime.strptime(args.end, "%Y-%m-%d")
            else:
                end_date = datetime.now()
            
            # Align custom dates to timeframe boundaries
            start_date = align_time_to_timeframe(start_date, timeframe, is_start=True)
            end_date = align_time_to_timeframe(end_date, timeframe, is_start=False)
                
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}")
    
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")
        
    print(f"🕐 Aligned time range: {start_date} to {end_date} (timeframe: {timeframe})")
        
    return start_date, end_date


def parse_timeframes(args) -> List[str]:
    """Parse timeframes from arguments."""
    
    if args.all_timeframes:
        return ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]
    elif args.timeframes:
        return [tf.strip() for tf in args.timeframes.split(",")]
    elif args.timeframe:
        return [args.timeframe]
    else:
        raise ValueError("No timeframes specified")


async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Parse arguments
        start_date, end_date = parse_date_range(args)
        timeframes = parse_timeframes(args)
        output_format = OutputFormat(args.format.upper())
        
        # Load configuration
        if args.config:
            config = DataCollectorConfig.from_file(args.config)
        else:
            config = DataCollectorConfig(
                output_directory=args.output_dir,
                testnet=args.testnet,
                rate_limit=args.rate_limit
            )
        
        # Create collector and run
        collector = HistoricalDataCollector(config)
        results = await collector.collect_data(
            symbol=args.symbol.upper(),
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            output_format=output_format
        )
        
        if results:
            print(f"\n✅ Successfully collected data for {len(results)} timeframes")
            for timeframe, path in results.items():
                print(f"   {timeframe}: {path}")
        else:
            print("❌ No data was collected")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Collection cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())