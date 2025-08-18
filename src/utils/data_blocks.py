"""Fixed data blocks management for consistent testing and debugging."""

import json
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

from .logger import TradingLogger
from .types import TimeFrame


class DataBlockType(Enum):
    """Types of data blocks."""
    TREND = "trend"
    VOLATILE = "volatile"
    SIDEWAYS = "sideways"
    MIXED = "mixed"
    HIGH_VOLUME = "high_volume"
    LOW_VOLUME = "low_volume"


@dataclass
class DataBlockInfo:
    """Information about a data block."""
    block_id: str
    name: str
    description: str
    start_time: datetime
    end_time: datetime
    symbol: str
    timeframes: List[str]
    block_type: DataBlockType
    characteristics: Dict[str, Any]
    data_hash: str
    created_at: datetime
    file_paths: Dict[str, str]  # timeframe -> file_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        data['end_time'] = self.end_time.isoformat()
        data['created_at'] = self.created_at.isoformat()
        data['block_type'] = self.block_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataBlockInfo':
        """Create from dictionary."""
        data = data.copy()
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        data['end_time'] = datetime.fromisoformat(data['end_time'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['block_type'] = DataBlockType(data['block_type'])
        return cls(**data)


class DataBlockManager:
    """Manager for fixed data blocks."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize data block manager.
        
        Args:
            cache_dir: Directory for cached blocks. Defaults to data/blocks/
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "blocks"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Directories for different components
        self.blocks_dir = self.cache_dir / "data"
        self.metadata_dir = self.cache_dir / "metadata"
        self.processed_dir = self.cache_dir / "processed"
        
        for dir_path in [self.blocks_dir, self.metadata_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = TradingLogger("data_blocks")
        
        # Registry of available blocks
        self.blocks_registry: Dict[str, DataBlockInfo] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load blocks registry from file."""
        registry_file = self.metadata_dir / "registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                for block_id, block_data in registry_data.items():
                    self.blocks_registry[block_id] = DataBlockInfo.from_dict(block_data)
                    
                self.logger.log_system_event(
                    event_type="registry_loaded",
                    component="data_blocks",
                    status="success",
                    details={"blocks_count": len(self.blocks_registry)}
                )
            except Exception as e:
                self.logger.log_error(
                    error_type="registry_load_failed",
                    component="data_blocks",
                    error_message=str(e)
                )
        else:
            self.logger.log_system_event(
                event_type="registry_not_found",
                component="data_blocks",
                status="info",
                details={"creating_new": True}
            )
    
    def _save_registry(self) -> None:
        """Save blocks registry to file."""
        registry_file = self.metadata_dir / "registry.json"
        
        try:
            registry_data = {
                block_id: block_info.to_dict()
                for block_id, block_info in self.blocks_registry.items()
            }
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
                
            self.logger.log_system_event(
                event_type="registry_saved",
                component="data_blocks",
                status="success",
                details={"blocks_count": len(self.blocks_registry)}
            )
        except Exception as e:
            self.logger.log_error(
                error_type="registry_save_failed",
                component="data_blocks",
                error_message=str(e)
            )
    
    def create_block(
        self,
        block_id: str,
        name: str,
        description: str,
        start_time: datetime,
        end_time: datetime,
        symbol: str = "SOLUSDT",
        timeframes: Optional[List[str]] = None,
        block_type: DataBlockType = DataBlockType.MIXED,
        source_data_dir: Optional[Path] = None
    ) -> DataBlockInfo:
        """Create a new data block.
        
        Args:
            block_id: Unique identifier for the block
            name: Human-readable name
            description: Description of the block characteristics
            start_time: Start datetime
            end_time: End datetime
            symbol: Trading symbol
            timeframes: List of timeframes to include
            block_type: Type of the block
            source_data_dir: Source directory with original data files
            
        Returns:
            DataBlockInfo object
        """
        if timeframes is None:
            timeframes = ["1m", "5m", "15m", "1h"]
        
        if source_data_dir is None:
            source_data_dir = Path(__file__).parent.parent.parent / "data"
        
        # Create block directory
        block_dir = self.blocks_dir / block_id
        block_dir.mkdir(parents=True, exist_ok=True)
        
        file_paths = {}
        all_data = {}
        
        # Process each timeframe
        for timeframe in timeframes:
            # Find source file
            source_pattern = f"{symbol}_{timeframe}_real_*.csv"
            source_files = list(Path(source_data_dir).glob(source_pattern))
            
            if not source_files:
                self.logger.log_error(
                    error_type="source_file_not_found",
                    component="data_blocks",
                    error_message=f"No source file found for {symbol}_{timeframe}"
                )
                continue
            
            source_file = source_files[0]  # Take the first match
            
            # Load and filter data
            try:
                df = pd.read_csv(source_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Filter by time range
                mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
                filtered_df = df[mask].copy()
                
                if filtered_df.empty:
                    self.logger.log_system_event(
                        event_type="no_data_in_range",
                        component="data_blocks",
                        status="warning",
                        details={
                            "timeframe": timeframe,
                            "start_time": start_time.isoformat(),
                            "end_time": end_time.isoformat()
                        }
                    )
                    continue
                
                # Save filtered data
                block_file = block_dir / f"{symbol}_{timeframe}_{block_id}.csv"
                filtered_df.to_csv(block_file, index=False)
                file_paths[timeframe] = str(block_file)
                all_data[timeframe] = filtered_df
                
                self.logger.log_system_event(
                    event_type="timeframe_processed",
                    component="data_blocks",
                    status="success",
                    details={
                        "timeframe": timeframe,
                        "records_count": len(filtered_df),
                        "file_path": str(block_file)
                    }
                )
                
            except Exception as e:
                self.logger.log_error(
                    error_type="timeframe_processing_failed",
                    component="data_blocks",
                    error_message=f"Failed to process {timeframe}: {str(e)}"
                )
        
        # Calculate characteristics
        characteristics = self._calculate_characteristics(all_data, timeframes)
        
        # Calculate data hash for integrity
        data_hash = self._calculate_data_hash(all_data)
        
        # Create block info
        block_info = DataBlockInfo(
            block_id=block_id,
            name=name,
            description=description,
            start_time=start_time,
            end_time=end_time,
            symbol=symbol,
            timeframes=list(file_paths.keys()),
            block_type=block_type,
            characteristics=characteristics,
            data_hash=data_hash,
            created_at=datetime.now(),
            file_paths=file_paths
        )
        
        # Save to registry
        self.blocks_registry[block_id] = block_info
        self._save_registry()
        
        # Save block metadata separately
        metadata_file = self.metadata_dir / f"{block_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(block_info.to_dict(), f, indent=2, default=str)
        
        self.logger.log_system_event(
            event_type="block_created",
            component="data_blocks",
            status="success",
            details={
                "block_id": block_id,
                "name": name,
                "timeframes": list(file_paths.keys()),
                "characteristics": characteristics
            }
        )
        
        return block_info
    
    def _calculate_characteristics(
        self, 
        all_data: Dict[str, pd.DataFrame], 
        timeframes: List[str]
    ) -> Dict[str, Any]:
        """Calculate characteristics of the data block."""
        if not all_data:
            return {}
        
        # Use 1-minute data for detailed analysis, fallback to available timeframes
        primary_tf = "1m" if "1m" in all_data else list(all_data.keys())[0]
        df = all_data[primary_tf]
        
        if df.empty:
            return {}
        
        try:
            # Price characteristics
            price_range = df['high'].max() - df['low'].min()
            price_volatility = df['close'].pct_change().std()
            avg_volume = df['volume'].mean()
            max_volume = df['volume'].max()
            
            # Trend characteristics
            first_price = df['close'].iloc[0]
            last_price = df['close'].iloc[-1]
            total_return = (last_price - first_price) / first_price
            
            # Volatility periods
            rolling_vol = df['close'].pct_change().rolling(window=60).std()
            high_vol_periods = (rolling_vol > rolling_vol.quantile(0.8)).sum()
            
            return {
                "records_count": len(df),
                "duration_hours": (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 3600,
                "price_range": float(price_range),
                "price_volatility": float(price_volatility),
                "total_return": float(total_return),
                "avg_volume": float(avg_volume),
                "max_volume": float(max_volume),
                "high_volatility_periods": int(high_vol_periods),
                "start_price": float(first_price),
                "end_price": float(last_price),
                "min_price": float(df['low'].min()),
                "max_price": float(df['high'].max()),
            }
        
        except Exception as e:
            self.logger.log_error(
                error_type="characteristics_calculation_failed",
                component="data_blocks",
                error_message=str(e)
            )
            return {"error": str(e)}
    
    def _calculate_data_hash(self, all_data: Dict[str, pd.DataFrame]) -> str:
        """Calculate hash of the data for integrity checking."""
        try:
            # Combine all data into a single string for hashing
            combined_data = ""
            for timeframe in sorted(all_data.keys()):
                df = all_data[timeframe]
                combined_data += df.to_csv(index=False)
            
            return hashlib.md5(combined_data.encode()).hexdigest()
        
        except Exception:
            return "unknown"
    
    def load_block(self, block_id: str, timeframes: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load data from a cached block.
        
        Args:
            block_id: Block identifier
            timeframes: Specific timeframes to load (all if None)
            
        Returns:
            Dictionary with timeframe -> DataFrame
        """
        if block_id not in self.blocks_registry:
            raise ValueError(f"Block {block_id} not found in registry")
        
        block_info = self.blocks_registry[block_id]
        
        if timeframes is None:
            timeframes = block_info.timeframes
        
        data = {}
        
        for timeframe in timeframes:
            if timeframe not in block_info.file_paths:
                self.logger.log_system_event(
                    event_type="timeframe_not_available",
                    component="data_blocks",
                    status="warning",
                    details={
                        "block_id": block_id,
                        "timeframe": timeframe,
                        "available": block_info.timeframes
                    }
                )
                continue
            
            file_path = Path(block_info.file_paths[timeframe])
            
            if not file_path.exists():
                raise FileNotFoundError(f"Block data file not found: {file_path}")
            
            try:
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                data[timeframe] = df
                
                self.logger.log_system_event(
                    event_type="timeframe_loaded",
                    component="data_blocks",
                    status="success",
                    details={
                        "block_id": block_id,
                        "timeframe": timeframe,
                        "records_count": len(df)
                    }
                )
                
            except Exception as e:
                self.logger.log_error(
                    error_type="timeframe_load_failed",
                    component="data_blocks",
                    error_message=f"Failed to load {timeframe} from {block_id}: {str(e)}"
                )
        
        return data
    
    def get_block_info(self, block_id: str) -> Optional[DataBlockInfo]:
        """Get information about a block."""
        return self.blocks_registry.get(block_id)
    
    def list_blocks(
        self, 
        block_type: Optional[DataBlockType] = None,
        symbol: Optional[str] = None
    ) -> List[DataBlockInfo]:
        """List available blocks with optional filtering."""
        blocks = list(self.blocks_registry.values())
        
        if block_type is not None:
            blocks = [b for b in blocks if b.block_type == block_type]
        
        if symbol is not None:
            blocks = [b for b in blocks if b.symbol == symbol]
        
        return blocks
    
    def delete_block(self, block_id: str) -> bool:
        """Delete a block and its data."""
        if block_id not in self.blocks_registry:
            return False
        
        block_info = self.blocks_registry[block_id]
        
        try:
            # Delete data files
            for file_path in block_info.file_paths.values():
                Path(file_path).unlink(missing_ok=True)
            
            # Delete block directory if empty
            block_dir = self.blocks_dir / block_id
            if block_dir.exists() and not any(block_dir.iterdir()):
                block_dir.rmdir()
            
            # Delete metadata file
            metadata_file = self.metadata_dir / f"{block_id}.json"
            metadata_file.unlink(missing_ok=True)
            
            # Remove from registry
            del self.blocks_registry[block_id]
            self._save_registry()
            
            self.logger.log_system_event(
                event_type="block_deleted",
                component="data_blocks",
                status="success",
                details={"block_id": block_id}
            )
            
            return True
            
        except Exception as e:
            self.logger.log_error(
                error_type="block_deletion_failed",
                component="data_blocks",
                error_message=str(e)
            )
            return False
    
    def verify_block_integrity(self, block_id: str) -> bool:
        """Verify integrity of a block by checking data hash."""
        if block_id not in self.blocks_registry:
            return False
        
        block_info = self.blocks_registry[block_id]
        
        try:
            # Load all data
            all_data = self.load_block(block_id)
            
            # Calculate current hash
            current_hash = self._calculate_data_hash(all_data)
            
            # Compare with stored hash
            integrity_ok = current_hash == block_info.data_hash
            
            self.logger.log_system_event(
                event_type="block_integrity_checked",
                component="data_blocks",
                status="success" if integrity_ok else "error",
                details={
                    "block_id": block_id,
                    "integrity_ok": integrity_ok,
                    "stored_hash": block_info.data_hash[:8],
                    "current_hash": current_hash[:8]
                }
            )
            
            return integrity_ok
            
        except Exception as e:
            self.logger.log_error(
                error_type="integrity_check_failed",
                component="data_blocks",
                error_message=str(e)
            )
            return False


# Predefined block configurations
PREDEFINED_BLOCKS = {
    "august_10_17_full": {
        "name": "August 10-17 Full Week",
        "description": "Complete week of SOL/USDT data with various market conditions",
        "start_time": datetime(2025, 8, 10, 0, 0, 0),
        "end_time": datetime(2025, 8, 17, 23, 59, 59),
        "block_type": DataBlockType.MIXED,
    },
    "august_10_13_trend": {
        "name": "August 10-13 Trend Period",
        "description": "4-day period with strong trending behavior",
        "start_time": datetime(2025, 8, 10, 0, 0, 0),
        "end_time": datetime(2025, 8, 13, 23, 59, 59),
        "block_type": DataBlockType.TREND,
    },
    "august_14_17_volatile": {
        "name": "August 14-17 Volatile Period", 
        "description": "4-day period with high volatility",
        "start_time": datetime(2025, 8, 14, 0, 0, 0),
        "end_time": datetime(2025, 8, 17, 23, 59, 59),
        "block_type": DataBlockType.VOLATILE,
    },
    "august_12_single_day": {
        "name": "August 12 Single Day",
        "description": "Single day for detailed intraday testing",
        "start_time": datetime(2025, 8, 12, 0, 0, 0),
        "end_time": datetime(2025, 8, 12, 23, 59, 59),
        "block_type": DataBlockType.MIXED,
    },
}


def create_predefined_blocks() -> None:
    """Create all predefined blocks."""
    manager = DataBlockManager()
    
    for block_id, config in PREDEFINED_BLOCKS.items():
        if block_id not in manager.blocks_registry:
            print(f"Creating block: {block_id}")
            try:
                block_info = manager.create_block(
                    block_id=block_id,
                    name=config["name"],
                    description=config["description"],
                    start_time=config["start_time"],
                    end_time=config["end_time"],
                    block_type=config["block_type"]
                )
                print(f"✅ Created block '{block_id}' with {len(block_info.timeframes)} timeframes")
            except Exception as e:
                print(f"❌ Failed to create block '{block_id}': {e}")
        else:
            print(f"⚠️ Block '{block_id}' already exists")


if __name__ == "__main__":
    # CLI interface for creating blocks
    import argparse
    
    parser = argparse.ArgumentParser(description="Data blocks management")
    parser.add_argument("--create-predefined", action="store_true", help="Create predefined blocks")
    parser.add_argument("--list", action="store_true", help="List existing blocks")
    parser.add_argument("--verify", type=str, help="Verify block integrity")
    
    args = parser.parse_args()
    
    if args.create_predefined:
        create_predefined_blocks()
    
    if args.list:
        manager = DataBlockManager()
        blocks = manager.list_blocks()
        print(f"\nFound {len(blocks)} blocks:")
        for block in blocks:
            print(f"- {block.block_id}: {block.name} ({block.block_type.value})")
            print(f"  Period: {block.start_time} - {block.end_time}")
            print(f"  Timeframes: {', '.join(block.timeframes)}")
            print(f"  Records: {block.characteristics.get('records_count', 'N/A')}")
            print()
    
    if args.verify:
        manager = DataBlockManager()
        integrity_ok = manager.verify_block_integrity(args.verify)
        print(f"Block '{args.verify}' integrity: {'✅ OK' if integrity_ok else '❌ FAILED'}")