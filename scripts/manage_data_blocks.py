#!/usr/bin/env python3
"""Utility script for managing fixed data blocks."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import click
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_blocks import DataBlockManager, DataBlockType, DataBlockInfo, PREDEFINED_BLOCKS, create_predefined_blocks
from src.utils.logger import setup_logging, TradingLogger


def format_characteristics(chars: Dict[str, Any]) -> str:
    """Format characteristics for display."""
    if not chars:
        return "N/A"
    
    parts = []
    if 'records_count' in chars:
        parts.append(f"Records: {chars['records_count']:,}")
    if 'total_return' in chars:
        parts.append(f"Return: {chars['total_return']:.2%}")
    if 'price_volatility' in chars:
        parts.append(f"Vol: {chars['price_volatility']:.4f}")
    
    return " | ".join(parts)


@click.group()
def cli():
    """Data blocks management utility."""
    setup_logging(log_level="INFO", log_format="simple")


@cli.command()
def create_predefined():
    """Create all predefined data blocks."""
    click.echo("🔄 Creating predefined data blocks...")
    
    manager = DataBlockManager()
    created_count = 0
    skipped_count = 0
    
    for block_id, config in PREDEFINED_BLOCKS.items():
        if block_id in manager.blocks_registry:
            click.echo(f"⚠️  Block '{block_id}' already exists - skipping")
            skipped_count += 1
            continue
        
        try:
            click.echo(f"📦 Creating block: {block_id}")
            block_info = manager.create_block(
                block_id=block_id,
                name=config["name"],
                description=config["description"],
                start_time=config["start_time"],
                end_time=config["end_time"],
                block_type=config["block_type"]
            )
            
            click.echo(f"✅ Created '{block_id}' with {len(block_info.timeframes)} timeframes")
            click.echo(f"   Period: {block_info.start_time} - {block_info.end_time}")
            click.echo(f"   Type: {block_info.block_type.value}")
            click.echo(f"   Records: {block_info.characteristics.get('records_count', 'N/A'):,}")
            click.echo()
            
            created_count += 1
            
        except Exception as e:
            click.echo(f"❌ Failed to create block '{block_id}': {e}")
    
    click.echo(f"🎉 Summary: Created {created_count} blocks, skipped {skipped_count} existing")


@cli.command()
@click.option("--type", "block_type", type=click.Choice([t.value for t in DataBlockType]), help="Filter by block type")
@click.option("--symbol", default=None, help="Filter by symbol")
def list_blocks(block_type: Optional[str], symbol: Optional[str]):
    """List all available data blocks."""
    manager = DataBlockManager()
    
    # Apply filters
    filter_type = DataBlockType(block_type) if block_type else None
    blocks = manager.list_blocks(block_type=filter_type, symbol=symbol)
    
    if not blocks:
        click.echo("No blocks found matching the criteria.")
        return
    
    # Prepare table data
    table_data = []
    for block in blocks:
        duration = block.end_time - block.start_time
        table_data.append([
            block.block_id,
            block.name[:30] + "..." if len(block.name) > 30 else block.name,
            block.block_type.value,
            block.start_time.strftime("%m-%d %H:%M"),
            block.end_time.strftime("%m-%d %H:%M"), 
            f"{duration.days}d {duration.seconds//3600}h",
            ", ".join(block.timeframes),
            format_characteristics(block.characteristics)
        ])
    
    headers = ["ID", "Name", "Type", "Start", "End", "Duration", "Timeframes", "Stats"]
    click.echo(f"\n📊 Found {len(blocks)} data blocks:")
    click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))


@cli.command()
@click.argument("block_id")
def info(block_id: str):
    """Show detailed information about a data block."""
    manager = DataBlockManager()
    block_info = manager.get_block_info(block_id)
    
    if not block_info:
        click.echo(f"❌ Block '{block_id}' not found")
        return
    
    click.echo(f"\n📦 Block Information: {block_id}")
    click.echo("=" * 50)
    click.echo(f"Name: {block_info.name}")
    click.echo(f"Description: {block_info.description}")
    click.echo(f"Type: {block_info.block_type.value}")
    click.echo(f"Symbol: {block_info.symbol}")
    click.echo(f"Created: {block_info.created_at}")
    click.echo(f"Data Hash: {block_info.data_hash}")
    click.echo()
    
    click.echo("📅 Time Range:")
    click.echo(f"  Start: {block_info.start_time}")
    click.echo(f"  End:   {block_info.end_time}")
    duration = block_info.end_time - block_info.start_time
    click.echo(f"  Duration: {duration.days} days, {duration.seconds // 3600} hours")
    click.echo()
    
    click.echo("📈 Timeframes:")
    for tf in block_info.timeframes:
        file_path = block_info.file_paths[tf]
        file_exists = Path(file_path).exists()
        status = "✅" if file_exists else "❌"
        click.echo(f"  {tf}: {status} {file_path}")
    click.echo()
    
    click.echo("📊 Characteristics:")
    for key, value in block_info.characteristics.items():
        if isinstance(value, float):
            if key.endswith('_return') or key.endswith('_volatility'):
                click.echo(f"  {key}: {value:.4f}")
            elif key.endswith('_price') or key.endswith('_volume'):
                click.echo(f"  {key}: {value:,.2f}")
            else:
                click.echo(f"  {key}: {value:.2f}")
        else:
            click.echo(f"  {key}: {value}")


@cli.command()
@click.argument("block_id")
@click.option("--timeframe", "-t", multiple=True, help="Specific timeframes to load")
def load(block_id: str, timeframe: List[str]):
    """Load and display basic statistics of a data block."""
    manager = DataBlockManager()
    
    if block_id not in manager.blocks_registry:
        click.echo(f"❌ Block '{block_id}' not found")
        return
    
    try:
        click.echo(f"📥 Loading block: {block_id}")
        
        timeframes = list(timeframe) if timeframe else None
        data = manager.load_block(block_id, timeframes)
        
        if not data:
            click.echo("❌ No data loaded")
            return
        
        click.echo(f"✅ Loaded {len(data)} timeframes")
        click.echo()
        
        # Display statistics for each timeframe
        for tf, df in data.items():
            click.echo(f"📊 {tf} Statistics:")
            click.echo(f"  Records: {len(df):,}")
            click.echo(f"  Time range: {df['timestamp'].min()} - {df['timestamp'].max()}")
            click.echo(f"  Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
            click.echo(f"  Average volume: {df['volume'].mean():,.0f}")
            click.echo(f"  Total volume: {df['volume'].sum():,.0f}")
            
            # Price movement
            price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
            click.echo(f"  Price change: {price_change:.2%}")
            click.echo()
    
    except Exception as e:
        click.echo(f"❌ Failed to load block: {e}")


@cli.command()
@click.argument("block_id")
def verify(block_id: str):
    """Verify integrity of a data block."""
    manager = DataBlockManager()
    
    if block_id not in manager.blocks_registry:
        click.echo(f"❌ Block '{block_id}' not found")
        return
    
    click.echo(f"🔍 Verifying block: {block_id}")
    
    try:
        integrity_ok = manager.verify_block_integrity(block_id)
        
        if integrity_ok:
            click.echo("✅ Block integrity verified - data is consistent")
        else:
            click.echo("❌ Block integrity check failed - data may be corrupted")
        
    except Exception as e:
        click.echo(f"❌ Verification failed: {e}")


@cli.command()
@click.argument("block_id")
@click.confirmation_option(prompt="Are you sure you want to delete this block?")
def delete(block_id: str):
    """Delete a data block."""
    manager = DataBlockManager()
    
    if block_id not in manager.blocks_registry:
        click.echo(f"❌ Block '{block_id}' not found")
        return
    
    try:
        success = manager.delete_block(block_id)
        
        if success:
            click.echo(f"✅ Block '{block_id}' deleted successfully")
        else:
            click.echo(f"❌ Failed to delete block '{block_id}'")
    
    except Exception as e:
        click.echo(f"❌ Deletion failed: {e}")


@cli.command()
@click.argument("block_id")
@click.argument("name")
@click.argument("description")
@click.option("--start", required=True, help="Start datetime (YYYY-MM-DD HH:MM:SS)")
@click.option("--end", required=True, help="End datetime (YYYY-MM-DD HH:MM:SS)")
@click.option("--symbol", default="SOLUSDT", help="Trading symbol")
@click.option("--type", "block_type", type=click.Choice([t.value for t in DataBlockType]), 
              default=DataBlockType.MIXED.value, help="Block type")
@click.option("--timeframes", default="1m,5m,15m,1h", help="Comma-separated timeframes")
def create(
    block_id: str, 
    name: str, 
    description: str, 
    start: str, 
    end: str,
    symbol: str,
    block_type: str,
    timeframes: str
):
    """Create a custom data block."""
    manager = DataBlockManager()
    
    if block_id in manager.blocks_registry:
        click.echo(f"❌ Block '{block_id}' already exists")
        return
    
    try:
        # Parse dates
        start_time = datetime.fromisoformat(start)
        end_time = datetime.fromisoformat(end)
        
        # Parse timeframes
        tf_list = [tf.strip() for tf in timeframes.split(",")]
        
        # Parse block type
        bt = DataBlockType(block_type)
        
        click.echo(f"🔄 Creating block: {block_id}")
        click.echo(f"  Period: {start_time} - {end_time}")
        click.echo(f"  Timeframes: {', '.join(tf_list)}")
        
        block_info = manager.create_block(
            block_id=block_id,
            name=name,
            description=description,
            start_time=start_time,
            end_time=end_time,
            symbol=symbol,
            timeframes=tf_list,
            block_type=bt
        )
        
        click.echo(f"✅ Block created successfully!")
        click.echo(f"  Records: {block_info.characteristics.get('records_count', 'N/A'):,}")
        click.echo(f"  Timeframes: {len(block_info.timeframes)}")
        
    except Exception as e:
        click.echo(f"❌ Failed to create block: {e}")


@cli.command()
def cleanup():
    """Clean up invalid or corrupted blocks."""
    manager = DataBlockManager()
    blocks = manager.list_blocks()
    
    if not blocks:
        click.echo("No blocks to check")
        return
    
    click.echo(f"🔍 Checking {len(blocks)} blocks...")
    
    corrupted_blocks = []
    for block in blocks:
        # Check if files exist
        missing_files = []
        for tf, file_path in block.file_paths.items():
            if not Path(file_path).exists():
                missing_files.append(tf)
        
        if missing_files:
            click.echo(f"⚠️  Block '{block.block_id}' has missing files: {', '.join(missing_files)}")
            corrupted_blocks.append(block.block_id)
            continue
        
        # Check data integrity
        if not manager.verify_block_integrity(block.block_id):
            click.echo(f"⚠️  Block '{block.block_id}' failed integrity check")
            corrupted_blocks.append(block.block_id)
    
    if corrupted_blocks:
        click.echo(f"\n❌ Found {len(corrupted_blocks)} corrupted blocks:")
        for block_id in corrupted_blocks:
            click.echo(f"  - {block_id}")
        
        if click.confirm("Delete corrupted blocks?"):
            for block_id in corrupted_blocks:
                manager.delete_block(block_id)
                click.echo(f"🗑️  Deleted {block_id}")
    else:
        click.echo("✅ All blocks are valid")


if __name__ == "__main__":
    cli()