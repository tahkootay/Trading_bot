#!/usr/bin/env python3
"""
Trading Bot - Main Entry Point

Cryptocurrency data collection and trading analysis system.
"""

import argparse
import sys
from pathlib import Path

def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Trading Bot - Cryptocurrency Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  collect    Collect historical data from exchanges
  train      Train ML models for price prediction  
  predict    Generate trading predictions
  
Examples:
  # Collect SOLUSDT data for last week
  python main.py collect --symbol SOLUSDT --timeframe 5m --period week
  
  # Train new model
  python main.py train --data data/processed --version v2
  
  # Generate predictions
  python main.py predict --symbol SOLUSDT --model v1
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data collection command
    collect_parser = subparsers.add_parser('collect', help='Collect historical data')
    collect_parser.add_argument('--symbol', '-s', required=True, help='Trading pair (e.g., SOLUSDT)')
    collect_parser.add_argument('--timeframe', '-t', help='Timeframe (5m, 1h, etc.)')
    collect_parser.add_argument('--period', '-p', help='Time period (week, month, etc.)')
    collect_parser.add_argument('--format', '-f', default='csv', choices=['csv', 'json', 'parquet'], help='Output format')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument('--data', required=True, help='Path to processed data')
    train_parser.add_argument('--version', required=True, help='Model version')
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--symbol', required=True, help='Trading pair')
    predict_parser.add_argument('--model', required=True, help='Model version to use')
    
    return parser

def run_data_collection(args):
    """Run data collection module."""
    print(f"🔄 Delegating to data collection module...")
    
    import subprocess
    
    cmd = [
        sys.executable, '-m', 'modules.data_collector',
        '--symbol', args.symbol
    ]
    
    if args.timeframe:
        cmd.extend(['--timeframe', args.timeframe])
    if args.period:
        cmd.extend(['--period', args.period])
    if args.format:
        cmd.extend(['--format', args.format])
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

def run_training(args):
    """Run model training."""
    print(f"🧠 Starting model training...")
    
    # Check if processed data exists
    train_file = Path(args.data) / "train.csv"
    if not train_file.exists():
        print(f"❌ Training data not found: {train_file}")
        print(f"   Please prepare data first using ml_data_prep.py")
        sys.exit(1)
    
    # Run training script
    import subprocess
    result = subprocess.run([sys.executable, 'scripts/train_model.py'])
    sys.exit(result.returncode)

def run_prediction(args):
    """Run prediction generation."""
    print(f"🔮 Generating predictions...")
    print(f"🚧 Prediction functionality not implemented yet")
    print(f"   Use demo_model_usage.py for now")
    sys.exit(1)

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print(f"🤖 Trading Bot System")
    print(f"📋 Command: {args.command}")
    
    try:
        if args.command == 'collect':
            run_data_collection(args)
        elif args.command == 'train':
            run_training(args)
        elif args.command == 'predict':
            run_prediction(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()