#!/usr/bin/env python3
"""
Trading Bot - Main Entry Point

Cryptocurrency data collection and trading analysis system with modular architecture.
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
  backtest   Run backtesting on historical data
  
Examples:
  # Collect SOLUSDT data for last week
  python main.py collect --symbol SOLUSDT --timeframe 5m --period week
  
  # Train LSTM model with custom horizon
  python main.py train --model lstm --horizon 5 --window 30
  
  # Train Random Forest model
  python main.py train --model rf --horizon 3 --window 20
  
  # Run backtest on LSTM model  
  python main.py backtest --model lstm --horizon 3 --period 2025-08:2025-10
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data collection command
    collect_parser = subparsers.add_parser('collect', help='Collect historical data')
    collect_parser.add_argument('--symbol', '-s', default='SOLUSDT', help='Trading pair (default: SOLUSDT)')
    collect_parser.add_argument('--timeframe', '-t', default='5m', help='Timeframe (default: 5m)')
    collect_parser.add_argument('--period', '-p', default='week', help='Time period (default: week)')
    collect_parser.add_argument('--format', '-f', default='csv', choices=['csv', 'json', 'parquet'], help='Output format')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument('--model', required=True, choices=['lstm', 'rf'], help='Model type: lstm or rf (Random Forest)')
    train_parser.add_argument('--horizon', type=int, default=3, help='Prediction horizon in bars (default: 3)')
    train_parser.add_argument('--window', type=int, default=30, help='Data window size in bars (default: 30)')
    train_parser.add_argument('--data', help='Path to data file (default: auto-detect latest)')
    train_parser.add_argument('--epochs', type=int, default=30, help='Training epochs for LSTM (default: 30)')
    train_parser.add_argument('--sample-size', type=int, help='Limit training data size for faster experiments')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('--model', required=True, choices=['lstm', 'rf'], help='Model type to backtest')
    backtest_parser.add_argument('--horizon', type=int, default=3, help='Prediction horizon in bars (default: 3)')  
    backtest_parser.add_argument('--period', help='Test period in format YYYY-MM:YYYY-MM (e.g., 2025-08:2025-10)')
    backtest_parser.add_argument('--model-path', help='Path to specific model file (default: auto-detect latest)')
    
    return parser

def run_data_collection(args):
    """Run data collection module."""
    print(f"🔄 Collecting {args.symbol} data ({args.timeframe}, {args.period})...")
    
    import subprocess
    
    cmd = [
        sys.executable, '-m', 'src.data_collection',
        '--symbol', args.symbol,
        '--timeframe', args.timeframe,
        '--period', args.period,
        '--format', args.format
    ]
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

def run_training(args):
    """Run model training."""
    print(f"🧠 Starting {args.model.upper()} model training...")
    print(f"📊 Parameters: horizon={args.horizon}, window={args.window}")
    
    if args.sample_size:
        print(f"🔬 Using sample size: {args.sample_size}")
    
    # Determine data path
    if args.data:
        data_path = args.data
    else:
        # Auto-detect latest data file
        data_path = "data/raw/SOLUSDT_5m_20250101_20251031_advanced_indicators.csv"
        print(f"📁 Using default data: {data_path}")
    
    # Check if data exists
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        print(f"   Please collect data first: python main.py collect")
        sys.exit(1)
    
    # Run appropriate training script
    import subprocess
    
    if args.model == 'lstm':
        script_path = "scripts/training/train_lstm_parameterized.py"
        
        # Check if parameterized script exists, otherwise create it
        if not Path(script_path).exists():
            print(f"📝 Creating parameterized LSTM training script...")
            create_parameterized_lstm_script(script_path)
        
        cmd = [
            sys.executable, script_path,
            '--horizon', str(args.horizon),
            '--window', str(args.window),
            '--epochs', str(args.epochs),
            '--data', data_path
        ]
        
        if args.sample_size:
            cmd.extend(['--sample-size', str(args.sample_size)])
            
    elif args.model == 'rf':
        print(f"🌲 Random Forest training not yet parameterized")
        print(f"   Using existing script for now...")
        cmd = [sys.executable, 'scripts/training/train_horizon5_regularized.py']
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

def run_backtest(args):
    """Run backtesting."""
    print(f"📈 Running backtest for {args.model.upper()} model...")
    print(f"🎯 Horizon: {args.horizon} bars")
    
    if args.period:
        print(f"📅 Period: {args.period}")
    
    # Run appropriate backtest script
    import subprocess
    
    if args.model == 'lstm':
        # Use existing LSTM backtest script for now
        script_path = "scripts/backtesting/backtest_real_lstm.py"
    elif args.model == 'rf':
        script_path = "scripts/backtesting/backtest_regularized_horizon5.py" 
    
    if not Path(script_path).exists():
        print(f"❌ Backtest script not found: {script_path}")
        print(f"   Available scripts in scripts/backtesting/:")
        for script in Path("scripts/backtesting").glob("*.py"):
            print(f"     - {script.name}")
        sys.exit(1)
    
    result = subprocess.run([sys.executable, script_path])
    sys.exit(result.returncode)

def create_parameterized_lstm_script(script_path):
    """Create a parameterized version of LSTM training script."""
    script_content = '''#!/usr/bin/env python3
"""
Parameterized LSTM Training Script
Automatically generated by main.py
"""

import argparse
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.lstm.real_lstm_model import RealLSTMPredictor

def main():
    parser = argparse.ArgumentParser(description='Train LSTM model with custom parameters')
    parser.add_argument('--horizon', type=int, required=True, help='Prediction horizon')
    parser.add_argument('--window', type=int, required=True, help='Window size')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--data', required=True, help='Path to data file')
    parser.add_argument('--sample-size', type=int, help='Sample size for faster training')
    
    args = parser.parse_args()
    
    print(f"🚀 Training LSTM with horizon={args.horizon}, window={args.window}")
    
    # Initialize predictor with custom parameters
    predictor = RealLSTMPredictor(window_size=args.window, horizon=args.horizon)
    
    # Load data
    df = predictor.load_and_prepare_data(args.data, sample_size=args.sample_size)
    
    # Prepare data splits
    train_data, val_data, test_data = predictor.prepare_data_splits(df)
    
    # Build model
    input_shape = (predictor.window_size, len(predictor.features))
    predictor.build_model(input_shape)
    
    # Train model
    history = predictor.train_model(train_data, val_data, epochs=args.epochs, batch_size=64)
    
    # Plot training history
    predictor.plot_training_history()
    
    # Evaluate model
    test_accuracy, cm, report = predictor.evaluate_model(test_data)
    
    # Save artifacts
    predictor.save_artifacts()
    
    print(f"✅ Training completed! Test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
'''
    
    # Create directory if needed
    Path(script_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write script
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    Path(script_path).chmod(0o755)
    
    print(f"✅ Created {script_path}")

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print(f"🤖 Trading Bot System v2.0")
    print(f"📋 Command: {args.command}")
    
    try:
        if args.command == 'collect':
            run_data_collection(args)
        elif args.command == 'train':
            run_training(args)
        elif args.command == 'backtest':
            run_backtest(args)
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