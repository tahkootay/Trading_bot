#!/usr/bin/env python3
"""
ML Training Module CLI

Command line interface for model training and management.
"""

import argparse
import sys
from pathlib import Path

def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="ML Training and Model Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train new model
  python -m modules.ml_training train --data data/processed --version v2
  
  # List available models
  python -m modules.ml_training list
  
  # Compare model versions
  python -m modules.ml_training compare v1 v2
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train new model')
    train_parser.add_argument('--data', required=True, help='Path to processed data directory')
    train_parser.add_argument('--version', required=True, help='Model version (e.g., v1, v2)')
    train_parser.add_argument('--model-type', default='random_forest', choices=['random_forest', 'xgboost'], help='Model type')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available models')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare model versions')
    compare_parser.add_argument('versions', nargs='+', help='Model versions to compare')
    
    return parser

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    from .model_manager import ModelManager
    
    manager = ModelManager()
    
    if args.command == 'list':
        print("📋 Available models:")
        models_df = manager.list_models()
        if not models_df.empty:
            print(models_df.to_string(index=False))
        else:
            print("   No models found")
    
    elif args.command == 'compare':
        print(f"📊 Comparing models: {', '.join(args.versions)}")
        comparison_df = manager.compare_models(args.versions)
        if not comparison_df.empty:
            print(comparison_df.to_string(index=False))
        else:
            print("   No comparison data available")
    
    elif args.command == 'train':
        print(f"🚧 Training functionality not implemented yet")
        print(f"   Use train_model.py for now")
        sys.exit(1)

if __name__ == "__main__":
    main()