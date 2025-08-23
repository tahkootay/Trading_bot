#!/usr/bin/env python3
"""
PROPER ML Backtest using actual trained models on August 10-17 data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import joblib

# Add src to path
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)

# from src.models.ml_models import MLModelPredictor  # Not used in this script

def calculate_technical_indicators(df):
    """Calculate the exact same technical indicators used in training."""
    # EMA indicators
    df['ema_8'] = df['close'].ewm(span=8).mean()
    df['ema_13'] = df['close'].ewm(span=13).mean() 
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_34'] = df['close'].ewm(span=34).mean()
    df['ema_55'] = df['close'].ewm(span=55).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # ADX (simplified)
    df['adx'] = 25.0  # Default value used in training
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    bb_middle = df['close'].rolling(window=bb_period).mean()
    bb_std_dev = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = bb_middle + (bb_std_dev * bb_std)
    df['bb_lower'] = bb_middle - (bb_std_dev * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_middle
    
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Volume features
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price features
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_15'] = df['close'].pct_change(15)
    
    # Volatility
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    
    # Momentum
    df['momentum'] = df['close'] / df['close'].shift(10) - 1
    
    # Support/Resistance
    df['high_20'] = df['high'].rolling(window=20).max()
    df['low_20'] = df['low'].rolling(window=20).min()
    df['distance_to_high'] = (df['high_20'] - df['close']) / df['close']
    df['distance_to_low'] = (df['close'] - df['low_20']) / df['close']
    
    # Trend features
    df['ema_trend'] = (df['ema_8'] > df['ema_21']).astype(int)
    df['price_vs_vwap'] = (df['close'] > df['vwap']).astype(int)
    
    return df

def prepare_features_dict(row):
    """Prepare features dictionary in the same format as training."""
    feature_names = [
        'ema_8', 'ema_13', 'ema_21', 'ema_34', 'ema_55',
        'rsi', 'macd', 'macd_signal', 'macd_hist', 'atr', 'adx',
        'bb_width', 'volume_ratio', 'price_change_1', 'price_change_5', 
        'price_change_15', 'volatility', 'momentum', 'distance_to_high', 
        'distance_to_low', 'ema_trend', 'price_vs_vwap'
    ]
    
    features = {}
    for feature in feature_names:
        features[feature] = float(row[feature]) if not pd.isna(row[feature]) else 0.0
    
    return features

def run_proper_ml_backtest():
    print("🔬 PROPER ML Backtest using trained models on August 10-17")
    print("=" * 60)
    
    # Load data
    print("📊 Loading August 10-17 dataset...")
    try:
        df = pd.read_csv('data/raw/SOLUSDT_5m_aug10_17.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        print(f"✅ Loaded {len(df)} records")
        print(f"📅 Period: {df.index[0]} → {df.index[-1]}")
        print(f"💰 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Calculate indicators
    print("🔧 Calculating technical indicators (same as training)...")
    df = calculate_technical_indicators(df)
    
    # Remove NaN values
    df = df.dropna()
    print(f"📊 After indicators: {len(df)} records")
    
    # Load trained ML models
    print("🤖 Loading trained ML models...")
    try:
        # Load the actual trained models
        models_path = Path("models/90d_enhanced/20250823_134813")
        
        # Load models and preprocessors
        scaler = joblib.load(models_path / "scaler.joblib")
        feature_names = joblib.load(models_path / "feature_names.joblib")
        catboost_model = joblib.load(models_path / "catboost.joblib")
        rf_model = joblib.load(models_path / "random_forest.joblib")
        
        print(f"✅ Models loaded successfully")
        print(f"📋 Features: {len(feature_names)}")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Backtest settings following CLAUDE.md
    initial_balance = 10000.0
    commission_rate = 0.001  # 0.1% commission
    max_position_pct = 0.02  # Max 2% of account per trade (CLAUDE.md)
    stop_loss_pct = 0.015    # 1.5% stop loss (CLAUDE.md) 
    take_profit_pct = 0.04   # 4% take profit
    min_confidence = 0.15    # P(UP) - P(DOWN) > 0.15 (CLAUDE.md)
    
    # Trading state
    balance = initial_balance
    position = 0  # 0 = no position, 1 = long, -1 = short
    position_size_usd = 0
    entry_price = 0
    
    trades = []
    equity_curve = [initial_balance]
    
    print(f"💰 Initial Balance: ${initial_balance:,.2f}")
    print(f"💸 Commission: {commission_rate*100:.1f}%") 
    print(f"📊 Max position per trade: {max_position_pct*100:.1f}% of balance")
    print(f"🛑 Stop Loss: {stop_loss_pct*100:.1f}%")
    print(f"🎯 Take Profit: {take_profit_pct*100:.1f}%")
    print(f"🧠 Min ML Confidence: P(direction) - P(other) > {min_confidence:.2f}")
    print("\n🚀 Running backtest with actual trained models...")
    
    successful_predictions = 0
    failed_predictions = 0
    trades_opened = 0
    
    for i in range(100, len(df)):  # Start from 100 for indicator stability
        current_data = df.iloc[i]
        current_price = current_data['close']
        
        try:
            # Prepare features for ML prediction
            features = prepare_features_dict(current_data)
            
            # Create feature array in the same order as training
            feature_array = np.array([features[name] for name in feature_names]).reshape(1, -1)
            
            # Scale features
            feature_array_scaled = scaler.transform(feature_array)
            
            # Get predictions from both models
            catboost_probs = catboost_model.predict_proba(feature_array_scaled)[0]
            rf_probs = rf_model.predict_proba(feature_array_scaled)[0]
            
            # Ensemble prediction (average probabilities)
            ensemble_probs = (catboost_probs + rf_probs) / 2
            
            # Map to trading signals
            # 0 = flat/sideways, 1 = up, 2 = down (from training)
            p_flat = ensemble_probs[0] if len(ensemble_probs) > 0 else 0.5
            p_up = ensemble_probs[1] if len(ensemble_probs) > 1 else 0.25
            p_down = ensemble_probs[2] if len(ensemble_probs) > 2 else 0.25
            
            successful_predictions += 1
            
            # Trading logic following CLAUDE.md
            signal = None
            confidence = 0
            
            # BUY signal conditions from CLAUDE.md
            if (p_up - p_down) > min_confidence:
                signal = 'buy'
                confidence = p_up - p_down
                
            # SELL signal conditions from CLAUDE.md  
            elif (p_down - p_up) > min_confidence:
                signal = 'sell'
                confidence = p_down - p_up
                
        except Exception as e:
            failed_predictions += 1
            signal = None
            confidence = 0
            
        # Execute trading logic
        if position == 0 and signal:  # No position, got signal
            
            # Additional filters from CLAUDE.md (simplified)
            ema_aligned = False
            volume_confirmation = current_data['volume_ratio'] > 1.0
            
            if signal == 'buy':
                ema_aligned = (current_data['ema_8'] > current_data['ema_21'] and 
                              current_price >= current_data['vwap'])
            elif signal == 'sell':
                ema_aligned = (current_data['ema_8'] < current_data['ema_21'] and 
                              current_price <= current_data['vwap'])
            
            # Open position if all conditions met
            if ema_aligned and volume_confirmation:
                position_size_usd = balance * max_position_pct
                commission_cost = position_size_usd * commission_rate
                
                if balance >= commission_cost:  # Check sufficient balance for commission only
                    # For futures, we don't subtract position_size_usd from balance
                    # Only subtract commission cost
                    balance -= commission_cost
                    
                    if signal == 'buy':
                        position = 1
                        entry_price = current_price
                        trades_opened += 1
                        print(f"📈 LONG ${position_size_usd:.0f} @${current_price:.2f} (conf: {confidence:.3f}) - {current_data.name.strftime('%m-%d %H:%M')}")
                        
                    elif signal == 'sell':
                        position = -1 
                        entry_price = current_price
                        trades_opened += 1
                        print(f"📉 SHORT ${position_size_usd:.0f} @${current_price:.2f} (conf: {confidence:.3f}) - {current_data.name.strftime('%m-%d %H:%M')}")
        
        elif position != 0:  # Have position, check exit conditions
            should_exit = False
            exit_reason = ""
            
            if position == 1:  # LONG position
                pnl_pct = (current_price - entry_price) / entry_price
                
                # Exit conditions
                if pnl_pct <= -stop_loss_pct:
                    should_exit = True
                    exit_reason = "Stop Loss"
                elif pnl_pct >= take_profit_pct:
                    should_exit = True  
                    exit_reason = "Take Profit"
                elif signal == 'sell' and confidence > min_confidence:
                    should_exit = True
                    exit_reason = "ML Signal"
                    
            elif position == -1:  # SHORT position
                pnl_pct = (entry_price - current_price) / entry_price
                
                # Exit conditions
                if pnl_pct <= -stop_loss_pct:
                    should_exit = True
                    exit_reason = "Stop Loss"
                elif pnl_pct >= take_profit_pct:
                    should_exit = True
                    exit_reason = "Take Profit" 
                elif signal == 'buy' and confidence > min_confidence:
                    should_exit = True
                    exit_reason = "ML Signal"
            
            if should_exit:
                # Calculate P&L correctly
                if position == 1:  # Close LONG
                    pnl_pct = (current_price - entry_price) / entry_price
                else:  # Close SHORT
                    pnl_pct = (entry_price - current_price) / entry_price
                
                gross_pnl = position_size_usd * pnl_pct
                commission_cost = position_size_usd * commission_rate
                net_pnl = gross_pnl - commission_cost
                
                # Update balance correctly - add back only the net result
                # (position_size_usd was already deducted when opening the position)
                balance += net_pnl
                
                # Record trade
                trades.append({
                    'entry_time': str(current_data.name),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position_size_usd': position_size_usd,
                    'side': 'LONG' if position == 1 else 'SHORT',
                    'pnl': net_pnl,
                    'pnl_pct': pnl_pct * 100,
                    'exit_reason': exit_reason
                })
                
                print(f"🔄 EXIT @${current_price:.2f} | PnL: ${net_pnl:+.2f} ({pnl_pct*100:+.2f}%) [{exit_reason}]")
                
                # Reset position
                position = 0
                position_size_usd = 0
                entry_price = 0
        
        # Update equity curve
        if position == 0:
            current_equity = balance
        else:
            if position == 1:  # LONG
                unrealized_pnl = position_size_usd * ((current_price - entry_price) / entry_price)
            else:  # SHORT
                unrealized_pnl = position_size_usd * ((entry_price - current_price) / entry_price)
            current_equity = balance + unrealized_pnl
            
        equity_curve.append(current_equity)
    
    # Close final position if still open
    if position != 0:
        final_price = df.iloc[-1]['close']
        if position == 1:
            pnl_pct = (final_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - final_price) / entry_price
        
        gross_pnl = position_size_usd * pnl_pct
        commission_cost = position_size_usd * commission_rate  
        net_pnl = gross_pnl - commission_cost
        balance += net_pnl
        
        trades.append({
            'entry_time': str(df.index[-1]),
            'entry_price': entry_price,
            'exit_price': final_price,
            'position_size_usd': position_size_usd,
            'side': 'LONG' if position == 1 else 'SHORT',
            'pnl': net_pnl,
            'pnl_pct': pnl_pct * 100,
            'exit_reason': 'End of backtest'
        })
    
    # Results
    final_balance = balance
    total_pnl = final_balance - initial_balance
    total_return_pct = (total_pnl / initial_balance) * 100
    
    print("\n" + "="*60)
    print("📊 PROPER ML BACKTEST RESULTS:")
    print(f"💰 Initial Balance: ${initial_balance:,.2f}")
    print(f"💸 Final Balance: ${final_balance:,.2f}")
    print(f"📈 Total P&L: ${total_pnl:+,.2f} ({total_return_pct:+.2f}%)")
    print(f"🤖 ML Predictions: {successful_predictions} success, {failed_predictions} failed")
    print(f"🔄 Trades Opened: {trades_opened}")
    print(f"📋 Total Trades: {len(trades)}")
    
    if trades:
        profitable_trades = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = (profitable_trades / len(trades)) * 100
        avg_pnl = sum(t['pnl'] for t in trades) / len(trades)
        avg_pnl_pct = sum(t['pnl_pct'] for t in trades) / len(trades)
        total_commissions = len(trades) * 2 * (initial_balance * max_position_pct * commission_rate)  # 2 = entry + exit
        
        print(f"✅ Profitable Trades: {profitable_trades} ({win_rate:.1f}%)")
        print(f"❌ Losing Trades: {len(trades) - profitable_trades}")
        print(f"📊 Average PnL: ${avg_pnl:+.2f} ({avg_pnl_pct:+.2f}%)")
        print(f"💸 Total Commissions: ${total_commissions:.2f}")
        
        if trades:
            best_trade = max(trades, key=lambda x: x['pnl'])
            worst_trade = min(trades, key=lambda x: x['pnl'])
            
            print(f"🏆 Best Trade: ${best_trade['pnl']:+.2f} ({best_trade['pnl_pct']:+.2f}%)")
            print(f"💀 Worst Trade: ${worst_trade['pnl']:+.2f} ({worst_trade['pnl_pct']:+.2f}%)")
    
    # Max drawdown
    peak = initial_balance
    max_drawdown = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    print(f"📉 Max Drawdown: {max_drawdown*100:.2f}%")
    
    # Validation check
    calculated_balance_check = initial_balance + sum(t['pnl'] for t in trades) - len(trades) * 2 * (initial_balance * max_position_pct * commission_rate)
    print(f"✅ Balance validation: Expected ${calculated_balance_check:.2f}, Got ${final_balance:.2f}")
    
    # Save results
    results = {
        'backtest_date': datetime.now().isoformat(),
        'model_version': '90d_enhanced_actual',
        'period': 'august_10_17_2025',
        'dataset_records': len(df),
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_pnl': total_pnl,
        'total_return_pct': total_return_pct,
        'successful_predictions': successful_predictions,
        'failed_predictions': failed_predictions,
        'trades_opened': trades_opened,
        'total_trades': len(trades),
        'profitable_trades': profitable_trades if trades else 0,
        'win_rate': win_rate if trades else 0,
        'max_drawdown_pct': max_drawdown * 100,
        'max_position_pct': max_position_pct * 100,
        'commission_rate': commission_rate,
        'stop_loss_pct': stop_loss_pct * 100,
        'take_profit_pct': take_profit_pct * 100,
        'min_confidence': min_confidence,
        'trades': trades
    }
    
    Path('output').mkdir(exist_ok=True)
    output_file = f"output/proper_ml_backtest_august_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved to {output_file}")
    print("\n🎉 PROPER ML backtest with actual trained models completed!")

if __name__ == "__main__":
    run_proper_ml_backtest()