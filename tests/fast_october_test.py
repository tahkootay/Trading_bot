#!/usr/bin/env python3
"""
Быстрый тест модели на октябрьских данных с оптимизированным бэктестером.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

def load_model_and_scaler():
    """Загрузка модели и скалера."""
    
    model_path = "models/optimized/rf_horizon_10_optimized.pkl"
    scaler_path = "models/optimized/scaler_horizon_10_optimized.pkl"
    
    print(f"📥 Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"📥 Loading scaler: {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def prepare_all_features(data):
    """Подготовка всех фичей сразу для ускорения."""
    
    # Исключаем служебные колонки
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Добавляем target колонки если есть
    target_cols = [col for col in data.columns if col.startswith('target')]
    exclude_cols.extend(target_cols)
    
    # Получаем фичи
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    if not feature_cols:
        raise ValueError("No feature columns found in data")
    
    print(f"🔍 Using {len(feature_cols)} features")
    
    features = data[feature_cols].values
    
    return features, feature_cols

def fast_backtest():
    """Быстрый бэктест с пакетным предсказанием."""
    
    print("🗓️  БЫСТРЫЙ ОКТЯБРЬСКИЙ БЭКТЕСТ")
    print("=" * 40)
    
    # Параметры
    initial_balance = 1000.0
    current_balance = initial_balance
    fee = 0.001
    buy_threshold = 0.6
    sell_threshold = 0.4
    
    # Загружаем модель и данные
    try:
        model, scaler = load_model_and_scaler()
        
        data_path = "data/processed/october_2025_test.csv"
        data = pd.read_csv(data_path)
        print(f"📊 Loaded {len(data)} rows of test data")
        
        # Подготавливаем все фичи сразу
        features, feature_cols = prepare_all_features(data)
        
        # Масштабируем все данные сразу
        print("🔄 Scaling features...")
        features_scaled = scaler.transform(features)
        
        # Получаем все предсказания сразу
        print("🤖 Getting all predictions...")
        probabilities = model.predict_proba(features_scaled)
        prob_up = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        
        print("📈 Running fast backtest...")
        
        # Состояние торговли
        position = None
        entry_price = 0.0
        trades = []
        
        # Проходим по всем предсказаниям
        for i in range(len(data)):
            price = data.iloc[i]['close']
            prediction = prob_up[i]
            
            # Торговые сигналы
            if prediction > buy_threshold and position is None:
                # Открываем лонг
                position = 'long'
                entry_price = price
                print(f"📈 OPEN LONG at {price:.2f} (prob: {prediction:.3f})")
                
            elif prediction < sell_threshold and position == 'long':
                # Закрываем лонг
                exit_price = price
                
                # Рассчитываем прибыль
                gross_profit = current_balance / entry_price * exit_price
                fee_amount = gross_profit * fee
                net_profit = gross_profit - fee_amount
                
                profit_pct = (net_profit - current_balance) / current_balance
                current_balance = net_profit
                
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'balance': current_balance
                })
                
                print(f"📉 CLOSE LONG at {price:.2f} | Profit: {profit_pct:+.2%} | Balance: {current_balance:.2f}")
                
                position = None
                entry_price = 0.0
        
        # Закрываем открытую позицию в конце
        if position == 'long':
            final_price = data.iloc[-1]['close']
            gross_profit = current_balance / entry_price * final_price
            fee_amount = gross_profit * fee
            net_profit = gross_profit - fee_amount
            profit_pct = (net_profit - current_balance) / current_balance
            current_balance = net_profit
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': final_price,
                'profit_pct': profit_pct,
                'balance': current_balance
            })
            
            print(f"📉 FINAL CLOSE at {final_price:.2f} | Profit: {profit_pct:+.2%}")
        
        # Статистика
        print(f"\n📊 РЕЗУЛЬТАТЫ БЭКТЕСТА")
        print("=" * 30)
        
        if trades:
            total_return = (current_balance / initial_balance) - 1
            winning_trades = [t for t in trades if t['profit_pct'] > 0]
            win_rate = len(winning_trades) / len(trades) * 100
            
            profits = [t['profit_pct'] for t in trades]
            best_trade = max(profits) * 100
            worst_trade = min(profits) * 100
            
            # Buy & Hold для сравнения
            start_price = data.iloc[0]['close']
            end_price = data.iloc[-1]['close']
            buy_hold_return = (end_price / start_price) - 1
            
            print(f"💰 Initial balance:  ${initial_balance:,.2f}")
            print(f"📈 Final balance:    ${current_balance:,.2f}")
            print(f"🎯 Total return:     {total_return:+.2%}")
            print(f"📊 Total trades:     {len(trades)}")
            print(f"✅ Win rate:         {win_rate:.1f}%")
            print(f"🚀 Best trade:       {best_trade:+.2f}%")
            print(f"💀 Worst trade:      {worst_trade:+.2f}%")
            print(f"\n🔄 СРАВНЕНИЕ")
            print(f"📊 Buy & Hold:       {buy_hold_return:+.2%}")
            print(f"🤖 ML Strategy:      {total_return:+.2%}")
            print(f"🎯 Difference:       {total_return - buy_hold_return:+.2%}")
            
            if total_return > buy_hold_return:
                print("✅ ML strategy outperformed Buy & Hold!")
            else:
                print("❌ ML strategy underperformed Buy & Hold")
            
            # Сохраняем результаты
            output_dir = Path("results/october_test")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            trades_df = pd.DataFrame(trades)
            trades_file = output_dir / "backtest_trades.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"\n💾 Results saved: {trades_file}")
            
        else:
            print("❌ No trades executed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Главная функция."""
    
    print("🚀 ЗАПУСК БЫСТРОГО ОКТЯБРЬСКОГО ТЕСТА")
    print("=" * 50)
    
    # Создаём октябрьские данные если их нет
    october_file = "data/processed/october_2025_test.csv"
    if not Path(october_file).exists():
        print("📋 Creating October data...")
        import subprocess
        result = subprocess.run(['python3', 'quick_october_test.py'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Failed to create October data")
            return
    
    # Запускаем быстрый бэктест
    fast_backtest()
    
    print(f"\n🎉 ТЕСТ ЗАВЕРШЁН!")

if __name__ == "__main__":
    main()