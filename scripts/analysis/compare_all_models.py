#!/usr/bin/env python3
"""
Сравнение всех обученных моделей на октябрьских данных.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

def load_model_and_scaler(horizon):
    """Загрузка модели и скалера для конкретного горизонта."""
    
    model_path = f"models/optimized/rf_horizon_{horizon}_optimized.pkl"
    scaler_path = f"models/optimized/scaler_horizon_{horizon}_optimized.pkl"
    
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
    
    features = data[feature_cols].values
    
    return features, feature_cols

def test_model(horizon, data):
    """Тестирование модели с конкретным горизонтом."""
    
    print(f"\n🤖 ТЕСТИРОВАНИЕ МОДЕЛИ HORIZON_{horizon}")
    print("=" * 50)
    
    # Параметры
    initial_balance = 1000.0
    current_balance = initial_balance
    fee = 0.001
    buy_threshold = 0.6
    sell_threshold = 0.4
    
    try:
        # Загружаем модель
        model, scaler = load_model_and_scaler(horizon)
        
        # Подготавливаем фичи
        features, feature_cols = prepare_all_features(data)
        print(f"🔍 Using {len(feature_cols)} features")
        
        # Масштабируем данные
        features_scaled = scaler.transform(features)
        
        # Получаем предсказания
        probabilities = model.predict_proba(features_scaled)
        prob_up = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        
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
                position = 'long'
                entry_price = price
                
            elif prediction < sell_threshold and position == 'long':
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
                    'profit_pct': profit_pct
                })
                
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
                'profit_pct': profit_pct
            })
        
        # Рассчитываем статистику
        if trades:
            total_return = (current_balance / initial_balance) - 1
            winning_trades = [t for t in trades if t['profit_pct'] > 0]
            win_rate = len(winning_trades) / len(trades) * 100
            
            profits = [t['profit_pct'] for t in trades]
            best_trade = max(profits) * 100
            worst_trade = min(profits) * 100
            avg_profit = np.mean(profits) * 100
            
            # Volatility (std of returns)
            volatility = np.std(profits) * 100
            
            # Sharpe ratio approximation
            sharpe = avg_profit / volatility if volatility > 0 else 0
            
            results = {
                'horizon': horizon,
                'total_return': total_return * 100,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'avg_profit': avg_profit,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'final_balance': current_balance
            }
            
            print(f"✅ Результаты:")
            print(f"   💰 Доходность: {total_return:+.2%}")
            print(f"   📊 Сделок: {len(trades)}")
            print(f"   ✅ Win rate: {win_rate:.1f}%")
            print(f"   📈 Среднее: {avg_profit:+.2f}%")
            print(f"   📊 Волатильность: {volatility:.2f}%")
            print(f"   📉 Sharpe: {sharpe:.2f}")
            
            return results
            
        else:
            print("❌ No trades executed!")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Главная функция сравнения всех моделей."""
    
    print("🔍 СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ НА ОКТЯБРЬСКИХ ДАННЫХ")
    print("=" * 60)
    
    # Загружаем октябрьские данные
    data_path = "data/processed/october_2025_test.csv"
    if not Path(data_path).exists():
        print("❌ Нет октябрьских данных, создаём...")
        import subprocess
        subprocess.run(['python3', 'quick_october_test.py'], capture_output=True)
    
    data = pd.read_csv(data_path)
    print(f"📊 Loaded {len(data)} rows of October test data")
    
    # Buy & Hold для сравнения
    start_price = data.iloc[0]['close']
    end_price = data.iloc[-1]['close']
    buy_hold_return = (end_price / start_price) - 1
    
    # Тестируем все модели
    horizons = [3, 5, 7, 10]
    results = []
    
    for horizon in horizons:
        result = test_model(horizon, data)
        if result:
            results.append(result)
    
    # Сводная таблица
    if results:
        print(f"\n📊 СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("=" * 80)
        print(f"📊 Buy & Hold: {buy_hold_return:+.2%}")
        print("-" * 80)
        print(f"{'Horizon':<8} {'Return %':<10} {'Trades':<8} {'Win %':<8} {'Avg %':<8} {'Sharpe':<8} {'vs B&H':<8}")
        print("-" * 80)
        
        # Сортируем по доходности
        results_sorted = sorted(results, key=lambda x: x['total_return'], reverse=True)
        
        for result in results_sorted:
            vs_bh = result['total_return'] - (buy_hold_return * 100)
            print(f"{result['horizon']:<8} "
                  f"{result['total_return']:+<10.2f} "
                  f"{result['total_trades']:<8} "
                  f"{result['win_rate']:<8.1f} "
                  f"{result['avg_profit']:+<8.2f} "
                  f"{result['sharpe_ratio']:<8.2f} "
                  f"{vs_bh:+<8.2f}")
        
        print("-" * 80)
        
        # Лучшая модель
        best_model = results_sorted[0]
        print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: HORIZON_{best_model['horizon']}")
        print(f"   💰 Доходность: {best_model['total_return']:+.2f}%")
        print(f"   📊 Превосходство над B&H: {best_model['total_return'] - (buy_hold_return * 100):+.2f}%")
        print(f"   ✅ Win rate: {best_model['win_rate']:.1f}%")
        print(f"   📉 Sharpe ratio: {best_model['sharpe_ratio']:.2f}")
        
        # Сохраняем результаты
        results_df = pd.DataFrame(results)
        results_df.to_csv("results/models_comparison_october.csv", index=False)
        print(f"\n💾 Результаты сохранены: results/models_comparison_october.csv")
    
    print(f"\n🎉 СРАВНЕНИЕ ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main()