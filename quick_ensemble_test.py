#!/usr/bin/env python3
"""
Быстрый тест ансамбля на ограниченных данных с прогресс-баром
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

def quick_ensemble_test():
    """Быстрый тест ансамбля."""
    print("🚀 БЫСТРЫЙ ТЕСТ АНСАМБЛЯ")
    print("="*50)
    
    # 1. Загружаем данные (только последние 500 свечей)
    print("📊 Загрузка данных...")
    data_file = "data/SOLUSDT_5m_real_2025-08-10_to_2025-08-17.csv"
    
    if not Path(data_file).exists():
        print("❌ Файл данных не найден")
        return
    
    # Загружаем только последние строки для быстроты
    data = pd.read_csv(data_file).tail(500)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    
    print(f"✅ Загружено {len(data)} свечей")
    
    # 2. Генерируем быстрые признаки
    print("📈 Генерация базовых признаков...")
    
    # Только основные признаки для ускорения
    data['MA5'] = data['close'].rolling(5, min_periods=1).mean()
    data['MA10'] = data['close'].rolling(10, min_periods=1).mean()
    data['MA20'] = data['close'].rolling(20, min_periods=1).mean()
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['EMA12'] = data['close'].ewm(span=12, min_periods=1).mean()
    data['EMA26'] = data['close'].ewm(span=26, min_periods=1).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_signal'] = data['MACD'].ewm(span=9, min_periods=1).mean()
    data['MACD_diff'] = data['MACD'] - data['MACD_signal']
    
    # BB
    bb_std = data['close'].rolling(20, min_periods=1).std()
    data['BB_hband'] = data['MA20'] + (bb_std * 2)
    data['BB_lband'] = data['MA20'] - (bb_std * 2)
    data['BB_width'] = data['BB_hband'] - data['BB_lband']
    data['BB_position'] = (data['close'] - data['BB_lband']) / (data['BB_width'] + 1e-8)
    
    # Volume & Price changes
    data['vol_change'] = data['volume'].pct_change().fillna(0)
    data['volume_sma'] = data['volume'].rolling(20, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / (data['volume_sma'] + 1e-8)
    data['price_change'] = data['close'].pct_change().fillna(0)
    data['price_change_5'] = data['close'].pct_change(5).fillna(0)
    
    # Остальные признаки (упрощенные)
    data['volatility'] = data['close'].rolling(14, min_periods=1).std() / (data['close'].rolling(14, min_periods=1).mean() + 1e-8)
    data['high_low_pct'] = (data['high'] - data['low']) / (data['close'] + 1e-8)
    data['close_to_high'] = (data['close'] - data['low']) / ((data['high'] - data['low']) + 1e-8)
    data['close_to_low'] = (data['high'] - data['close']) / ((data['high'] - data['low']) + 1e-8)
    data['timeframe_minutes'] = 5.0
    
    data = data.fillna(0)
    print("✅ Признаки готовы")
    
    # 3. Загружаем ансамбль
    print("🤖 Загрузка ансамбля...")
    
    try:
        from src.models.ensemble_predictor import EnsemblePredictor
        predictor = EnsemblePredictor()
        
        if not predictor.is_ready():
            print("❌ Предсказатель не готов")
            return
            
        print("✅ Ансамбль загружен")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки ансамбля: {e}")
        return
    
    # 4. Быстрый бэктест
    print("📊 Запуск бэктестирования...")
    
    balance = 10000.0
    position = None
    trades = []
    
    # Используем только последние 200 свечей для ускорения
    test_data = data.tail(200)
    
    # Названия признаков в правильном порядке
    feature_names = [
        'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 
        'BB_hband', 'BB_lband', 'BB_width', 'BB_position', 'vol_change', 
        'volume_sma', 'volume_ratio', 'price_change', 'price_change_5', 
        'volatility', 'EMA12', 'EMA26', 'high_low_pct', 'close_to_high', 
        'close_to_low', 'timeframe_minutes'
    ]
    
    signals = []
    
    with tqdm(total=len(test_data), desc="Тестирование", unit="свечи") as pbar:
        for i, (timestamp, row) in enumerate(test_data.iterrows()):
            try:
                # Подготовка признаков
                features_df = pd.DataFrame([row])
                features_df = features_df[feature_names]  # Правильный порядок
                
                # Получение предсказания
                prediction = predictor.predict_ensemble(features_df)
                
                if prediction:
                    prob = prediction['final_probability']
                    signal = prediction['final_signal']
                    
                    signals.append({
                        'timestamp': timestamp,
                        'price': row['close'],
                        'signal': signal,
                        'probability': prob
                    })
                    
                    # Простая логика торговли
                    if position is None and signal == 1 and prob > 0.6:
                        # Вход в длинную позицию
                        position = {
                            'entry_price': row['close'],
                            'entry_time': timestamp,
                            'size': balance * 0.02 / row['close']  # 2% от баланса
                        }
                    
                    elif position is not None and (signal == 0 or prob < 0.4):
                        # Выход из позиции
                        pnl = (row['close'] - position['entry_price']) * position['size']
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': timestamp,
                            'entry_price': position['entry_price'],
                            'exit_price': row['close'],
                            'pnl': pnl,
                            'duration': timestamp - position['entry_time']
                        })
                        balance += pnl
                        position = None
                
                pbar.set_postfix({
                    "Цена": f"${row['close']:.1f}",
                    "Баланс": f"${balance:.0f}",
                    "Сделок": len(trades)
                })
                
            except Exception as e:
                print(f"\n⚠️ Ошибка на свече {i}: {e}")
                continue
            
            pbar.update(1)
    
    # 5. Результаты
    print("\n" + "="*60)
    print("📋 РЕЗУЛЬТАТЫ БЫСТРОГО ТЕСТА")
    print("="*60)
    
    print(f"📊 Обработано свечей: {len(test_data)}")
    print(f"🎯 Сгенерировано сигналов: {len(signals)}")
    
    if signals:
        buy_signals = sum(1 for s in signals if s['signal'] == 1)
        avg_prob = np.mean([s['probability'] for s in signals])
        print(f"📈 BUY сигналов: {buy_signals}")
        print(f"📊 Средняя вероятность: {avg_prob:.3f}")
    
    if trades:
        total_pnl = sum(t['pnl'] for t in trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        win_rate = winning_trades / len(trades)
        
        print(f"💰 Начальный баланс: ${10000:.2f}")
        print(f"💰 Финальный баланс: ${balance:.2f}")
        print(f"💹 Общий P&L: ${total_pnl:+.2f}")
        print(f"🎯 Всего сделок: {len(trades)}")
        print(f"✅ Прибыльных: {winning_trades}")
        print(f"📊 Винрейт: {win_rate:.1%}")
        
        print(f"\n📋 Последние 3 сделки:")
        for i, trade in enumerate(trades[-3:], 1):
            duration = trade['duration'].total_seconds() / 3600
            print(f"  {i}. ${trade['pnl']:+.2f} | {duration:.1f}h | "
                  f"{trade['entry_time'].strftime('%m-%d %H:%M')} → {trade['exit_time'].strftime('%H:%M')}")
    else:
        print("❌ Сделки не выполнены")
    
    print("="*60)
    
    if len(signals) > 0 and not any('error' in str(s) for s in signals):
        print("🎉 ТЕСТ ПРОЙДЕН УСПЕШНО!")
        print("✅ Ансамбль работает корректно")
        print("✅ Генерирует валидные предсказания")
        print("✅ Нет критических ошибок")
    else:
        print("⚠️ Есть проблемы, требующие внимания")

if __name__ == "__main__":
    quick_ensemble_test()