#!/usr/bin/env python3
"""
Простой бэктест на данных 10-17 августа с использованием свежих ML моделей
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add src to path
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)

from src.utils.data_blocks import DataBlockManager
from src.models.ml_predictor import MLPredictor
from src.feature_engine.technical_indicators import TechnicalIndicatorCalculator

def run_simple_backtest():
    print("🔬 Бэктест на данных 10-17 августа с ML моделями 90d_enhanced")
    print("=" * 65)
    
    # Загрузка данных
    print("📊 Загрузка данных блока august_10_17_full...")
    data_manager = DataBlockManager()
    block_data = data_manager.load_block("august_10_17_full", ["5m"])
    
    if not block_data or '5m' not in block_data:
        print("❌ Не удалось загрузить блок данных")
        return
        
    # Используем 5-минутные данные для тестирования
    df = block_data['5m']
    print(f"✅ Загружено {len(df)} 5-минутных свечей")
    print(f"📅 Период: {df.index[0]} → {df.index[-1]}")
    
    # Расчет индикаторов
    print("🔧 Расчет технических индикаторов...")
    calc = TechnicalIndicatorCalculator()
    indicators = calc.calculate_all(df, "SOLUSDT", "5m")
    
    # Добавляем индикаторы в датафрейм
    for name, series in indicators.items():
        df[name] = series
    
    # Удаление NaN значений
    df = df.dropna()
    print(f"📊 После расчета индикаторов: {len(df)} записей")
    
    # Инициализация ML предиктора
    print("🤖 Загрузка ML моделей 90d_enhanced...")
    try:
        predictor = MLPredictor(models_dir="models/90d_enhanced")
        print("✅ ML модели загружены успешно")
    except Exception as e:
        print(f"❌ Ошибка загрузки моделей: {e}")
        return
    
    # Настройки бэктеста
    initial_balance = 10000.0
    commission = 0.001  # 0.1%
    balance = initial_balance
    position = 0  # 0 = нет позиции, 1 = long, -1 = short
    position_size = 0
    entry_price = 0
    
    trades = []
    equity_curve = [initial_balance]
    
    print(f"💰 Начальный баланс: ${initial_balance:,.2f}")
    print(f"💸 Комиссия: {commission*100:.1f}%")
    print("🚀 Запуск симуляции...")
    
    for i in range(100, len(df)):  # Начинаем с 100 для стабильности индикаторов
        current_data = df.iloc[i]
        current_price = current_data['close']
        
        # Подготовка данных для предсказания
        features_df = df.iloc[i-50:i+1].copy()  # Берем окно для стабильности
        
        try:
            # Получение предсказания ML модели
            predictions = predictor.predict(features_df.tail(1))
            
            if predictions and len(predictions) > 0:
                pred = predictions[0]
                confidence = pred.get('confidence', 0.5)
                signal = pred.get('signal', 'hold')
                
                # Торговая логика
                if position == 0:  # Нет позиции
                    if signal == 'buy' and confidence > 0.7:
                        # Открываем LONG позицию
                        position = 1
                        position_size = (balance * 0.95) / current_price  # Используем 95% баланса
                        entry_price = current_price
                        commission_cost = position_size * current_price * commission
                        balance -= commission_cost
                        
                        print(f"📈 LONG @${current_price:.2f} (confidence: {confidence:.3f}) - {current_data.name}")
                        
                    elif signal == 'sell' and confidence > 0.7:
                        # Открываем SHORT позицию (упрощенная логика)
                        position = -1
                        position_size = (balance * 0.95) / current_price
                        entry_price = current_price
                        commission_cost = position_size * current_price * commission
                        balance -= commission_cost
                        
                        print(f"📉 SHORT @${current_price:.2f} (confidence: {confidence:.3f}) - {current_data.name}")
                
                elif position != 0:  # Есть позиция
                    # Простая логика выхода: противоположный сигнал или stop loss
                    should_exit = False
                    exit_reason = ""
                    
                    if position == 1:  # LONG позиция
                        if signal == 'sell' and confidence > 0.6:
                            should_exit = True
                            exit_reason = "ML signal"
                        elif current_price <= entry_price * 0.98:  # 2% stop loss
                            should_exit = True
                            exit_reason = "Stop Loss"
                        elif current_price >= entry_price * 1.04:  # 4% take profit
                            should_exit = True
                            exit_reason = "Take Profit"
                            
                    elif position == -1:  # SHORT позиция
                        if signal == 'buy' and confidence > 0.6:
                            should_exit = True
                            exit_reason = "ML signal"
                        elif current_price >= entry_price * 1.02:  # 2% stop loss for short
                            should_exit = True
                            exit_reason = "Stop Loss"
                        elif current_price <= entry_price * 0.96:  # 4% take profit for short
                            should_exit = True
                            exit_reason = "Take Profit"
                    
                    if should_exit:
                        # Закрываем позицию
                        if position == 1:  # Закрываем LONG
                            pnl = position_size * (current_price - entry_price)
                        else:  # Закрываем SHORT
                            pnl = position_size * (entry_price - current_price)
                        
                        commission_cost = position_size * current_price * commission
                        net_pnl = pnl - commission_cost
                        balance += position_size * current_price - commission_cost
                        
                        # Записываем сделку
                        trades.append({
                            'entry_time': current_data.name,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position_size': position_size,
                            'side': 'LONG' if position == 1 else 'SHORT',
                            'pnl': net_pnl,
                            'exit_reason': exit_reason
                        })
                        
                        print(f"🔄 EXIT @${current_price:.2f} | PnL: ${net_pnl:+.2f} ({exit_reason})")
                        
                        position = 0
                        position_size = 0
                        entry_price = 0
            
        except Exception as e:
            pass  # Игнорируем ошибки предсказания
        
        # Обновляем equity curve
        if position == 0:
            current_equity = balance
        elif position == 1:  # LONG
            current_equity = balance + position_size * (current_price - entry_price)
        else:  # SHORT
            current_equity = balance + position_size * (entry_price - current_price)
            
        equity_curve.append(current_equity)
    
    # Закрываем открытую позицию в конце
    if position != 0:
        final_price = df.iloc[-1]['close']
        if position == 1:
            pnl = position_size * (final_price - entry_price)
        else:
            pnl = position_size * (entry_price - final_price)
        
        commission_cost = position_size * final_price * commission
        net_pnl = pnl - commission_cost
        balance += position_size * final_price - commission_cost
        
        trades.append({
            'entry_time': df.index[-1],
            'entry_price': entry_price,
            'exit_price': final_price,
            'position_size': position_size,
            'side': 'LONG' if position == 1 else 'SHORT',
            'pnl': net_pnl,
            'exit_reason': 'End of backtest'
        })
    
    # Результаты
    final_balance = balance
    total_pnl = final_balance - initial_balance
    total_return_pct = (total_pnl / initial_balance) * 100
    
    print("\n" + "="*65)
    print("📊 РЕЗУЛЬТАТЫ БЭКТЕСТА:")
    print(f"💰 Начальный баланс: ${initial_balance:,.2f}")
    print(f"💸 Финальный баланс: ${final_balance:,.2f}")
    print(f"📈 Общий P&L: ${total_pnl:+,.2f} ({total_return_pct:+.2f}%)")
    print(f"🔄 Количество сделок: {len(trades)}")
    
    if trades:
        profitable_trades = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = (profitable_trades / len(trades)) * 100
        avg_pnl = sum(t['pnl'] for t in trades) / len(trades)
        
        print(f"✅ Прибыльных сделок: {profitable_trades} ({win_rate:.1f}%)")
        print(f"❌ Убыточных сделок: {len(trades) - profitable_trades}")
        print(f"📊 Средний PnL: ${avg_pnl:+.2f}")
        
        best_trade = max(trades, key=lambda x: x['pnl'])
        worst_trade = min(trades, key=lambda x: x['pnl'])
        
        print(f"🏆 Лучшая сделка: ${best_trade['pnl']:+.2f}")
        print(f"💀 Худшая сделка: ${worst_trade['pnl']:+.2f}")
    
    # Максимальная просадка
    max_balance = initial_balance
    max_drawdown = 0
    for equity in equity_curve:
        if equity > max_balance:
            max_balance = equity
        drawdown = (max_balance - equity) / max_balance
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    print(f"📉 Максимальная просадка: {max_drawdown*100:.2f}%")
    
    # Сохранение результатов
    results = {
        'backtest_date': datetime.now().isoformat(),
        'model_version': '90d_enhanced',
        'data_block': 'august_10_17_full',
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_pnl': total_pnl,
        'total_return_pct': total_return_pct,
        'total_trades': len(trades),
        'profitable_trades': profitable_trades if trades else 0,
        'win_rate': win_rate if trades else 0,
        'max_drawdown_pct': max_drawdown * 100,
        'trades': trades
    }
    
    output_file = f"output/ml_backtest_august_10_17_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Результаты сохранены в {output_file}")
    print("\n🎉 Бэктест завершен!")

if __name__ == "__main__":
    run_simple_backtest()