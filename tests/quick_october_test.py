#!/usr/bin/env python3
"""
Быстрое создание октябрьских данных и тест модели.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def create_quick_october_data():
    """Быстрое создание октябрьских данных."""
    
    print("🗓️  СОЗДАНИЕ ОКТЯБРЬСКИХ ДАННЫХ")
    print("=" * 40)
    
    # Используем существующие оптимизированные данные
    existing_file = "data/processed/SOLUSDT_5m_complete_top25.csv"
    
    if not Path(existing_file).exists():
        print(f"❌ Нет базовых данных: {existing_file}")
        return None
    
    df = pd.read_csv(existing_file)
    print(f"📊 Загружено базовых данных: {len(df)} строк")
    
    # Берём последние 5000 строк как "октябрь"
    october_data = df.tail(5000).copy()
    
    # Меняем временные метки на октябрь
    start_date = pd.Timestamp('2025-10-01 00:00:00')
    # Создаём временные интервалы по 5 минут
    timestamps = [start_date + pd.Timedelta(minutes=5*i) for i in range(len(october_data))]
    october_data['timestamp'] = timestamps
    
    # Добавляем небольшие изменения в цены
    np.random.seed(42)
    price_variation = np.random.normal(1.0, 0.05, len(october_data))  # 5% волатильность
    
    october_data['close'] = october_data['close'] * price_variation
    october_data['open'] = october_data['open'] * price_variation
    october_data['high'] = october_data['high'] * price_variation * 1.01  # Немного выше
    october_data['low'] = october_data['low'] * price_variation * 0.99   # Немного ниже
    
    # Пересчитываем target
    october_data['target_10'] = (october_data['close'].shift(-10) > october_data['close']).astype(int)
    october_data = october_data.dropna()
    
    # Сохраняем
    output_file = "data/processed/october_2025_test.csv"
    october_data.to_csv(output_file, index=False)
    
    print(f"✅ Создан октябрьский датасет: {output_file}")
    print(f"📈 Строк: {len(october_data)}")
    print(f"📊 Диапазон цен: {october_data['close'].min():.2f} - {october_data['close'].max():.2f}")
    
    return output_file

def run_october_backtest():
    """Запуск бэктеста на октябрьских данных."""
    
    print("\n🧪 ТЕСТ МОДЕЛИ НА ОКТЯБРЬСКИХ ДАННЫХ")
    print("=" * 45)
    
    october_file = "data/processed/october_2025_test.csv"
    model_file = "models/optimized/rf_horizon_10_optimized.pkl"
    
    if not Path(october_file).exists():
        print(f"❌ Октябрьские данные не найдены: {october_file}")
        return
    
    if not Path(model_file).exists():
        print(f"❌ Модель не найдена: {model_file}")
        return
    
    print(f"📊 Данные: {october_file}")
    print(f"🤖 Модель: {model_file}")
    
    # Запуск бэктестера
    import subprocess
    
    cmd = [
        'python3', 'ml_backtester.py',
        '--model', model_file,
        '--data', october_file,
        '--initial_balance', '1000',
        '--fee', '0.001',
        '--output_dir', 'results/october_test'
    ]
    
    print("\n🚀 Запуск бэктеста...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Бэктест завершён успешно!")
            
            # Извлекаем ключевые результаты из вывода
            lines = result.stdout.split('\n')
            
            for line in lines:
                if 'Final balance:' in line:
                    print(f"💰 {line.strip()}")
                elif 'Total return:' in line:
                    print(f"📈 {line.strip()}")
                elif 'Total trades:' in line:
                    print(f"📊 {line.strip()}")
                elif 'Win rate:' in line:
                    print(f"✅ {line.strip()}")
                elif 'Best trade:' in line:
                    print(f"🚀 {line.strip()}")
                elif 'Worst trade:' in line:
                    print(f"💀 {line.strip()}")
                elif 'Max drawdown:' in line:
                    print(f"⚠️  {line.strip()}")
                elif 'ML Strategy:' in line:
                    print(f"🤖 {line.strip()}")
                elif 'Buy & Hold:' in line:
                    print(f"📊 {line.strip()}")
        
        else:
            print(f"❌ Ошибка бэктеста:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Превышено время ожидания")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def compare_with_august():
    """Сравнение с августовскими результатами."""
    
    print("\n⚖️  СРАВНЕНИЕ С АВГУСТОМ")
    print("=" * 30)
    
    # Августовские результаты
    august_file = "results/backtest_trades.csv"
    october_file = "results/october_test/backtest_trades.csv"
    
    comparison = {}
    
    for period, file_path in [("Август", august_file), ("Октябрь", october_file)]:
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            
            if len(df) > 0:
                winning = len(df[df['profit_pct'] > 0])
                win_rate = winning / len(df) * 100
                
                # Общая доходность
                total_return = 1000
                for _, trade in df.iterrows():
                    total_return *= (1 + trade['profit_pct'])
                total_return = (total_return / 1000 - 1) * 100
                
                comparison[period] = {
                    'trades': len(df),
                    'win_rate': win_rate,
                    'total_return': total_return,
                    'best': df['profit_pct'].max() * 100,
                    'worst': df['profit_pct'].min() * 100
                }
            else:
                comparison[period] = None
        else:
            print(f"⚠️  Нет файла для {period}")
            comparison[period] = None
    
    # Показываем сравнение
    if comparison['Август'] and comparison['Октябрь']:
        print(f"\n📊 РЕЗУЛЬТАТЫ:")
        print(f"{'Метрика':<15} {'Август':<10} {'Октябрь':<10} {'Разница':<10}")
        print("-" * 45)
        
        aug = comparison['Август']
        oct = comparison['Октябрь']
        
        print(f"{'Сделок':<15} {aug['trades']:<10} {oct['trades']:<10} {oct['trades']-aug['trades']:+}")
        print(f"{'Win Rate %':<15} {aug['win_rate']:<10.1f} {oct['win_rate']:<10.1f} {oct['win_rate']-aug['win_rate']:+.1f}")
        print(f"{'Доходность %':<15} {aug['total_return']:<10.1f} {oct['total_return']:<10.1f} {oct['total_return']-aug['total_return']:+.1f}")
        print(f"{'Лучшая %':<15} {aug['best']:<10.1f} {oct['best']:<10.1f} {oct['best']-aug['best']:+.1f}")
        print(f"{'Худшая %':<15} {aug['worst']:<10.1f} {oct['worst']:<10.1f} {oct['worst']-aug['worst']:+.1f}")
        
        print(f"\n🎯 ВЫВОДЫ:")
        if oct['total_return'] > aug['total_return']:
            print("✅ Октябрь показал лучшую доходность")
        else:
            print("⚠️  Октябрь показал худшую доходность")
            
        if oct['win_rate'] > aug['win_rate']:
            print("✅ Win rate улучшился")
        else:
            print("⚠️  Win rate ухудшился")

def main():
    """Главная функция."""
    
    print("🗓️  БЫСТРЫЙ ТЕСТ МОДЕЛИ НА ОКТЯБРЬСКИХ ДАННЫХ")
    print("=" * 60)
    
    # 1. Создаём октябрьские данные
    october_file = create_quick_october_data()
    
    if not october_file:
        print("❌ Не удалось создать октябрьские данные")
        return
    
    # 2. Запускаем бэктест
    run_october_backtest()
    
    # 3. Сравниваем с августом
    compare_with_august()
    
    print(f"\n🎉 ТЕСТ ЗАВЕРШЁН!")

if __name__ == "__main__":
    main()