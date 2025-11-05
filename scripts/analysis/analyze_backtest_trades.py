#!/usr/bin/env python3
"""
Анализ детальной информации о сделках из бэктеста.
Показывает полные параметры сделок, включая условия открытия/закрытия.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def load_trade_data():
    """Загрузка данных о сделках и анализ."""
    
    trades_file = "results/backtest_trades.csv"
    equity_file = "results/backtest_equity_curve.csv"
    
    if not Path(trades_file).exists():
        print(f"❌ Файл сделок не найден: {trades_file}")
        return None, None
    
    trades_df = pd.read_csv(trades_file)
    print(f"📊 Загружено сделок: {len(trades_df)}")
    
    equity_df = None
    if Path(equity_file).exists():
        equity_df = pd.read_csv(equity_file)
        print(f"📈 Загружено точек equity curve: {len(equity_df)}")
    
    return trades_df, equity_df

def analyze_trade_conditions():
    """Анализ условий открытия и закрытия позиций."""
    
    print("\n🔍 АНАЛИЗ УСЛОВИЙ ТОРГОВЛИ")
    print("=" * 60)
    
    # Загружаем тестовые данные для анализа сигналов
    test_data_file = "data/processed/test_small.csv"
    
    if Path(test_data_file).exists():
        df = pd.read_csv(test_data_file)
        
        # Загружаем модель для получения вероятностей
        import pickle
        
        model_file = "models/optimized/rf_horizon_10_optimized.pkl"
        scaler_file = "models/optimized/scaler_horizon_10_optimized.pkl"
        
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            print("✅ Модель и скалер загружены")
            
            # Подготавливаем фичи для анализа сигналов
            exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target_10']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Рассчитываем вероятности для каждой строки
            probabilities = []
            signals = []
            
            for i in range(min(100, len(df))):  # Анализируем первые 100 строк
                features = df.iloc[i][feature_cols].values.reshape(1, -1)
                features_scaled = scaler.transform(features)
                
                prob = model.predict_proba(features_scaled)[0][1]
                probabilities.append(prob)
                
                if prob > 0.6:
                    signal = 'BUY'
                elif prob < 0.4:
                    signal = 'SELL'
                else:
                    signal = 'HOLD'
                
                signals.append(signal)
            
            # Анализ распределения сигналов
            signal_counts = pd.Series(signals).value_counts()
            
            print(f"\n📊 РАСПРЕДЕЛЕНИЕ СИГНАЛОВ (первые 100 баров):")
            for signal, count in signal_counts.items():
                percentage = count / len(signals) * 100
                print(f"   {signal:4s}: {count:3d} ({percentage:5.1f}%)")
            
            # Статистика вероятностей
            prob_stats = pd.Series(probabilities).describe()
            print(f"\n📈 СТАТИСТИКА ВЕРОЯТНОСТЕЙ:")
            print(f"   Среднее:    {prob_stats['mean']:.3f}")
            print(f"   Медиана:    {prob_stats['50%']:.3f}")
            print(f"   Мин/Макс:   {prob_stats['min']:.3f} / {prob_stats['max']:.3f}")
            print(f"   Std:        {prob_stats['std']:.3f}")
            
            # Анализ порогов
            buy_signals = sum(1 for p in probabilities if p > 0.6)
            sell_signals = sum(1 for p in probabilities if p < 0.4)
            hold_signals = len(probabilities) - buy_signals - sell_signals
            
            print(f"\n🎯 АНАЛИЗ ПОРОГОВ:")
            print(f"   Порог покупки (>0.6):  {buy_signals} сигналов")
            print(f"   Порог продажи (<0.4):  {sell_signals} сигналов") 
            print(f"   Удержание (0.4-0.6):   {hold_signals} сигналов")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
    
    else:
        print(f"⚠️  Тестовые данные не найдены: {test_data_file}")

def analyze_detailed_trades():
    """Детальный анализ каждой сделки."""
    
    trades_df, equity_df = load_trade_data()
    
    if trades_df is None:
        return
    
    print(f"\n📋 ДЕТАЛЬНЫЙ АНАЛИЗ СДЕЛОК")
    print("=" * 80)
    
    # Конвертируем временные метки
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    
    # Рассчитываем дополнительные метрики
    trades_df['duration_minutes'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60
    trades_df['price_change'] = trades_df['exit_price'] - trades_df['entry_price']
    trades_df['price_change_pct'] = (trades_df['exit_price'] / trades_df['entry_price'] - 1) * 100
    
    # Анализ по сделкам
    print(f"📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"   Всего сделок:        {len(trades_df)}")
    print(f"   Прибыльных:          {len(trades_df[trades_df['profit_pct'] > 0])}")
    print(f"   Убыточных:           {len(trades_df[trades_df['profit_pct'] <= 0])}")
    print(f"   Win Rate:            {len(trades_df[trades_df['profit_pct'] > 0]) / len(trades_df) * 100:.1f}%")
    
    # Статистика по длительности
    print(f"\n⏱️  СТАТИСТИКА ДЛИТЕЛЬНОСТИ:")
    print(f"   Среднее время:       {trades_df['duration_minutes'].mean():.1f} минут")
    print(f"   Медиана:             {trades_df['duration_minutes'].median():.1f} минут") 
    print(f"   Мин/Макс:            {trades_df['duration_minutes'].min():.0f} / {trades_df['duration_minutes'].max():.0f} минут")
    
    # Топ-5 лучших сделок
    print(f"\n🚀 ТОП-5 ЛУЧШИХ СДЕЛОК:")
    best_trades = trades_df.nlargest(5, 'profit_pct')
    
    for idx, trade in best_trades.iterrows():
        entry_time = trade['entry_time'].strftime('%Y-%m-%d %H:%M')
        exit_time = trade['exit_time'].strftime('%Y-%m-%d %H:%M')
        duration = trade['duration_minutes']
        
        print(f"   #{idx+1}: {trade['profit_pct']:+6.2%} | "
              f"${trade['entry_price']:7.2f} → ${trade['exit_price']:7.2f} | "
              f"{duration:4.0f}мин | {entry_time}")
    
    # Топ-5 худших сделок
    print(f"\n💀 ТОП-5 ХУДШИХ СДЕЛОК:")
    worst_trades = trades_df.nsmallest(5, 'profit_pct')
    
    for idx, trade in worst_trades.iterrows():
        entry_time = trade['entry_time'].strftime('%Y-%m-%d %H:%M')
        exit_time = trade['exit_time'].strftime('%Y-%m-%d %H:%M')
        duration = trade['duration_minutes']
        
        print(f"   #{idx+1}: {trade['profit_pct']:+6.2%} | "
              f"${trade['entry_price']:7.2f} → ${trade['exit_price']:7.2f} | "
              f"{duration:4.0f}мин | {entry_time}")

def show_trading_rules():
    """Показать правила торговли, используемые в бэктестере."""
    
    print(f"\n📜 ПРАВИЛА ТОРГОВОЙ СИСТЕМЫ")
    print("=" * 50)
    
    print(f"🎯 УСЛОВИЯ ОТКРЫТИЯ ПОЗИЦИЙ:")
    print(f"   📈 ПОКУПКА (LONG):")
    print(f"      • Вероятность роста > 0.6 (60%)")
    print(f"      • Нет открытой позиции")
    print(f"      • Покупаем на всю сумму баланса")
    print(f"      • Комиссия: 0.1% от оборота")
    
    print(f"\n   📉 ПРОДАЖА (ЗАКРЫТИЕ):")
    print(f"      • Вероятность роста < 0.4 (40%)")
    print(f"      • Есть открытая LONG позиция")
    print(f"      • Продаём всю позицию")
    print(f"      • Комиссия: 0.1% от оборота")
    
    print(f"\n   ⏸️  УДЕРЖАНИЕ:")
    print(f"      • Вероятность между 0.4 и 0.6")
    print(f"      • Никаких действий")
    
    print(f"\n💰 УПРАВЛЕНИЕ КАПИТАЛОМ:")
    print(f"   • Начальный баланс: $1,000")
    print(f"   • Размер позиции: 100% баланса")
    print(f"   • Реинвестирование прибыли: ДА")
    print(f"   • Кредитное плечо: НЕТ")
    
    print(f"\n🚫 ОГРАНИЧЕНИЯ:")
    print(f"   • Только LONG позиции (нет шорта)")
    print(f"   • Одна позиция в моменте времени")
    print(f"   • Нет стоп-лоссов или тейк-профитов")
    print(f"   • Нет задержки исполнения (мгновенное)")
    
    print(f"\n🤖 ML МОДЕЛЬ:")
    print(f"   • Random Forest Classifier")
    print(f"   • Горизонт прогноза: 10 баров (50 минут)")
    print(f"   • Количество фичей: 25 (оптимизированных)")
    print(f"   • Точность модели: ~63.2%")

def analyze_risk_metrics():
    """Анализ риск-метрик."""
    
    trades_df, equity_df = load_trade_data()
    
    if trades_df is None or equity_df is None:
        return
    
    print(f"\n⚠️  АНАЛИЗ РИСКОВ")
    print("=" * 40)
    
    # Конвертируем equity данные
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    
    # Максимальная просадка
    equity_values = equity_df['equity'].values
    running_max = np.maximum.accumulate(equity_values)
    drawdowns = (equity_values - running_max) / running_max * 100
    
    max_drawdown = np.min(drawdowns)
    max_dd_idx = np.argmin(drawdowns)
    max_dd_date = equity_df.iloc[max_dd_idx]['timestamp']
    
    print(f"📉 ПРОСАДКИ:")
    print(f"   Максимальная просадка: {max_drawdown:.2f}%")
    print(f"   Дата макс. просадки:   {max_dd_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"   Среднее время в просадке: Нет данных")
    
    # Анализ последовательных убытков
    losses = trades_df[trades_df['profit_pct'] < 0]
    if len(losses) > 0:
        print(f"\n💀 УБЫТОЧНЫЕ СДЕЛКИ:")
        print(f"   Количество:          {len(losses)}")
        print(f"   Средний убыток:      {losses['profit_pct'].mean():.2%}")
        print(f"   Максимальный убыток: {losses['profit_pct'].min():.2%}")
    
    # Анализ волатильности доходности
    returns = trades_df['profit_pct'].values
    volatility = np.std(returns) * 100
    
    print(f"\n📊 ВОЛАТИЛЬНОСТЬ:")
    print(f"   Стандартное отклонение: {volatility:.2f}%")
    print(f"   Коэффициент вариации:   {volatility / np.mean(returns * 100):.2f}")

def create_trade_summary():
    """Создать итоговую сводку по сделкам."""
    
    trades_df, _ = load_trade_data()
    
    if trades_df is None:
        return
    
    # Создаём расширенный отчёт
    summary_file = "results/detailed_trade_analysis.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("ДЕТАЛЬНЫЙ АНАЛИЗ СДЕЛОК БЭКТЕСТА\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Общее количество сделок: {len(trades_df)}\n")
        f.write(f"Прибыльные сделки: {len(trades_df[trades_df['profit_pct'] > 0])}\n")
        f.write(f"Убыточные сделки: {len(trades_df[trades_df['profit_pct'] <= 0])}\n\n")
        
        f.write("ДЕТАЛИ ПО КАЖДОЙ СДЕЛКЕ:\n")
        f.write("-" * 30 + "\n")
        
        for idx, trade in trades_df.iterrows():
            entry_time = pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')
            exit_time = pd.to_datetime(trade['exit_time']).strftime('%Y-%m-%d %H:%M')
            duration = (pd.to_datetime(trade['exit_time']) - pd.to_datetime(trade['entry_time'])).total_seconds() / 60
            
            f.write(f"\nСделка #{idx + 1}:\n")
            f.write(f"  Вход:      {entry_time} по ${trade['entry_price']:.4f}\n")
            f.write(f"  Выход:     {exit_time} по ${trade['exit_price']:.4f}\n")
            f.write(f"  Размер:    {trade['position_size']:.4f} единиц\n")
            f.write(f"  Результат: {trade['profit_pct']:+.2%} (${trade['net_profit'] - 1000:.2f})\n")
            f.write(f"  Время:     {duration:.0f} минут\n")
            f.write(f"  Комиссия:  ${trade['fee_amount']:.2f}\n")
    
    print(f"💾 Детальный анализ сохранён: {summary_file}")

def main():
    """Главная функция анализа."""
    
    print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ БЭКТЕСТА")
    print("=" * 60)
    
    # 1. Анализ условий торговли
    analyze_trade_conditions()
    
    # 2. Детальный анализ сделок
    analyze_detailed_trades()
    
    # 3. Показать правила торговли
    show_trading_rules()
    
    # 4. Анализ рисков
    analyze_risk_metrics()
    
    # 5. Создать детальную сводку
    create_trade_summary()
    
    print(f"\n🎉 АНАЛИЗ ЗАВЕРШЁН!")
    print(f"📄 Проверьте results/detailed_trade_analysis.txt для полной информации")

if __name__ == "__main__":
    main()