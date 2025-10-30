#!/usr/bin/env python3
"""
Анализ результатов предсказаний модели

Анализирует файл test_with_preds.csv и показывает детальную статистику
по торговым сигналам и точности модели.
"""

import pandas as pd
import numpy as np

def analyze_predictions():
    """Анализирует результаты предсказаний."""
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ ПРЕДСКАЗАНИЙ")
    print("=" * 50)
    
    # Загружаем результаты
    df = pd.read_csv("data/processed/test_with_preds.csv")
    
    print(f"📁 Загружено {len(df)} образцов с предсказаниями")
    
    # Общая статистика
    print(f"\n📈 ОБЩАЯ СТАТИСТИКА:")
    print(f"   Реальные классы:")
    target_counts = df['target'].value_counts().sort_index()
    print(f"     0 (Down): {target_counts[0]} ({target_counts[0]/len(df)*100:.1f}%)")
    print(f"     1 (Up):   {target_counts[1]} ({target_counts[1]/len(df)*100:.1f}%)")
    
    print(f"\n   Предсказанные классы:")
    pred_counts = df['pred'].value_counts().sort_index()
    print(f"     0 (Down): {pred_counts[0]} ({pred_counts[0]/len(df)*100:.1f}%)")
    print(f"     1 (Up):   {pred_counts[1]} ({pred_counts[1]/len(df)*100:.1f}%)")
    
    # Общая точность
    accuracy = (df['target'] == df['pred']).mean()
    print(f"\n🎯 Общая точность: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Анализ торговых сигналов
    print(f"\n💰 АНАЛИЗ ТОРГОВЫХ СИГНАЛОВ:")
    signal_counts = df['signal'].value_counts()
    print(f"   🟢 Buy:  {signal_counts['Buy']} ({signal_counts['Buy']/len(df)*100:.1f}%)")
    print(f"   🔴 Sell: {signal_counts['Sell']} ({signal_counts['Sell']/len(df)*100:.1f}%)")
    print(f"   🟡 Hold: {signal_counts['Hold']} ({signal_counts['Hold']/len(df)*100:.1f}%)")
    
    # Точность по сигналам
    print(f"\n🎯 ТОЧНОСТЬ ПО СИГНАЛАМ:")
    
    for signal in ['Buy', 'Sell', 'Hold']:
        signal_mask = df['signal'] == signal
        if signal_mask.sum() > 0:
            signal_df = df[signal_mask]
            
            if signal == 'Buy':
                # Для Buy сигналов проверяем, сколько действительно выросло
                correct = (signal_df['target'] == 1).sum()
            elif signal == 'Sell':
                # Для Sell сигналов проверяем, сколько действительно упало
                correct = (signal_df['target'] == 0).sum()
            else:  # Hold
                # Для Hold не считаем точность, это нейтральный сигнал
                print(f"   🟡 {signal:4s}: {signal_mask.sum():4d} сигналов (нейтральные)")
                continue
            
            accuracy = correct / signal_mask.sum()
            print(f"   {'🟢' if signal == 'Buy' else '🔴'} {signal:4s}: {signal_mask.sum():4d} сигналов, точность: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Анализ вероятностей
    print(f"\n📊 АНАЛИЗ ВЕРОЯТНОСТЕЙ:")
    print(f"   Средняя вероятность UP: {df['prob_up'].mean():.4f}")
    print(f"   Медианная вероятность:  {df['prob_up'].median():.4f}")
    print(f"   Мин/Макс вероятности:   {df['prob_up'].min():.4f} / {df['prob_up'].max():.4f}")
    
    # Распределение вероятностей
    print(f"\n📈 РАСПРЕДЕЛЕНИЕ УВЕРЕННОСТИ:")
    very_low = (df['prob_up'] < 0.2).sum()
    low = ((df['prob_up'] >= 0.2) & (df['prob_up'] < 0.4)).sum()
    neutral = ((df['prob_up'] >= 0.4) & (df['prob_up'] < 0.6)).sum()
    high = ((df['prob_up'] >= 0.6) & (df['prob_up'] < 0.8)).sum()
    very_high = (df['prob_up'] >= 0.8).sum()
    
    print(f"   Очень низкая (<0.2):  {very_low} ({very_low/len(df)*100:.1f}%)")
    print(f"   Низкая (0.2-0.4):     {low} ({low/len(df)*100:.1f}%)")
    print(f"   Нейтральная (0.4-0.6): {neutral} ({neutral/len(df)*100:.1f}%)")
    print(f"   Высокая (0.6-0.8):    {high} ({high/len(df)*100:.1f}%)")
    print(f"   Очень высокая (>0.8): {very_high} ({very_high/len(df)*100:.1f}%)")
    
    # Точность для высокоуверенных предсказаний
    print(f"\n⭐ ВЫСОКОУВЕРЕННЫЕ ПРЕДСКАЗАНИЯ:")
    
    # Очень высокая уверенность в росте
    high_up_mask = df['prob_up'] > 0.8
    if high_up_mask.sum() > 0:
        high_up_accuracy = (df[high_up_mask]['target'] == 1).mean()
        print(f"   📈 Очень уверенный рост (>0.8): {high_up_mask.sum()} случаев, точность: {high_up_accuracy:.4f}")
    
    # Очень высокая уверенность в падении
    high_down_mask = df['prob_up'] < 0.2
    if high_down_mask.sum() > 0:
        high_down_accuracy = (df[high_down_mask]['target'] == 0).mean()
        print(f"   📉 Очень уверенное падение (<0.2): {high_down_mask.sum()} случаев, точность: {high_down_accuracy:.4f}")
    
    # Примеры лучших предсказаний
    print(f"\n🏆 ПРИМЕРЫ ЛУЧШИХ ПРЕДСКАЗАНИЙ:")
    
    # Лучшие Buy сигналы (высокая вероятность + правильно предсказано)
    best_buy = df[(df['signal'] == 'Buy') & (df['target'] == 1) & (df['prob_up'] > 0.8)]
    if len(best_buy) > 0:
        top_buy = best_buy.nlargest(3, 'prob_up')
        print(f"   🟢 Топ-3 Buy сигнала:")
        for i, (_, row) in enumerate(top_buy.iterrows()):
            print(f"      {i+1}. Вероятность: {row['prob_up']:.4f}, Результат: {'✅' if row['target'] == 1 else '❌'}")
    
    # Лучшие Sell сигналы
    best_sell = df[(df['signal'] == 'Sell') & (df['target'] == 0) & (df['prob_up'] < 0.2)]
    if len(best_sell) > 0:
        top_sell = best_sell.nsmallest(3, 'prob_up')
        print(f"   🔴 Топ-3 Sell сигнала:")
        for i, (_, row) in enumerate(top_sell.iterrows()):
            print(f"      {i+1}. Вероятность: {row['prob_up']:.4f}, Результат: {'✅' if row['target'] == 0 else '❌'}")
    
    # Итоговая оценка
    print(f"\n🎯 ИТОГОВАЯ ОЦЕНКА МОДЕЛИ:")
    buy_accuracy = (df[df['signal'] == 'Buy']['target'] == 1).mean()
    sell_accuracy = (df[df['signal'] == 'Sell']['target'] == 0).mean()
    
    print(f"   📊 Общая точность:     {accuracy:.1%}")
    print(f"   🟢 Точность Buy:       {buy_accuracy:.1%}")
    print(f"   🔴 Точность Sell:      {sell_accuracy:.1%}")
    print(f"   💰 Торговых сигналов:  {(signal_counts['Buy'] + signal_counts['Sell'])/len(df):.1%}")
    
    if buy_accuracy > 0.5 or sell_accuracy > 0.5:
        print(f"   ✅ Модель показывает потенциал для торговли!")
    else:
        print(f"   ⚠️ Модель требует доработки для эффективной торговли")
    
    print(f"\n🚀 РЕКОМЕНДАЦИИ:")
    print(f"   1. Используйте только высокоуверенные сигналы (>0.7 или <0.3)")
    print(f"   2. Комбинируйте с другими техническими индикаторами")
    print(f"   3. Применяйте стоп-лоссы и тейк-профиты")
    print(f"   4. Протестируйте на live данных с небольшими суммами")

if __name__ == "__main__":
    analyze_predictions()