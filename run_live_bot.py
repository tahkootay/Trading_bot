#!/usr/bin/env python3
"""
Скрипт запуска Live Trading Bot согласно спецификации.
Интегрирует все компоненты: WebSocket, ансамбль моделей, торговые сигналы.
"""

import asyncio
import sys
from pathlib import Path

# Добавляем src в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.live_trading_bot import LiveTradingBot, BotConfig


def get_bot_config() -> BotConfig:
    """Конфигурация бота согласно спецификации."""
    return BotConfig(
        # Торговые параметры
        symbol="SOL/USDT:USDT",  # Фьючерсы как в спецификации
        timeframe="5",  # 5-минутные свечи
        
        # API ключи Bybit (заполните для реальной торговли)
        bybit_api_key="",  # Ваш API ключ
        bybit_api_secret="",  # Ваш API секрет
        bybit_testnet=True,  # True = testnet, False = mainnet
        
        # ML модели (ансамбль из спецификации)
        models_dir="models/ensemble_live",
        min_confidence=0.65,  # Минимальная уверенность для торговли
        prediction_cooldown=60,  # Секунды между предсказаниями
        
        # Риск-менеджмент (как в спецификации)
        max_position_size_usd=100.0,  # Максимальный размер позиции
        max_total_exposure_usd=500.0,  # Максимальная экспозиция
        max_daily_loss_usd=50.0,  # Максимальная дневная потеря
        stop_loss_pct=0.02,  # Stop-loss 2%
        take_profit_pct=0.04,  # Take-profit 4%
        min_trade_interval_sec=300,  # 5 минут между сделками
        
        # Общие настройки
        initial_balance=1000.0,
        dry_run=True,  # ВАЖНО: Установите False для реальной торговли
        log_level="INFO"
    )


async def main():
    """Основная функция запуска бота."""
    print("=" * 60)
    print("🤖 LIVE CRYPTO TRADING BOT с АНСАМБЛЕМ ML-МОДЕЛЕЙ")
    print("=" * 60)
    print("📋 Согласно спецификации:")
    print("   • Ансамбль: Random Forest + LightGBM + XGBoost + CatBoost")
    print("   • Метамодель: Logistic Regression") 
    print("   • Признаки: MA5/10/20, RSI, MACD, Bollinger Bands, Volume change")
    print("   • Live данные: WebSocket Bybit фьючерсы")
    print("   • Риск-менеджмент: Stop-loss, Take-profit, ограничения экспозиции")
    print("=" * 60)
    
    # Создаем конфигурацию
    config = get_bot_config()
    
    # Отображаем конфигурацию
    print(f"🎯 Символ: {config.symbol}")
    print(f"⏱️  Таймфрейм: {config.timeframe} минут")
    print(f"🔄 Режим: {'DRY RUN (симуляция)' if config.dry_run else 'LIVE TRADING (реальные деньги!)'}")
    print(f"🧠 Минимальная уверенность: {config.min_confidence}")
    print(f"💰 Максимальная позиция: ${config.max_position_size_usd}")
    print(f"🛡️  Максимальная дневная потеря: ${config.max_daily_loss_usd}")
    print("=" * 60)
    
    if not config.dry_run and (not config.bybit_api_key or not config.bybit_api_secret):
        print("❌ ОШИБКА: Для реальной торговли необходимо указать API ключи Bybit!")
        print("   Заполните bybit_api_key и bybit_api_secret в конфигурации")
        return
    
    # Создаем и запускаем бота
    try:
        print("🚀 Инициализация бота...")
        bot = LiveTradingBot(config)
        
        print("✅ Бот успешно инициализирован!")
        print("🎯 Начинаем live торговлю...")
        print("📊 Нажмите Ctrl+C для graceful остановки")
        print("=" * 60)
        
        # Запускаем бота
        await bot.start()
        
    except KeyboardInterrupt:
        print("\n⚠️  Получен сигнал остановки от пользователя")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
    finally:
        if 'bot' in locals():
            print("🛑 Останавливаем бота...")
            await bot.stop()
        print("👋 Бот остановлен")


if __name__ == "__main__":
    print("🤖 Запуск Live Trading Bot...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Программа завершена пользователем")
    except Exception as e:
        print(f"❌ Критическая ошибка при запуске: {e}")