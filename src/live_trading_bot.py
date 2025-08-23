#!/usr/bin/env python3
"""
Live крипто-бот с ансамблем ML-моделей для Bybit Futures
Согласно полной спецификации из /Users/alexey/Documents/4модели.md

Основные компоненты:
- WebSocket поток данных в реальном времени
- Генерация признаков на лету (MA, RSI, MACD, Bollinger Bands, Volume change)
- Ансамбль из 4 базовых моделей (RF, LightGBM, XGBoost, CatBoost)
- Метамодель (Logistic Regression) для комбинирования
- Торговое исполнение с интеграцией к Bybit API
- Риск-менеджмент с stop-loss, max exposure, задержками между сделками
"""

import asyncio
import logging
import signal
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Импорты компонентов
from .data_collector.websocket_client import BybitWebSocketClient, LiveDataManager
from .feature_engine.live_features import LiveFeatureGenerator
from .models.live_predictor import LiveTradingPredictor
from .execution.live_bybit_executor import BybitLiveExecutor
from .risk_manager.live_risk_manager import LiveRiskManager, RiskLimits, RiskAction
from .utils.logger import setup_logger


@dataclass
class BotConfig:
    """Конфигурация бота."""
    # Торговые параметры
    symbol: str = "SOL/USDT:USDT"
    timeframe: str = "5"  # 5 минут
    
    # API ключи Bybit
    bybit_api_key: str = ""
    bybit_api_secret: str = ""
    bybit_testnet: bool = True
    
    # ML модели
    models_dir: str = "models/ensemble_live"
    min_confidence: float = 0.65  # Минимальная уверенность для торговых сигналов
    prediction_cooldown: int = 60  # Секунды между предсказаниями
    
    # Риск-менеджмент
    max_position_size_usd: float = 100.0
    max_total_exposure_usd: float = 500.0
    max_daily_loss_usd: float = 50.0
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    min_trade_interval_sec: int = 300  # 5 минут между сделками
    
    # Общие настройки
    initial_balance: float = 1000.0
    dry_run: bool = True  # Режим симуляции
    log_level: str = "INFO"


class LiveTradingBot:
    """
    Главный класс live торгового бота.
    Объединяет все компоненты согласно архитектуре из спецификации.
    """
    
    def __init__(self, config: BotConfig):
        """
        Инициализация бота.
        
        Args:
            config: Конфигурация бота
        """
        self.config = config
        self.running = False
        self.start_time = None
        
        # Настройка логирования
        self.logger = setup_logger("LiveTradingBot", config.log_level)
        
        # Состояние
        self.stats = {
            'uptime': 0,
            'total_predictions': 0,
            'trading_signals': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'current_balance': config.initial_balance,
            'daily_pnl': 0.0
        }
        
        # Инициализация компонентов
        self._initialize_components()
        
        self.logger.info("🤖 Live Trading Bot initialized")
        self._log_configuration()
    
    def _initialize_components(self):
        """Инициализация всех компонентов бота."""
        try:
            # 1. WebSocket клиент для получения данных
            self.ws_client = BybitWebSocketClient(
                symbol=self.config.symbol.replace("/", "").replace(":USDT", ""),
                timeframe=self.config.timeframe
            )
            
            # 2. Live предсказатель (включает генерацию признаков и ансамбль моделей)
            self.predictor = LiveTradingPredictor(
                models_dir=self.config.models_dir,
                min_confidence=self.config.min_confidence,
                prediction_cooldown=self.config.prediction_cooldown
            )
            
            # 3. Исполнитель ордеров Bybit
            if not self.config.dry_run:
                self.executor = BybitLiveExecutor(
                    api_key=self.config.bybit_api_key,
                    api_secret=self.config.bybit_api_secret,
                    testnet=self.config.bybit_testnet
                )
            else:
                self.executor = None
                self.logger.info("🔄 Running in DRY RUN mode - no real trades")
            
            # 4. Риск-менеджер
            risk_limits = RiskLimits(
                max_position_size_usd=self.config.max_position_size_usd,
                max_total_exposure_usd=self.config.max_total_exposure_usd,
                max_daily_loss_usd=self.config.max_daily_loss_usd,
                stop_loss_pct=self.config.stop_loss_pct,
                take_profit_pct=self.config.take_profit_pct,
                min_trade_interval_sec=self.config.min_trade_interval_sec
            )
            
            self.risk_manager = LiveRiskManager(
                risk_limits=risk_limits,
                initial_balance=self.config.initial_balance
            )
            
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _log_configuration(self):
        """Логирование конфигурации бота."""
        config_info = {
            'symbol': self.config.symbol,
            'timeframe': f"{self.config.timeframe}m",
            'dry_run': self.config.dry_run,
            'testnet': self.config.bybit_testnet if not self.config.dry_run else "N/A",
            'min_confidence': self.config.min_confidence,
            'max_position_usd': self.config.max_position_size_usd,
            'max_daily_loss_usd': self.config.max_daily_loss_usd
        }
        self.logger.info(f"📋 Bot Configuration: {config_info}")
    
    def _setup_callbacks(self):
        """Настройка колбэков между компонентами."""
        
        # Колбэк для новых данных от WebSocket
        def on_kline_data(df):
            """Обработка новых свечных данных."""
            try:
                # Генерируем предсказание
                prediction_result = self.predictor.process_new_data(df)
                
                if prediction_result and prediction_result.get('is_trading_signal', False):
                    self.stats['trading_signals'] += 1
                    asyncio.create_task(self._handle_trading_signal(prediction_result, df))
                
                if prediction_result:
                    self.stats['total_predictions'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing kline data: {e}")
        
        self.ws_client.set_kline_callback(on_kline_data)
        
        # Колбэки риск-менеджера
        def on_stop_loss(symbol, position):
            """Обработка срабатывания stop-loss."""
            self.logger.warning(f"🛑 STOP LOSS triggered for {symbol}")
            asyncio.create_task(self._execute_stop_loss(symbol, position))
        
        def on_take_profit(symbol, position):
            """Обработка срабатывания take-profit."""
            self.logger.info(f"🎯 TAKE PROFIT triggered for {symbol}")
            asyncio.create_task(self._execute_take_profit(symbol, position))
        
        def on_emergency_stop(reason, positions):
            """Обработка экстренной остановки."""
            self.logger.critical(f"🚨 EMERGENCY STOP: {reason}")
            asyncio.create_task(self._handle_emergency_stop(reason, positions))
        
        self.risk_manager.set_callbacks(
            on_stop_loss=on_stop_loss,
            on_take_profit=on_take_profit,
            on_emergency_stop=on_emergency_stop
        )
    
    async def _handle_trading_signal(self, prediction_result: Dict[str, Any], market_data):
        """
        Обработка торгового сигнала.
        
        Args:
            prediction_result: Результат предсказания
            market_data: Рыночные данные
        """
        try:
            prediction = prediction_result['prediction']
            current_price = prediction_result.get('price', 0)
            
            # Определяем направление сделки
            signal = prediction['final_signal']  # 1 = BUY, 0 = SELL/HOLD
            probability = prediction['final_probability']
            
            if signal != 1 or probability < self.config.min_confidence:
                self.logger.debug(f"Signal ignored: signal={signal}, prob={probability:.3f}")
                return
            
            # Рассчитываем размер позиции (простая логика)
            position_size_usd = min(
                self.config.max_position_size_usd,
                self.risk_manager.current_balance * 0.1  # 10% от баланса
            )
            
            if current_price > 0:
                position_size = position_size_usd / current_price
            else:
                self.logger.error("Invalid current price for position sizing")
                return
            
            # Валидация через риск-менеджер
            risk_action, risk_reason = self.risk_manager.validate_new_trade(
                symbol=self.config.symbol,
                side='buy',
                size=position_size,
                price=current_price
            )
            
            if risk_action == RiskAction.BLOCK:
                self.logger.warning(f"🚫 Trade blocked by risk manager: {risk_reason}")
                return
            
            elif risk_action == RiskAction.REDUCE_SIZE:
                self.logger.info(f"⚠️ Position size reduced: {risk_reason}")
                # Извлекаем предложенный размер из сообщения
                if "suggested:" in risk_reason:
                    try:
                        suggested_size = float(risk_reason.split("suggested:")[1].strip())
                        position_size = suggested_size
                    except:
                        position_size *= 0.5  # Уменьшаем в 2 раза как fallback
            
            # Исполняем сделку
            await self._execute_trade({
                'symbol': self.config.symbol,
                'side': 'buy',
                'size': position_size,
                'price': current_price,
                'prediction': prediction_result,
                'risk_validated': True
            })
            
        except Exception as e:
            self.logger.error(f"Error handling trading signal: {e}")
    
    async def _execute_trade(self, trade_params: Dict[str, Any]):
        """
        Исполнение торговой сделки.
        
        Args:
            trade_params: Параметры сделки
        """
        try:
            symbol = trade_params['symbol']
            side = trade_params['side']
            size = trade_params['size']
            price = trade_params['price']
            
            self.logger.info(f"🎯 Executing trade: {side} {size:.4f} {symbol} @ {price:.4f}")
            
            if self.config.dry_run:
                # Симуляция сделки
                await self._simulate_trade(trade_params)
            else:
                # Реальная сделка через Bybit
                if not self.executor or not self.executor.is_connected():
                    self.logger.error("Executor not available for real trading")
                    self.stats['failed_trades'] += 1
                    return
                
                # Исполняем лимитный ордер
                result = await self.executor.execute_limit_order(
                    symbol=symbol,
                    side=side,
                    amount=size,
                    price=price * 1.001 if side == 'buy' else price * 0.999  # Небольшая коррекция цены
                )
                
                if result and result.status in ['FILLED', 'PARTIALLY_FILLED']:
                    # Добавляем позицию в риск-менеджер
                    self.risk_manager.add_position(
                        symbol=symbol,
                        side=side,
                        size=result.filled_size,
                        entry_price=result.avg_price
                    )
                    
                    self.stats['successful_trades'] += 1
                    self.logger.info(f"✅ Trade executed successfully: {result.order_id}")
                else:
                    self.stats['failed_trades'] += 1
                    self.logger.error(f"❌ Trade execution failed")
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            self.stats['failed_trades'] += 1
    
    async def _simulate_trade(self, trade_params: Dict[str, Any]):
        """Симуляция сделки в dry run режиме."""
        try:
            symbol = trade_params['symbol']
            side = trade_params['side']
            size = trade_params['size']
            price = trade_params['price']
            
            # Добавляем позицию в риск-менеджер для отслеживания
            success = self.risk_manager.add_position(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=price
            )
            
            if success:
                self.stats['successful_trades'] += 1
                self.logger.info(f"✅ [SIMULATION] Trade executed: {side} {size:.4f} {symbol} @ {price:.4f}")
            else:
                self.stats['failed_trades'] += 1
                self.logger.error(f"❌ [SIMULATION] Trade simulation failed")
            
        except Exception as e:
            self.logger.error(f"Error in trade simulation: {e}")
    
    async def _execute_stop_loss(self, symbol: str, position):
        """Исполнение stop-loss."""
        try:
            if self.config.dry_run:
                # Симуляция закрытия
                self.risk_manager.close_position(
                    symbol=symbol,
                    close_price=position.current_price,
                    reason="stop_loss_simulation"
                )
                self.logger.info(f"✅ [SIMULATION] Stop-loss executed for {symbol}")
            else:
                # Реальное закрытие позиции
                if self.executor:
                    result = await self.executor.close_position(symbol)
                    if result:
                        self.risk_manager.close_position(
                            symbol=symbol,
                            close_price=result.avg_price,
                            reason="stop_loss"
                        )
        
        except Exception as e:
            self.logger.error(f"Error executing stop-loss: {e}")
    
    async def _execute_take_profit(self, symbol: str, position):
        """Исполнение take-profit."""
        try:
            if self.config.dry_run:
                # Симуляция закрытия
                self.risk_manager.close_position(
                    symbol=symbol,
                    close_price=position.current_price,
                    reason="take_profit_simulation"
                )
                self.logger.info(f"✅ [SIMULATION] Take-profit executed for {symbol}")
            else:
                # Реальное закрытие позиции
                if self.executor:
                    result = await self.executor.close_position(symbol)
                    if result:
                        self.risk_manager.close_position(
                            symbol=symbol,
                            close_price=result.avg_price,
                            reason="take_profit"
                        )
        
        except Exception as e:
            self.logger.error(f"Error executing take-profit: {e}")
    
    async def _handle_emergency_stop(self, reason: str, positions: Dict):
        """Обработка экстренной остановки."""
        try:
            self.logger.critical(f"🚨 EMERGENCY STOP ACTIVATED: {reason}")
            
            # Закрываем все открытые позиции
            for symbol, position in positions.items():
                if self.config.dry_run:
                    self.risk_manager.close_position(
                        symbol=symbol,
                        close_price=position.current_price,
                        reason="emergency_stop_simulation"
                    )
                else:
                    if self.executor:
                        await self.executor.close_position(symbol)
            
            # Останавливаем бота
            self.logger.critical("🛑 Bot shutting down due to emergency stop")
            await self.stop()
            
        except Exception as e:
            self.logger.error(f"Error handling emergency stop: {e}")
    
    async def _monitor_positions(self):
        """Мониторинг позиций и обновление цен."""
        while self.running:
            try:
                # Получаем текущую цену
                current_price = self.ws_client.get_latest_price()
                
                if current_price and self.config.symbol in self.risk_manager.active_positions:
                    # Обновляем цены позиций
                    price_updates = {self.config.symbol: current_price}
                    self.risk_manager.update_position_prices(price_updates)
                
                # Обновляем статистику
                self._update_stats()
                
                await asyncio.sleep(10)  # Проверка каждые 10 секунд
                
            except Exception as e:
                self.logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(10)
    
    def _update_stats(self):
        """Обновление статистики бота."""
        if self.start_time:
            self.stats['uptime'] = int((datetime.now() - self.start_time).total_seconds())
        
        self.stats['current_balance'] = self.risk_manager.current_balance
        self.stats['daily_pnl'] = self.risk_manager.daily_pnl
        self.stats['active_positions'] = len(self.risk_manager.active_positions)
    
    async def _status_reporter(self):
        """Периодическое сообщение о статусе."""
        while self.running:
            try:
                # Логируем статус каждые 5 минут
                await asyncio.sleep(300)
                
                if self.running:
                    self._log_status()
                
            except Exception as e:
                self.logger.error(f"Error in status reporting: {e}")
    
    def _log_status(self):
        """Логирование текущего статуса бота."""
        risk_summary = self.risk_manager.get_risk_summary()
        predictor_stats = self.predictor.get_stats()
        
        status = {
            'uptime_minutes': self.stats['uptime'] // 60,
            'total_predictions': self.stats['total_predictions'],
            'trading_signals': self.stats['trading_signals'],
            'successful_trades': self.stats['successful_trades'],
            'failed_trades': self.stats['failed_trades'],
            'current_balance': f"{self.stats['current_balance']:.2f}",
            'daily_pnl': f"{self.stats['daily_pnl']:.2f}",
            'active_positions': len(self.risk_manager.active_positions),
            'trading_blocked': risk_summary['trading_status']['blocked']
        }
        
        self.logger.info(f"📊 Bot Status: {status}")
    
    def _setup_signal_handlers(self):
        """Настройка обработчиков системных сигналов."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self):
        """Запуск бота."""
        try:
            if self.running:
                self.logger.warning("Bot is already running")
                return
            
            self.logger.info("🚀 Starting Live Trading Bot...")
            
            # Проверяем готовность компонентов
            if not self.predictor.is_ready:
                self.logger.error("❌ Predictor not ready - models not loaded")
                return
            
            # Настройка
            self._setup_callbacks()
            self._setup_signal_handlers()
            
            # Запуск
            self.running = True
            self.start_time = datetime.now()
            
            # Запускаем все задачи
            tasks = [
                asyncio.create_task(self.ws_client.connect()),
                asyncio.create_task(self._monitor_positions()),
                asyncio.create_task(self._status_reporter())
            ]
            
            self.logger.info("✅ Live Trading Bot started successfully!")
            self.logger.info("=" * 60)
            self.logger.info("🎯 Bot is now actively trading according to ML predictions")
            self.logger.info("🛡️ Risk management is active")
            self.logger.info("📊 Use Ctrl+C to stop gracefully")
            self.logger.info("=" * 60)
            
            # Ждем завершения всех задач
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"Error starting bot: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Остановка бота."""
        if not self.running:
            return
        
        self.logger.info("🛑 Stopping Live Trading Bot...")
        
        try:
            self.running = False
            
            # Отключаем WebSocket
            if hasattr(self, 'ws_client'):
                await self.ws_client.disconnect()
            
            # Закрываем все позиции при остановке (в dry run просто логируем)
            active_positions = list(self.risk_manager.active_positions.keys())
            for symbol in active_positions:
                position = self.risk_manager.active_positions[symbol]
                if self.config.dry_run:
                    self.risk_manager.close_position(
                        symbol=symbol,
                        close_price=position.current_price,
                        reason="bot_shutdown_simulation"
                    )
                else:
                    if hasattr(self, 'executor') and self.executor:
                        await self.executor.close_position(symbol)
            
            # Финальная статистика
            self._log_final_stats()
            
            self.logger.info("✅ Bot stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}")
    
    def _log_final_stats(self):
        """Логирование финальной статистики."""
        total_time = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        final_stats = {
            'session_duration_minutes': int(total_time // 60),
            'total_predictions': self.stats['total_predictions'],
            'trading_signals_generated': self.stats['trading_signals'],
            'successful_trades': self.stats['successful_trades'],
            'failed_trades': self.stats['failed_trades'],
            'final_balance': f"{self.risk_manager.current_balance:.2f}",
            'session_pnl': f"{self.risk_manager.current_balance - self.config.initial_balance:.2f}",
            'success_rate': f"{(self.stats['successful_trades'] / max(1, self.stats['successful_trades'] + self.stats['failed_trades'])) * 100:.1f}%"
        }
        
        self.logger.info("=" * 60)
        self.logger.info("📊 FINAL SESSION STATS")
        self.logger.info("=" * 60)
        for key, value in final_stats.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 60)


# Основная функция для запуска
async def main():
    """Основная функция для запуска бота."""
    
    # Конфигурация бота (можно загружать из файла)
    config = BotConfig(
        symbol="SOL/USDT:USDT",
        timeframe="5",
        
        # API ключи (заполните реальными для торговли)
        bybit_api_key="",  # Ваш API ключ
        bybit_api_secret="",  # Ваш API секрет
        bybit_testnet=True,
        
        # Модели
        models_dir="models/ensemble_live",
        min_confidence=0.65,
        
        # Риски
        max_position_size_usd=50.0,  # Начинаем с малых сумм
        max_total_exposure_usd=200.0,
        max_daily_loss_usd=25.0,
        
        # Режим
        dry_run=True,  # Установите False для реальной торговли
        initial_balance=1000.0,
        log_level="INFO"
    )
    
    # Создаем и запускаем бота
    bot = LiveTradingBot(config)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\n⚠️ Keyboard interrupt received")
    except Exception as e:
        print(f"❌ Bot error: {e}")
    finally:
        await bot.stop()


if __name__ == "__main__":
    # Запуск бота
    print("🤖 Starting Live Crypto Trading Bot...")
    print("🔄 Press Ctrl+C to stop gracefully")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")