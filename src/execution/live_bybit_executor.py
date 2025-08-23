#!/usr/bin/env python3
"""
Live Bybit API исполнитель для торгового бота
Согласно спецификации: торговля на фьючерсном рынке Bybit (USDT Perpetual)
"""

import ccxt
import asyncio
import time
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
import json

from .order_executor import OrderExecutor, OrderRequest, OrderResult, OrderStatus, OrderType, TimeInForce


class BybitLiveExecutor:
    """
    Исполнитель ордеров для Bybit фьючерсов в реальном времени.
    Интегрируется с существующим OrderExecutor.
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        Инициализация Bybit исполнителя.
        
        Args:
            api_key: API ключ Bybit
            api_secret: API секрет Bybit  
            testnet: Использовать testnet (True) или mainnet (False)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Инициализация CCXT клиента
        self.exchange = self._init_exchange()
        
        # Основной исполнитель ордеров
        self.order_executor = OrderExecutor(self, self._get_executor_config())
        
        # Состояние
        self.connected = False
        self.account_info = {}
        self.positions = {}
        self.symbol_info = {}
        
        # Логирование
        self.logger = logging.getLogger(__name__)
        
        # Автоподключение
        asyncio.create_task(self._initialize())
    
    def _init_exchange(self) -> ccxt.bybit:
        """Инициализация CCXT клиента для Bybit."""
        try:
            exchange = ccxt.bybit({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'sandbox': self.testnet,  # testnet
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'linear',  # Фьючерсы USDT Perpetual
                    'unified': True
                }
            })
            
            return exchange
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Bybit exchange: {e}")
            raise
    
    def _get_executor_config(self) -> Dict[str, Any]:
        """Получение конфигурации для OrderExecutor."""
        return {
            'entry_offset_pct': 0.0005,  # 0.05%
            'fill_timeout_sec': 30,
            'partial_fill_min': 0.5,
            'iceberg_show_pct': 0.2,
            'retry_attempts': 3,
            'retry_delay_sec': 5,
            'max_slippage_pct': 0.001  # 0.1%
        }
    
    async def _initialize(self):
        """Инициализация соединения и загрузка информации."""
        try:
            self.logger.info("🔌 Connecting to Bybit...")
            
            # Загружаем рынки
            await self._load_markets()
            
            # Загружаем информацию об аккаунте
            await self._load_account_info()
            
            # Загружаем позиции
            await self._load_positions()
            
            self.connected = True
            self.logger.info("✅ Connected to Bybit successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Bybit connection: {e}")
            self.connected = False
    
    async def _load_markets(self):
        """Загрузка информации о рынках."""
        try:
            markets = await self.exchange.load_markets()
            
            # Сохраняем информацию о символах
            for symbol, market in markets.items():
                if market.get('linear') and market.get('active'):  # Фьючерсы USDT
                    self.symbol_info[symbol] = {
                        'min_amount': market.get('limits', {}).get('amount', {}).get('min', 0),
                        'max_amount': market.get('limits', {}).get('amount', {}).get('max', 0),
                        'amount_precision': market.get('precision', {}).get('amount', 8),
                        'price_precision': market.get('precision', {}).get('price', 8),
                        'tick_size': market.get('info', {}).get('tickSize', '0.01'),
                        'contract_size': market.get('contractSize', 1)
                    }
            
            self.logger.info(f"📊 Loaded {len(self.symbol_info)} symbols")
            
        except Exception as e:
            self.logger.error(f"Failed to load markets: {e}")
            raise
    
    async def _load_account_info(self):
        """Загрузка информации об аккаунте."""
        try:
            balance = await self.exchange.fetch_balance()
            
            self.account_info = {
                'total_balance': balance.get('total', {}),
                'free_balance': balance.get('free', {}),
                'used_balance': balance.get('used', {}),
                'usdt_balance': balance.get('USDT', {}).get('total', 0),
                'usdt_free': balance.get('USDT', {}).get('free', 0)
            }
            
            self.logger.info(f"💰 Account loaded: {self.account_info['usdt_balance']:.2f} USDT")
            
        except Exception as e:
            self.logger.error(f"Failed to load account info: {e}")
            raise
    
    async def _load_positions(self):
        """Загрузка текущих позиций."""
        try:
            positions = await self.exchange.fetch_positions()
            
            self.positions = {}
            for position in positions:
                if position['size'] != 0:  # Только открытые позиции
                    symbol = position['symbol']
                    self.positions[symbol] = {
                        'symbol': symbol,
                        'side': position['side'],
                        'size': position['size'],
                        'entry_price': position['entryPrice'],
                        'mark_price': position['markPrice'],
                        'unrealized_pnl': position['unrealizedPnl'],
                        'percentage': position['percentage']
                    }
            
            self.logger.info(f"📈 Loaded {len(self.positions)} open positions")
            
        except Exception as e:
            self.logger.error(f"Failed to load positions: {e}")
    
    async def execute_market_order(self, symbol: str, side: str, amount: float, 
                                  reduce_only: bool = False) -> Optional[OrderResult]:
        """
        Исполнение рыночного ордера.
        
        Args:
            symbol: Торговая пара (например, "SOL/USDT:USDT")
            side: Направление ('buy' или 'sell')
            amount: Размер позиции
            reduce_only: Только закрытие позиции
            
        Returns:
            OrderResult или None
        """
        try:
            # Создаем запрос на ордер
            setup = {
                'symbol': symbol,
                'direction': 'long' if side == 'buy' else 'short',
                'position_size': amount,
                'entry': None,  # Рыночная цена
                'confidence': 1.0  # Максимальная уверенность для рыночного ордера
            }
            
            if reduce_only:
                # Для закрытия позиции используем execute_exit
                position = self.positions.get(symbol, {
                    'symbol': symbol,
                    'direction': 'long' if side == 'sell' else 'short',  # Обратное направление
                    'size': amount
                })
                return await self.order_executor.execute_exit(position, reason="market_close")
            else:
                return await self.order_executor.execute_entry(setup)
                
        except Exception as e:
            self.logger.error(f"Error executing market order: {e}")
            return None
    
    async def execute_limit_order(self, symbol: str, side: str, amount: float, 
                                 price: float, time_in_force: str = "GTC") -> Optional[OrderResult]:
        """
        Исполнение лимитного ордера.
        
        Args:
            symbol: Торговая пара
            side: Направление ('buy' или 'sell')
            amount: Размер позиции
            price: Цена ордера
            time_in_force: Время действия ордера
            
        Returns:
            OrderResult или None
        """
        try:
            setup = {
                'symbol': symbol,
                'direction': 'long' if side == 'buy' else 'short',
                'position_size': amount,
                'entry': price,
                'confidence': 0.7  # Средняя уверенность для лимитного ордера
            }
            
            return await self.order_executor.execute_entry(setup)
            
        except Exception as e:
            self.logger.error(f"Error executing limit order: {e}")
            return None
    
    async def close_position(self, symbol: str, size: Optional[float] = None) -> Optional[OrderResult]:
        """
        Закрытие позиции по рынку.
        
        Args:
            symbol: Символ для закрытия
            size: Размер для закрытия (None = вся позиция)
            
        Returns:
            OrderResult или None
        """
        try:
            position = self.positions.get(symbol)
            if not position:
                self.logger.warning(f"No position found for {symbol}")
                return None
            
            close_size = size or abs(position['size'])
            
            if size and size < abs(position['size']):
                # Частичное закрытие
                return await self.order_executor.execute_partial_close(
                    position, close_size, reason="partial_close"
                )
            else:
                # Полное закрытие
                return await self.order_executor.execute_exit(position, reason="full_close")
                
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return None
    
    def format_amount(self, symbol: str, amount: float) -> float:
        """Форматирование размера ордера согласно требованиям символа."""
        try:
            symbol_data = self.symbol_info.get(symbol, {})
            precision = symbol_data.get('amount_precision', 8)
            
            # Округляем до нужной точности
            formatted = float(Decimal(str(amount)).quantize(
                Decimal('0.1') ** precision, 
                rounding=ROUND_DOWN
            ))
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Error formatting amount: {e}")
            return amount
    
    def format_price(self, symbol: str, price: float) -> float:
        """Форматирование цены согласно требованиям символа."""
        try:
            symbol_data = self.symbol_info.get(symbol, {})
            precision = symbol_data.get('price_precision', 8)
            
            # Округляем до нужной точности
            formatted = float(Decimal(str(price)).quantize(
                Decimal('0.1') ** precision,
                rounding=ROUND_DOWN
            ))
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Error formatting price: {e}")
            return price
    
    # Методы для интеграции с OrderExecutor
    
    async def place_order(self, **params) -> Optional[Dict[str, Any]]:
        """Размещение ордера через Bybit API."""
        try:
            symbol = params['symbol']
            side = params['side']
            order_type = params['type']
            amount = self.format_amount(symbol, params['quantity'])
            
            order_params = {
                'symbol': symbol,
                'type': 'market' if order_type == 'market' else 'limit',
                'side': side,
                'amount': amount,
            }
            
            if 'price' in params:
                order_params['price'] = self.format_price(symbol, params['price'])
            
            if 'timeInForce' in params:
                order_params['timeInForce'] = params['timeInForce']
            
            # Размещаем ордер
            result = await self.exchange.create_order(**order_params)
            
            return {
                'order_id': result['id'],
                'symbol': result['symbol'],
                'status': result['status'],
                'type': result['type'],
                'side': result['side'],
                'amount': result['amount'],
                'filled': result['filled'],
                'remaining': result['remaining'],
                'price': result.get('price', 0),
                'timestamp': result['timestamp']
            }
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Получение статуса ордера."""
        try:
            order = await self.exchange.fetch_order(order_id)
            
            return {
                'id': order['id'],
                'status': order['status'].upper(),  # FILLED, OPEN, CANCELED, etc.
                'symbol': order['symbol'],
                'type': order['type'],
                'side': order['side'],
                'amount': order['amount'],
                'filled': order['filled'],
                'remaining': order['remaining'],
                'executedQty': order['filled'],
                'price': order.get('price', 0),
                'avgPrice': order.get('average', 0),
                'commission': order.get('fee', {}).get('cost', 0),
                'timestamp': order['timestamp']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return None
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None):
        """Отмена ордера."""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Order {order_id} cancelled successfully")
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
    
    def get_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Получение стакана ордеров."""
        try:
            # Используем синхронный вызов для простоты
            orderbook = self.exchange.fetch_order_book(symbol)
            
            return {
                'symbol': symbol,
                'bids': orderbook['bids'],
                'asks': orderbook['asks'],
                'timestamp': orderbook['timestamp']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting orderbook for {symbol}: {e}")
            return None
    
    async def refresh_positions(self):
        """Обновление позиций."""
        await self._load_positions()
    
    async def refresh_account(self):
        """Обновление информации об аккаунте."""
        await self._load_account_info()
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Получение позиции по символу."""
        return self.positions.get(symbol)
    
    def get_account_balance(self) -> Dict[str, Any]:
        """Получение баланса аккаунта."""
        return self.account_info
    
    def is_connected(self) -> bool:
        """Проверка соединения."""
        return self.connected
    
    def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Получение торговых комиссий."""
        try:
            market = self.exchange.market(symbol)
            return {
                'maker': market.get('maker', 0.0002),  # 0.02% по умолчанию
                'taker': market.get('taker', 0.0005)   # 0.05% по умолчанию
            }
        except Exception:
            return {'maker': 0.0002, 'taker': 0.0005}
    
    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Получение тикера."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'last': ticker['last'],
                'volume': ticker['baseVolume'],
                'timestamp': ticker['timestamp']
            }
        except Exception as e:
            self.logger.error(f"Error getting ticker for {symbol}: {e}")
            return None


# Пример использования
async def example_usage():
    """Демонстрация использования Bybit исполнителя."""
    
    # ВНИМАНИЕ: Используйте реальные API ключи
    executor = BybitLiveExecutor(
        api_key="your_api_key",
        api_secret="your_api_secret",
        testnet=True  # Используем testnet для тестов
    )
    
    # Ждем подключения
    while not executor.is_connected():
        print("Waiting for connection...")
        await asyncio.sleep(1)
    
    print("✅ Connected!")
    
    # Получаем баланс
    balance = executor.get_account_balance()
    print(f"💰 Balance: {balance.get('usdt_balance', 0):.2f} USDT")
    
    # Получаем позиции
    await executor.refresh_positions()
    print(f"📈 Open positions: {len(executor.positions)}")
    
    # Пример ордера (не выполняется на реальном рынке без средств)
    if False:  # Установите True для тестирования ордеров
        result = await executor.execute_limit_order(
            symbol="SOL/USDT:USDT",
            side="buy", 
            amount=0.1,
            price=100.0
        )
        
        if result:
            print(f"🎯 Order result: {executor.order_executor.ensemble_predictor.get_prediction_summary(result)}")


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Запуск примера (требуются API ключи)
    # asyncio.run(example_usage())