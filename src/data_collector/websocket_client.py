#!/usr/bin/env python3
"""
Live WebSocket клиент для получения данных с Bybit в реальном времени
Согласно спецификации: фьючерсы USDT Perpetual, таймфреймы 5M-1H
"""

import asyncio
import json
import logging
import pandas as pd
import websockets
from datetime import datetime, timezone
from typing import Callable, Dict, Any, Optional
from collections import deque
import numpy as np


class BybitWebSocketClient:
    """WebSocket клиент для получения live данных с Bybit фьючерсов."""
    
    def __init__(self, symbol: str = "SOLUSDT", timeframe: str = "5"):
        """
        Инициализация клиента.
        
        Args:
            symbol: Торговая пара (например, "SOLUSDT", "BTCUSDT")
            timeframe: Таймфрейм в минутах ("1", "5", "15", "30", "60")
        """
        self.symbol = symbol
        self.timeframe = timeframe
        
        # URL для Bybit WebSocket (фьючерсы)
        self.ws_url = "wss://stream.bybit.com/v5/public/linear"
        
        # Буфер для хранения свечных данных
        self.data_buffer = deque(maxlen=100)  # Скользящее окно последних 100 свечей
        
        # Колбэки для обработки данных
        self.on_kline_callback: Optional[Callable] = None
        
        # Статус подключения
        self.connected = False
        self.websocket = None
        
        # Логирование
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Параметры переподключения
        self.reconnect_attempts = 5
        self.reconnect_delay = 5  # секунды
        
    def set_kline_callback(self, callback: Callable[[pd.DataFrame], None]):
        """Установка колбэка для обработки новых свечей."""
        self.on_kline_callback = callback
    
    def _format_kline_data(self, kline_data: dict) -> dict:
        """Форматирование данных свечи в стандартный формат."""
        return {
            'timestamp': pd.to_datetime(int(kline_data['start']), unit='ms'),
            'open': float(kline_data['open']),
            'high': float(kline_data['high']),
            'low': float(kline_data['low']),
            'close': float(kline_data['close']),
            'volume': float(kline_data['volume']),
            'confirm': kline_data['confirm']  # True если свеча закрыта
        }
    
    def _add_to_buffer(self, kline_dict: dict):
        """Добавление свечи в буфер данных."""
        # Обновляем или добавляем свечу
        timestamp = kline_dict['timestamp']
        
        # Если это обновление последней свечи (не закрыта)
        if not kline_dict['confirm'] and len(self.data_buffer) > 0:
            # Проверяем, нужно ли обновить последнюю свечу
            if self.data_buffer[-1]['timestamp'] == timestamp:
                self.data_buffer[-1] = kline_dict
            else:
                self.data_buffer.append(kline_dict)
        else:
            # Добавляем новую закрытую свечу
            self.data_buffer.append(kline_dict)
        
        # Создаем DataFrame для передачи в колбэк
        if len(self.data_buffer) >= 20:  # Минимум данных для индикаторов
            df = pd.DataFrame(list(self.data_buffer))
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Вызываем колбэк если установлен
            if self.on_kline_callback:
                try:
                    self.on_kline_callback(df)
                except Exception as e:
                    self.logger.error(f"Error in kline callback: {e}")
    
    async def _handle_message(self, message: str):
        """Обработка входящих WebSocket сообщений."""
        try:
            data = json.loads(message)
            
            # Обработка kline данных
            if data.get("topic", "").startswith("kline"):
                kline_data = data.get("data", [])
                if kline_data:
                    for kline in kline_data:
                        formatted_kline = self._format_kline_data(kline)
                        self._add_to_buffer(formatted_kline)
            
            # Обработка системных сообщений
            elif "success" in data:
                if data["success"]:
                    self.logger.info(f"Successfully subscribed to {data.get('ret_msg', 'topic')}")
                else:
                    self.logger.error(f"Subscription failed: {data.get('ret_msg', 'Unknown error')}")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse WebSocket message: {e}")
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}")
    
    async def _subscribe_to_kline(self):
        """Подписка на получение kline данных."""
        subscribe_message = {
            "op": "subscribe",
            "args": [f"kline.{self.timeframe}.{self.symbol}"]
        }
        
        await self.websocket.send(json.dumps(subscribe_message))
        self.logger.info(f"Subscribed to kline.{self.timeframe}.{self.symbol}")
    
    async def _ping_pong(self):
        """Отправка ping сообщений для поддержания соединения."""
        while self.connected:
            try:
                await asyncio.sleep(20)  # Отправляем ping каждые 20 секунд
                if self.websocket and not self.websocket.closed:
                    ping_message = {"op": "ping"}
                    await self.websocket.send(json.dumps(ping_message))
            except Exception as e:
                self.logger.error(f"Ping failed: {e}")
                break
    
    async def connect(self):
        """Основное подключение к WebSocket."""
        attempt = 0
        while attempt < self.reconnect_attempts:
            try:
                self.logger.info(f"Connecting to Bybit WebSocket... (attempt {attempt + 1})")
                
                self.websocket = await websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                )
                
                self.connected = True
                self.logger.info("Connected to Bybit WebSocket successfully!")
                
                # Подписываемся на данные
                await self._subscribe_to_kline()
                
                # Запускаем ping-pong в фоне
                ping_task = asyncio.create_task(self._ping_pong())
                
                # Основной цикл получения сообщений
                async for message in self.websocket:
                    await self._handle_message(message)
                
            except websockets.exceptions.ConnectionClosed as e:
                self.logger.warning(f"WebSocket connection closed: {e}")
                self.connected = False
                
            except Exception as e:
                self.logger.error(f"WebSocket connection error: {e}")
                self.connected = False
                
            # Переподключение
            attempt += 1
            if attempt < self.reconnect_attempts:
                self.logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)
            else:
                self.logger.error("Max reconnection attempts reached. Stopping.")
                break
        
        self.connected = False
    
    async def disconnect(self):
        """Отключение от WebSocket."""
        self.connected = False
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
            self.logger.info("Disconnected from WebSocket")
    
    def get_current_data(self) -> Optional[pd.DataFrame]:
        """Получение текущих данных из буфера."""
        if len(self.data_buffer) == 0:
            return None
            
        df = pd.DataFrame(list(self.data_buffer))
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def get_latest_price(self) -> Optional[float]:
        """Получение последней цены."""
        if len(self.data_buffer) == 0:
            return None
        return self.data_buffer[-1]['close']


class LiveDataManager:
    """Менеджер для управления live данными от нескольких источников."""
    
    def __init__(self):
        self.clients: Dict[str, BybitWebSocketClient] = {}
        self.data_callbacks: Dict[str, Callable] = {}
        
    def add_symbol(self, symbol: str, timeframe: str = "5", 
                   callback: Optional[Callable] = None) -> BybitWebSocketClient:
        """Добавление символа для отслеживания."""
        key = f"{symbol}_{timeframe}"
        
        if key in self.clients:
            return self.clients[key]
        
        client = BybitWebSocketClient(symbol=symbol, timeframe=timeframe)
        
        if callback:
            client.set_kline_callback(callback)
        
        self.clients[key] = client
        return client
    
    async def start_all(self):
        """Запуск всех WebSocket клиентов."""
        tasks = []
        for key, client in self.clients.items():
            task = asyncio.create_task(client.connect())
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_all(self):
        """Остановка всех WebSocket клиентов."""
        tasks = []
        for client in self.clients.values():
            task = asyncio.create_task(client.disconnect())
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_client(self, symbol: str, timeframe: str = "5") -> Optional[BybitWebSocketClient]:
        """Получение клиента по символу и таймфрейму."""
        key = f"{symbol}_{timeframe}"
        return self.clients.get(key)


# Пример использования
async def example_usage():
    """Пример использования WebSocket клиента."""
    
    def on_new_kline(df: pd.DataFrame):
        """Колбэк для обработки новых данных."""
        latest = df.iloc[-1]
        print(f"New kline: {latest['timestamp']} | "
              f"O: {latest['open']:.4f} | "
              f"H: {latest['high']:.4f} | "
              f"L: {latest['low']:.4f} | "
              f"C: {latest['close']:.4f} | "
              f"V: {latest['volume']:.2f} | "
              f"Confirm: {latest['confirm']}")
    
    # Создание клиента
    client = BybitWebSocketClient(symbol="SOLUSDT", timeframe="5")
    client.set_kline_callback(on_new_kline)
    
    try:
        # Подключение и получение данных
        await client.connect()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Запуск примера
    asyncio.run(example_usage())