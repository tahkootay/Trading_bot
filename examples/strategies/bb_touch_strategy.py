from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import BollingerBands
from typing import Optional


class BBTouchStrategy(StrategyBase):
    """
    Bollinger Bands Touch Strategy - только лонги с улучшенным R/R.
    
    Правила:
    - LONG: При касании нижней BB → TP=3 USDT, SL=6 USDT (R/R 1:2)
    - SHORT: ОТКЛЮЧЕНЫ
    """
    
    def _initialize(self):
        """Initialize strategy parameters and state."""
        self.parameters = {
            'bb_period': 20,           # Период Bollinger Bands
            'bb_deviation': 2.0,       # Стандартное отклонение
            'take_profit_usdt': 3.0,   # Тейк-профит: 3 USDT 
            'stop_loss_usdt': 6.0,     # Стоп-лосс: 6 USDT (R/R 1:2)
            'position_size_pct': 0.05, # 5% от капитала
        }
        
        # Initialize indicators
        self.indicators = {
            'bb': BollingerBands(
                period=self.parameters['bb_period'], 
                std_dev=self.parameters['bb_deviation']
            )
        }
        
        # Strategy state
        self.state = {
            'bb_values': None,
            'entry_price': 0.0,
            'trades_count': 0,
            'bar_count': 0
        }

    def update_indicators(self, data: MarketData) -> None:
        """Update technical indicators with new market data."""
        self.state['bb_values'] = self.indicators['bb'].update(data.close)

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Process new market data and generate trading signal."""
        self.state['bar_count'] += 1
        
        # Update indicators first
        self.update_indicators(data)
        
        # Check if indicators are ready
        if not self._indicators_ready():
            if self.state['bar_count'] <= 25:
                print(f"Bar {self.state['bar_count']}: Ожидаем готовности BB (нужно {self.parameters['bb_period']} баров)")
            return TradeSignal(signal=Signal.HOLD)
        
        # Position management for open positions
        if position.direction != "NONE":
            return self._manage_position(data, position)
        
        # Entry signal generation
        return self._generate_entry_signal(data)
    
    def _indicators_ready(self) -> bool:
        """Check if all indicators are ready."""
        return (
            self.indicators['bb'].is_ready() and 
            self.state['bb_values'] is not None
        )
    
    def _generate_entry_signal(self, data: MarketData) -> TradeSignal:
        """Generate entry signals based on BB touches."""
        
        current_price = data.close
        bb = self.state['bb_values']
        
        # LONG: Касание нижней линии BB
        if current_price <= bb['lower']:
            self.state['trades_count'] += 1
            self.state['entry_price'] = current_price
            
            print(f"🟢 LONG сигнал на баре {self.state['bar_count']}: Касание нижней BB")
            print(f"   Цена: ${current_price:.2f}, BB Lower: ${bb['lower']:.2f}")
            print(f"   TP: ${current_price + self.parameters['take_profit_usdt']:.2f} (+3 USDT)")
            print(f"   SL: ${current_price - self.parameters['stop_loss_usdt']:.2f} (-6 USDT)")
            
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"BB Touch LONG: Цена {current_price:.2f} <= BB Lower {bb['lower']:.2f}",
                stop_loss=current_price - self.parameters['stop_loss_usdt'],
                take_profit=current_price + self.parameters['take_profit_usdt']
            )
        
        # SHORT: ОТКЛЮЧЕНЫ в данной версии стратегии
        
        return TradeSignal(signal=Signal.HOLD)
    
    def _manage_position(self, data: MarketData, position: Position) -> TradeSignal:
        """Управление открытыми позициями."""
        
        current_price = data.close
        entry_price = self.state['entry_price']
        
        if position.direction == "LONG":
            # Take Profit: рост на 3 USDT
            if current_price >= entry_price + self.parameters['take_profit_usdt']:
                profit = current_price - entry_price
                print(f"🎯 Закрываем LONG с прибылью: +${profit:.2f} USDT")
                return TradeSignal(
                    signal=Signal.CLOSE_LONG,
                    reason=f"TP LONG: Рост на {profit:.2f} USDT (цель: +3 USDT)"
                )
            
            # Stop Loss: падение на 6 USDT
            if current_price <= entry_price - self.parameters['stop_loss_usdt']:
                loss = entry_price - current_price
                print(f"🛑 Закрываем LONG со стоп-лоссом: -${loss:.2f} USDT")
                return TradeSignal(
                    signal=Signal.CLOSE_LONG,
                    reason=f"SL LONG: Падение на {loss:.2f} USDT (лимит: -6 USDT)"
                )
        
        # SHORT позиции не используются в данной стратегии
        
        return TradeSignal(signal=Signal.HOLD)
    
    def get_position_size(self, data: MarketData, signal: Signal, capital: float) -> float:
        """Calculate position size for trade."""
        return capital * self.parameters['position_size_pct']