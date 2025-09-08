"""
Простейшая тестовая стратегия - покупать каждые 100 свечей
"""

from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal


class SimpleTestStrategy(StrategyBase):
    """Простейшая стратегия для тестирования бэктестера"""
    
    def _initialize(self):
        self.parameters = {
            'position_size_pct': 0.1
        }
        self.state = {
            'bar_count': 0,
            'in_position': False
        }

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        self.state['bar_count'] += 1
        
        # Покупаем каждые 100 свечей если нет позиции
        if self.state['bar_count'] % 100 == 0 and position.quantity == 0:
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"Test buy at bar {self.state['bar_count']}"
            )
        
        # Продаём через 50 свечей после покупки
        if position.quantity > 0 and self.state['bar_count'] % 50 == 0:
            return TradeSignal(
                signal=Signal.SELL,
                reason=f"Test sell at bar {self.state['bar_count']}"
            )
        
        return TradeSignal(signal=Signal.HOLD)