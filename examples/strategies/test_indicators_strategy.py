from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import MACD, RSI, EMA
from typing import Optional


class TestIndicatorsStrategy(StrategyBase):
    """
    Test strategy to debug indicators initialization.
    
    This strategy only tests if indicators are working correctly,
    without any trading logic.
    """
    
    def _initialize(self):
        """Initialize indicators for testing."""
        self.parameters = {
            'test_only': True
        }
        
        # Initialize indicators
        self.indicators = {
            'macd': MACD(fast_period=12, slow_period=26, signal_period=9),
            'rsi': RSI(period=14),
            'ema50': EMA(period=50)
        }
        
        # Strategy state
        self.state = {
            'bar_count': 0,
            'macd_ready_bar': None,
            'rsi_ready_bar': None,
            'ema_ready_bar': None
        }

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Test indicator updates."""
        self.state['bar_count'] += 1
        
        # Update all indicators
        macd_result = self.indicators['macd'].update(data.close)
        rsi_result = self.indicators['rsi'].update(data.close)
        ema_result = self.indicators['ema50'].update(data.close)
        
        # Check readiness status
        macd_ready = self.indicators['macd'].is_ready()
        rsi_ready = self.indicators['rsi'].is_ready()
        ema_ready = self.indicators['ema50'].is_ready()
        
        # Track when each becomes ready
        if macd_ready and self.state['macd_ready_bar'] is None:
            self.state['macd_ready_bar'] = self.state['bar_count']
            print(f"✅ MACD became ready at bar {self.state['bar_count']}")
            print(f"   MACD values: {macd_result}")
        
        if rsi_ready and self.state['rsi_ready_bar'] is None:
            self.state['rsi_ready_bar'] = self.state['bar_count']
            print(f"✅ RSI became ready at bar {self.state['bar_count']}")
            print(f"   RSI value: {rsi_result}")
        
        if ema_ready and self.state['ema_ready_bar'] is None:
            self.state['ema_ready_bar'] = self.state['bar_count']
            print(f"✅ EMA50 became ready at bar {self.state['bar_count']}")
            print(f"   EMA value: {ema_result}")
        
        # Debug output every 50 bars
        if self.state['bar_count'] % 50 == 0:
            print(f"\n📊 Bar {self.state['bar_count']} Status:")
            print(f"   MACD: ready={macd_ready}, result={macd_result}")
            print(f"   RSI:  ready={rsi_ready}, result={rsi_result}")
            print(f"   EMA:  ready={ema_ready}, result={ema_result}")
            
            # Detailed debug for MACD
            print(f"\n🔍 MACD Debug:")
            print(f"   Fast EMA ready: {self.indicators['macd'].fast_ema.is_ready()}")
            print(f"   Slow EMA ready: {self.indicators['macd'].slow_ema.is_ready()}")
            print(f"   Signal EMA ready: {self.indicators['macd'].signal_ema.is_ready()}")
            print(f"   MACD._initialized: {self.indicators['macd']._initialized}")
            print(f"   MACD values length: {len(self.indicators['macd'].values)}")
            print(f"   MACD line length: {len(self.indicators['macd'].macd_line)}")
            
            # Detailed debug for RSI  
            print(f"\n🔍 RSI Debug:")
            print(f"   RSI._initialized: {self.indicators['rsi']._initialized}")
            print(f"   RSI values length: {len(self.indicators['rsi'].values)}")
            print(f"   RSI gains length: {len(self.indicators['rsi'].gains)}")
            print(f"   RSI losses length: {len(self.indicators['rsi'].losses)}")
            
            # Detailed debug for EMA
            print(f"\n🔍 EMA Debug:")
            print(f"   EMA._initialized: {self.indicators['ema50']._initialized}")
            print(f"   EMA values length: {len(self.indicators['ema50'].values)}")
            print(f"   EMA current value: {self.indicators['ema50'].ema_value}")
        
        # Stop after some bars to avoid infinite output
        if self.state['bar_count'] > 300:
            all_ready = macd_ready and rsi_ready and ema_ready
            print(f"\n🏁 Test completed at bar {self.state['bar_count']}")
            print(f"   All indicators ready: {all_ready}")
            if all_ready:
                print(f"   MACD ready at bar: {self.state['macd_ready_bar']}")
                print(f"   RSI ready at bar: {self.state['rsi_ready_bar']}")  
                print(f"   EMA ready at bar: {self.state['ema_ready_bar']}")
            return TradeSignal(signal=Signal.HOLD)
        
        return TradeSignal(signal=Signal.HOLD)
    
    def get_position_size(self, data: MarketData, signal: Signal, capital: float) -> float:
        """No position sizing needed for test."""
        return 0.0