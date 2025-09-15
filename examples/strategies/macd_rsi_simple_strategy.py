from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import MACD, RSI, EMA
from typing import Optional


class MacdRsiSimpleStrategy(StrategyBase):
    """
    Simplified MACD + RSI Strategy with shorter periods.
    
    Uses smaller periods to work with limited data:
    - EMA50 instead of EMA200 for trend filter
    - Standard MACD (12/26/9) and RSI (14)
    - Extensive debug output to understand market conditions
    """
    
    def _initialize(self):
        """Initialize strategy parameters and state."""
        self.parameters = {
            'macd_fast': 12,           # MACD fast period
            'macd_slow': 26,           # MACD slow period  
            'macd_signal': 9,          # MACD signal period
            'rsi_period': 14,          # RSI period
            'ema_period': 50,          # EMA trend filter (reduced from 200)
            'rsi_overbought': 70,      # RSI overbought level
            'rsi_oversold': 30,        # RSI oversold level
            'take_profit_pct': 0.8,    # Take profit: 0.8%
            'stop_loss_pct': 0.6,      # Stop loss: 0.6%
            'position_size_pct': 0.05, # 5% of capital
        }
        
        # Initialize indicators
        self.indicators = {
            'macd': MACD(
                fast_period=self.parameters['macd_fast'],
                slow_period=self.parameters['macd_slow'],
                signal_period=self.parameters['macd_signal']
            ),
            'rsi': RSI(period=self.parameters['rsi_period']),
            'ema50': EMA(period=self.parameters['ema_period'])
        }
        
        # Strategy state
        self.state = {
            'macd_values': None,
            'rsi_value': None,
            'ema50_value': None,
            'prev_macd_values': None,
            'entry_price': 0.0,
            'trades_count': 0,
            'bar_count': 0,
            'debug_counter': 0,
            'first_debug': True
        }

    def update_indicators(self, data: MarketData) -> None:
        """Update technical indicators with new market data."""
        # Store previous MACD for crossover detection
        self.state['prev_macd_values'] = self.state['macd_values']
        
        # Update all indicators
        self.state['macd_values'] = self.indicators['macd'].update(data.close)
        self.state['rsi_value'] = self.indicators['rsi'].update(data.close)
        self.state['ema50_value'] = self.indicators['ema50'].update(data.close)

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Process new market data and generate trading signal."""
        self.state['bar_count'] += 1
        
        # Update indicators first
        self.update_indicators(data)
        
        # Check if indicators are ready
        if not self._indicators_ready():
            if self.state['bar_count'] % 10 == 0:
                print(f"Bar {self.state['bar_count']}: Waiting for indicators (need ~{self.parameters['ema_period']} bars)")
            return TradeSignal(signal=Signal.HOLD)
        
        # Debug output immediately when indicators become ready, then every 200 bars
        if self.state['first_debug']:
            print(f"✅ Indicators ready at bar {self.state['bar_count']}! Starting analysis...")
            self.state['first_debug'] = False
            self._debug_conditions(data)
        
        self.state['debug_counter'] += 1
        if self.state['debug_counter'] % 200 == 0:
            self._debug_conditions(data)
        
        # Position management for open positions
        if position.direction != "NONE":
            return self._manage_position(data, position)
        
        # Entry signal generation
        return self._generate_entry_signal(data)
    
    def _debug_conditions(self, data: MarketData) -> None:
        """Debug output to understand why no signals are generated."""
        current_price = data.close
        macd = self.state['macd_values']
        prev_macd = self.state['prev_macd_values']
        rsi = self.state['rsi_value']
        ema50 = self.state['ema50_value']
        
        print(f"\n🔍 DEBUG Bar {self.state['bar_count']}: Market Analysis")
        print(f"   Price: ${current_price:.2f}, EMA50: ${ema50:.2f}")
        print(f"   Price vs EMA50: {'Above' if current_price > ema50 else 'Below'} ({((current_price/ema50-1)*100):+.2f}%)")
        print(f"   MACD Line: {macd['macd']:.6f}")
        print(f"   MACD Signal: {macd['signal']:.6f}")
        print(f"   MACD Diff: {(macd['macd'] - macd['signal']):.6f}")
        print(f"   Prev MACD Line: {prev_macd['macd']:.6f}")
        print(f"   Prev MACD Signal: {prev_macd['signal']:.6f}")
        print(f"   Prev MACD Diff: {(prev_macd['macd'] - prev_macd['signal']):.6f}")
        print(f"   RSI: {rsi:.2f}")
        
        # Check MACD crossover conditions
        macd_bullish_cross = (
            prev_macd['macd'] <= prev_macd['signal'] and
            macd['macd'] > macd['signal']
        )
        
        macd_bearish_cross = (
            prev_macd['macd'] >= prev_macd['signal'] and
            macd['macd'] < macd['signal']
        )
        
        print(f"   🔄 MACD Crossovers:")
        print(f"      Bullish Cross: {macd_bullish_cross}")
        print(f"      Bearish Cross: {macd_bearish_cross}")
        
        # Check individual LONG conditions
        long_cond1 = macd_bullish_cross
        long_cond2 = rsi < self.parameters['rsi_overbought']
        long_cond3 = current_price > ema50
        
        print(f"   📈 LONG Conditions:")
        print(f"      1. MACD Bullish Cross: {long_cond1}")
        print(f"      2. RSI < {self.parameters['rsi_overbought']}: {long_cond2} (RSI: {rsi:.1f})")
        print(f"      3. Price > EMA50: {long_cond3} (${current_price:.2f} vs ${ema50:.2f})")
        print(f"      🎯 LONG Signal: {long_cond1 and long_cond2 and long_cond3}")
        
        # Check individual SHORT conditions  
        short_cond1 = macd_bearish_cross
        short_cond2 = rsi > self.parameters['rsi_oversold']
        short_cond3 = current_price < ema50
        
        print(f"   📉 SHORT Conditions:")
        print(f"      1. MACD Bearish Cross: {short_cond1}")
        print(f"      2. RSI > {self.parameters['rsi_oversold']}: {short_cond2} (RSI: {rsi:.1f})")
        print(f"      3. Price < EMA50: {short_cond3} (${current_price:.2f} vs ${ema50:.2f})")
        print(f"      🎯 SHORT Signal: {short_cond1 and short_cond2 and short_cond3}")
        
        # Show recent MACD trend
        macd_trend = "Rising" if macd['macd'] > prev_macd['macd'] else "Falling"
        signal_trend = "Rising" if macd['signal'] > prev_macd['signal'] else "Falling"
        print(f"   📊 MACD Trends: Line {macd_trend}, Signal {signal_trend}")
    
    def _indicators_ready(self) -> bool:
        """Check if all indicators are ready."""
        return (
            self.indicators['macd'].is_ready() and
            self.indicators['rsi'].is_ready() and
            self.indicators['ema50'].is_ready() and
            self.state['macd_values'] is not None and
            self.state['rsi_value'] is not None and
            self.state['ema50_value'] is not None and
            self.state['prev_macd_values'] is not None
        )
    
    def _generate_entry_signal(self, data: MarketData) -> TradeSignal:
        """Generate entry signals based on MACD crossover + RSI filter + EMA trend."""
        
        current_price = data.close
        macd = self.state['macd_values']
        prev_macd = self.state['prev_macd_values']
        rsi = self.state['rsi_value']
        ema50 = self.state['ema50_value']
        
        # Check for MACD crossover signals
        macd_bullish_cross = (
            prev_macd['macd'] <= prev_macd['signal'] and
            macd['macd'] > macd['signal']
        )
        
        macd_bearish_cross = (
            prev_macd['macd'] >= prev_macd['signal'] and
            macd['macd'] < macd['signal']
        )
        
        # LONG: MACD bullish cross + RSI < 70 + Price > EMA50
        if (macd_bullish_cross and 
            rsi < self.parameters['rsi_overbought'] and 
            current_price > ema50):
            
            self.state['trades_count'] += 1
            self.state['entry_price'] = current_price
            
            tp_price = current_price * (1 + self.parameters['take_profit_pct'] / 100)
            sl_price = current_price * (1 - self.parameters['stop_loss_pct'] / 100)
            
            print(f"\n🟢 LONG SIGNAL on bar {self.state['bar_count']}!")
            print(f"   Price: ${current_price:.2f}, RSI: {rsi:.1f}, EMA50: ${ema50:.2f}")
            print(f"   MACD: {macd['macd']:.6f}, Signal: {macd['signal']:.6f}")
            print(f"   TP: ${tp_price:.2f} (+{self.parameters['take_profit_pct']:.1f}%)")
            print(f"   SL: ${sl_price:.2f} (-{self.parameters['stop_loss_pct']:.1f}%)")
            
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"MACD Bullish Cross: MACD {macd['macd']:.4f} > Signal {macd['signal']:.4f}, RSI {rsi:.1f} < 70, Price ${current_price:.2f} > EMA50 ${ema50:.2f}",
                stop_loss=sl_price,
                take_profit=tp_price
            )
        
        # SHORT: MACD bearish cross + RSI > 30 + Price < EMA50
        if (macd_bearish_cross and 
            rsi > self.parameters['rsi_oversold'] and 
            current_price < ema50):
            
            self.state['trades_count'] += 1
            self.state['entry_price'] = current_price
            
            tp_price = current_price * (1 - self.parameters['take_profit_pct'] / 100)
            sl_price = current_price * (1 + self.parameters['stop_loss_pct'] / 100)
            
            print(f"\n🔴 SHORT SIGNAL on bar {self.state['bar_count']}!")
            print(f"   Price: ${current_price:.2f}, RSI: {rsi:.1f}, EMA50: ${ema50:.2f}")
            print(f"   MACD: {macd['macd']:.6f}, Signal: {macd['signal']:.6f}")
            print(f"   TP: ${tp_price:.2f} (-{self.parameters['take_profit_pct']:.1f}%)")
            print(f"   SL: ${sl_price:.2f} (+{self.parameters['stop_loss_pct']:.1f}%)")
            
            return TradeSignal(
                signal=Signal.SELL,
                reason=f"MACD Bearish Cross: MACD {macd['macd']:.4f} < Signal {macd['signal']:.4f}, RSI {rsi:.1f} > 30, Price ${current_price:.2f} < EMA50 ${ema50:.2f}",
                stop_loss=sl_price,
                take_profit=tp_price
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def _manage_position(self, data: MarketData, position: Position) -> TradeSignal:
        """Manage open positions with fixed TP/SL."""
        
        current_price = data.close
        entry_price = self.state['entry_price']
        
        if position.direction == "LONG":
            # Take Profit
            tp_price = entry_price * (1 + self.parameters['take_profit_pct'] / 100)
            if current_price >= tp_price:
                profit_pct = ((current_price - entry_price) / entry_price) * 100
                print(f"🎯 Closing LONG with profit: +{profit_pct:.2f}%")
                return TradeSignal(
                    signal=Signal.CLOSE_LONG,
                    reason=f"TP LONG: {profit_pct:.2f}% profit (target: +{self.parameters['take_profit_pct']:.1f}%)"
                )
            
            # Stop Loss
            sl_price = entry_price * (1 - self.parameters['stop_loss_pct'] / 100)
            if current_price <= sl_price:
                loss_pct = ((entry_price - current_price) / entry_price) * 100
                print(f"🛑 Closing LONG with stop loss: -{loss_pct:.2f}%")
                return TradeSignal(
                    signal=Signal.CLOSE_LONG,
                    reason=f"SL LONG: -{loss_pct:.2f}% loss (limit: -{self.parameters['stop_loss_pct']:.1f}%)"
                )
        
        elif position.direction == "SHORT":
            # Take Profit
            tp_price = entry_price * (1 - self.parameters['take_profit_pct'] / 100)
            if current_price <= tp_price:
                profit_pct = ((entry_price - current_price) / entry_price) * 100
                print(f"🎯 Closing SHORT with profit: +{profit_pct:.2f}%")
                return TradeSignal(
                    signal=Signal.CLOSE_SHORT,
                    reason=f"TP SHORT: {profit_pct:.2f}% profit (target: +{self.parameters['take_profit_pct']:.1f}%)"
                )
            
            # Stop Loss
            sl_price = entry_price * (1 + self.parameters['stop_loss_pct'] / 100)
            if current_price >= sl_price:
                loss_pct = ((current_price - entry_price) / entry_price) * 100
                print(f"🛑 Closing SHORT with stop loss: -{loss_pct:.2f}%")
                return TradeSignal(
                    signal=Signal.CLOSE_SHORT,
                    reason=f"SL SHORT: -{loss_pct:.2f}% loss (limit: -{self.parameters['stop_loss_pct']:.1f}%)"
                )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def get_position_size(self, data: MarketData, signal: Signal, capital: float) -> float:
        """Calculate position size for trade."""
        return capital * self.parameters['position_size_pct']