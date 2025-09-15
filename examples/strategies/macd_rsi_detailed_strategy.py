from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import MACD, RSI, EMA
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime


class MacdRsiDetailedStrategy(StrategyBase):
    """
    MACD + RSI Strategy with detailed logging for analysis.
    
    Records all candle data, indicator values, and trading signals
    for comprehensive post-backtest analysis.
    """
    
    def _initialize(self):
        """Initialize strategy parameters and state."""
        self.parameters = {
            'macd_fast': 12,           # MACD fast period
            'macd_slow': 26,           # MACD slow period  
            'macd_signal': 9,          # MACD signal period
            'rsi_period': 14,          # RSI period
            'ema_period': 50,          # EMA trend filter
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
            'current_position': 'NONE',
            'position_entry_bar': 0,
        }
        
        # Data logging
        self.detailed_log = []

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
            # Log even when indicators are not ready
            self._log_bar_data(data, position, None, "INDICATORS_NOT_READY")
            return TradeSignal(signal=Signal.HOLD)
        
        # Generate signal
        signal_result = None
        
        # Position management for open positions
        if position.direction != "NONE":
            signal_result = self._manage_position(data, position)
        else:
            # Entry signal generation
            signal_result = self._generate_entry_signal(data)
        
        # Log all data
        self._log_bar_data(data, position, signal_result, "NORMAL")
        
        return signal_result if signal_result else TradeSignal(signal=Signal.HOLD)
    
    def _log_bar_data(self, data: MarketData, position: Position, signal: Optional[TradeSignal], status: str) -> None:
        """Log detailed bar data for analysis."""
        
        # Get indicator values
        macd_line = self.state['macd_values']['macd'] if self.state['macd_values'] else np.nan
        macd_signal = self.state['macd_values']['signal'] if self.state['macd_values'] else np.nan
        macd_histogram = self.state['macd_values']['histogram'] if self.state['macd_values'] else np.nan
        rsi_value = self.state['rsi_value'] if self.state['rsi_value'] is not None else np.nan
        ema50_value = self.state['ema50_value'] if self.state['ema50_value'] is not None else np.nan
        
        prev_macd_line = self.state['prev_macd_values']['macd'] if self.state['prev_macd_values'] else np.nan
        prev_macd_signal = self.state['prev_macd_values']['signal'] if self.state['prev_macd_values'] else np.nan
        
        # Calculate crossovers
        macd_bullish_cross = False
        macd_bearish_cross = False
        
        if not (np.isnan(macd_line) or np.isnan(macd_signal) or np.isnan(prev_macd_line) or np.isnan(prev_macd_signal)):
            macd_bullish_cross = prev_macd_line <= prev_macd_signal and macd_line > macd_signal
            macd_bearish_cross = prev_macd_line >= prev_macd_signal and macd_line < macd_signal
        
        # Calculate signal conditions
        long_condition_1 = macd_bullish_cross
        long_condition_2 = rsi_value < self.parameters['rsi_overbought'] if not np.isnan(rsi_value) else False
        long_condition_3 = data.close > ema50_value if not np.isnan(ema50_value) else False
        long_signal_valid = long_condition_1 and long_condition_2 and long_condition_3
        
        short_condition_1 = macd_bearish_cross
        short_condition_2 = rsi_value > self.parameters['rsi_oversold'] if not np.isnan(rsi_value) else False
        short_condition_3 = data.close < ema50_value if not np.isnan(ema50_value) else False
        short_signal_valid = short_condition_1 and short_condition_2 and short_condition_3
        
        # Determine signal type
        signal_type = "HOLD"
        signal_reason = ""
        
        if signal and hasattr(signal, 'signal'):
            if signal.signal == Signal.BUY:
                signal_type = "BUY"
                signal_reason = getattr(signal, 'reason', '')
            elif signal.signal == Signal.SELL:
                signal_type = "SELL"  
                signal_reason = getattr(signal, 'reason', '')
            elif signal.signal == Signal.CLOSE_LONG:
                signal_type = "CLOSE_LONG"
                signal_reason = getattr(signal, 'reason', '')
            elif signal.signal == Signal.CLOSE_SHORT:
                signal_type = "CLOSE_SHORT"
                signal_reason = getattr(signal, 'reason', '')
        
        # Log entry
        log_entry = {
            'bar': self.state['bar_count'],
            'timestamp': f"2025-06-01 00:00:00",  # Placeholder, will be updated by backtester
            'open': data.open,
            'high': data.high,
            'low': data.low,
            'close': data.close,
            'volume': data.volume,
            
            # Indicators
            'macd_line': macd_line,
            'macd_signal_line': macd_signal,
            'macd_histogram': macd_histogram,
            'rsi': rsi_value,
            'ema50': ema50_value,
            
            # Previous values for crossover
            'prev_macd_line': prev_macd_line,
            'prev_macd_signal_line': prev_macd_signal,
            
            # Crossover signals
            'macd_bullish_cross': macd_bullish_cross,
            'macd_bearish_cross': macd_bearish_cross,
            
            # Signal conditions
            'long_cond_1_macd_cross': long_condition_1,
            'long_cond_2_rsi_below_70': long_condition_2,
            'long_cond_3_price_above_ema': long_condition_3,
            'long_signal_valid': long_signal_valid,
            
            'short_cond_1_macd_cross': short_condition_1,
            'short_cond_2_rsi_above_30': short_condition_2,
            'short_cond_3_price_below_ema': short_condition_3,
            'short_signal_valid': short_signal_valid,
            
            # Position information
            'position_direction': position.direction if position else 'NONE',
            'position_size': position.size if position and hasattr(position, 'size') else 0,
            'position_entry_price': getattr(position, 'entry_price', 0) if position else 0,
            
            # Signal information  
            'signal_type': signal_type,
            'signal_reason': signal_reason,
            'stop_loss': getattr(signal, 'stop_loss', None) if signal else None,
            'take_profit': getattr(signal, 'take_profit', None) if signal else None,
            
            # Strategy state
            'indicators_ready': self._indicators_ready(),
            'status': status,
            'trades_count': self.state['trades_count']
        }
        
        self.detailed_log.append(log_entry)
    
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
                return TradeSignal(
                    signal=Signal.CLOSE_LONG,
                    reason=f"TP LONG: {profit_pct:.2f}% profit (target: +{self.parameters['take_profit_pct']:.1f}%)"
                )
            
            # Stop Loss
            sl_price = entry_price * (1 - self.parameters['stop_loss_pct'] / 100)
            if current_price <= sl_price:
                loss_pct = ((entry_price - current_price) / entry_price) * 100
                return TradeSignal(
                    signal=Signal.CLOSE_LONG,
                    reason=f"SL LONG: -{loss_pct:.2f}% loss (limit: -{self.parameters['stop_loss_pct']:.1f}%)"
                )
        
        elif position.direction == "SHORT":
            # Take Profit
            tp_price = entry_price * (1 - self.parameters['take_profit_pct'] / 100)
            if current_price <= tp_price:
                profit_pct = ((entry_price - current_price) / entry_price) * 100
                return TradeSignal(
                    signal=Signal.CLOSE_SHORT,
                    reason=f"TP SHORT: {profit_pct:.2f}% profit (target: +{self.parameters['take_profit_pct']:.1f}%)"
                )
            
            # Stop Loss
            sl_price = entry_price * (1 + self.parameters['stop_loss_pct'] / 100)
            if current_price >= sl_price:
                loss_pct = ((current_price - entry_price) / entry_price) * 100
                return TradeSignal(
                    signal=Signal.CLOSE_SHORT,
                    reason=f"SL SHORT: -{loss_pct:.2f}% loss (limit: -{self.parameters['stop_loss_pct']:.1f}%)"
                )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def get_position_size(self, data: MarketData, signal: Signal, capital: float) -> float:
        """Calculate position size for trade."""
        return capital * self.parameters['position_size_pct']
    
    def export_detailed_log(self, filename: str = None) -> str:
        """Export detailed log to CSV file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output/detailed_backtest_{timestamp}.csv"
        
        # Convert to DataFrame
        df = pd.DataFrame(self.detailed_log)
        
        # Save to CSV
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        
        print(f"📊 Detailed log exported to: {filename}")
        print(f"📈 Total bars recorded: {len(df)}")
        print(f"🔄 Total trades: {self.state['trades_count']}")
        
        return filename