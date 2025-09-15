from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import BollingerBands, RSI
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import zscore


class BbRsiCsiStrategy(StrategyBase):
    """
    Bollinger Bands + RSI + CSI Strategy from Habr article.
    
    Complex multi-indicator strategy combining:
    - Bollinger Bands (40 period, 1 std deviation)
    - RSI (450 period for stability)
    - CSI (Custom Strength Index) - proprietary momentum indicator
    - CSC (Cluster Signal Confirmation) - cluster analysis
    
    Entry Rules:
    - LONG: price < BB lower + CSI > 0 + CSI rising + bull cluster + RSI < 60
    - SHORT: price > BB upper + CSI < 0 + CSI falling + bear cluster + RSI > 40
    
    Exit Rules:
    - Stop Loss: 0.4%
    - Time Exit: 15 bars (75 minutes on 5M)
    """
    
    def _initialize(self):
        """Initialize strategy parameters and state."""
        self.parameters = {
            'bb_period': 40,           # Bollinger Bands period (from article)
            'bb_std': 1.0,             # BB standard deviation (from article)
            'rsi_period': 450,         # RSI period (from article)
            'rsi_threshold': 60,       # RSI filter level
            'stop_loss_pct': 0.4,      # Stop loss: 0.4%
            'time_exit_bars': 15,      # Time exit: 15 bars
            'position_size_pct': 0.05, # 5% of capital
            'min_cluster': 3,          # Minimum cluster size
            'bull_quant': 0.75,        # Bull cluster quantile
            'bear_quant': 0.25,        # Bear cluster quantile
            'atr_period': 14,          # ATR smoothing period
            'vol_window': 50,          # Volume analysis window
        }
        
        # Initialize basic indicators
        self.indicators = {
            'bb': BollingerBands(
                period=self.parameters['bb_period'], 
                std_dev=self.parameters['bb_std']
            ),
            'rsi': RSI(period=self.parameters['rsi_period'])
        }
        
        # Strategy state
        self.state = {
            'bb_values': None,
            'rsi_value': None,
            'csi_values': [],
            'prev_csi': None,
            'cluster_data': None,
            'entry_price': 0.0,
            'entry_bar': 0,
            'trades_count': 0,
            'bar_count': 0,
            'price_history': [],
            'volume_history': [],
            'ohlc_history': []
        }

    def update_indicators(self, data: MarketData) -> None:
        """Update all technical indicators with new market data."""
        # Store OHLC history for CSI calculation
        self.state['ohlc_history'].append({
            'open': data.open,
            'high': data.high, 
            'low': data.low,
            'close': data.close,
            'volume': data.volume
        })
        
        # Keep only recent history for performance
        if len(self.state['ohlc_history']) > 1000:
            self.state['ohlc_history'] = self.state['ohlc_history'][-1000:]
        
        # Update basic indicators
        self.state['bb_values'] = self.indicators['bb'].update(data.close)
        self.state['rsi_value'] = self.indicators['rsi'].update(data.close)
        
        # Calculate CSI if we have enough data
        if len(self.state['ohlc_history']) >= 50:
            self._calculate_csi()
            self._calculate_clusters()

    def _calculate_csi(self) -> None:
        """Calculate Custom Strength Index (CSI) from article."""
        if len(self.state['ohlc_history']) < 50:
            return
            
        df_temp = pd.DataFrame(self.state['ohlc_history'][-100:])  # Use last 100 bars
        
        # Body ratio calculation
        body = (df_temp['close'] - df_temp['open']).abs()
        rng = (df_temp['high'] - df_temp['low']).replace(0, np.nan)
        body_ratio = body / rng
        
        # Direction (bullish/bearish)
        direction = np.where(df_temp['close'] > df_temp['open'], 1, -1)
        
        # Volume score (relative to recent max)
        vol_max = df_temp['volume'].rolling(self.parameters['vol_window']).max()
        vol_score = df_temp['volume'] / vol_max
        
        # Range z-score (volatility measure)
        range_vals = df_temp['high'] - df_temp['low']
        range_z = zscore(range_vals).clip(-3, 3)
        
        # True Range and ATR calculation
        tr_hl = df_temp['high'] - df_temp['low']
        tr_hc = (df_temp['high'] - df_temp['close'].shift(1)).abs()
        tr_lc = (df_temp['low'] - df_temp['close'].shift(1)).abs()
        tr = pd.concat([tr_hl, tr_hc, tr_lc], axis=1).max(axis=1)
        atr = tr.rolling(self.parameters['atr_period']).mean().bfill()
        
        # CSI formula from article (with ATR normalization)
        csi = direction * (0.5 * body_ratio + 0.3 * vol_score + 0.2 * range_z) / (atr + 1e-8)
        
        # Store current CSI
        self.state['prev_csi'] = self.state['csi_values'][-1] if self.state['csi_values'] else 0
        current_csi = csi.iloc[-1] if not pd.isna(csi.iloc[-1]) else 0
        self.state['csi_values'].append(current_csi)
        
        # Keep CSI history manageable
        if len(self.state['csi_values']) > 200:
            self.state['csi_values'] = self.state['csi_values'][-200:]

    def _calculate_clusters(self) -> None:
        """Calculate cluster analysis (CSC) from article."""
        if len(self.state['csi_values']) < 50:
            self.state['cluster_data'] = None
            return
            
        csi_series = pd.Series(self.state['csi_values'][-100:])  # Use last 100 values
        
        # Calculate quantile thresholds
        bull_threshold = csi_series.quantile(self.parameters['bull_quant'])
        bear_threshold = csi_series.quantile(self.parameters['bear_quant'])
        
        # Classify sentiment
        sentiment = np.where(
            csi_series >= bull_threshold, 'bull',
            np.where(csi_series <= bear_threshold, 'bear', 'neutral')
        )
        
        # Find clusters (consecutive bull/bear periods)
        current_cluster = None
        cluster_length = 0
        cluster_start = 0
        
        for i, s in enumerate(sentiment):
            if s == current_cluster and s in ['bull', 'bear']:
                cluster_length += 1
            else:
                # Check if previous cluster meets minimum size
                if (current_cluster in ['bull', 'bear'] and 
                    cluster_length >= self.parameters['min_cluster']):
                    # We have a valid cluster
                    pass
                
                if s in ['bull', 'bear']:
                    current_cluster = s
                    cluster_start = i
                    cluster_length = 1
                else:
                    current_cluster = None
                    cluster_length = 0
        
        # Store current cluster info
        if (current_cluster in ['bull', 'bear'] and 
            cluster_length >= self.parameters['min_cluster']):
            self.state['cluster_data'] = {
                'type': current_cluster,
                'length': cluster_length,
                'valid': True
            }
        else:
            self.state['cluster_data'] = {'valid': False}

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Process new market data and generate trading signal."""
        self.state['bar_count'] += 1
        
        # Update indicators first
        self.update_indicators(data)
        
        # Check if indicators are ready
        if not self._indicators_ready():
            if self.state['bar_count'] <= 500:
                if self.state['bar_count'] % 50 == 0:
                    print(f"Bar {self.state['bar_count']}: Waiting for indicators (need ~{self.parameters['rsi_period']} bars)")
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
            self.indicators['rsi'].is_ready() and
            self.state['bb_values'] is not None and
            self.state['rsi_value'] is not None and
            len(self.state['csi_values']) >= 10 and
            self.state['cluster_data'] is not None
        )
    
    def _generate_entry_signal(self, data: MarketData) -> TradeSignal:
        """Generate entry signals based on BB + RSI + CSI + Clusters."""
        
        current_price = data.close
        bb = self.state['bb_values']
        rsi = self.state['rsi_value']
        current_csi = self.state['csi_values'][-1] if self.state['csi_values'] else 0
        prev_csi = self.state['prev_csi'] if self.state['prev_csi'] is not None else 0
        cluster = self.state['cluster_data']
        
        # LONG conditions from article
        long_conditions = [
            current_price < bb['lower'],              # Price below BB lower
            current_csi > 0,                          # CSI positive
            current_csi > prev_csi,                   # CSI rising
            cluster.get('valid', False),              # Valid cluster exists
            cluster.get('type') == 'bull',            # Bull cluster
            rsi < self.parameters['rsi_threshold']    # RSI < 60
        ]
        
        # SHORT conditions from article  
        short_conditions = [
            current_price > bb['upper'],              # Price above BB upper
            current_csi < 0,                          # CSI negative
            current_csi < prev_csi,                   # CSI falling
            cluster.get('valid', False),              # Valid cluster exists
            cluster.get('type') == 'bear',            # Bear cluster
            rsi > (100 - self.parameters['rsi_threshold'])  # RSI > 40
        ]
        
        if all(long_conditions):
            self.state['trades_count'] += 1
            self.state['entry_price'] = current_price
            self.state['entry_bar'] = self.state['bar_count']
            
            sl_price = current_price * (1 - self.parameters['stop_loss_pct'] / 100)
            
            print(f"🟢 LONG signal on bar {self.state['bar_count']}: BB + RSI + CSI strategy")
            print(f"   Price: ${current_price:.2f}, BB Lower: ${bb['lower']:.2f}")
            print(f"   RSI: {rsi:.1f}, CSI: {current_csi:.4f} (prev: {prev_csi:.4f})")
            print(f"   Cluster: {cluster.get('type', 'none')} (length: {cluster.get('length', 0)})")
            print(f"   SL: ${sl_price:.2f} (-{self.parameters['stop_loss_pct']:.1f}%)")
            
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"BB+RSI+CSI LONG: Price {current_price:.2f} < BB Lower {bb['lower']:.2f}, RSI {rsi:.1f}, CSI {current_csi:.4f} rising, Bull cluster",
                stop_loss=sl_price,
                take_profit=None  # No fixed TP, using time exit
            )
        
        elif all(short_conditions):
            self.state['trades_count'] += 1
            self.state['entry_price'] = current_price
            self.state['entry_bar'] = self.state['bar_count']
            
            sl_price = current_price * (1 + self.parameters['stop_loss_pct'] / 100)
            
            print(f"🔴 SHORT signal on bar {self.state['bar_count']}: BB + RSI + CSI strategy")
            print(f"   Price: ${current_price:.2f}, BB Upper: ${bb['upper']:.2f}")
            print(f"   RSI: {rsi:.1f}, CSI: {current_csi:.4f} (prev: {prev_csi:.4f})")
            print(f"   Cluster: {cluster.get('type', 'none')} (length: {cluster.get('length', 0)})")
            print(f"   SL: ${sl_price:.2f} (+{self.parameters['stop_loss_pct']:.1f}%)")
            
            return TradeSignal(
                signal=Signal.SELL,
                reason=f"BB+RSI+CSI SHORT: Price {current_price:.2f} > BB Upper {bb['upper']:.2f}, RSI {rsi:.1f}, CSI {current_csi:.4f} falling, Bear cluster",
                stop_loss=sl_price,
                take_profit=None  # No fixed TP, using time exit
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def _manage_position(self, data: MarketData, position: Position) -> TradeSignal:
        """Manage open positions with stop loss and time exit."""
        
        current_price = data.close
        entry_price = self.state['entry_price']
        bars_in_position = self.state['bar_count'] - self.state['entry_bar']
        
        if position.direction == "LONG":
            # Stop Loss
            sl_price = entry_price * (1 - self.parameters['stop_loss_pct'] / 100)
            if current_price <= sl_price:
                loss_pct = ((entry_price - current_price) / entry_price) * 100
                print(f"🛑 Closing LONG with stop loss: -{loss_pct:.2f}%")
                return TradeSignal(
                    signal=Signal.CLOSE_LONG,
                    reason=f"SL LONG: -{loss_pct:.2f}% loss (limit: -{self.parameters['stop_loss_pct']:.1f}%)"
                )
            
            # Time Exit
            if bars_in_position >= self.parameters['time_exit_bars']:
                profit_pct = ((current_price - entry_price) / entry_price) * 100
                print(f"⏰ Closing LONG by time: {profit_pct:+.2f}% after {bars_in_position} bars")
                return TradeSignal(
                    signal=Signal.CLOSE_LONG,
                    reason=f"Time Exit LONG: {profit_pct:+.2f}% after {bars_in_position} bars"
                )
        
        elif position.direction == "SHORT":
            # Stop Loss
            sl_price = entry_price * (1 + self.parameters['stop_loss_pct'] / 100)
            if current_price >= sl_price:
                loss_pct = ((current_price - entry_price) / entry_price) * 100
                print(f"🛑 Closing SHORT with stop loss: -{loss_pct:.2f}%")
                return TradeSignal(
                    signal=Signal.CLOSE_SHORT,
                    reason=f"SL SHORT: -{loss_pct:.2f}% loss (limit: -{self.parameters['stop_loss_pct']:.1f}%)"
                )
            
            # Time Exit
            if bars_in_position >= self.parameters['time_exit_bars']:
                profit_pct = ((entry_price - current_price) / entry_price) * 100
                print(f"⏰ Closing SHORT by time: {profit_pct:+.2f}% after {bars_in_position} bars")
                return TradeSignal(
                    signal=Signal.CLOSE_SHORT,
                    reason=f"Time Exit SHORT: {profit_pct:+.2f}% after {bars_in_position} bars"
                )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def get_position_size(self, data: MarketData, signal: Signal, capital: float) -> float:
        """Calculate position size for trade."""
        return capital * self.parameters['position_size_pct']