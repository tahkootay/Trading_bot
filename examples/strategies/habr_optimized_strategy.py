from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from modules.indicators import BollingerBands, RSI
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import zscore
from collections import deque


class HabrOptimizedStrategy(StrategyBase):
    """
    Optimized version of original Habr strategy.
    
    Uses existing indicators for BB and RSI, implements custom CSI and clustering.
    All parameters and logic remain exactly as in the original article.
    
    Performance optimizations:
    - Use existing BollingerBands and RSI indicators
    - Efficient rolling calculations for CSI
    - Deque-based data storage
    - Minimal pandas operations
    """
    
    def _initialize(self):
        """Initialize strategy with exact parameters from article."""
        # Exact config from article
        self.parameters = {
            'bb_period': 40,           # BB period from article
            'bb_std': 1,               # BB std deviation from article  
            'rsi_period': 450,         # RSI period from article
            'stop_loss_pct': 0.4,      # 0.004 = 0.4% stop loss
            'position_size_pct': 0.05, # 5% position size
            'min_cluster': 3,          # min cluster size from config
            'bull_quant': 0.75,        # bull quantile from config
            'bear_quant': 0.25,        # bear quantile from config
            'rsi_threshold': 60,       # RSI threshold from config
            'exit_after_bars': 15,     # EXIT_AFTER_BARS = 15 from article
            'vol_window': 50,          # volume analysis window
            'atr_period': 14,          # ATR period
        }
        
        # Initialize existing indicators
        self.indicators = {
            'bb': BollingerBands(
                period=self.parameters['bb_period'], 
                std_dev=self.parameters['bb_std']
            ),
            'rsi': RSI(period=self.parameters['rsi_period'])
        }
        
        # Efficient data storage using deques
        self.state = {
            'bb_values': None,
            'rsi_value': None,
            'ohlcv_data': deque(maxlen=1000),  # Store recent OHLCV
            'csi_values': deque(maxlen=200),   # Store CSI history
            'sentiment_data': deque(maxlen=100),  # Store sentiment
            'cluster_info': None,
            'prev_csi': 0.0,
            'entry_price': 0.0,
            'entry_bar': 0,
            'trades_count': 0,
            'bar_count': 0,
        }

    def update_indicators(self, data: MarketData) -> None:
        """Update all indicators efficiently."""
        # Update existing indicators
        self.state['bb_values'] = self.indicators['bb'].update(data.close)
        self.state['rsi_value'] = self.indicators['rsi'].update(data.close)
        
        # Store OHLCV data
        self.state['ohlcv_data'].append({
            'open': data.open,
            'high': data.high,
            'low': data.low,
            'close': data.close,
            'volume': data.volume
        })
        
        # Calculate CSI if we have enough data
        if len(self.state['ohlcv_data']) >= self.parameters['vol_window']:
            self._update_csi()
            self._update_clusters()

    def _update_csi(self) -> None:
        """Calculate CSI efficiently using deque data."""
        if len(self.state['ohlcv_data']) < self.parameters['vol_window']:
            return
            
        # Get recent data for calculation
        recent_data = list(self.state['ohlcv_data'])[-self.parameters['vol_window']:]
        current_bar = recent_data[-1]
        
        # Calculate CSI components for current bar
        body = abs(current_bar['close'] - current_bar['open'])
        bar_range = current_bar['high'] - current_bar['low']
        if bar_range == 0:
            bar_range = 1e-8  # Avoid division by zero
            
        body_ratio = body / bar_range
        direction = 1 if current_bar['close'] > current_bar['open'] else -1
        
        # Volume score (relative to recent max)
        volumes = [bar['volume'] for bar in recent_data]
        max_volume = max(volumes) if volumes else 1
        vol_score = current_bar['volume'] / max_volume
        
        # Range z-score
        ranges = [bar['high'] - bar['low'] for bar in recent_data]
        if len(ranges) > 3:  # Need minimum data for zscore
            try:
                range_z = float(zscore([bar_range] + ranges[:-1])[-1])  # zscore of current range
                range_z = max(-3, min(3, range_z))  # Clip to [-3, 3]
            except:
                range_z = 0.0
        else:
            range_z = 0.0
        
        # ATR calculation
        tr_values = []
        for i in range(1, len(recent_data)):
            curr = recent_data[i]
            prev = recent_data[i-1]
            
            tr = max(
                curr['high'] - curr['low'],
                abs(curr['high'] - prev['close']),
                abs(curr['low'] - prev['close'])
            )
            tr_values.append(tr)
        
        if len(tr_values) >= self.parameters['atr_period']:
            atr = sum(tr_values[-self.parameters['atr_period']:]) / self.parameters['atr_period']
        else:
            atr = sum(tr_values) / len(tr_values) if tr_values else 1e-8
        
        # CSI formula from article
        csi = direction * (0.5 * body_ratio + 0.3 * vol_score + 0.2 * range_z) / (atr + 1e-8)
        
        # Store CSI
        self.state['prev_csi'] = self.state['csi_values'][-1] if self.state['csi_values'] else 0.0
        self.state['csi_values'].append(csi)

    def _update_clusters(self) -> None:
        """Update cluster analysis efficiently."""
        if len(self.state['csi_values']) < 20:  # Need minimum CSI history
            self.state['cluster_info'] = None
            return
        
        # Calculate quantile thresholds from recent CSI values
        csi_list = list(self.state['csi_values'])
        csi_array = np.array(csi_list)
        
        bull_threshold = np.quantile(csi_array, self.parameters['bull_quant'])
        bear_threshold = np.quantile(csi_array, self.parameters['bear_quant'])
        
        # Classify current sentiment
        current_csi = csi_list[-1]
        if current_csi >= bull_threshold:
            sentiment = 'bull'
        elif current_csi <= bear_threshold:
            sentiment = 'bear'
        else:
            sentiment = 'neutral'
        
        self.state['sentiment_data'].append(sentiment)
        
        # Find current cluster
        if len(self.state['sentiment_data']) < self.parameters['min_cluster']:
            self.state['cluster_info'] = None
            return
        
        # Check if we have a valid cluster (consecutive same sentiment)
        recent_sentiment = list(self.state['sentiment_data'])
        cluster_length = 0
        cluster_type = recent_sentiment[-1]
        
        # Count consecutive occurrences from the end
        for i in range(len(recent_sentiment) - 1, -1, -1):
            if recent_sentiment[i] == cluster_type and cluster_type in ['bull', 'bear']:
                cluster_length += 1
            else:
                break
        
        # Store cluster info
        if cluster_type in ['bull', 'bear'] and cluster_length >= self.parameters['min_cluster']:
            self.state['cluster_info'] = {
                'type': cluster_type,
                'length': cluster_length,
                'valid': True
            }
        else:
            self.state['cluster_info'] = {'valid': False}

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Process new bar using optimized logic."""
        self.state['bar_count'] += 1
        
        # Update indicators
        self.update_indicators(data)
        
        # Check if indicators are ready
        if not self._indicators_ready():
            if self.state['bar_count'] % 50 == 0:
                print(f"Bar {self.state['bar_count']}: Building indicators (need {self.parameters['rsi_period']} bars)")
            return TradeSignal(signal=Signal.HOLD)
        
        # Position management
        if position.direction != "NONE":
            return self._manage_position(data, position)
        
        # Generate entry signals
        return self._generate_entry_signal(data)
    
    def _indicators_ready(self) -> bool:
        """Check if all indicators are ready."""
        return (
            self.indicators['bb'].is_ready() and
            self.indicators['rsi'].is_ready() and
            self.state['bb_values'] is not None and
            self.state['rsi_value'] is not None and
            len(self.state['csi_values']) >= 5 and
            self.state['cluster_info'] is not None
        )
    
    def _generate_entry_signal(self, data: MarketData) -> TradeSignal:
        """Generate entry signal using exact article logic."""
        current_price = data.close
        bb = self.state['bb_values'] 
        rsi = self.state['rsi_value']
        current_csi = self.state['csi_values'][-1] if self.state['csi_values'] else 0
        prev_csi = self.state['prev_csi']
        cluster = self.state['cluster_info']
        
        # Check for NaN values
        if (pd.isna(bb['lower']) or pd.isna(bb['upper']) or pd.isna(rsi) or 
            pd.isna(current_csi) or not cluster or not cluster.get('valid', False)):
            return TradeSignal(signal=Signal.HOLD)
        
        # LONG conditions from article
        long_cond = (
            current_price < bb['lower'] and                    # Price below BB lower
            current_csi > 0 and                               # CSI positive
            current_csi > prev_csi and                        # CSI rising
            cluster.get('type') == 'bull' and                 # Bull cluster
            rsi < self.parameters['rsi_threshold']            # RSI < 60
        )
        
        # SHORT conditions from article
        short_cond = (
            current_price > bb['upper'] and                    # Price above BB upper
            current_csi < 0 and                               # CSI negative
            current_csi < prev_csi and                        # CSI falling
            cluster.get('type') == 'bear' and                 # Bear cluster
            rsi > (100 - self.parameters['rsi_threshold'])    # RSI > 40
        )
        
        if long_cond:
            self.state['trades_count'] += 1
            self.state['entry_price'] = current_price
            self.state['entry_bar'] = self.state['bar_count']
            
            stop_price = current_price * (1 - self.parameters['stop_loss_pct'] / 100)
            
            print(f"🟢 LONG signal on bar {self.state['bar_count']}: Optimized Habr strategy")
            print(f"   Price: ${current_price:.2f}, BB Lower: ${bb['lower']:.2f}")
            print(f"   RSI: {rsi:.1f}, CSI: {current_csi:.4f} (prev: {prev_csi:.4f})")
            print(f"   Cluster: {cluster.get('type', 'none')} (length: {cluster.get('length', 0)})")
            print(f"   SL: ${stop_price:.2f} (-{self.parameters['stop_loss_pct']:.1f}%)")
            
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"Habr LONG: Price {current_price:.2f} < BB Lower {bb['lower']:.2f}, CSI {current_csi:.4f} rising, Bull cluster, RSI {rsi:.1f}",
                stop_loss=stop_price,
                take_profit=None
            )
        
        elif short_cond:
            self.state['trades_count'] += 1
            self.state['entry_price'] = current_price
            self.state['entry_bar'] = self.state['bar_count']
            
            stop_price = current_price * (1 + self.parameters['stop_loss_pct'] / 100)
            
            print(f"🔴 SHORT signal on bar {self.state['bar_count']}: Optimized Habr strategy")
            print(f"   Price: ${current_price:.2f}, BB Upper: ${bb['upper']:.2f}")
            print(f"   RSI: {rsi:.1f}, CSI: {current_csi:.4f} (prev: {prev_csi:.4f})")
            print(f"   Cluster: {cluster.get('type', 'none')} (length: {cluster.get('length', 0)})")
            print(f"   SL: ${stop_price:.2f} (+{self.parameters['stop_loss_pct']:.1f}%)")
            
            return TradeSignal(
                signal=Signal.SELL,
                reason=f"Habr SHORT: Price {current_price:.2f} > BB Upper {bb['upper']:.2f}, CSI {current_csi:.4f} falling, Bear cluster, RSI {rsi:.1f}",
                stop_loss=stop_price,
                take_profit=None
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def _manage_position(self, data: MarketData, position: Position) -> TradeSignal:
        """Manage positions using exact article logic."""
        current_price = data.close
        entry_price = self.state['entry_price']
        bars_in_position = self.state['bar_count'] - self.state['entry_bar']
        
        if position.direction == "LONG":
            # Stop Loss
            stop_price = entry_price * (1 - self.parameters['stop_loss_pct'] / 100)
            if current_price <= stop_price:
                loss_pct = ((entry_price - current_price) / entry_price) * 100
                print(f"🛑 Closing LONG with stop loss: -{loss_pct:.2f}%")
                return TradeSignal(
                    signal=Signal.CLOSE_LONG,
                    reason=f"Stop Loss LONG: -{loss_pct:.2f}%"
                )
            
            # Time Exit (15 bars as in article)
            if bars_in_position >= self.parameters['exit_after_bars']:
                profit_pct = ((current_price - entry_price) / entry_price) * 100
                print(f"⏰ Closing LONG by time: {profit_pct:+.2f}% after {bars_in_position} bars")
                return TradeSignal(
                    signal=Signal.CLOSE_LONG,
                    reason=f"Time Exit LONG: {profit_pct:+.2f}% after {bars_in_position} bars"
                )
        
        elif position.direction == "SHORT":
            # Stop Loss
            stop_price = entry_price * (1 + self.parameters['stop_loss_pct'] / 100)
            if current_price >= stop_price:
                loss_pct = ((current_price - entry_price) / entry_price) * 100
                print(f"🛑 Closing SHORT with stop loss: -{loss_pct:.2f}%")
                return TradeSignal(
                    signal=Signal.CLOSE_SHORT,
                    reason=f"Stop Loss SHORT: -{loss_pct:.2f}%"
                )
            
            # Time Exit (15 bars as in article)
            if bars_in_position >= self.parameters['exit_after_bars']:
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