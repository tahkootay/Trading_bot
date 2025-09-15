from modules.backtester import StrategyBase, MarketData, Position, TradeSignal, Signal
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import zscore


class HabrOriginalStrategy(StrategyBase):
    """
    Original Bollinger Bands + RSI + CSI Strategy from Habr article.
    
    Exact implementation based on the provided code in the article.
    All parameters, formulas, and logic match the original.
    
    Original config from article:
    - symbol = "ETHUSDT"
    - interval = "5m" 
    - bb_period = 40, bb_std = 1
    - STOP_LOSS_PCT = 0.004 (0.4%)
    - config = {'min_cluster': 3, 'bull_quant': 0.75, 'bear_quant': 0.25, 'rsi': 60}
    - RSI period = 450
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
        
        # Strategy state - store full OHLCV history for calculations
        self.state = {
            'df_history': pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']),
            'entry_price': 0.0,
            'entry_bar': 0,
            'trades_count': 0,
            'bar_count': 0,
        }

    def on_bar(self, data: MarketData, position: Position) -> TradeSignal:
        """Process new bar using exact article logic."""
        self.state['bar_count'] += 1
        
        # Add current bar to history
        new_row = pd.DataFrame({
            'timestamp': [self.state['bar_count']],  # Use bar number as timestamp
            'open': [data.open],
            'high': [data.high], 
            'low': [data.low],
            'close': [data.close],
            'volume': [data.volume]
        })
        
        self.state['df_history'] = pd.concat([self.state['df_history'], new_row], ignore_index=True)
        
        # Keep reasonable history size for performance
        if len(self.state['df_history']) > 1000:
            self.state['df_history'] = self.state['df_history'].tail(1000).reset_index(drop=True)
        
        # Need minimum bars for indicators
        if len(self.state['df_history']) < max(self.parameters['rsi_period'], 100):
            if self.state['bar_count'] % 50 == 0:
                print(f"Bar {self.state['bar_count']}: Building history (need {self.parameters['rsi_period']} bars)")
            return TradeSignal(signal=Signal.HOLD)
        
        # Calculate all indicators using exact article functions
        df = self._calculate_all_indicators()
        
        # Position management
        if position.direction != "NONE":
            return self._manage_position(data, position, df)
        
        # Generate entry signals
        return self._generate_entry_signal(data, df)
    
    def _calculate_all_indicators(self) -> pd.DataFrame:
        """Calculate all indicators using exact formulas from article."""
        df = self.state['df_history'].copy()
        
        # 1. Compute RSI - exact from article
        df = self._compute_rsi(df, self.parameters['rsi_period'])
        
        # 2. Compute Bollinger Bands - exact from article  
        df = self._compute_bollinger(df)
        
        # 3. Compute CSI - exact from article
        df = self._get_csi(df)
        
        # 4. Compute CSC (clusters) - exact from article
        df = self._compute_csc(df, self.parameters['min_cluster'], 
                              self.parameters['bull_quant'], self.parameters['bear_quant'])
        
        return df
    
    def _compute_rsi(self, df: pd.DataFrame, period: int = 450) -> pd.DataFrame:
        """Exact RSI calculation from article."""
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period, min_periods=1).mean()
        avg_loss = loss.rolling(period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(method='bfill')
        return df
    
    def _compute_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exact Bollinger Bands calculation from article."""
        bb_period = self.parameters['bb_period']  # 40
        bb_std = self.parameters['bb_std']        # 1
        
        df['ma'] = df['close'].rolling(bb_period).mean()
        df['std'] = df['close'].rolling(bb_period).std()
        df['upper'] = df['ma'] + bb_std * df['std']
        df['lower'] = df['ma'] - bb_std * df['std']
        return df
    
    def _get_csi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exact CSI calculation from article."""
        body = (df['close'] - df['open']).abs()
        rng = (df['high'] - df['low']).replace(0, np.nan)
        body_ratio = body / rng
        direction = np.where(df['close'] > df['open'], 1, -1)
        vol_score = df['volume'] / df['volume'].rolling(self.parameters['vol_window']).max()
        range_z = zscore(df['high'] - df['low']).clip(-3, 3)

        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': (df['high'] - df['close'].shift(1)).abs(),
            'lc': (df['low'] - df['close'].shift(1)).abs()
        }).max(axis=1)

        atr = tr.rolling(self.parameters['atr_period']).mean().bfill()
        df['CSI'] = direction * (0.5 * body_ratio + 0.3 * vol_score + 0.2 * range_z) / atr
        return df
    
    def _compute_csc(self, df: pd.DataFrame, min_cluster: int, bull_quant: float, bear_quant: float) -> pd.DataFrame:
        """Exact CSC (cluster) calculation from article."""
        bull_thr = df['CSI'].quantile(bull_quant)
        bear_thr = df['CSI'].quantile(bear_quant)

        df['sentiment'] = np.where(df['CSI'] >= bull_thr, 'bull',
                            np.where(df['CSI'] <= bear_thr, 'bear', 'neutral'))
        df['cluster_id'] = pd.Series(dtype='object')
        curr_type, curr_start, length = None, None, 0

        for i, s in df['sentiment'].items():
            if s == curr_type and s in ['bull', 'bear']:
                length += 1
            else:
                if curr_type in ['bull', 'bear'] and length >= min_cluster:
                    df.loc[curr_start:i-1, 'cluster_id'] = f"{curr_type}_{curr_start}"
                if s in ['bull', 'bear']:
                    curr_type, curr_start, length = s, i, 1
                else:
                    curr_type, length = None, 0

        if curr_type in ['bull', 'bear'] and length >= min_cluster:
            df.loc[curr_start:df.index[-1], 'cluster_id'] = f"{curr_type}_{curr_start}"

        return df
    
    def _check_signal_row(self, row: pd.Series, prev_row: pd.Series) -> Optional[str]:
        """Exact signal check from article."""
        if (np.isnan(row['lower']) or np.isnan(prev_row['CSI']) or 
            np.isnan(row['CSI']) or pd.isna(row['cluster_id'])):
            return None
            
        cluster = row['cluster_id']
        if not isinstance(cluster, str):
            return None

        # Exact conditions from article
        long_cond = (
            row['close'] < row['lower'] and
            row['CSI'] > 0 and row['CSI'] > prev_row['CSI'] and
            cluster.startswith('bull') and row['RSI'] < self.parameters['rsi_threshold']
        )
        short_cond = (
            row['close'] > row['upper'] and
            row['CSI'] < 0 and row['CSI'] < prev_row['CSI'] and
            cluster.startswith('bear') and row['RSI'] > (100 - self.parameters['rsi_threshold'])
        )

        if long_cond:
            return 'buy'
        elif short_cond:
            return 'sell'
        return None
    
    def _generate_entry_signal(self, data: MarketData, df: pd.DataFrame) -> TradeSignal:
        """Generate entry signal using exact article logic."""
        if len(df) < 2:
            return TradeSignal(signal=Signal.HOLD)
        
        current_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        signal = self._check_signal_row(current_row, prev_row)
        
        if signal == 'buy':
            self.state['trades_count'] += 1
            self.state['entry_price'] = data.close
            self.state['entry_bar'] = self.state['bar_count']
            
            stop_price = data.close * (1 - self.parameters['stop_loss_pct'] / 100)
            
            print(f"🟢 LONG signal on bar {self.state['bar_count']}: Original Habr strategy")
            print(f"   Price: ${data.close:.2f}, BB Lower: ${current_row['lower']:.2f}")
            print(f"   RSI: {current_row['RSI']:.1f}, CSI: {current_row['CSI']:.4f} (prev: {prev_row['CSI']:.4f})")
            print(f"   Cluster: {current_row['cluster_id']}")
            print(f"   SL: ${stop_price:.2f} (-{self.parameters['stop_loss_pct']:.1f}%)")
            
            return TradeSignal(
                signal=Signal.BUY,
                reason=f"Habr LONG: Price {data.close:.2f} < BB Lower {current_row['lower']:.2f}, CSI {current_row['CSI']:.4f} rising, Bull cluster, RSI {current_row['RSI']:.1f}",
                stop_loss=stop_price,
                take_profit=None  # No fixed TP, using time exit
            )
        
        elif signal == 'sell':
            self.state['trades_count'] += 1
            self.state['entry_price'] = data.close
            self.state['entry_bar'] = self.state['bar_count']
            
            stop_price = data.close * (1 + self.parameters['stop_loss_pct'] / 100)
            
            print(f"🔴 SHORT signal on bar {self.state['bar_count']}: Original Habr strategy")
            print(f"   Price: ${data.close:.2f}, BB Upper: ${current_row['upper']:.2f}")
            print(f"   RSI: {current_row['RSI']:.1f}, CSI: {current_row['CSI']:.4f} (prev: {prev_row['CSI']:.4f})")
            print(f"   Cluster: {current_row['cluster_id']}")
            print(f"   SL: ${stop_price:.2f} (+{self.parameters['stop_loss_pct']:.1f}%)")
            
            return TradeSignal(
                signal=Signal.SELL,
                reason=f"Habr SHORT: Price {data.close:.2f} > BB Upper {current_row['upper']:.2f}, CSI {current_row['CSI']:.4f} falling, Bear cluster, RSI {current_row['RSI']:.1f}",
                stop_loss=stop_price,
                take_profit=None  # No fixed TP, using time exit
            )
        
        return TradeSignal(signal=Signal.HOLD)
    
    def _manage_position(self, data: MarketData, position: Position, df: pd.DataFrame) -> TradeSignal:
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