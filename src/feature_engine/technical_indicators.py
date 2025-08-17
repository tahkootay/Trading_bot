"""Technical indicators calculation module."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
import warnings

# Suppress pandas warnings
warnings.filterwarnings('ignore')

from ..utils.types import TechnicalIndicators, TimeFrame
from ..utils.logger import TradingLogger


class TechnicalIndicatorCalculator:
    """Calculate various technical indicators from OHLCV data."""
    
    def __init__(self):
        self.logger = TradingLogger("technical_indicators")
    
    def calculate_all(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: TimeFrame,
    ) -> Optional[TechnicalIndicators]:
        """Calculate all technical indicators for the given data."""
        if df.empty or len(df) < 55:  # Need minimum data for longest EMA
            return None
        
        try:
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            # Calculate all indicators
            indicators = {}
            
            # Moving averages
            indicators.update(self._calculate_moving_averages(df))
            
            # Momentum indicators
            indicators.update(self._calculate_momentum_indicators(df))
            
            # Volatility indicators
            indicators.update(self._calculate_volatility_indicators(df))
            
            # Volume indicators
            indicators.update(self._calculate_volume_indicators(df))
            
            # Volume profile and liquidity (temporarily disabled for debugging)
            # indicators.update(self._calculate_volume_profile(df))
            # indicators.update(self._calculate_liquidity_zones(df))
            
            # Trend indicators
            indicators.update(self._calculate_trend_indicators(df))
            
            # Support/Resistance levels
            indicators.update(self._calculate_support_resistance(df))
            
            # Swing points analysis (algorithm requirement)
            indicators.update(self._identify_swing_points(df))
            
            # Z-score volume (algorithm requirement)
            indicators.update(self._calculate_z_score_volume(df))
            
            # VWAP with bands (enhanced version)
            indicators.update(self._calculate_vwap_with_bands(df))
            
            # Create TechnicalIndicators object with latest values - filter to known fields
            known_fields = {
                'ema_8', 'ema_13', 'ema_21', 'ema_34', 'ema_55', 'sma_20', 'hull_ma', 'vwma',
                'rsi', 'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d', 'williams_r', 'roc', 'mfi',
                'atr', 'bb_upper', 'bb_middle', 'bb_lower', 'keltner_upper', 'keltner_middle', 'keltner_lower',
                'vwap', 'vwap_std1_upper', 'vwap_std1_lower', 'vwap_std2_upper', 'vwap_std2_lower', 'volume_sma', 'volume_ratio',
                'adx', 'supertrend', 'supertrend_direction', 'parabolic_sar',
                'pivot_point', 'resistance_1', 'resistance_2', 'support_1', 'support_2'
            }
            
            filtered_indicators = {}
            for k, v in indicators.items():
                if k in known_fields:
                    filtered_indicators[k] = v.iloc[-1] if hasattr(v, 'iloc') else v
            
            latest_indicators = TechnicalIndicators(
                symbol=symbol,
                timestamp=df.index[-1],
                timeframe=timeframe,
                **filtered_indicators
            )
            
            return latest_indicators
        
        except Exception as e:
            self.logger.log_error(
                error_type="indicator_calculation_failed",
                component="technical_indicators",
                error_message=str(e),
                details={"symbol": symbol, "timeframe": timeframe.value, "data_length": len(df)},
            )
            return None
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate moving averages."""
        close = df['close']
        volume = df['volume']
        
        indicators = {}
        
        # Exponential Moving Averages
        indicators['ema_8'] = close.ewm(span=8).mean()
        indicators['ema_13'] = close.ewm(span=13).mean()
        indicators['ema_21'] = close.ewm(span=21).mean()
        indicators['ema_34'] = close.ewm(span=34).mean()
        indicators['ema_55'] = close.ewm(span=55).mean()
        
        # Simple Moving Average
        indicators['sma_20'] = close.rolling(window=20).mean()
        
        # Hull Moving Average
        indicators['hull_ma'] = self._hull_ma(close, period=21)
        
        # Volume Weighted Moving Average
        indicators['vwma'] = self._vwma(close, volume, period=20)
        
        return indicators
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate momentum indicators."""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        indicators = {}
        
        # RSI
        indicators['rsi'] = self._rsi(close, period=14)
        
        # MACD
        macd_line, macd_signal, macd_hist = self._macd(close)
        indicators['macd'] = macd_line
        indicators['macd_signal'] = macd_signal
        indicators['macd_hist'] = macd_hist
        
        # Stochastic
        stoch_k, stoch_d = self._stochastic(high, low, close)
        indicators['stoch_k'] = stoch_k
        indicators['stoch_d'] = stoch_d
        
        # Williams %R
        indicators['williams_r'] = self._williams_r(high, low, close, period=14)
        
        # Rate of Change
        indicators['roc'] = self._roc(close, period=12)
        
        # Money Flow Index
        indicators['mfi'] = self._mfi(high, low, close, volume, period=14)
        
        return indicators
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volatility indicators."""
        close = df['close']
        high = df['high']
        low = df['low']
        
        indicators = {}
        
        # Average True Range
        indicators['atr'] = self._atr(high, low, close, period=14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(close, period=20, std=2)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        # Keltner Channels
        kc_upper, kc_middle, kc_lower = self._keltner_channels(high, low, close, period=20)
        indicators['keltner_upper'] = kc_upper
        indicators['keltner_middle'] = kc_middle
        indicators['keltner_lower'] = kc_lower
        
        return indicators
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume indicators."""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        indicators = {}
        
        # VWAP with standard deviations
        vwap, vwap_std = self._vwap_with_std(high, low, close, volume)
        indicators['vwap'] = vwap
        indicators['vwap_std1_upper'] = vwap + vwap_std
        indicators['vwap_std1_lower'] = vwap - vwap_std
        indicators['vwap_std2_upper'] = vwap + (2 * vwap_std)
        indicators['vwap_std2_lower'] = vwap - (2 * vwap_std)
        
        # Volume indicators
        indicators['volume_sma'] = volume.rolling(window=20).mean()
        indicators['volume_ratio'] = volume / indicators['volume_sma']
        
        return indicators
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate trend indicators."""
        close = df['close']
        high = df['high']
        low = df['low']
        
        indicators = {}
        
        # ADX (Average Directional Index)
        indicators['adx'] = self._adx(high, low, close, period=14)
        
        # SuperTrend
        supertrend, supertrend_direction = self._supertrend(high, low, close, period=10, factor=3)
        indicators['supertrend'] = supertrend
        indicators['supertrend_direction'] = supertrend_direction
        
        # Parabolic SAR
        indicators['parabolic_sar'] = self._parabolic_sar(high, low, close)
        
        return indicators
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate support/resistance levels."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        indicators = {}
        
        # Pivot Points (Classical)
        pivot_point = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
        indicators['pivot_point'] = pivot_point
        indicators['resistance_1'] = 2 * pivot_point - low.iloc[-1]
        indicators['support_1'] = 2 * pivot_point - high.iloc[-1]
        indicators['resistance_2'] = pivot_point + (high.iloc[-1] - low.iloc[-1])
        indicators['support_2'] = pivot_point - (high.iloc[-1] - low.iloc[-1])
        
        return indicators
    
    # Individual indicator calculations
    
    def _rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _macd(self, close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_hist = macd_line - macd_signal
        return macd_line.fillna(0), macd_signal.fillna(0), macd_hist.fillna(0)
    
    def _stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator."""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        stoch_d = stoch_k.rolling(window=3).mean()
        return stoch_k.fillna(50), stoch_d.fillna(50)
    
    def _williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r.fillna(-50)
    
    def _roc(self, close: pd.Series, period: int = 12) -> pd.Series:
        """Calculate Rate of Change."""
        roc = ((close - close.shift(period)) / close.shift(period)) * 100
        return roc.fillna(0)
    
    def _mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
        
        money_flow_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_flow_ratio))
        return mfi.fillna(50)
    
    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = high - low
        high_close_prev = (high - close.shift(1)).abs()
        low_close_prev = (low - close.shift(1)).abs()
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr.fillna(0)
    
    def _bollinger_bands(self, close: pd.Series, period: int = 20, std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = close.rolling(window=period).mean()
        std_dev = close.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper.fillna(0), middle.fillna(0), lower.fillna(0)
    
    def _keltner_channels(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Keltner Channels."""
        middle = close.ewm(span=period).mean()
        atr = self._atr(high, low, close, period)
        upper = middle + (2 * atr)
        lower = middle - (2 * atr)
        return upper.fillna(0), middle.fillna(0), lower.fillna(0)
    
    def _vwap_with_std(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate VWAP with standard deviation."""
        typical_price = (high + low + close) / 3
        
        # Calculate cumulative values for VWAP
        cum_vol = volume.cumsum()
        cum_vol_price = (typical_price * volume).cumsum()
        vwap = cum_vol_price / cum_vol
        
        # Calculate VWAP standard deviation
        squared_diff = ((typical_price - vwap) ** 2) * volume
        cum_squared_diff = squared_diff.cumsum()
        vwap_variance = cum_squared_diff / cum_vol
        vwap_std = np.sqrt(vwap_variance)
        
        return vwap.fillna(0), vwap_std.fillna(0)
    
    def _adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index)."""
        # Calculate True Range
        tr = self._atr(high, low, close, 1)
        
        # Calculate Directional Movement
        dm_plus = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
        dm_minus = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
        
        dm_plus = pd.Series(dm_plus, index=high.index)
        dm_minus = pd.Series(dm_minus, index=low.index)
        
        # Smooth the values
        tr_smooth = tr.rolling(window=period).mean()
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()
        
        # Calculate DI+ and DI-
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # Calculate DX and ADX
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return adx.fillna(0)
    
    def _supertrend(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, factor: float = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate SuperTrend."""
        hl2 = (high + low) / 2
        atr = self._atr(high, low, close, period)
        
        upper_band = hl2 + (factor * atr)
        lower_band = hl2 - (factor * atr)
        
        # Initialize arrays
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)
        
        for i in range(len(close)):
            if i == 0:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                if close.iloc[i] > supertrend.iloc[i-1]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
                elif close.iloc[i] < supertrend.iloc[i-1]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = supertrend.iloc[i-1]
                    direction.iloc[i] = direction.iloc[i-1]
        
        return supertrend.fillna(0), direction.fillna(1)
    
    def _parabolic_sar(self, high: pd.Series, low: pd.Series, close: pd.Series, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR."""
        length = len(close)
        psar = pd.Series(index=close.index, dtype=float)
        bull = True
        af = af_start
        ep = low.iloc[0]
        hp = high.iloc[0]
        lp = low.iloc[0]
        
        for i in range(length):
            if i == 0:
                psar.iloc[i] = close.iloc[i]
                continue
            
            if bull:
                psar.iloc[i] = psar.iloc[i-1] + af * (hp - psar.iloc[i-1])
            else:
                psar.iloc[i] = psar.iloc[i-1] + af * (lp - psar.iloc[i-1])
            
            reverse = False
            
            if bull:
                if low.iloc[i] <= psar.iloc[i]:
                    bull = False
                    reverse = True
                    psar.iloc[i] = hp
                    lp = low.iloc[i]
                    af = af_start
            else:
                if high.iloc[i] >= psar.iloc[i]:
                    bull = True
                    reverse = True
                    psar.iloc[i] = lp
                    hp = high.iloc[i]
                    af = af_start
            
            if not reverse:
                if bull:
                    if high.iloc[i] > hp:
                        hp = high.iloc[i]
                        af = min(af + af_increment, af_max)
                else:
                    if low.iloc[i] < lp:
                        lp = low.iloc[i]
                        af = min(af + af_increment, af_max)
        
        return psar.fillna(0)
    
    def _hull_ma(self, close: pd.Series, period: int = 21) -> pd.Series:
        """Calculate Hull Moving Average."""
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = close.rolling(window=half_period).apply(lambda x: np.average(x, weights=np.arange(1, half_period + 1)), raw=True)
        wma_full = close.rolling(window=period).apply(lambda x: np.average(x, weights=np.arange(1, period + 1)), raw=True)
        
        hull_ma_raw = 2 * wma_half - wma_full
        hull_ma = hull_ma_raw.rolling(window=sqrt_period).apply(lambda x: np.average(x, weights=np.arange(1, sqrt_period + 1)), raw=True)
        
        return hull_ma.fillna(0)
    
    def _vwma(self, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Volume Weighted Moving Average."""
        vwma = (close * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        return vwma.fillna(0)
    
    def _identify_swing_points(self, df: pd.DataFrame, window: int = 5) -> Dict[str, Union[float, list]]:
        """
        Identify swing highs and lows for structure analysis
        Required by algorithm for SetupDetector
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        swing_points = []
        
        # Find swing highs
        for i in range(window, len(df) - window):
            is_swing_high = all(high.iloc[i] > high.iloc[i-j] for j in range(1, window+1)) and \
                          all(high.iloc[i] > high.iloc[i+j] for j in range(1, window+1))
            
            if is_swing_high:
                swing_points.append({
                    'type': 'high',
                    'price': high.iloc[i],
                    'index': i,
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
        
        # Find swing lows
        for i in range(window, len(df) - window):
            is_swing_low = all(low.iloc[i] < low.iloc[i-j] for j in range(1, window+1)) and \
                         all(low.iloc[i] < low.iloc[i+j] for j in range(1, window+1))
            
            if is_swing_low:
                swing_points.append({
                    'type': 'low',
                    'price': low.iloc[i],
                    'index': i,
                    'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
                })
        
        # Sort by index
        swing_points.sort(key=lambda x: x['index'])
        
        # Calculate nearest swing levels to current price
        current_price = close.iloc[-1]
        swing_highs = [sp['price'] for sp in swing_points if sp['type'] == 'high']
        swing_lows = [sp['price'] for sp in swing_points if sp['type'] == 'low']
        
        nearest_resistance = min([h for h in swing_highs if h > current_price], default=current_price * 1.02)
        nearest_support = max([l for l in swing_lows if l < current_price], default=current_price * 0.98)
        
        return {
            'swing_points': swing_points,
            'nearest_swing_high': nearest_resistance,
            'nearest_swing_low': nearest_support,
            'swing_high_count': len([sp for sp in swing_points if sp['type'] == 'high']),
            'swing_low_count': len([sp for sp in swing_points if sp['type'] == 'low'])
        }
    
    def _calculate_z_score_volume(self, df: pd.DataFrame, window: int = 50) -> Dict[str, float]:
        """
        Calculate volume z-score as required by algorithm
        Used in entry filters and regime classification
        """
        volume = df['volume']
        
        if len(volume) < window:
            return {'zvol': 0.0, 'volume_percentile': 0.5}
        
        # Calculate rolling statistics
        vol_mean = volume.rolling(window=window).mean()
        vol_std = volume.rolling(window=window).std()
        
        # Calculate z-score
        current_volume = volume.iloc[-1]
        mean_vol = vol_mean.iloc[-1]
        std_vol = vol_std.iloc[-1]
        
        if std_vol > 0:
            z_score = (current_volume - mean_vol) / std_vol
        else:
            z_score = 0.0
        
        # Calculate volume percentile
        recent_volumes = volume.tail(window).values
        percentile = (recent_volumes < current_volume).sum() / len(recent_volumes)
        
        return {
            'zvol': z_score,
            'volume_percentile': percentile,
            'volume_mean': mean_vol,
            'volume_std': std_vol
        }
    
    def _calculate_vwap_with_bands(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Enhanced VWAP with multiple standard deviation bands
        Required by algorithm for mean reversion setups
        """
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Typical price
        typical_price = (high + low + close) / 3
        
        # Calculate cumulative values
        cum_vol = volume.cumsum()
        cum_vol_price = (typical_price * volume).cumsum()
        vwap = cum_vol_price / cum_vol
        
        # Calculate VWAP variance and standard deviation
        price_diff_sq = ((typical_price - vwap) ** 2) * volume
        cum_price_diff_sq = price_diff_sq.cumsum()
        vwap_variance = cum_price_diff_sq / cum_vol
        vwap_std = np.sqrt(vwap_variance)
        
        # Create bands
        indicators = {
            'vwap': vwap.fillna(method='ffill'),
            'vwap_std': vwap_std.fillna(0),
            'vwap_upper_1sigma': (vwap + vwap_std).fillna(method='ffill'),
            'vwap_lower_1sigma': (vwap - vwap_std).fillna(method='ffill'),
            'vwap_upper_2sigma': (vwap + 2 * vwap_std).fillna(method='ffill'),
            'vwap_lower_2sigma': (vwap - 2 * vwap_std).fillna(method='ffill'),
            'vwap_upper_3sigma': (vwap + 3 * vwap_std).fillna(method='ffill'),
            'vwap_lower_3sigma': (vwap - 3 * vwap_std).fillna(method='ffill')
        }
        
        return indicators


class FeatureEngine:
    """Advanced feature engineering for ML models."""
    
    def __init__(self):
        self.indicator_calc = TechnicalIndicatorCalculator()
        self.logger = TradingLogger("feature_engine")
    
    def generate_features(
        self,
        market_data: dict,
        symbol: str,
        primary_timeframe: TimeFrame = TimeFrame.M5,
    ) -> Optional[Dict[str, float]]:
        """Generate comprehensive feature set for ML models."""
        try:
            if symbol not in market_data or primary_timeframe not in market_data[symbol]:
                return None
            
            df_primary = market_data[symbol][primary_timeframe]
            if df_primary.empty or len(df_primary) < 55:
                return None
            
            features = {}
            
            # Basic price features
            features.update(self._price_features(df_primary))
            
            # Technical indicator features
            features.update(self._technical_features(df_primary, symbol, primary_timeframe))
            
            # Multi-timeframe features
            features.update(self._multi_timeframe_features(market_data, symbol))
            
            # Volume features
            features.update(self._volume_features(df_primary))
            
            # Volatility features
            features.update(self._volatility_features(df_primary))
            
            # Pattern features
            features.update(self._pattern_features(df_primary))
            
            # Time-based features
            features.update(self._time_features(df_primary))
            
            return features
        
        except Exception as e:
            self.logger.log_error(
                error_type="feature_generation_failed",
                component="feature_engine",
                error_message=str(e),
                details={"symbol": symbol, "timeframe": primary_timeframe.value},
            )
            return None
    
    def _price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate price-based features."""
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        
        features = {}
        
        # Current price relative to recent highs/lows
        features['price_vs_5_high'] = close.iloc[-1] / high.tail(5).max() - 1
        features['price_vs_5_low'] = close.iloc[-1] / low.tail(5).min() - 1
        features['price_vs_20_high'] = close.iloc[-1] / high.tail(20).max() - 1
        features['price_vs_20_low'] = close.iloc[-1] / low.tail(20).min() - 1
        
        # Returns
        features['return_1'] = close.pct_change(1).iloc[-1]
        features['return_3'] = close.pct_change(3).iloc[-1]
        features['return_5'] = close.pct_change(5).iloc[-1]
        features['return_10'] = close.pct_change(10).iloc[-1]
        
        # Candle patterns
        body = abs(close - open_price)
        upper_shadow = high - np.maximum(close, open_price)
        lower_shadow = np.minimum(close, open_price) - low
        
        features['body_ratio'] = (body / (high - low)).iloc[-1]
        features['upper_shadow_ratio'] = (upper_shadow / (high - low)).iloc[-1]
        features['lower_shadow_ratio'] = (lower_shadow / (high - low)).iloc[-1]
        
        return features
    
    def _technical_features(self, df: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> Dict[str, float]:
        """Generate technical indicator features."""
        indicators = self.indicator_calc.calculate_all(df, symbol, timeframe)
        if not indicators:
            return {}
        
        features = {}
        close = df['close'].iloc[-1]
        
        # RSI features
        features['rsi'] = indicators.rsi
        features['rsi_oversold'] = 1 if indicators.rsi < 30 else 0
        features['rsi_overbought'] = 1 if indicators.rsi > 70 else 0
        
        # MACD features
        features['macd'] = indicators.macd
        features['macd_signal'] = indicators.macd_signal
        features['macd_hist'] = indicators.macd_hist
        features['macd_bullish'] = 1 if indicators.macd > indicators.macd_signal else 0
        
        # EMA features
        features['ema_8'] = indicators.ema_8
        features['ema_21'] = indicators.ema_21
        features['ema_55'] = indicators.ema_55
        features['price_vs_ema_8'] = close / indicators.ema_8 - 1
        features['price_vs_ema_21'] = close / indicators.ema_21 - 1
        features['price_vs_ema_55'] = close / indicators.ema_55 - 1
        
        # EMA alignment
        features['ema_bullish_alignment'] = 1 if (
            indicators.ema_8 > indicators.ema_21 > indicators.ema_55
        ) else 0
        features['ema_bearish_alignment'] = 1 if (
            indicators.ema_8 < indicators.ema_21 < indicators.ema_55
        ) else 0
        
        # Bollinger Bands
        features['bb_position'] = (close - indicators.bb_lower) / (indicators.bb_upper - indicators.bb_lower)
        features['bb_squeeze'] = 1 if (indicators.bb_upper - indicators.bb_lower) / close < 0.04 else 0
        
        # VWAP
        features['price_vs_vwap'] = close / indicators.vwap - 1
        features['vwap_std1_position'] = (close - indicators.vwap) / (indicators.vwap_std1_upper - indicators.vwap)
        
        # ADX
        features['adx'] = indicators.adx
        features['strong_trend'] = 1 if indicators.adx > 25 else 0
        
        # SuperTrend
        features['supertrend_bullish'] = 1 if indicators.supertrend_direction > 0 else 0
        features['price_vs_supertrend'] = close / indicators.supertrend - 1
        
        return features
    
    def _multi_timeframe_features(self, market_data: dict, symbol: str) -> Dict[str, float]:
        """Generate multi-timeframe features."""
        features = {}
        
        if symbol not in market_data:
            return features
        
        timeframes = [TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.H1]
        
        for tf in timeframes:
            if tf in market_data[symbol]:
                df = market_data[symbol][tf]
                if len(df) >= 20:
                    close = df['close']
                    
                    # Trend direction on different timeframes
                    ema_8 = close.ewm(span=8).mean()
                    ema_21 = close.ewm(span=21).mean()
                    
                    features[f'{tf.value}_trend_bullish'] = 1 if ema_8.iloc[-1] > ema_21.iloc[-1] else 0
                    features[f'{tf.value}_momentum'] = close.pct_change(5).iloc[-1]
        
        return features
    
    def _volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate volume-based features."""
        volume = df['volume']
        close = df['close']
        
        features = {}
        
        # Volume ratios
        vol_sma_20 = volume.rolling(20).mean()
        features['volume_ratio'] = volume.iloc[-1] / vol_sma_20.iloc[-1] if vol_sma_20.iloc[-1] > 0 else 1
        
        # Volume trend
        features['volume_trend'] = volume.tail(5).mean() / volume.tail(20).mean() - 1
        
        # Volume-price divergence
        price_change = close.pct_change(5).iloc[-1]
        volume_change = (volume.tail(5).mean() / volume.tail(10).mean()) - 1
        
        if price_change > 0 and volume_change < 0:
            features['volume_divergence'] = -1  # Bearish divergence
        elif price_change < 0 and volume_change > 0:
            features['volume_divergence'] = 1   # Bullish divergence
        else:
            features['volume_divergence'] = 0   # No divergence
        
        return features
    
    def _volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate volatility features."""
        close = df['close']
        high = df['high']
        low = df['low']
        
        features = {}
        
        # ATR-based features
        atr = self.indicator_calc._atr(high, low, close, 14)
        features['atr_ratio'] = atr.iloc[-1] / close.iloc[-1]
        
        # Volatility percentile
        returns = close.pct_change().tail(20)
        current_vol = returns.std()
        vol_percentile = (returns.rolling(20).std() <= current_vol).sum() / 20
        features['volatility_percentile'] = vol_percentile
        
        # Range features
        daily_range = (high - low) / close
        features['daily_range'] = daily_range.iloc[-1]
        features['range_vs_avg'] = daily_range.iloc[-1] / daily_range.tail(20).mean() - 1
        
        return features
    
    def _pattern_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate pattern-based features."""
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        
        features = {}
        
        # Support/Resistance breaks
        recent_high = high.tail(20).max()
        recent_low = low.tail(20).min()
        
        features['near_resistance'] = 1 if close.iloc[-1] >= recent_high * 0.98 else 0
        features['near_support'] = 1 if close.iloc[-1] <= recent_low * 1.02 else 0
        
        # Consecutive candles
        green_candles = (close > open_price).astype(int)
        features['consecutive_green'] = self._count_consecutive(green_candles)
        
        red_candles = (close < open_price).astype(int)
        features['consecutive_red'] = self._count_consecutive(red_candles)
        
        # Higher highs / Lower lows
        features['making_higher_highs'] = 1 if high.iloc[-1] > high.tail(5).max() else 0
        features['making_lower_lows'] = 1 if low.iloc[-1] < low.tail(5).min() else 0
        
        return features
    
    def _time_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate time-based features."""
        features = {}
        
        if df.index[-1]:
            timestamp = df.index[-1]
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Hour of day (UTC)
            features['hour'] = hour
            features['is_us_session'] = 1 if 13 <= hour <= 22 else 0
            features['is_asia_session'] = 1 if 1 <= hour <= 9 else 0
            features['is_europe_session'] = 1 if 7 <= hour <= 16 else 0
            
            # Day of week
            features['day_of_week'] = day_of_week
            features['is_weekend'] = 1 if day_of_week >= 5 else 0
        
        return features
    
    def _calculate_volume_profile(self, df: pd.DataFrame, bins: int = 20) -> Dict[str, Union[float, pd.Series]]:
        """Calculate volume profile with POC, VAH, VAL."""
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        indicators = {}
        
        # Calculate price range and bin size
        price_min = low.min()
        price_max = high.max()
        bin_size = (price_max - price_min) / bins
        
        if bin_size == 0:
            return {
                'poc': close.iloc[-1],
                'vah': close.iloc[-1],
                'val': close.iloc[-1],
                'volume_at_price': pd.Series(index=close.index, dtype=float).fillna(0)
            }
        
        # Create volume profile
        profile = {}
        for i in range(len(df)):
            price = close.iloc[i]
            vol = volume.iloc[i]
            bin_idx = int((price - price_min) / bin_size)
            bin_idx = min(bin_idx, bins - 1)  # Ensure within bounds
            
            if bin_idx not in profile:
                profile[bin_idx] = {'price': price_min + bin_idx * bin_size, 'volume': 0}
            profile[bin_idx]['volume'] += vol
        
        if not profile:
            return {
                'poc': close.iloc[-1],
                'vah': close.iloc[-1],
                'val': close.iloc[-1],
                'volume_at_price': pd.Series(index=close.index, dtype=float).fillna(0)
            }
        
        # Find POC (Point of Control) - highest volume bin
        poc_bin = max(profile.items(), key=lambda x: x[1]['volume'])
        poc = poc_bin[1]['price']
        
        # Calculate Value Area (70% of total volume)
        total_volume = sum([p['volume'] for p in profile.values()])
        value_area_volume = total_volume * 0.70
        
        # Sort bins by volume
        sorted_profile = sorted(profile.items(), key=lambda x: x[1]['volume'], reverse=True)
        accumulated = 0
        value_area_bins = []
        
        for bin_idx, data in sorted_profile:
            accumulated += data['volume']
            value_area_bins.append(data['price'])
            if accumulated >= value_area_volume:
                break
        
        vah = max(value_area_bins) if value_area_bins else poc  # Value Area High
        val = min(value_area_bins) if value_area_bins else poc  # Value Area Low
        
        # Create volume at price series for current prices
        volume_at_price = pd.Series(index=close.index, dtype=float)
        for i in range(len(df)):
            price = close.iloc[i]
            bin_idx = int((price - price_min) / bin_size)
            bin_idx = min(bin_idx, bins - 1)
            volume_at_price.iloc[i] = profile.get(bin_idx, {'volume': 0})['volume']
        
        indicators['poc'] = poc
        indicators['vah'] = vah
        indicators['val'] = val
        indicators['volume_at_price'] = volume_at_price.fillna(0)
        
        return indicators
    
    def _calculate_liquidity_zones(self, df: pd.DataFrame, lookback: int = 100) -> Dict[str, Union[float, list]]:
        """Identify liquidity zones from price action and volume."""
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        indicators = {}
        liquidity_pools = []
        
        # Find resistance levels (highs with multiple touches)
        for i in range(20, len(df) - 1):
            level = high.iloc[i]
            touches = 0
            volume_at_level = 0
            
            # Check for touches within tolerance
            start_idx = max(0, i - lookback)
            end_idx = min(len(df), i + lookback)
            
            for j in range(start_idx, end_idx):
                if abs(high.iloc[j] - level) / level < 0.001:  # 0.1% tolerance
                    touches += 1
                    volume_at_level += volume.iloc[j]
            
            if touches >= 3:  # At least 3 touches
                strength = touches * (volume_at_level / volume.mean())
                liquidity_pools.append({
                    'level': level,
                    'touches': touches,
                    'volume': volume_at_level,
                    'type': 'resistance',
                    'strength': strength
                })
        
        # Find support levels (lows with multiple touches)
        for i in range(20, len(df) - 1):
            level = low.iloc[i]
            touches = 0
            volume_at_level = 0
            
            start_idx = max(0, i - lookback)
            end_idx = min(len(df), i + lookback)
            
            for j in range(start_idx, end_idx):
                if abs(low.iloc[j] - level) / level < 0.001:
                    touches += 1
                    volume_at_level += volume.iloc[j]
            
            if touches >= 3:
                strength = touches * (volume_at_level / volume.mean())
                liquidity_pools.append({
                    'level': level,
                    'touches': touches,
                    'volume': volume_at_level,
                    'type': 'support',
                    'strength': strength
                })
        
        # Sort by strength and filter overlapping levels
        liquidity_pools.sort(key=lambda x: x['strength'], reverse=True)
        filtered_pools = []
        
        for pool in liquidity_pools:
            too_close = False
            for existing in filtered_pools:
                if abs(pool['level'] - existing['level']) / existing['level'] < 0.002:  # 0.2% tolerance
                    too_close = True
                    break
            if not too_close:
                filtered_pools.append(pool)
        
        # Keep top 10 strongest pools
        top_pools = filtered_pools[:10]
        
        # Calculate nearest support and resistance
        current_price = df['close'].iloc[-1]
        resistances = [p['level'] for p in top_pools if p['type'] == 'resistance' and p['level'] > current_price]
        supports = [p['level'] for p in top_pools if p['type'] == 'support' and p['level'] < current_price]
        
        nearest_resistance = min(resistances) if resistances else current_price * 1.02
        nearest_support = max(supports) if supports else current_price * 0.98
        
        indicators['liquidity_pools'] = top_pools
        indicators['nearest_resistance'] = nearest_resistance
        indicators['nearest_support'] = nearest_support
        indicators['support_strength'] = max([p['strength'] for p in top_pools if p['type'] == 'support'], default=0)
        indicators['resistance_strength'] = max([p['strength'] for p in top_pools if p['type'] == 'resistance'], default=0)
        
        return indicators
    
    def _count_consecutive(self, series: pd.Series) -> int:
        """Count consecutive 1s from the end of series."""
        count = 0
        for i in range(len(series) - 1, -1, -1):
            if series.iloc[i] == 1:
                count += 1
            else:
                break
        return count