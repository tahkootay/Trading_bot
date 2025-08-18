"""
Algorithm Constants and Configurations
Exact constants from SOL/USDT trading algorithm specification
"""

from typing import Dict, List, Any

# Time Constants (all in minutes unless specified)
TIMEFRAMES = {
    'tick': '1s',
    'micro': '1m',
    'fast': '3m', 
    'primary': '5m',
    'medium': '15m',
    'slow': '1h',
    'macro': '4h'
}

# Market Hours (UTC)
SESSIONS = {
    'asia': (1, 9),
    'europe': (7, 16),
    'us': (13, 22),
    'active_hours': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
}

# Technical Indicators Windows
INDICATOR_PARAMS = {
    'ema_ribbon': [8, 13, 21, 34, 55],
    'atr_period': 14,
    'rsi_period': 14,
    'adx_period': 14,
    'volume_window': 50,
    'correlation_windows': [30, 60, 120, 240],  # minutes
    'macd': {'fast': 12, 'slow': 26, 'signal': 9}
}

# Market Regime Thresholds
MARKET_REGIMES = {
    'low_volatility_range': {
        'adx_max': 20,
        'atr_ratio_max': 0.8,
        'volume_ratio_max': 0.7
    },
    'normal_range': {
        'adx_range': (20, 25),
        'atr_ratio_range': (0.8, 1.2)
    },
    'trending': {
        'adx_min': 25,
        'adx_max': 40,
        'ema_aligned': True
    },
    'strong_trend': {
        'adx_min': 40,
        'volume_trending': True,
        'ema_strongly_aligned': True
    },
    'volatile_choppy': {
        'adx_max': 25,
        'atr_ratio_min': 1.5,
        'whipsaw_count_min': 3
    }
}

# Entry Filters - Adaptive Based on Market Regime
ENTRY_FILTERS = {
    'default': {
        'adx_min': 25,
        'zvol_min': 0.5,
        'rsi_range': (35, 65),
        'btc_corr_max': 0.80,
        'ml_margin_min': 0.15,
        'ml_conf_min': 0.70,
        'ml_agreement_min': 0.60
    },
    'trending': {
        'adx_min': 25,
        'zvol_min': 0.4,
        'rsi_range': (30, 70),
        'btc_corr_max': 0.85,
        'ml_margin_min': 0.12,
        'ml_conf_min': 0.65,
        'ml_agreement_min': 0.55
    },
    'ranging': {
        'adx_min': 15,
        'zvol_min': 0.6,
        'rsi_range': (35, 65),
        'btc_corr_max': 0.75,
        'ml_margin_min': 0.18,
        'ml_conf_min': 0.75,
        'ml_agreement_min': 0.65
    },
    'high_volatility': {
        'adx_min': 20,
        'zvol_min': 0.8,
        'rsi_range': (40, 60),
        'btc_corr_max': 0.70,
        'ml_margin_min': 0.20,
        'ml_conf_min': 0.75,
        'ml_agreement_min': 0.70
    }
}

# Risk Management
RISK_PARAMS = {
    'position_risk_pct': 0.02,  # 2% per trade
    'daily_loss_limit': 0.03,   # 3%
    'max_corr_positions': 3,
    'max_portfolio_risk': 0.10,  # 10%
    'kelly_cap': 0.25,           # 25% of Kelly
    'series_loss_pause': 3,      # pause after 3 consecutive losses
    'pause_duration_min': 120,   # 2 hours
    
    # Drawdown Management
    'dd_levels': {
        0.05: {'position_mult': 0.5, 'conf_boost': 0},      # 5% DD
        0.08: {'position_mult': 0.3, 'conf_boost': 0.10},   # 8% DD
        0.10: {'position_mult': 0, 'conf_boost': None}      # 10% DD - stop
    },
    
    # Emergency Stops
    'extreme_1m_move': 0.05,    # 5% in 1 minute
    'black_swan_dd': 0.15,      # 15% drawdown
    'max_slippage_pct': 0.001,  # 0.1%
}

# Stop Loss and Take Profit Multipliers (ATR-based)
SL_TP_PARAMS = {
    'sl_atr_range': (1.0, 1.5),
    'tp_levels': [1.0, 1.5, 2.5],
    'tp_allocations': [0.4, 0.3, 0.2],  # 10% reserved for runner
    'trailing_activation': 1.0,  # Activate trailing after 1 ATR profit
    'profit_lock_pct': 0.70      # Lock 70% of max profit
}

# Time Stops - Adaptive by Setup Type
TIME_STOPS = {
    'scalp': 30,
    'momentum': 60,
    'breakout': 180,
    'mean_reversion': 240,
    'position': 360,
    'default': 120
}

# Liquidity Parameters
LIQUIDITY_PARAMS = {
    'min_pool_size_usd': 50000,
    'max_position_vs_volume': 0.01,  # 1% of 5m volume
    'orderbook_depth_pct': 0.002,    # 0.2% from mid price
    'max_position_vs_depth': 0.10,   # 10% of available depth
    'large_order_multiplier': 10,    # 10x average trade size
    'absorption_volume_ratio': 3.0,  # 3x average with no price move
}

# Order Execution
EXECUTION_PARAMS = {
    'entry_type': 'limit',
    'entry_offset_pct': 0.0005,  # 0.05%
    'fill_timeout_sec': 30,
    'partial_fill_min': 0.5,
    'iceberg_enabled': True,
    'iceberg_show_pct': 0.2,     # Show 20% of total
    'retry_attempts': 3,
    'retry_delay_sec': 5
}

def get_regime_entry_filters(regime: str) -> Dict[str, Any]:
    """Get entry filters for specific market regime"""
    regime_mapping = {
        'strong_trend': 'trending',
        'trending': 'trending', 
        'normal_range': 'default',
        'low_volatility_range': 'ranging',
        'volatile_choppy': 'high_volatility'
    }
    
    filter_key = regime_mapping.get(regime, 'default')
    return ENTRY_FILTERS[filter_key]

def get_regime_multipliers(regime: str) -> Dict[str, float]:
    """Get position sizing and filter multipliers for market regime"""
    multipliers = {
        'strong_trend': {
            'position_size': 1.2,
            'confidence_mult': 0.9,
            'margin_mult': 0.8,
            'agreement_mult': 0.9
        },
        'trending': {
            'position_size': 1.0,
            'confidence_mult': 0.95,
            'margin_mult': 0.9,
            'agreement_mult': 0.95
        },
        'normal_range': {
            'position_size': 0.8,
            'confidence_mult': 1.0,
            'margin_mult': 1.0,
            'agreement_mult': 1.0
        },
        'low_volatility_range': {
            'position_size': 0.6,
            'confidence_mult': 1.1,
            'margin_mult': 1.2,
            'agreement_mult': 1.1
        },
        'volatile_choppy': {
            'position_size': 0.4,
            'confidence_mult': 1.2,
            'margin_mult': 1.3,
            'agreement_mult': 1.2
        }
    }
    
    return multipliers.get(regime, multipliers['normal_range'])

def get_time_stop_minutes(setup_type: str) -> int:
    """Get time stop in minutes for setup type"""
    return TIME_STOPS.get(setup_type, TIME_STOPS['default'])

def validate_trading_session() -> bool:
    """Check if current time is within active trading hours"""
    from datetime import datetime, timezone
    current_hour = datetime.now(timezone.utc).hour
    return current_hour in SESSIONS['active_hours']

def get_sl_tp_params_for_regime(regime: str) -> Dict[str, Any]:
    """Get stop loss and take profit parameters adjusted for regime"""
    base_params = SL_TP_PARAMS.copy()
    
    # Adjust SL range based on regime
    if regime == 'strong_trend':
        base_params['sl_atr_range'] = (0.8, 1.2)  # Tighter stops in strong trends
    elif regime == 'volatile_choppy':
        base_params['sl_atr_range'] = (1.5, 2.0)  # Wider stops in choppy markets
    
    return base_params