"""
Entry Filters System for SOL/USDT Trading Algorithm

Implements adaptive entry filters based on market regime with exact
algorithm specification thresholds and validation logic.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import logging

class FilterResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"

@dataclass
class FilterCheck:
    """Individual filter check result"""
    name: str
    result: FilterResult
    value: float
    threshold: float
    weight: float
    description: str

@dataclass
class EntryFilterResult:
    """Complete entry filter result"""
    overall_pass: bool
    total_score: float
    required_score: float
    individual_checks: List[FilterCheck]
    failed_critical: List[str]
    warnings: List[str]
    regime_adjustments: Dict[str, float]

class EntryFilterSystem:
    """
    Centralized entry filter system implementing algorithm specification
    
    Filters all potential trades through multiple layers:
    - Session filters (time-based)
    - Technical filters (indicators)
    - Market structure filters
    - ML filters
    - Risk filters
    - Regime-adaptive thresholds
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Algorithm constants from specification
        self.BASE_FILTERS = {
            # Session filters
            'session_active': {'weight': 1.0, 'critical': True},
            
            # Technical filters
            'trend_strength': {'threshold': 20, 'weight': 0.8, 'critical': False},  # ADX >= 20
            'momentum_range': {'min': 35, 'max': 65, 'weight': 0.6, 'critical': False},  # RSI 35-65
            'volume_confirmation': {'threshold': 0.5, 'weight': 0.7, 'critical': False},  # ZVol >= 0.5
            'trend_alignment': {'weight': 0.9, 'critical': True},  # EMA alignment
            
            # Market structure filters
            'price_vs_vwap': {'weight': 0.5, 'critical': False},
            'distance_to_structure': {'min_pct': 0.002, 'weight': 0.4, 'critical': False},  # 0.2% min
            
            # ML filters
            'ml_confidence': {'threshold': 0.70, 'weight': 1.0, 'critical': True},
            'ml_margin': {'threshold': 0.15, 'weight': 1.0, 'critical': True},
            'ml_agreement': {'threshold': 0.60, 'weight': 0.8, 'critical': True},
            
            # Risk filters
            'correlation_check': {'max_correlation': 0.8, 'weight': 0.6, 'critical': False},
            'position_limit': {'weight': 1.0, 'critical': True},
            'volatility_regime': {'weight': 0.5, 'critical': False}
        }
        
        # Regime-specific adjustments as per algorithm
        self.REGIME_ADJUSTMENTS = {
            'strong_trend': {
                'trend_strength': 0.9,      # Lower ADX required
                'ml_confidence': 0.9,       # Lower confidence required
                'ml_margin': 0.8,          # Lower margin required
                'volume_confirmation': 0.8  # Lower volume required
            },
            'trending': {
                'trend_strength': 0.95,
                'ml_confidence': 0.95,
                'ml_margin': 0.9,
                'volume_confirmation': 0.9
            },
            'normal_range': {
                'trend_strength': 1.0,
                'ml_confidence': 1.0,
                'ml_margin': 1.0,
                'volume_confirmation': 1.0
            },
            'low_volatility_range': {
                'trend_strength': 1.1,
                'ml_confidence': 1.1,
                'ml_margin': 1.2,
                'volume_confirmation': 1.1
            },
            'volatile_choppy': {
                'trend_strength': 1.2,
                'ml_confidence': 1.2,
                'ml_margin': 1.3,
                'volume_confirmation': 1.2
            }
        }
        
        # Active hours as per algorithm specification
        self.ACTIVE_SESSIONS = {
            'asia': (1, 9),
            'europe': (7, 16),
            'us': (13, 22),
            'active_hours': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        }
        
    def check_entry_filters(
        self, 
        setup: Dict, 
        market_data: Dict, 
        ml_prediction: Dict, 
        regime: str,
        current_positions: List[Dict] = None
    ) -> EntryFilterResult:
        """
        Comprehensive entry filter check as per algorithm specification
        
        Args:
            setup: Trading setup details
            market_data: Current market state
            ml_prediction: ML prediction results
            regime: Current market regime
            current_positions: List of current open positions
            
        Returns:
            EntryFilterResult with detailed breakdown
        """
        try:
            current_positions = current_positions or []
            checks = []
            critical_failures = []
            warnings = []
            
            # Get regime adjustments
            adjustments = self.REGIME_ADJUSTMENTS.get(regime, self.REGIME_ADJUSTMENTS['normal_range'])
            
            # 1. Session filters
            session_check = self._check_session_filters()
            checks.append(session_check)
            if session_check.result == FilterResult.FAIL and self.BASE_FILTERS['session_active']['critical']:
                critical_failures.append(session_check.name)
            
            # 2. Technical filters
            tech_checks = self._check_technical_filters(market_data, adjustments)
            checks.extend(tech_checks)
            
            # 3. Market structure filters
            structure_checks = self._check_structure_filters(setup, market_data, adjustments)
            checks.extend(structure_checks)
            
            # 4. ML filters
            ml_checks = self._check_ml_filters(ml_prediction, adjustments)
            checks.extend(ml_checks)
            for check in ml_checks:
                if check.result == FilterResult.FAIL and self.BASE_FILTERS.get(check.name, {}).get('critical', False):
                    critical_failures.append(check.name)
            
            # 5. Risk filters
            risk_checks = self._check_risk_filters(setup, current_positions, market_data)
            checks.extend(risk_checks)
            for check in risk_checks:
                if check.result == FilterResult.FAIL and self.BASE_FILTERS.get(check.name, {}).get('critical', False):
                    critical_failures.append(check.name)
            
            # Calculate total score
            total_score = sum(
                check.weight for check in checks 
                if check.result == FilterResult.PASS
            )
            max_score = sum(check.weight for check in checks)
            required_score = max_score * 0.7  # 70% of filters must pass
            
            # Overall pass/fail
            overall_pass = (
                len(critical_failures) == 0 and 
                total_score >= required_score
            )
            
            # Collect warnings
            for check in checks:
                if check.result == FilterResult.WARNING:
                    warnings.append(f"{check.name}: {check.description}")
            
            return EntryFilterResult(
                overall_pass=overall_pass,
                total_score=total_score,
                required_score=required_score,
                individual_checks=checks,
                failed_critical=critical_failures,
                warnings=warnings,
                regime_adjustments=adjustments
            )
            
        except Exception as e:
            self.logger.error(f"Error in entry filter check: {e}")
            return EntryFilterResult(
                overall_pass=False,
                total_score=0,
                required_score=1,
                individual_checks=[],
                failed_critical=["system_error"],
                warnings=[f"Filter system error: {str(e)}"],
                regime_adjustments={}
            )
    
    def _check_session_filters(self) -> FilterCheck:
        """Check if current time is within active trading sessions"""
        try:
            current_hour = datetime.now(timezone.utc).hour
            is_active = current_hour in self.ACTIVE_SESSIONS['active_hours']
            
            return FilterCheck(
                name="session_active",
                result=FilterResult.PASS if is_active else FilterResult.FAIL,
                value=float(is_active),
                threshold=1.0,
                weight=self.BASE_FILTERS['session_active']['weight'],
                description=f"Trading session active (hour {current_hour})" if is_active else f"Outside active hours (hour {current_hour})"
            )
            
        except Exception as e:
            return FilterCheck(
                name="session_active",
                result=FilterResult.FAIL,
                value=0.0,
                threshold=1.0,
                weight=self.BASE_FILTERS['session_active']['weight'],
                description=f"Session check error: {e}"
            )
    
    def _check_technical_filters(self, market_data: Dict, adjustments: Dict) -> List[FilterCheck]:
        """Check technical indicator filters"""
        checks = []
        
        try:
            # Trend strength (ADX)
            adx = market_data.get('adx', 0)
            adx_threshold = self.BASE_FILTERS['trend_strength']['threshold'] * adjustments.get('trend_strength', 1.0)
            
            checks.append(FilterCheck(
                name="trend_strength",
                result=FilterResult.PASS if adx >= adx_threshold else FilterResult.FAIL,
                value=adx,
                threshold=adx_threshold,
                weight=self.BASE_FILTERS['trend_strength']['weight'],
                description=f"ADX {adx:.1f} vs required {adx_threshold:.1f}"
            ))
            
            # Momentum range (RSI)
            rsi = market_data.get('rsi', 50)
            rsi_min = self.BASE_FILTERS['momentum_range']['min']
            rsi_max = self.BASE_FILTERS['momentum_range']['max']
            rsi_ok = rsi_min <= rsi <= rsi_max
            
            checks.append(FilterCheck(
                name="momentum_range",
                result=FilterResult.PASS if rsi_ok else FilterResult.WARNING,  # Warning not failure
                value=rsi,
                threshold=(rsi_min + rsi_max) / 2,
                weight=self.BASE_FILTERS['momentum_range']['weight'],
                description=f"RSI {rsi:.1f} in range [{rsi_min}-{rsi_max}]" if rsi_ok else f"RSI {rsi:.1f} outside range"
            ))
            
            # Volume confirmation
            zvol = market_data.get('zvol', 0)
            vol_threshold = self.BASE_FILTERS['volume_confirmation']['threshold'] * adjustments.get('volume_confirmation', 1.0)
            
            checks.append(FilterCheck(
                name="volume_confirmation",
                result=FilterResult.PASS if zvol >= vol_threshold else FilterResult.FAIL,
                value=zvol,
                threshold=vol_threshold,
                weight=self.BASE_FILTERS['volume_confirmation']['weight'],
                description=f"Volume z-score {zvol:.2f} vs required {vol_threshold:.2f}"
            ))
            
            # Trend alignment (EMAs)
            ema_aligned = self._check_ema_alignment(market_data)
            checks.append(FilterCheck(
                name="trend_alignment",
                result=FilterResult.PASS if ema_aligned else FilterResult.FAIL,
                value=float(ema_aligned),
                threshold=1.0,
                weight=self.BASE_FILTERS['trend_alignment']['weight'],
                description="EMA trend aligned" if ema_aligned else "EMA trend not aligned"
            ))
            
        except Exception as e:
            self.logger.error(f"Error in technical filters: {e}")
            
        return checks
    
    def _check_structure_filters(self, setup: Dict, market_data: Dict, adjustments: Dict) -> List[FilterCheck]:
        """Check market structure filters"""
        checks = []
        
        try:
            # Price vs VWAP
            close_price = market_data.get('close', 0)
            if isinstance(close_price, list):
                close_price = close_price[-1] if close_price else 0
            
            vwap = market_data.get('vwap', close_price)
            direction = setup.get('direction', 'long')
            
            vwap_ok = (
                (direction == 'long' and close_price >= vwap) or
                (direction == 'short' and close_price <= vwap)
            )
            
            checks.append(FilterCheck(
                name="price_vs_vwap",
                result=FilterResult.PASS if vwap_ok else FilterResult.WARNING,
                value=close_price / vwap if vwap > 0 else 1.0,
                threshold=1.0,
                weight=self.BASE_FILTERS['price_vs_vwap']['weight'],
                description=f"Price {close_price:.2f} vs VWAP {vwap:.2f} for {direction}"
            ))
            
            # Distance to structure
            distance_check = self._check_distance_to_structure(setup, market_data)
            checks.append(distance_check)
            
        except Exception as e:
            self.logger.error(f"Error in structure filters: {e}")
            
        return checks
    
    def _check_ml_filters(self, ml_prediction: Dict, adjustments: Dict) -> List[FilterCheck]:
        """Check ML-based filters"""
        checks = []
        
        try:
            # ML confidence
            confidence = ml_prediction.get('confidence', 0)
            conf_threshold = self.BASE_FILTERS['ml_confidence']['threshold'] * adjustments.get('ml_confidence', 1.0)
            
            checks.append(FilterCheck(
                name="ml_confidence",
                result=FilterResult.PASS if confidence >= conf_threshold else FilterResult.FAIL,
                value=confidence,
                threshold=conf_threshold,
                weight=self.BASE_FILTERS['ml_confidence']['weight'],
                description=f"ML confidence {confidence:.3f} vs required {conf_threshold:.3f}"
            ))
            
            # ML margin
            margin = ml_prediction.get('margin', 0)
            margin_threshold = self.BASE_FILTERS['ml_margin']['threshold'] * adjustments.get('ml_margin', 1.0)
            
            checks.append(FilterCheck(
                name="ml_margin",
                result=FilterResult.PASS if margin >= margin_threshold else FilterResult.FAIL,
                value=margin,
                threshold=margin_threshold,
                weight=self.BASE_FILTERS['ml_margin']['weight'],
                description=f"ML margin {margin:.3f} vs required {margin_threshold:.3f}"
            ))
            
            # ML agreement
            agreement = ml_prediction.get('agreement', 0)
            agreement_threshold = self.BASE_FILTERS['ml_agreement']['threshold'] * adjustments.get('ml_agreement', 1.0)
            
            checks.append(FilterCheck(
                name="ml_agreement",
                result=FilterResult.PASS if agreement >= agreement_threshold else FilterResult.FAIL,
                value=agreement,
                threshold=agreement_threshold,
                weight=self.BASE_FILTERS['ml_agreement']['weight'],
                description=f"ML agreement {agreement:.3f} vs required {agreement_threshold:.3f}"
            ))
            
        except Exception as e:
            self.logger.error(f"Error in ML filters: {e}")
            
        return checks
    
    def _check_risk_filters(self, setup: Dict, current_positions: List[Dict], market_data: Dict) -> List[FilterCheck]:
        """Check risk management filters"""
        checks = []
        
        try:
            # Position limit check
            max_positions = self.config.get('max_positions', 3)
            current_count = len(current_positions)
            position_ok = current_count < max_positions
            
            checks.append(FilterCheck(
                name="position_limit",
                result=FilterResult.PASS if position_ok else FilterResult.FAIL,
                value=current_count,
                threshold=max_positions,
                weight=self.BASE_FILTERS['position_limit']['weight'],
                description=f"Positions {current_count}/{max_positions}"
            ))
            
            # Correlation check
            correlation_ok = self._check_correlation_limit(setup, current_positions)
            checks.append(FilterCheck(
                name="correlation_check",
                result=FilterResult.PASS if correlation_ok else FilterResult.WARNING,
                value=float(correlation_ok),
                threshold=1.0,
                weight=self.BASE_FILTERS['correlation_check']['weight'],
                description="Correlation acceptable" if correlation_ok else "High correlation detected"
            ))
            
            # Volatility regime check
            volatility_ok = self._check_volatility_regime(market_data)
            checks.append(FilterCheck(
                name="volatility_regime",
                result=FilterResult.PASS if volatility_ok else FilterResult.WARNING,
                value=float(volatility_ok),
                threshold=1.0,
                weight=self.BASE_FILTERS['volatility_regime']['weight'],
                description="Volatility regime acceptable" if volatility_ok else "Extreme volatility detected"
            ))
            
        except Exception as e:
            self.logger.error(f"Error in risk filters: {e}")
            
        return checks
    
    def _check_ema_alignment(self, market_data: Dict) -> bool:
        """Check if EMAs are properly aligned for trend"""
        try:
            ema_8 = market_data.get('ema_8', 0)
            ema_21 = market_data.get('ema_21', 0)
            ema_55 = market_data.get('ema_55', 0)
            
            if all([ema_8, ema_21, ema_55]):
                # For uptrend: EMA8 > EMA21 > EMA55 or similar bullish alignment
                # For downtrend: EMA8 < EMA21 < EMA55 or similar bearish alignment
                uptrend = ema_8 > ema_21 > ema_55
                downtrend = ema_8 < ema_21 < ema_55
                return uptrend or downtrend
            
            return False
            
        except Exception:
            return False
    
    def _check_distance_to_structure(self, setup: Dict, market_data: Dict) -> FilterCheck:
        """Check distance to major structure levels"""
        try:
            entry_price = setup.get('entry', 0)
            direction = setup.get('direction', 'long')
            
            # Get nearest structure levels
            resistance = market_data.get('nearest_resistance', entry_price * 1.02)
            support = market_data.get('nearest_support', entry_price * 0.98)
            
            if direction == 'long':
                distance = (resistance - entry_price) / entry_price
            else:
                distance = (entry_price - support) / entry_price
            
            min_distance = self.BASE_FILTERS['distance_to_structure']['min_pct']
            distance_ok = distance >= min_distance
            
            return FilterCheck(
                name="distance_to_structure",
                result=FilterResult.PASS if distance_ok else FilterResult.WARNING,
                value=distance,
                threshold=min_distance,
                weight=self.BASE_FILTERS['distance_to_structure']['weight'],
                description=f"Distance to structure {distance:.3%} vs min {min_distance:.3%}"
            )
            
        except Exception as e:
            return FilterCheck(
                name="distance_to_structure",
                result=FilterResult.WARNING,
                value=0.0,
                threshold=0.002,
                weight=self.BASE_FILTERS['distance_to_structure']['weight'],
                description=f"Structure distance check error: {e}"
            )
    
    def _check_correlation_limit(self, setup: Dict, current_positions: List[Dict]) -> bool:
        """Check if new position would exceed correlation limits"""
        try:
            if not current_positions:
                return True
            
            direction = setup.get('direction', 'long')
            symbol = setup.get('symbol', 'SOLUSDT')
            
            # Count positions in same direction
            same_direction = sum(
                1 for pos in current_positions 
                if pos.get('direction') == direction
            )
            
            # Simple correlation check - in production would use actual correlation calculation
            max_same_direction = self.config.get('max_correlated_positions', 2)
            return same_direction < max_same_direction
            
        except Exception:
            return True  # Default to allow if check fails
    
    def _check_volatility_regime(self, market_data: Dict) -> bool:
        """Check if volatility regime is suitable for trading"""
        try:
            atr = market_data.get('atr', 0)
            atr_avg = market_data.get('atr_20d_avg', atr)
            
            if atr_avg > 0:
                volatility_ratio = atr / atr_avg
                # Extreme volatility (>300% of normal) may indicate unstable conditions
                return volatility_ratio <= 3.0
            
            return True
            
        except Exception:
            return True
    
    def get_filter_summary(self, result: EntryFilterResult) -> str:
        """Generate human-readable summary of filter results"""
        try:
            if result.overall_pass:
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            
            summary = f"{status} | Score: {result.total_score:.1f}/{result.required_score:.1f}\n"
            
            if result.failed_critical:
                summary += f"Critical failures: {', '.join(result.failed_critical)}\n"
            
            if result.warnings:
                summary += f"Warnings: {len(result.warnings)}\n"
            
            # Top failed checks
            failed_checks = [
                check for check in result.individual_checks 
                if check.result == FilterResult.FAIL
            ]
            
            if failed_checks:
                summary += "Failed filters:\n"
                for check in failed_checks[:3]:  # Top 3
                    summary += f"  - {check.name}: {check.description}\n"
            
            return summary
            
        except Exception as e:
            return f"Error generating summary: {e}"
    
    def update_regime_adjustments(self, regime: str, custom_adjustments: Dict = None):
        """Update regime-specific filter adjustments"""
        if custom_adjustments:
            if regime not in self.REGIME_ADJUSTMENTS:
                self.REGIME_ADJUSTMENTS[regime] = {}
            self.REGIME_ADJUSTMENTS[regime].update(custom_adjustments)
            
            self.logger.info(f"Updated regime adjustments for {regime}: {custom_adjustments}")
    
    def get_regime_adjustments(self, regime: str) -> Dict[str, float]:
        """Get current regime adjustments"""
        return self.REGIME_ADJUSTMENTS.get(regime, self.REGIME_ADJUSTMENTS['normal_range'])