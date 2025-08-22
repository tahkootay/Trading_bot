"""Machine Learning models for trading signal generation."""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Try to import advanced ML libraries, fallback to scikit-learn
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, Exception):
    HAS_XGBOOST = False
    xgb = None

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, Exception):
    HAS_LIGHTGBM = False
    lgb = None

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except (ImportError, Exception):
    HAS_CATBOOST = False
    CatBoostClassifier = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..utils.types import Signal, SignalType
from ..utils.logger import TradingLogger


class MLModelPredictor:
    """Ensemble ML model for trading predictions."""
    
    def __init__(
        self,
        models_path: Path,
        ensemble_weights: Dict[str, float] = None,
    ):
        self.models_path = Path(models_path)
        # Dynamic ensemble weights based on available libraries
        default_weights = {}
        if HAS_XGBOOST:
            default_weights["xgboost"] = 0.4
        if HAS_LIGHTGBM:
            default_weights["lightgbm"] = 0.3
        if HAS_CATBOOST:
            default_weights["catboost"] = 0.3
        
        # Always include scikit-learn models
        if not default_weights:  # If no advanced libraries available
            default_weights.update({
                "random_forest": 0.4,
                "gradient_boosting": 0.4,
                "decision_tree": 0.2,
            })
        else:
            default_weights["random_forest"] = 0.2  # Add as backup
        
        self.ensemble_weights = ensemble_weights or default_weights
        
        self.logger = TradingLogger("ml_models")
        
        # Models and preprocessors
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: Optional[List[str]] = None
        self.is_trained = False
        
        # Performance tracking
        self.prediction_history: List[Dict] = []
        self.performance_metrics: Dict[str, float] = {}
    
    def load_models(self, model_version: str = "latest") -> bool:
        """Load trained models from disk."""
        try:
            model_dir = self.models_path / model_version
            
            # If latest symlink doesn't exist, try to find any available model directory
            if not model_dir.exists() and model_version == "latest":
                # Look for timestamp directories
                model_dirs = [d for d in self.models_path.iterdir() if d.is_dir() and d.name != "latest"]
                if model_dirs:
                    # Use the most recent one
                    model_dir = sorted(model_dirs)[-1]
                    self.logger.log_system_event(
                        event_type="model_fallback",
                        component="ml_models",
                        status="info",
                        details={"fallback_version": model_dir.name}
                    )
            
            if not model_dir.exists():
                self.logger.log_error(
                    error_type="model_load_failed",
                    component="ml_models",
                    error_message=f"Model directory not found: {model_dir}",
                )
                return False
            
            # Load models
            for model_name in self.ensemble_weights.keys():
                model_path = model_dir / f"{model_name}.joblib"
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    self.logger.log_system_event(
                        event_type="model_loaded",
                        component="ml_models",
                        status="success",
                        details={"model": model_name, "path": str(model_path)},
                    )
            
            # Load scalers
            scaler_path = model_dir / "scaler.joblib"
            if scaler_path.exists():
                self.scalers["main"] = joblib.load(scaler_path)
            
            # Load feature names
            features_path = model_dir / "feature_names.joblib"
            if features_path.exists():
                self.feature_names = joblib.load(features_path)
            
            # Load performance metrics
            metrics_path = model_dir / "performance_metrics.joblib"
            if metrics_path.exists():
                self.performance_metrics = joblib.load(metrics_path)
            
            self.is_trained = len(self.models) > 0
            
            self.logger.log_system_event(
                event_type="models_loaded",
                component="ml_models",
                status="success",
                details={
                    "models_count": len(self.models),
                    "has_scaler": "main" in self.scalers,
                    "feature_count": len(self.feature_names) if self.feature_names else 0,
                },
            )
            
            return self.is_trained
        
        except Exception as e:
            self.logger.log_error(
                error_type="model_load_error",
                component="ml_models",
                error_message=str(e),
                details={"model_version": model_version},
            )
            return False
    
    def predict(
        self,
        features: Dict[str, float],
        symbol: str,
        current_price: float,
    ) -> Optional[Tuple[float, float, Dict[str, float]]]:
        """Make ensemble prediction."""
        if not self.is_trained or not self.feature_names:
            return None
        
        try:
            # Prepare features
            feature_vector = self._prepare_features(features)
            if feature_vector is None:
                return None
            
            # Get predictions from each model
            predictions = {}
            confidences = {}
            
            for model_name, model in self.models.items():
                try:
                    # All models use the same sklearn-style predict_proba interface
                    pred_proba = model.predict_proba(feature_vector)[0]
                    
                    # Handle different number of classes
                    if len(pred_proba) == 3:  # 3-class: [down, flat, up]
                        prediction = pred_proba[2] - pred_proba[0]  # P(up) - P(down)
                        confidence = max(pred_proba) - min(pred_proba)
                    elif len(pred_proba) == 2:  # 2-class: [down, up]
                        prediction = pred_proba[1] - pred_proba[0]  # P(up) - P(down)
                        confidence = max(pred_proba) - min(pred_proba)
                    else:
                        # Fallback
                        prediction = 0.0
                        confidence = 0.5
                    
                    predictions[model_name] = prediction
                    confidences[model_name] = confidence
                
                except Exception as e:
                    self.logger.log_error(
                        error_type="model_prediction_failed",
                        component="ml_models",
                        error_message=str(e),
                        details={"model": model_name, "symbol": symbol},
                    )
                    continue
            
            if not predictions:
                return None
            
            # Ensemble prediction
            ensemble_prediction = sum(
                predictions[model] * self.ensemble_weights.get(model, 0)
                for model in predictions
            )
            
            # Ensemble confidence
            ensemble_confidence = sum(
                confidences[model] * self.ensemble_weights.get(model, 0)
                for model in confidences
            )
            
            # Track prediction
            self.prediction_history.append({
                "timestamp": datetime.now(),
                "symbol": symbol,
                "prediction": ensemble_prediction,
                "confidence": ensemble_confidence,
                "individual_predictions": predictions,
                "features": features,
            })
            
            # Keep only last 1000 predictions
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            self.logger.log_model_prediction(
                model_name="ensemble",
                symbol=symbol,
                prediction=ensemble_prediction,
                confidence=ensemble_confidence,
                features_count=len(features),
                inference_time_ms=0.0,  # Could add timing
            )
            
            return ensemble_prediction, ensemble_confidence, predictions
        
        except Exception as e:
            self.logger.log_error(
                error_type="ensemble_prediction_failed",
                component="ml_models",
                error_message=str(e),
                details={"symbol": symbol},
            )
            return None
    
    def _prepare_features(self, features: Dict[str, float]) -> Optional[np.ndarray]:
        """Prepare feature vector for prediction."""
        if not self.feature_names:
            return None
        
        try:
            # Create feature vector in correct order
            feature_vector = []
            for feature_name in self.feature_names:
                if feature_name in features:
                    value = features[feature_name]
                    # Handle NaN values
                    if pd.isna(value) or np.isinf(value):
                        value = 0.0
                    feature_vector.append(value)
                else:
                    # Missing feature, use default value
                    feature_vector.append(0.0)
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Apply scaling if available
            if "main" in self.scalers:
                feature_vector = self.scalers["main"].transform(feature_vector)
            
            return feature_vector
        
        except Exception as e:
            self.logger.log_error(
                error_type="feature_preparation_failed",
                component="ml_models",
                error_message=str(e),
            )
            return None
    
    def train_models(
        self,
        training_data: pd.DataFrame,
        target_column: str = "target",
        test_size: float = 0.2,
        model_version: str = None,
    ) -> bool:
        """Train ensemble models."""
        try:
            if model_version is None:
                model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare data
            X = training_data.drop(columns=[target_column])
            y = training_data[target_column]
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Split data (time-aware)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers["main"] = scaler
            
            # Train models based on available libraries
            trained_models = {}
            model_scores = {}
            
            # XGBoost (if available)
            if "xgboost" in self.ensemble_weights and HAS_XGBOOST:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric="logloss",
                )
                xgb_model.fit(X_train_scaled, y_train)
                
                y_pred = xgb_model.predict(X_test_scaled)
                score = accuracy_score(y_test, y_pred)
                
                trained_models["xgboost"] = xgb_model
                model_scores["xgboost"] = score
            
            # LightGBM (if available)
            if "lightgbm" in self.ensemble_weights and HAS_LIGHTGBM:
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1,
                )
                lgb_model.fit(X_train_scaled, y_train)
                
                y_pred = lgb_model.predict(X_test_scaled)
                score = accuracy_score(y_test, y_pred)
                
                trained_models["lightgbm"] = lgb_model
                model_scores["lightgbm"] = score
            
            # CatBoost (if available)
            if "catboost" in self.ensemble_weights and HAS_CATBOOST:
                cat_model = CatBoostClassifier(
                    iterations=100,
                    depth=6,
                    learning_rate=0.1,
                    random_seed=42,
                    verbose=False,
                )
                cat_model.fit(X_train_scaled, y_train)
                
                y_pred = cat_model.predict(X_test_scaled)
                score = accuracy_score(y_test, y_pred)
                
                trained_models["catboost"] = cat_model
                model_scores["catboost"] = score
            
            # Scikit-learn models (always available)
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.tree import DecisionTreeClassifier
            
            # Random Forest
            if "random_forest" in self.ensemble_weights:
                rf_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                )
                rf_model.fit(X_train, y_train)  # Tree models don't need scaling
                
                y_pred = rf_model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                
                trained_models["random_forest"] = rf_model
                model_scores["random_forest"] = score
            
            # Gradient Boosting
            if "gradient_boosting" in self.ensemble_weights:
                gb_model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                )
                gb_model.fit(X_train_scaled, y_train)
                
                y_pred = gb_model.predict(X_test_scaled)
                score = accuracy_score(y_test, y_pred)
                
                trained_models["gradient_boosting"] = gb_model
                model_scores["gradient_boosting"] = score
            
            # Decision Tree
            if "decision_tree" in self.ensemble_weights:
                dt_model = DecisionTreeClassifier(
                    max_depth=15,
                    min_samples_split=10,
                    random_state=42,
                )
                dt_model.fit(X_train, y_train)  # Tree models don't need scaling
                
                y_pred = dt_model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                
                trained_models["decision_tree"] = dt_model
                model_scores["decision_tree"] = score
            
            # Save models
            model_dir = self.models_path / model_version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            for model_name, model in trained_models.items():
                model_path = model_dir / f"{model_name}.joblib"
                joblib.dump(model, model_path)
            
            # Save scaler and feature names
            joblib.dump(scaler, model_dir / "scaler.joblib")
            joblib.dump(self.feature_names, model_dir / "feature_names.joblib")
            joblib.dump(model_scores, model_dir / "performance_metrics.joblib")
            
            # Update instance
            self.models = trained_models
            self.performance_metrics = model_scores
            self.is_trained = True
            
            self.logger.log_system_event(
                event_type="models_trained",
                component="ml_models",
                status="success",
                details={
                    "model_version": model_version,
                    "models_count": len(trained_models),
                    "training_samples": len(X_train),
                    "test_samples": len(X_test),
                    "scores": model_scores,
                },
            )
            
            return True
        
        except Exception as e:
            self.logger.log_error(
                error_type="model_training_failed",
                component="ml_models",
                error_message=str(e),
            )
            return False
    
    def get_feature_importance(self, model_name: str = "xgboost") -> Optional[Dict[str, float]]:
        """Get feature importance from specified model."""
        if model_name not in self.models or not self.feature_names:
            return None
        
        try:
            model = self.models[model_name]
            
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "get_feature_importance"):
                importances = model.get_feature_importance()
            else:
                return None
            
            importance_dict = dict(zip(self.feature_names, importances))
            
            # Sort by importance
            sorted_importance = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            
            return sorted_importance
        
        except Exception as e:
            self.logger.log_error(
                error_type="feature_importance_failed",
                component="ml_models",
                error_message=str(e),
                details={"model": model_name},
            )
            return None
    
    def evaluate_recent_performance(self, days: int = 7) -> Dict[str, float]:
        """Evaluate model performance over recent period."""
        if not self.prediction_history:
            return {}
        
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            recent_predictions = [
                p for p in self.prediction_history
                if p["timestamp"] >= cutoff_time
            ]
            
            if not recent_predictions:
                return {}
            
            # Calculate basic statistics
            predictions = [p["prediction"] for p in recent_predictions]
            confidences = [p["confidence"] for p in recent_predictions]
            
            performance = {
                "total_predictions": len(recent_predictions),
                "avg_prediction": np.mean(predictions),
                "avg_confidence": np.mean(confidences),
                "prediction_std": np.std(predictions),
                "confidence_std": np.std(confidences),
                "strong_signals": len([p for p in predictions if abs(p) > 0.2]),
                "weak_signals": len([p for p in predictions if abs(p) < 0.1]),
            }
            
            return performance
        
        except Exception as e:
            self.logger.log_error(
                error_type="performance_evaluation_failed",
                component="ml_models",
                error_message=str(e),
            )
            return {}
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get comprehensive model status."""
        status = {
            "is_trained": self.is_trained,
            "models_loaded": list(self.models.keys()),
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "ensemble_weights": self.ensemble_weights,
            "recent_predictions": len(self.prediction_history),
            "performance_metrics": self.performance_metrics,
        }
        
        # Add recent performance
        recent_perf = self.evaluate_recent_performance(days=1)
        if recent_perf:
            status["recent_performance"] = recent_perf
        
        return status