import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datetime import datetime
import json

@dataclass
class MarketRegime:
    """Market regime characteristics with enhanced metrics"""
    volatility: str  # 'low', 'medium', 'high'
    trend: str      # 'bull', 'bear', 'sideways'
    momentum: float # -1 to 1
    volume_profile: str # 'low', 'normal', 'high'
    confidence: float = 0.0  # Regime detection confidence
    duration: int = 0  # How long this regime has persisted

@dataclass 
class PerformanceMetrics:
    """Track performance over iterations"""
    mse: float
    mae: float
    directional_accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    volatility_forecast_error: float
    spike_detection_rate: float
    regime_stability: float

class FinancialSpikeNeuron:
    """Enhanced neuron with performance tracking and adaptive parameters"""
    
    def __init__(self, neuron_type: str = "momentum", adaptive: bool = True):
        self.neuron_type = neuron_type
        self.adaptive = adaptive
        
        # Multi-timescale state: [fast_voltage, slow_recovery, momentum_accumulator, regime_detector]
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Enhanced parameter sets with performance tracking
        self.base_params = self._get_base_params(neuron_type)
        self.params = self.base_params.copy()
        
        # Performance tracking
        self.spike_history = []
        self.input_buffer = np.zeros(50)  # Increased buffer for better adaptation
        self.output_buffer = np.zeros(50)
        self.adaptation_state = 1.0
        self.performance_history = []
        
        # Adaptive learning rates
        self.learning_rates = {
            'spike_threshold': 0.001,
            'recovery_rate': 0.0005,
            'momentum_decay': 0.0001,
        }
        
    def _get_base_params(self, neuron_type: str) -> Dict[str, float]:
        """Get optimized base parameters for each neuron type"""
        if neuron_type == "momentum":
            return {
                'spike_threshold': 0.7,
                'recovery_rate': 0.05,
                'momentum_decay': 0.02,
                'volume_sensitivity': 0.3,
                'volatility_adaptation': 0.1,
                'feedback_strength': 0.4,
                'noise_tolerance': 0.05,  # New: Handle market noise
            }
        elif neuron_type == "regime":
            return {
                'spike_threshold': 1.2,
                'recovery_rate': 0.01,
                'momentum_decay': 0.005,
                'volume_sensitivity': 0.2,
                'volatility_adaptation': 0.15,
                'feedback_strength': 0.6,
                'noise_tolerance': 0.02,
            }
        elif neuron_type == "volatility":
            return {
                'spike_threshold': 0.5,
                'recovery_rate': 0.1,
                'momentum_decay': 0.05,
                'volume_sensitivity': 0.5,
                'volatility_adaptation': 0.25,
                'feedback_strength': 0.2,
                'noise_tolerance': 0.1,
            }
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")
            
    def financial_dynamics(self, t: float, y: np.ndarray, price_input: float, 
                          volume_input: float, volatility_input: float) -> np.ndarray:
        """Enhanced financial dynamics with noise handling"""
        v, w, m, r = y
        
        # Adaptive parameters based on recent performance
        threshold = self.params['spike_threshold'] * (1 + 0.2 * volatility_input)
        recovery = self.params['recovery_rate'] * (1 + 0.3 * volatility_input)
        
        # Enhanced input processing with noise filtering
        noise_level = self.params['noise_tolerance']
        filtered_price_input = price_input if abs(price_input) > noise_level else price_input * 0.5
        
        # Volume-weighted input processing
        effective_input = filtered_price_input * (1 + self.params['volume_sensitivity'] * volume_input)
        
        # Enhanced FHN dynamics with cross-coupling
        dv_dt = (v - v**3/3 - w + effective_input + 
                self.params['feedback_strength'] * m * v +
                0.1 * r * effective_input +
                0.05 * np.sin(t * 10) * volatility_input)  # Market cycle component
        
        # Recovery with adaptive volatility
        dw_dt = recovery * (v + 0.7 - 0.8 * w + 0.1 * volatility_input)
        
        # Enhanced momentum with memory effects
        spike_indicator = 1.0 if v > threshold else 0.0
        momentum_boost = 0.2 if len(self.spike_history) > 0 and self.spike_history[-1]['time'] < 0.1 else 0.0
        dm_dt = (spike_indicator * (1 - m) - 
                self.params['momentum_decay'] * m + 
                0.1 * abs(effective_input) * (1 - m) +
                momentum_boost)
        
        # Enhanced regime detector with stability
        regime_smoothing = 0.9  # Prevent rapid regime switches
        dr_dt = 0.01 * (regime_smoothing * np.tanh(effective_input) + 
                       (1 - regime_smoothing) * r + 0.1 * m)
        
        return np.array([dv_dt, dw_dt, dm_dt, dr_dt])
    
    def forward(self, price_change: float, volume_ratio: float, volatility: float, 
               dt: float = 0.01) -> Dict[str, float]:
        """Enhanced forward pass with performance tracking"""
        
        # Update buffers
        self.input_buffer = np.roll(self.input_buffer, 1)
        self.input_buffer[0] = price_change
        
        # Enhanced input normalization with dynamic scaling
        price_scaling = 10 * (1 + volatility)  # Scale with volatility
        price_input = np.tanh(price_change * price_scaling)
        volume_input = np.tanh(volume_ratio - 1)
        volatility_input = np.tanh(volatility * 5)
        
        # Integrate dynamics
        def dynamics_wrapper(t, y):
            return self.financial_dynamics(t, y, price_input, volume_input, volatility_input)
        
        sol = solve_ivp(dynamics_wrapper, [0, dt], self.state, 
                       method='RK45', rtol=1e-6, atol=1e-8)
        self.state = sol.y[:, -1]
        
        # Enhanced spike detection with adaptive threshold
        spike_strength = 0.0
        adapted_threshold = self.params['spike_threshold'] * self.adaptation_state
        
        if self.state[0] > adapted_threshold:
            spike_strength = self.state[0] - adapted_threshold
            self.spike_history.append({
                'time': 0.0,  # Reset for each forward pass
                'strength': spike_strength,
                'momentum': self.state[2],
                'regime': self.state[3],
                'input_volatility': volatility_input,
                'adaptation_state': self.adaptation_state
            })
            
            # Keep recent history
            if len(self.spike_history) > 100:
                self.spike_history.pop(0)
        
        # Output with confidence estimation
        activation = np.tanh(self.state[0])
        confidence = 1.0 - abs(self.state[1]) / (abs(self.state[0]) + 1e-6)
        
        # Update output buffer
        self.output_buffer = np.roll(self.output_buffer, 1)
        self.output_buffer[0] = activation
        
        # Adaptive parameter updates if enabled
        if self.adaptive and len(self.output_buffer) == 50:
            self._update_adaptive_parameters()
        
        return {
            'activation': activation,
            'momentum': self.state[2],
            'regime': self.state[3],
            'spike_strength': spike_strength,
            'recovery_state': self.state[1],
            'confidence': confidence,
            'adaptation_state': self.adaptation_state
        }
    
    def _update_adaptive_parameters(self):
        """Update parameters based on recent performance"""
        # Analyze recent spike patterns
        recent_spikes = [s for s in self.spike_history[-10:]]
        
        if len(recent_spikes) > 0:
            avg_spike_strength = np.mean([s['strength'] for s in recent_spikes])
            
            # Adjust threshold based on spike frequency
            if len(recent_spikes) > 7:  # Too many spikes
                self.params['spike_threshold'] *= 1.01
            elif len(recent_spikes) < 3:  # Too few spikes
                self.params['spike_threshold'] *= 0.99
                
            # Adjust recovery based on spike strength
            if avg_spike_strength > 0.5:
                self.params['recovery_rate'] *= 1.005
            
        # Update adaptation state
        output_stability = 1.0 - np.std(self.output_buffer)
        self.adaptation_state = 0.9 * self.adaptation_state + 0.1 * output_stability

class FinancialLiquidLayer:
    """Enhanced layer with performance tracking and regime adaptation"""
    
    def __init__(self, input_size: int, output_size: int, layer_type: str = "mixed", 
                 adaptive: bool = True):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_type = layer_type
        self.adaptive = adaptive
        
        # Create specialized neurons
        self.neurons = self._create_neurons(output_size, layer_type, adaptive)
        
        # Enhanced weight initialization with Xavier/He initialization
        self.W_price = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        self.W_volume = np.random.randn(output_size, input_size) * np.sqrt(1.0 / input_size)
        self.W_volatility = np.random.randn(output_size, input_size) * np.sqrt(1.5 / input_size)
        self.W_recurrent = np.random.randn(output_size, output_size) * np.sqrt(1.0 / output_size)
        np.fill_diagonal(self.W_recurrent, 0)
        
        # Enhanced spike coupling with adaptation
        self.spike_coupling = np.random.rand(output_size, output_size) * 0.1
        np.fill_diagonal(self.spike_coupling, 0)
        
        # Performance tracking
        self.last_output = None
        self.output_history = []
        self.market_regime = MarketRegime("medium", "sideways", 0.0, "normal", 0.5, 0)
        self.regime_history = []
        self.performance_metrics = []
        
        # Learning rates for weight adaptation
        if adaptive:
            self.weight_learning_rates = {
                'price': 0.001,
                'volume': 0.0005,
                'volatility': 0.001,
                'recurrent': 0.0001
            }
    
    def _create_neurons(self, output_size: int, layer_type: str, adaptive: bool) -> List[FinancialSpikeNeuron]:
        """Create neurons with enhanced type distribution"""
        neurons = []
        
        if layer_type == "mixed":
            # Optimized neuron distribution
            momentum_count = max(1, output_size // 2)
            regime_count = max(1, output_size // 4)
            volatility_count = output_size - momentum_count - regime_count
            
            for i in range(momentum_count):
                neurons.append(FinancialSpikeNeuron("momentum", adaptive))
            for i in range(regime_count):
                neurons.append(FinancialSpikeNeuron("regime", adaptive))
            for i in range(volatility_count):
                neurons.append(FinancialSpikeNeuron("volatility", adaptive))
        else:
            for i in range(output_size):
                neurons.append(FinancialSpikeNeuron(layer_type, adaptive))
        
        return neurons
    
    def detect_market_regime(self, price_changes: np.ndarray, volumes: np.ndarray, 
                           volatilities: np.ndarray) -> MarketRegime:
        """Enhanced regime detection with confidence scoring"""
        
        # Enhanced volatility classification with percentiles
        vol_percentiles = np.percentile(volatilities, [25, 75])
        avg_vol = np.mean(volatilities)
        
        if avg_vol < vol_percentiles[0]:
            vol_regime = "low"
        elif avg_vol < vol_percentiles[1]:
            vol_regime = "medium"
        else:
            vol_regime = "high"
        
        # Enhanced trend classification with momentum
        price_trend = np.mean(price_changes)
        price_momentum = np.mean(price_changes[-5:]) - np.mean(price_changes[-10:-5])
        
        trend_threshold = np.std(price_changes) * 0.5
        if price_trend > trend_threshold:
            trend = "bull"
        elif price_trend < -trend_threshold:
            trend = "bear"
        else:
            trend = "sideways"
        
        # Enhanced momentum with acceleration
        momentum = np.tanh(price_trend * 100 + price_momentum * 50)
        
        # Volume regime with trend confirmation
        volume_trend = np.mean(volumes[-5:]) / np.mean(volumes[-20:-5])
        avg_volume = np.mean(volumes)
        
        if avg_volume < 0.8:
            volume_profile = "low"
        elif avg_volume < 1.2:
            volume_profile = "normal"
        else:
            volume_profile = "high"
        
        # Calculate confidence based on consistency
        trend_consistency = 1.0 - np.std(price_changes) / (abs(np.mean(price_changes)) + 1e-6)
        volume_consistency = 1.0 - np.std(volumes) / (np.mean(volumes) + 1e-6)
        confidence = (trend_consistency + volume_consistency) / 2.0
        
        # Calculate regime duration
        duration = 1
        if len(self.regime_history) > 0:
            last_regime = self.regime_history[-1]
            if (last_regime.trend == trend and last_regime.volatility == vol_regime):
                duration = last_regime.duration + 1
        
        return MarketRegime(vol_regime, trend, momentum, volume_profile, confidence, duration)
    
    def forward(self, price_features: np.ndarray, volume_features: np.ndarray, 
               volatility_features: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """Enhanced forward pass with adaptation"""
        
        # Update market regime
        self.market_regime = self.detect_market_regime(price_features, volume_features, volatility_features)
        self.regime_history.append(self.market_regime)
        
        # Keep recent regime history
        if len(self.regime_history) > 100:
            self.regime_history.pop(0)
        
        outputs = np.zeros(self.output_size)
        spike_influences = np.zeros(self.output_size)
        neuron_outputs = []
        
        for i, neuron in enumerate(self.neurons):
            # Compute weighted inputs with regime adaptation
            regime_multiplier = 1.0 + 0.2 * self.market_regime.confidence
            
            price_input = np.dot(self.W_price[i], price_features) * regime_multiplier
            volume_input = np.dot(self.W_volume[i], volume_features)
            volatility_input = np.dot(self.W_volatility[i], volatility_features)
            
            # Enhanced recurrent connections
            if self.last_output is not None:
                recurrent_input = np.dot(self.W_recurrent[i], self.last_output)
                price_input += recurrent_input * 0.5  # Reduced recurrent influence
            
            # Process through neuron
            neuron_output = neuron.forward(price_input, volume_input, volatility_input, dt)
            neuron_outputs.append(neuron_output)
            
            outputs[i] = neuron_output['activation']
            
            # Enhanced spike contagion with decay
            if neuron_output['spike_strength'] > 0:
                contagion_strength = neuron_output['spike_strength'] * neuron_output['confidence']
                spike_influences += self.spike_coupling[i] * contagion_strength
        
        # Apply spike contagion with saturation
        outputs += np.tanh(spike_influences)  # Prevent runaway activation
        outputs = np.tanh(outputs)  # Keep bounded
        
        # Store outputs for analysis
        self.output_history.append(outputs.copy())
        if len(self.output_history) > 100:
            self.output_history.pop(0)
        
        # Adaptive weight updates
        if self.adaptive and self.last_output is not None:
            self._update_weights(price_features, volume_features, volatility_features, outputs)
        
        self.last_output = outputs.copy()
        return outputs
    
    def _update_weights(self, price_features: np.ndarray, volume_features: np.ndarray,
                       volatility_features: np.ndarray, outputs: np.ndarray):
        """Adaptive weight updates based on output consistency"""
        if len(self.output_history) < 10:
            return
            
        # Calculate output stability
        recent_outputs = np.array(self.output_history[-10:])
        output_variance = np.var(recent_outputs, axis=0)
        
        # Update weights to improve stability
        for i in range(self.output_size):
            if output_variance[i] > 0.1:  # High variance indicates instability
                # Reduce learning rates for unstable neurons
                lr_multiplier = 0.5
            else:
                lr_multiplier = 1.0
            
            # Small gradient-free updates based on regime confidence
            confidence_factor = self.market_regime.confidence
            update_strength = self.weight_learning_rates['price'] * lr_multiplier * confidence_factor
            
            # Random walk updates with bias toward stability
            self.W_price[i] += np.random.randn(len(price_features)) * update_strength * 0.1
            self.W_volume[i] += np.random.randn(len(volume_features)) * update_strength * 0.05
            self.W_volatility[i] += np.random.randn(len(volatility_features)) * update_strength * 0.08
    
    def get_layer_insights(self) -> Dict[str, Any]:
        """Enhanced insights with performance metrics"""
        insights = {
            'market_regime': self.market_regime,
            'neuron_states': [],
            'spike_activity': 0,
            'momentum_levels': [],
            'regime_states': [],
            'confidence_levels': [],
            'adaptation_states': [],
            'regime_stability': self._calculate_regime_stability()
        }
        
        total_spikes = 0
        for neuron in self.neurons:
            if len(neuron.spike_history) > 0:
                recent_spikes = [s for s in neuron.spike_history[-10:]]
                total_spikes += len(recent_spikes)
            
            insights['momentum_levels'].append(neuron.state[2])
            insights['regime_states'].append(neuron.state[3])
            insights['confidence_levels'].append(getattr(neuron, 'confidence', 0.5))
            insights['adaptation_states'].append(neuron.adaptation_state)
        
        insights['spike_activity'] = total_spikes
        insights['avg_confidence'] = np.mean(insights['confidence_levels'])
        insights['avg_adaptation'] = np.mean(insights['adaptation_states'])
        
        return insights
    
    def _calculate_regime_stability(self) -> float:
        """Calculate how stable the regime detection has been"""
        if len(self.regime_history) < 5:
            return 0.5
        
        recent_regimes = self.regime_history[-10:]
        trend_changes = sum(1 for i in range(1, len(recent_regimes)) 
                           if recent_regimes[i].trend != recent_regimes[i-1].trend)
        vol_changes = sum(1 for i in range(1, len(recent_regimes))
                         if recent_regimes[i].volatility != recent_regimes[i-1].volatility)
        
        stability = 1.0 - (trend_changes + vol_changes) / (2 * len(recent_regimes))
        return max(0.0, stability)

class EnhancedFinancialLNN:
    """Enhanced LNN with comprehensive performance tracking and comparison capabilities"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, 
                 adaptive: bool = True, name: str = "FinancialLNN"):
        self.name = name
        self.adaptive = adaptive
        self.layers = []
        
        # Create enhanced layer architecture
        layer_configs = [
            ("momentum", "Primary momentum detection"),
            ("mixed", "Multi-signal processing"),
            ("regime", "Regime-aware processing"),
            ("volatility", "Volatility-sensitive output")
        ]
        
        # Input layer
        self.layers.append(FinancialLiquidLayer(input_size, hidden_sizes[0], 
                                               layer_configs[0][0], adaptive))
        
        # Hidden layers with alternating specializations
        for i in range(1, len(hidden_sizes)):
            layer_type = layer_configs[min(i, len(layer_configs)-1)][0]
            self.layers.append(FinancialLiquidLayer(hidden_sizes[i-1], hidden_sizes[i], 
                                                   layer_type, adaptive))
        
        # Output layer
        self.layers.append(FinancialLiquidLayer(hidden_sizes[-1], output_size, 
                                               "volatility", adaptive))
        
        # Performance tracking
        self.prediction_history = []
        self.performance_metrics = []
        self.training_iteration = 0
        
        # Comparison framework
        self.comparison_data = {
            'predictions': [],
            'actual_values': [],
            'timestamps': [],
            'market_regimes': [],
            'confidence_scores': []
        }
        
    def forward(self, price_data: np.ndarray, volume_data: np.ndarray, 
               volatility_data: np.ndarray, dt: float = 0.01, 
               actual_value: Optional[float] = None) -> Dict[str, Any]:
        """Enhanced forward pass with comprehensive tracking"""
        
        current_input_price = price_data
        current_input_volume = volume_data  
        current_input_vol = volatility_data
        
        layer_insights = []
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            current_input_price = layer.forward(current_input_price, current_input_volume, 
                                               current_input_vol, dt)
            
            # Propagate transformed features to next layer
            if i < len(self.layers) - 1:
                current_input_volume = current_input_price * 0.1
                current_input_vol = current_input_price * 0.1
            
            layer_insights.append(layer.get_layer_insights())
        
        prediction = current_input_price
        
        # Enhanced result compilation
        market_signals = self._extract_market_signals(layer_insights)
        
        # Calculate ensemble confidence
        confidence_scores = [insight['avg_confidence'] for insight in layer_insights]
        ensemble_confidence = np.mean(confidence_scores)
        
        result = {
            'prediction': prediction,
            'confidence': ensemble_confidence,
            'prediction_std': np.std(prediction),
            'layer_insights': layer_insights,
            'market_signals': market_signals,
            'model_name': self.name,
            'iteration': self.training_iteration,
            'regime_stability': np.mean([insight['regime_stability'] for insight in layer_insights])
        }
        
        # Store for comparison
        self.prediction_history.append(result)
        
        # Update comparison data
        self.comparison_data['predictions'].append(prediction[0] if len(prediction) > 1 else prediction)
        self.comparison_data['timestamps'].append(datetime.now())
        self.comparison_data['market_regimes'].append(layer_insights[0]['market_regime'].trend)
        self.comparison_data['confidence_scores'].append(ensemble_confidence)
        
        if actual_value is not None:
            self.comparison_data['actual_values'].append(actual_value)
            self._update_performance_metrics(prediction, actual_value, market_signals)
        
        self.training_iteration += 1
        return result
    
    def _extract_market_signals(self, layer_insights: List[Dict]) -> Dict[str, float]:
        """Enhanced market signal extraction"""
        
        # Aggregate metrics across layers
        total_spike_activity = sum(layer['spike_activity'] for layer in layer_insights)
        
        all_momentum = []
        all_regime = []
        all_confidence = []
        all_adaptation = []
        
        for layer in layer_insights:
            all_momentum.extend(layer['momentum_levels'])
            all_regime.extend(layer['regime_states'])
            all_confidence.extend(layer['confidence_levels'])
            all_adaptation.extend(layer['adaptation_states'])
        
        # Calculate enhanced signals
        avg_momentum = np.mean(all_momentum) if all_momentum else 0.0
        regime_consensus = np.mean(all_regime) if all_regime else 0.0
        avg_confidence = np.mean(all_confidence) if all_confidence else 0.5
        avg_adaptation = np.mean(all_adaptation) if all_adaptation else 1.0
        
        # Calculate regime stability across layers
        regime_stabilities = [layer['regime_stability'] for layer in layer_insights]
        avg_regime_stability = np.mean(regime_stabilities)
        
        return {
            'momentum_strength': avg_momentum,
            'spike_activity': total_spike_activity,
            'regime_signal': regime_consensus,
            'volatility_alert': 1.0 if total_spike_activity > 5 else 0.0,
            'trend_strength': abs(regime_consensus),
            'confidence_score': avg_confidence,
            'adaptation_level': avg_adaptation,
            'regime_stability': avg_regime_stability,
            'signal_quality': avg_confidence * avg_regime_stability
        }
    
    def _update_performance_metrics(self, prediction: np.ndarray, actual: float, 
                                   market_signals: Dict[str, float]):
        """Update performance metrics for tracking"""
        pred_value = prediction[0] if len(prediction) > 1 else float(prediction)
        
        # Calculate error metrics
        mse = (pred_value - actual) ** 2
        mae = abs(pred_value - actual)
        
        # Store metrics
        metrics = {
            'mse': mse,
            'mae': mae,
            'prediction': pred_value,
            'actual': actual,
            'confidence': market_signals.get('confidence_score', 0.5),
            'regime_stability': market_signals.get('regime_stability', 0.5),
            'iteration': self.training_iteration
        }
        
        self.performance_metrics.append(metrics)
    
    def get_performance_summary(self, last_n: int = 50) -> Dict[str, float]:
        """Get performance summary for comparison"""
        if not self.performance_metrics:
            return {'status': 'No performance data available'}
        
        recent_metrics = self.performance_metrics[-last_n:]
        
        mse_values = [m['mse'] for m in recent_metrics]
        mae_values = [m['mae'] for m in recent_metrics]
        confidence_values = [m['confidence'] for m in recent_metrics]
        
        return {
            'avg_mse': np.mean(mse_values),
            'avg_mae': np.mean(mae_values),
            'mse_std': np.std(mse_values),
            'mae_std': np.std(mae_values),
            'avg_confidence': np.mean(confidence_values),
            'total_predictions': len(self.performance_metrics),
            'recent_predictions': len(recent_metrics),
            'model_name': self.name
        }
    
    def compare_with_baseline(self, baseline_predictions: List[float], 
                            actual_values: List[float]) -> Dict[str, Any]:
        """Compare performance with baseline model"""
        if len(self.comparison_data['predictions']) != len(baseline_predictions):
            return {'error': 'Prediction arrays must have same length'}
        
        # Calculate metrics for both models
        lnn_predictions = self.comparison_data['predictions']
        
        lnn_mse = np.mean([(p - a)**2 for p, a in zip(lnn_predictions, actual_values)])
        baseline_mse = np.mean([(p - a)**2 for p, a in zip(baseline_predictions, actual_values)])
        
        lnn_mae = np.mean([abs(p - a) for p, a in zip(lnn_predictions, actual_values)])
        baseline_mae = np.mean([abs(p - a) for p, a in zip(baseline_predictions, actual_values)])
        
        # Directional accuracy
        lnn_directions = [(p > 0) == (a > 0) for p, a in zip(lnn_predictions, actual_values)]
        baseline_directions = [(p > 0) == (a > 0) for p, a in zip(baseline_predictions, actual_values)]
        
        lnn_directional_acc = np.mean(lnn_directions)
        baseline_directional_acc = np.mean(baseline_directions)
        
        # Performance comparison
        comparison = {
            'lnn_performance': {
                'mse': lnn_mse,
                'mae': lnn_mae,
                'directional_accuracy': lnn_directional_acc,
                'avg_confidence': np.mean(self.comparison_data['confidence_scores']),
                'model_name': self.name
            },
            'baseline_performance': {
                'mse': baseline_mse,
                'mae': baseline_mae,
                'directional_accuracy': baseline_directional_acc,
                'model_name': 'Baseline'
            },
            'improvements': {
                'mse_improvement': (baseline_mse - lnn_mse) / baseline_mse * 100,
                'mae_improvement': (baseline_mae - lnn_mae) / baseline_mae * 100,
                'directional_improvement': (lnn_directional_acc - baseline_directional_acc) * 100
            },
            'statistical_significance': self._calculate_significance_tests(
                lnn_predictions, baseline_predictions, actual_values
            )
        }
        
        return comparison
    
    def _calculate_significance_tests(self, lnn_preds: List[float], 
                                    baseline_preds: List[float], 
                                    actual_vals: List[float]) -> Dict[str, float]:
        """Calculate statistical significance of improvements"""
        from scipy import stats
        
        lnn_errors = [abs(p - a) for p, a in zip(lnn_preds, actual_vals)]
        baseline_errors = [abs(p - a) for p, a in zip(baseline_preds, actual_vals)]
        
        # Wilcoxon signed-rank test for paired samples
        try:
            statistic, p_value = stats.wilcoxon(lnn_errors, baseline_errors, alternative='less')
            return {
                'wilcoxon_statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        except Exception as e:
            return {'error': str(e), 'significant': False}

# Enhanced Feature Extractor with more sophisticated features
class EnhancedFinancialFeatureExtractor:
    """Enhanced feature extraction with technical indicators and regime detection"""
    
    @staticmethod
    def extract_features(price_data: pd.DataFrame, window: int = 20) -> Dict[str, np.ndarray]:
        """Extract comprehensive multi-timescale financial features"""
        
        # Ensure we have enough data
        if len(price_data) < window * 2:
            raise ValueError(f"Need at least {window * 2} data points, got {len(price_data)}")
        
        # Basic returns
        returns = price_data['Close'].pct_change().fillna(0)
        
        # Multi-timescale returns with technical indicators
        returns_1 = returns.values[-window:]
        returns_5 = price_data['Close'].pct_change(5).fillna(0).values[-window:]
        returns_20 = price_data['Close'].pct_change(20).fillna(0).values[-window:]
        
        # RSI-like momentum indicators
        rsi_short = EnhancedFinancialFeatureExtractor._calculate_rsi(price_data['Close'], 14)[-window:]
        rsi_long = EnhancedFinancialFeatureExtractor._calculate_rsi(price_data['Close'], 30)[-window:]
        
        # Price position within recent range
        price_position = EnhancedFinancialFeatureExtractor._calculate_price_position(
            price_data['Close'], window
        )[-window:]
        
        # Enhanced price features
        price_features = np.column_stack([
            returns_1, returns_5, returns_20, 
            rsi_short, rsi_long, price_position
        ])
        
        # Enhanced volume features
        volume_ratio = (price_data['Volume'] / price_data['Volume'].rolling(window).mean()).fillna(1.0)
        volume_change = volume_ratio.pct_change().fillna(0)
        volume_momentum = volume_ratio.rolling(5).mean() / volume_ratio.rolling(20).mean()
        
        # Volume-price relationship
        vp_ratio = (returns * volume_ratio).rolling(5).mean().fillna(0)
        
        volume_features = np.column_stack([
            volume_ratio.values[-window:],
            volume_change.values[-window:],
            volume_momentum.fillna(1.0).values[-window:],
            vp_ratio.values[-window:]
        ])
        
        # Enhanced volatility features
        volatility_short = returns.rolling(5).std().fillna(0).values[-window:]
        volatility_long = returns.rolling(20).std().fillna(0).values[-window:]
        volatility_ratio = (volatility_short / (volatility_long + 1e-8))
        
        # GARCH-like volatility clustering
        vol_clustering = EnhancedFinancialFeatureExtractor._calculate_vol_clustering(
            returns, window
        )[-window:]
        
        volatility_features = np.column_stack([
            volatility_short, volatility_long, volatility_ratio, vol_clustering
        ])
        
        return {
            'price_features': price_features,
            'volume_features': volume_features, 
            'volatility_features': volatility_features
        }
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50) / 100.0  # Normalize to 0-1
    
    @staticmethod
    def _calculate_price_position(prices: pd.Series, window: int) -> pd.Series:
        """Calculate price position within recent high-low range"""
        rolling_high = prices.rolling(window).max()
        rolling_low = prices.rolling(window).min()
        price_position = (prices - rolling_low) / (rolling_high - rolling_low + 1e-8)
        return price_position.fillna(0.5)
    
    @staticmethod
    def _calculate_vol_clustering(returns: pd.Series, window: int) -> pd.Series:
        """Calculate volatility clustering indicator"""
        vol = returns.rolling(5).std()
        vol_ma = vol.rolling(window).mean()
        clustering = vol / (vol_ma + 1e-8)
        return clustering.fillna(1.0)

# Comparison and Testing Framework
class ModelComparisonFramework:
    """Framework for comparing different LNN variants and baseline models"""
    
    def __init__(self):
        self.models = {}
        self.test_results = {}
        self.comparison_history = []
    
    def add_model(self, name: str, model):
        """Add a model to the comparison framework"""
        self.models[name] = model
        print(f"Added model '{name}' to comparison framework")
    
    def run_side_by_side_test(self, price_data: pd.DataFrame, 
                             test_window: int = 50, 
                             prediction_horizon: int = 1) -> Dict[str, Any]:
        """
        Run side-by-side testing of all models
        
        Args:
            price_data: DataFrame with financial data (must have 'Close' and 'Volume' columns)
            test_window: Number of recent data points to use for testing
            prediction_horizon: How many steps ahead to predict (1 = next period)
        """
        
        if len(self.models) == 0:
            raise ValueError("No models added to comparison framework")
        
        print(f"Running side-by-side test on {len(self.models)} models...")
        print(f"Test window: {test_window}, Prediction horizon: {prediction_horizon}")
        
        # Extract features
        extractor = EnhancedFinancialFeatureExtractor()
        features = extractor.extract_features(price_data, window=test_window + 20)
        
        # Prepare test data
        test_results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTesting model: {model_name}")
            
            model_predictions = []
            actual_values = []
            prediction_times = []
            
            # Rolling window prediction
            for i in range(test_window):
                try:
                    # Get current features
                    current_idx = -(test_window - i)
                    if current_idx == 0:
                        current_idx = len(features['price_features'])
                    
                    price_feat = features['price_features'][current_idx - 1]
                    volume_feat = features['volume_features'][current_idx - 1] 
                    vol_feat = features['volatility_features'][current_idx - 1]
                    
                    # Get actual value for comparison
                    actual_return = price_data['Close'].pct_change().iloc[-(test_window - i)]
                    
                    # Make prediction
                    if hasattr(model, 'forward'):
                        # Enhanced LNN model
                        result = model.forward(price_feat, volume_feat, vol_feat, 
                                             actual_value=actual_return)
                        prediction = result['prediction']
                        if isinstance(prediction, np.ndarray):
                            prediction = prediction[0] if len(prediction) > 0 else 0.0
                    else:
                        # Baseline model (simple function)
                        prediction = model(price_feat, volume_feat, vol_feat)
                    
                    model_predictions.append(float(prediction))
                    actual_values.append(float(actual_return))
                    prediction_times.append(i)
                    
                except Exception as e:
                    print(f"Error in prediction {i} for model {model_name}: {e}")
                    model_predictions.append(0.0)
                    actual_values.append(0.0)
            
            # Calculate performance metrics
            if len(model_predictions) > 0 and len(actual_values) > 0:
                mse = np.mean([(p - a)**2 for p, a in zip(model_predictions, actual_values)])
                mae = np.mean([abs(p - a) for p, a in zip(model_predictions, actual_values)])
                
                # Directional accuracy
                directions_correct = [(p > 0) == (a > 0) for p, a in zip(model_predictions, actual_values)]
                directional_accuracy = np.mean(directions_correct)
                
                # Correlation
                correlation = np.corrcoef(model_predictions, actual_values)[0, 1] if len(model_predictions) > 1 else 0.0
                
                test_results[model_name] = {
                    'predictions': model_predictions,
                    'actual_values': actual_values,
                    'mse': mse,
                    'mae': mae,
                    'directional_accuracy': directional_accuracy,
                    'correlation': correlation,
                    'total_predictions': len(model_predictions)
                }
                
                print(f"  MSE: {mse:.6f}")
                print(f"  MAE: {mse:.6f}")
                print(f"  Directional Accuracy: {directional_accuracy:.2%}")
                print(f"  Correlation: {correlation:.4f}")
            else:
                test_results[model_name] = {'error': 'No valid predictions generated'}
        
        # Store results
        self.test_results = test_results
        
        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(test_results)
        
        # Store in history
        self.comparison_history.append({
            'timestamp': datetime.now(),
            'test_window': test_window,
            'results': test_results,
            'summary': comparison_summary
        })
        
        return {
            'detailed_results': test_results,
            'summary': comparison_summary,
            'test_config': {
                'test_window': test_window,
                'prediction_horizon': prediction_horizon,
                'models_tested': list(self.models.keys())
            }
        }
    
    def _generate_comparison_summary(self, test_results: Dict) -> Dict[str, Any]:
        """Generate a summary comparing all models"""
        
        valid_results = {name: results for name, results in test_results.items() 
                        if 'error' not in results}
        
        if not valid_results:
            return {'error': 'No valid results to compare'}
        
        # Find best performing models
        best_mse = min(valid_results.items(), key=lambda x: x[1]['mse'])
        best_mae = min(valid_results.items(), key=lambda x: x[1]['mae'])
        best_directional = max(valid_results.items(), key=lambda x: x[1]['directional_accuracy'])
        best_correlation = max(valid_results.items(), key=lambda x: abs(x[1]['correlation']))
        
        # Calculate improvement percentages
        mse_values = [results['mse'] for results in valid_results.values()]
        baseline_mse = max(mse_values)  # Assume worst MSE is baseline
        
        summary = {
            'best_performers': {
                'mse': {'model': best_mse[0], 'value': best_mse[1]['mse']},
                'mae': {'model': best_mae[0], 'value': best_mae[1]['mae']},
                'directional_accuracy': {'model': best_directional[0], 'value': best_directional[1]['directional_accuracy']},
                'correlation': {'model': best_correlation[0], 'value': best_correlation[1]['correlation']}
            },
            'performance_rankings': self._rank_models(valid_results),
            'improvement_analysis': self._calculate_improvements(valid_results),
            'statistical_tests': self._run_statistical_tests(valid_results)
        }
        
        return summary
    
    def _rank_models(self, results: Dict) -> Dict[str, List]:
        """Rank models by different metrics"""
        
        models = list(results.keys())
        
        # Rank by MSE (lower is better)
        mse_ranking = sorted(models, key=lambda m: results[m]['mse'])
        
        # Rank by directional accuracy (higher is better)
        directional_ranking = sorted(models, key=lambda m: results[m]['directional_accuracy'], reverse=True)
        
        # Combined ranking (simple average of ranks)
        combined_scores = {}
        for model in models:
            mse_rank = mse_ranking.index(model) + 1
            dir_rank = directional_ranking.index(model) + 1
            combined_scores[model] = (mse_rank + dir_rank) / 2
        
        combined_ranking = sorted(models, key=lambda m: combined_scores[m])
        
        return {
            'by_mse': mse_ranking,
            'by_directional_accuracy': directional_ranking,
            'combined': combined_ranking,
            'combined_scores': combined_scores
        }
    
    def _calculate_improvements(self, results: Dict) -> Dict[str, Any]:
        """Calculate improvement percentages relative to worst performer"""
        
        mse_values = {name: res['mse'] for name, res in results.items()}
        mae_values = {name: res['mae'] for name, res in results.items()}
        dir_values = {name: res['directional_accuracy'] for name, res in results.items()}
        
        worst_mse = max(mse_values.values())
        worst_mae = max(mae_values.values())
        worst_dir = min(dir_values.values())
        
        improvements = {}
        for name in results.keys():
            improvements[name] = {
                'mse_improvement': (worst_mse - mse_values[name]) / worst_mse * 100,
                'mae_improvement': (worst_mae - mae_values[name]) / worst_mae * 100,
                'directional_improvement': (dir_values[name] - worst_dir) * 100
            }
        
        return improvements
    
    def _run_statistical_tests(self, results: Dict) -> Dict[str, Any]:
        """Run statistical significance tests between models"""
        
        # For now, return placeholder - would implement proper statistical tests
        return {
            'note': 'Statistical significance testing requires more sophisticated implementation',
            'tests_available': ['wilcoxon', 'paired_t_test', 'diebold_mariano'],
            'implementation_status': 'placeholder'
        }
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive comparison report"""
        
        if not self.test_results:
            return "No test results available. Run side-by-side test first."
        
        report = []
        report.append("=" * 60)
        report.append("FINANCIAL LNN MODEL COMPARISON REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Models tested: {len(self.test_results)}")
        report.append("")
        
        # Performance summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 30)
        
        for model_name, results in self.test_results.items():
            if 'error' in results:
                report.append(f"{model_name}: ERROR - {results['error']}")
                continue
                
            report.append(f"\n{model_name}:")
            report.append(f"  MSE: {results['mse']:.6f}")
            report.append(f"  MAE: {results['mae']:.6f}")
            report.append(f"  Directional Accuracy: {results['directional_accuracy']:.2%}")
            report.append(f"  Correlation: {results['correlation']:.4f}")
            report.append(f"  Predictions made: {results['total_predictions']}")
        
        # Add recommendations
        if len(self.comparison_history) > 0:
            latest_summary = self.comparison_history[-1]['summary']
            if 'best_performers' in latest_summary:
                report.append("\n\nRECOMMENDATIONS")
                report.append("-" * 30)
                
                best_overall = latest_summary['performance_rankings']['combined'][0]
                report.append(f"Best overall performer: {best_overall}")
                
                best_mse = latest_summary['best_performers']['mse']['model']
                report.append(f"Most accurate (MSE): {best_mse}")
                
                best_directional = latest_summary['best_performers']['directional_accuracy']['model']
                report.append(f"Best directional accuracy: {best_directional}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")
        
        return report_text

# Example usage and comprehensive demonstration
def create_baseline_models():
    """Create simple baseline models for comparison"""
    
    def simple_momentum_model(price_feat, volume_feat, vol_feat):
        """Simple momentum-based model"""
        if len(price_feat) >= 3:
            recent_momentum = np.mean(price_feat[-3:])
            return recent_momentum * 0.8  # Assume some mean reversion
        return 0.0
    
    def volume_weighted_model(price_feat, volume_feat, vol_feat):
        """Volume-weighted momentum model"""
        if len(price_feat) >= 2 and len(volume_feat) >= 2:
            price_momentum = price_feat[-1]
            volume_factor = np.tanh(volume_feat[-1] - 1.0)  # Normalized volume
            return price_momentum * (1 + 0.3 * volume_factor)
        return 0.0
    
    def volatility_adjusted_model(price_feat, volume_feat, vol_feat):
        """Volatility-adjusted prediction model"""
        if len(price_feat) >= 2 and len(vol_feat) >= 2:
            price_momentum = price_feat[-1]
            vol_adjustment = 1.0 / (1.0 + vol_feat[-1] * 5)  # Reduce prediction in high vol
            return price_momentum * vol_adjustment
        return 0.0
    
    return {
        'Simple_Momentum': simple_momentum_model,
        'Volume_Weighted': volume_weighted_model,
        'Volatility_Adjusted': volatility_adjusted_model
    }

if __name__ == "__main__":
    print("Enhanced Financial LNN with Comprehensive Comparison Framework")
    print("=" * 70)
    
    # Create enhanced sample financial data
    np.random.seed(42)
    n_days = 300  # More data for better testing
    
    # Simulate more realistic financial data with regime changes
    returns = []
    volatility = 0.02
    momentum = 0.0
    regime = 0  # 0=normal, 1=high_vol, 2=trending
    
    for i in range(n_days):
        # Regime switching
        if i % 50 == 0:  # Change regime every 50 days
            regime = np.random.choice([0, 1, 2])
        
        # Regime-dependent parameters
        if regime == 0:  # Normal market
            vol_target, momentum_persistence = 0.015, 0.9
        elif regime == 1:  # High volatility
            vol_target, momentum_persistence = 0.035, 0.7
        else:  # Trending market
            vol_target, momentum_persistence = 0.020, 0.95
        
        # Volatility clustering
        volatility = 0.8 * volatility + 0.2 * vol_target + 0.1 * vol_target * np.random.randn()
        volatility = max(0.005, volatility)  # Minimum volatility
        
        # Momentum with regime-dependent persistence
        momentum = momentum_persistence * momentum + 0.1 * np.random.randn() * 0.01
        
        # Generate return
        daily_return = momentum + volatility * np.random.randn()
        returns.append(daily_return)
    
    # Create realistic price and volume data
    prices = 100 * np.cumprod(1 + np.array(returns))
    
    # Volume correlation with volatility and price movements
    base_volume = 1000000
    volumes = []
    for i, ret in enumerate(returns):
        vol_boost = 1 + 2 * abs(ret) / 0.02  # Higher volume on big moves
        noise = 1 + 0.4 * np.random.randn()
        volume = base_volume * vol_boost * max(0.1, noise)
        volumes.append(volume)
    
    # Create comprehensive dataset
    financial_data = pd.DataFrame({
        'Close': prices,
        'Volume': volumes
    })
    
    print(f"Created financial dataset with {len(financial_data)} observations")
    print(f"Price range: ${financial_data['Close'].min():.2f} - ${financial_data['Close'].max():.2f}")
    print(f"Average daily return: {np.mean(returns):.4f}")
    print(f"Return volatility: {np.std(returns):.4f}")
    
    # Initialize comparison framework
    print("\nInitializing comparison framework...")
    comparison_framework = ModelComparisonFramework()
    
    # Add Enhanced Financial LNN models with different configurations
    print("Creating Enhanced Financial LNN models...")
    
    # Standard adaptive model
    lnn_adaptive = EnhancedFinancialLNN(
        input_size=6,  # Enhanced features
        hidden_sizes=[12, 8], 
        output_size=1,
        adaptive=True,
        name="Enhanced_LNN_Adaptive"
    )
    comparison_framework.add_model("Enhanced_LNN_Adaptive", lnn_adaptive)
    
    # Non-adaptive model for comparison
    lnn_static = EnhancedFinancialLNN(
        input_size=6, 
        hidden_sizes=[12, 8], 
        output_size=1,
        adaptive=False,
        name="Enhanced_LNN_Static"
    )
    comparison_framework.add_model("Enhanced_LNN_Static", lnn_static)
    
    # Smaller model
    lnn_small = EnhancedFinancialLNN(
        input_size=6,
        hidden_sizes=[6, 4],
        output_size=1,
        adaptive=True,
        name="Enhanced_LNN_Small"
    )
    comparison_framework.add_model("Enhanced_LNN_Small", lnn_small)
    
    # Add baseline models
    print("Adding baseline models...")
    baseline_models = create_baseline_models()
    for name, model in baseline_models.items():
        comparison_framework.add_model(name, model)
    
    # Run comprehensive side-by-side testing
    print("\nRunning comprehensive side-by-side testing...")
    test_results = comparison_framework.run_side_by_side_test(
        financial_data, 
        test_window=80,  # Test on last 80 data points
        prediction_horizon=1
    )
    
    # Display results
    print("\n" + "="*70)
    print("DETAILED TEST RESULTS")
    print("="*70)
    
    for model_name, results in test_results['detailed_results'].items():
        if 'error' in results:
            print(f"\n{model_name}: ERROR - {results['error']}")
            continue
            
        print(f"\n{model_name}:")
        print(f"  Mean Squared Error: {results['mse']:.6f}")
        print(f"  Mean Absolute Error: {results['mae']:.6f}")
        print(f"  Directional Accuracy: {results['directional_accuracy']:.2%}")
        print(f"  Correlation with actual: {results['correlation']:.4f}")
        print(f"  Total predictions: {results['total_predictions']}")
    
    # Show summary and recommendations
    if 'summary' in test_results:
        summary = test_results['summary']
        
        print("\n" + "="*70)
        print("PERFORMANCE RANKINGS AND RECOMMENDATIONS")
        print("="*70)
        
        if 'performance_rankings' in summary:
            rankings = summary['performance_rankings']
            print("\nOverall Performance Ranking (Combined Score):")
            for i, model in enumerate(rankings['combined'], 1):
                score = rankings['combined_scores'][model]
                print(f"  {i}. {model} (score: {score:.2f})")
        
        if 'best_performers' in summary:
            best = summary['best_performers']
            print(f"\nBest Performers by Metric:")
            print(f"  Lowest MSE: {best['mse']['model']} ({best['mse']['value']:.6f})")
            print(f"  Highest Directional Accuracy: {best['directional_accuracy']['model']} ({best['directional_accuracy']['value']:.2%})")
            print(f"  Highest Correlation: {best['correlation']['model']} ({best['correlation']['value']:.4f})")
    
    # Generate and display comprehensive report
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*70)
    
    report = comparison_framework.generate_report()
    print(report)
    
    # Performance tracking demonstration
    print("\n" + "="*70)
    print("PERFORMANCE TRACKING OVER ITERATIONS")
    print("="*70)
    
    # Show performance metrics for the best LNN model
    best_lnn_name = None
    best_lnn_model = None
    
    for name, model in comparison_framework.models.items():
        if hasattr(model, 'get_performance_summary') and 'Enhanced_LNN' in name:
            perf_summary = model.get_performance_summary()
            if perf_summary.get('total_predictions', 0) > 0:
                print(f"\n{name} Performance Summary:")
                print(f"  Average MSE: {perf_summary.get('avg_mse', 0):.6f}")
                print(f"  Average MAE: {perf_summary.get('avg_mae', 0):.6f}")
                print(f"  Average Confidence: {perf_summary.get('avg_confidence', 0):.4f}")
                print(f"  Total Predictions: {perf_summary.get('total_predictions', 0)}")
                
                if best_lnn_name is None:
                    best_lnn_name = name
                    best_lnn_model = model
    
    # Scaling opportunities and recommendations
    print("\n" + "="*70)
    print("SCALING OPPORTUNITIES AND NEXT STEPS")
    print("="*70)
    
    print("\nImmediate Improvements:")
    print("1. Fine-tune neuron type distributions based on market regime detection")
    print("2. Implement online learning for adaptive parameter updates")
    print("3. Add ensemble methods combining multiple LNN variants")
    print("4. Incorporate additional financial features (options data, sentiment)")
    
    print("\nScaling Opportunities:")
    print("1. Multi-asset portfolio optimization using LNN predictions")
    print("2. Real-time trading system integration")
    print("3. Risk management system with volatility forecasting")
    print("4. Market microstructure modeling for high-frequency trading")
    
    print("\nArchitecture Enhancements:")
    print("1. Hierarchical LNN with multiple timescales")
    print("2. Attention mechanisms for feature importance")
    print("3. Meta-learning for rapid adaptation to new market regimes")
    print("4. Uncertainty quantification for better risk assessment")
    
    print(f"\nFramework ready for integration with your existing architecture!")
    print(f"Total models tested: {len(comparison_framework.models)}")
    print(f"Test results stored and ready for further analysis.")
